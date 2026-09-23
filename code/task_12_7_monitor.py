"""
Task 12.7 — Conditioning monitor and Nystrom fallback trigger.

Aggregates the per-config numerical-health diagnostics from the Stage-5 pipeline (Tasks 12.1-12.6) into a
single standing monitor, and decides whether to activate the LIVE FALLBACK (the Task 10 Nystrom/quadrature
reference solver; locked decision, kept as benchmark/cross-check by Gate C, promoted to primary ONLY if the
FB-Galerkin path breaks down).

KEY DISTINCTION (the honest core of this task):
  * BENIGN ill-conditioning -- intrinsic to the prolate problem, NOT a trigger:
      - kappa(B) = lam_max/lam_min is astronomically large because the TAIL eigenvalues decay
        super-exponentially to 0 (Task 6.6); the tail modes are irrelevant (lam ~ 0), so this does
        not affect the leading/informative modes.
      - deep-plateau degeneracy (successive gaps ~1e-15): individual vectors rotate, but the leading
        SUBSPACE is well-conditioned (Task 12.2) -- benign.
      - kappa(Phi~) (sampling matrix) grows with n,P (Task 12.4): harmless; the PRIMARY coefficient
        route does not use Phi~.
      - nodal W-Gram defect eps_N (Task 12.6): a quadrature residual in the sample domain only.
  * MALIGN indicators -- signal actual numerical breakdown, TRIGGER the fallback:
      - assembly route-A/B disagreement (partial-fraction vs raw double-pole)   [Task 12.1]
      - quadrature non-convergence (G vs 2G)                                    [Task 12.1]
      - eigenvalues escaping [0,1] beyond tolerance                            [Task 12.2]
      - eigenpair residual ||Bv - lam v|| growth                              [Task 12.2]
      - loss of eigenvector orthonormality ||V^T V - I||                       [Task 12.2]

If any MALIGN indicator exceeds its threshold, fallback = True and the monitor reports the reference
solver's required node count Nq = ceil(c/2)+30 (Task 10.3). Otherwise the FB-Galerkin path stands.

Scope: the monitor + trigger. Mode tracking is Task 12.8; version-controlled tests are 12.9.
"""
import os
import numpy as np
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, assemble_routeA, Mn
from task_12_2_eigensolver import solve, degenerate_clusters
from task_12_4_synthesis import sampling_matrix_nd

# REPO root: parent of this code/ directory (override with env RDPSS_REPO if running out of tree).
REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MALIGN thresholds (well above observed-healthy ~1e-13; generous margin before crying wolf)
TOL = dict(route_ab=1e-10, quad_conv=1e-9, eig_excursion=1e-9, residual=1e-10, ortho=1e-10)


def required_Nq(c):
    """Reference-solver node count if fallback is activated (Task 10.3)."""
    return int(np.ceil(c / 2) + 30)


def monitor(n, c, P, G=24, B_override=None):
    """Compute the conditioning report for one config. If B_override is given, skip assembly checks
    (used to feed a deliberately-broken matrix through the eigensolver/malign checks)."""
    jz = jn_zeros(n, P)
    rep = dict(n=n, c=c, P=P, malign={}, benign={})

    if B_override is None:
        B, cache = assemble(n, c, P=P, G=G, return_cache=True)
        B2 = assemble(n, c, P=P, G=2 * G)                 # quadrature convergence G vs 2G
        BA = assemble_routeA(n, c, P, cache)              # independent assembly route
        rep["malign"]["route_ab"] = float(np.max(np.abs(B - BA)))
        rep["malign"]["quad_conv"] = float(np.max(np.abs(B - B2)))
    else:
        B = B_override
        rep["malign"]["route_ab"] = 0.0
        rep["malign"]["quad_conv"] = 0.0

    r = solve(B, "evr")
    lam = r["lam"]
    rep["malign"]["eig_excursion"] = max(-min(lam.min(), 0.0), max(lam.max() - 1.0, 0.0))
    rep["malign"]["residual"] = r["resid"]
    rep["malign"]["ortho"] = r["ortho"]

    # BENIGN diagnostics (reported, not gated)
    lam_pos = lam[lam > 1e-300]
    rep["benign"]["kappa_B"] = float(lam.max() / lam_pos.min()) if lam_pos.size else np.inf
    M = Mn(n, c)
    rep["benign"]["M"] = M
    # leading (informative) subspace: conditioning over leading modes, and plateau->plunge gap
    Klead = int((lam > 0.5).sum())
    rep["benign"]["kappa_lead"] = float(lam[0] / lam[Klead - 1]) if Klead >= 1 else np.inf
    rep["benign"]["plateau_plunge_gap"] = float(lam[M - 1] - lam[M]) if 0 < M < len(lam) else np.nan
    clusters = degenerate_clusters(lam, rel_tol=1e-8)
    rep["benign"]["deep_cluster_size"] = max((hi - lo) for (lo, hi) in clusters)
    if B_override is None:
        Phi = sampling_matrix_nd(n, jz, c)
        rep["benign"]["kappa_Phi"] = float(np.linalg.cond(Phi))

    # decision
    fired = {k: v for k, v in rep["malign"].items() if not (v <= TOL[k])}  # not(<=) also catches NaN
    rep["fallback"] = len(fired) > 0
    rep["fired"] = fired
    rep["required_Nq"] = required_Nq(c) if rep["fallback"] else None
    return rep


def _fmt(rep):
    m = rep["malign"]; b = rep["benign"]
    status = "FALLBACK->Nystrom(Nq=%d)" % rep["required_Nq"] if rep["fallback"] else "OK (FB-Galerkin)"
    return (f"  n={rep['n']} c={rep['c']:5.1f} P={rep['P']:3d} M={b['M']:2d}: {status}\n"
            f"     MALIGN  routeAB={m['route_ab']:.1e} quad={m['quad_conv']:.1e} "
            f"eig[0,1]={m['eig_excursion']:.1e} resid={m['residual']:.1e} ortho={m['ortho']:.1e}\n"
            f"     BENIGN  kappa(B)={b['kappa_B']:.1e} kappa_lead={b['kappa_lead']:.1e} "
            f"plateau->plunge gap={b['plateau_plunge_gap']:.2f} "
            f"deep-cluster={b['deep_cluster_size']}"
            + (f" kappa(Phi)={b['kappa_Phi']:.1e}" if 'kappa_Phi' in b else ""))


# ---- verification ------------------------------------------------------------------------------------
def verify():
    print("=" * 100)
    print("TASK 12.7 -- conditioning monitor + Nystrom fallback trigger")
    print("=" * 100)

    print("\n[A] Campaign-range sweep: MALIGN indicators healthy -> NO fallback; kappa(B) benign-large")
    any_fallback = False
    worst_malign = 0.0
    for (n, c, P) in [(0,20.,44),(0,40.,56),(1,40.,48),(2,40.,48),(4,40.,50),(4,80.,90),(2,120.,110)]:
        rep = monitor(n, c, P)
        any_fallback = any_fallback or rep["fallback"]
        worst_malign = max(worst_malign, max(rep["malign"].values()))
        print(_fmt(rep))

    print("\n[B] Benign vs malign separation: kappa(B) is huge (tail lam->0) yet the pipeline is healthy")
    rep = monitor(0, 40., 56)
    print(f"  n=0 c=40 P=56: kappa(B) = {rep['benign']['kappa_B']:.2e} (BENIGN, tail-driven) "
          f"but all MALIGN < {max(rep['malign'].values()):.1e}, leading kappa = {rep['benign']['kappa_lead']:.1e}")

    print("\n[C] Trigger discrimination: a deliberately-broken matrix MUST fire the fallback")
    jz = jn_zeros(0, 44); B = assemble(0, 20., P=44, G=32)
    Bbad = B.copy(); Bbad[0, 0] += 0.5             # pushes lam_max > 1 (eig excursion)
    repbad = monitor(0, 20., 44, B_override=Bbad)
    print(_fmt(repbad))
    fired_ok = repbad["fallback"] and "eig_excursion" in repbad["fired"]

    Bnan = B.copy(); Bnan[2, 3] = np.nan
    try:
        repnan = monitor(0, 20., 44, B_override=Bnan)
        nan_caught = repnan["fallback"] or not np.isfinite(repnan["malign"]["residual"])
    except Exception:
        nan_caught = True

    print("\n" + "=" * 100)
    print(f"  campaign sweep: any fallback triggered = {any_fallback}  (expected False)")
    print(f"  campaign worst MALIGN indicator        = {worst_malign:.2e}  (all < thresholds)")
    print(f"  broken-matrix fires fallback           = {fired_ok}  (expected True)")
    print(f"  NaN-injected matrix caught             = {nan_caught}  (expected True)")
    ok = (not any_fallback) and worst_malign < 1e-10 and fired_ok and nan_caught
    print(f"\n  TASK 12.7 CONDITIONING MONITOR VERIFIED: {ok}")


if __name__ == "__main__":
    verify()
