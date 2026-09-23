"""
Task 13.2 — Validation campaign: sweep the mode index m (per-mode accuracy at fixed c).

The m-resolved companion to the c-sweep (13.1). At fixed (n=0, c) the pipeline eigenpairs are compared to
the Task 10.4 benchmark mode-by-mode, exposing the plateau -> plunge -> tail structure vs m. Per mode:
  * EIGENVALUE error   |lam_m - lam_ref_m|,
  * INDIVIDUAL eigenfunction weighted-L2 error ||psi_N^{(m)} - psi_ref^{(m)}|| (sign-aligned),
  * a REGIME label from the reference eigenvalue:
       deep-plateau (lam_ref > 1-1e-6, degenerate),  separated (1e-3 <= lam_ref <= 1-1e-6),
       tail (lam_ref < 1e-3, dynamically irrelevant).
Plus the CONCENTRATED-SUBSPACE principal angle: the leading K=#{lam_ref>1-1e-6} modes (the degenerate
plateau core) compared as a subspace -- the correct metric where individual modes are interchangeable.

Key m-dependence findings:
  (1) EIGENVALUE error is a smooth monotone rise: machine in the deep plateau, ~1/P at the plunge knee,
      then small again in the tail (lam~0).
  (2) INDIVIDUAL eigenfunction L2 error is BIMODAL and only meaningful in the SEPARATED band: it is large
      in the deep-degenerate plateau (individual vectors rotate -> use the subspace) AND in the tail
      (lam~0, irrelevant), but small in the spectrally-separated shoulder, rising ~1/P into the plunge.
  (3) The concentrated plateau, taken as a SUBSPACE, is captured to a small principal angle (~1e-2);
      including the plunge shoulder (K=M) inflates it to ~1e-1 (those modes are ~1/P).

Weighted-L2 measure: x_gl * w_gl (benchmark quadrature). Scope: the m-profile at fixed c. The full
5-metric table is 13.4; convergence in N is 13.5.
"""
import os
import numpy as np
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize
from task_12_4_synthesis import synthesize_nd

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "data", "benchmark")
CAMPAIGN = os.path.join(REPO, "data", "campaign")


def subspace_sin(psiN, psi_ref, mu, K):
    """sin of the largest principal angle between the leading-K subspaces (orthonormal in mu)."""
    G = (psiN[:, :K] * mu[:, None]).T @ psi_ref[:, :K]
    sig = np.linalg.svd(G, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - min(sig) ** 2)))


def m_profile(n, c, buffer=20):
    d = np.load(os.path.join(BENCH, f"cpswf_n{n}_c{int(c)}.npz"))
    xg, wg, psi_ref, lam_ref = d["x_gl"], d["w_gl"], d["psi_gl"], d["lam"]
    nb = int(d["nmodes"]); M = int(d["Mn"]); mu = xg * wg
    P = M + buffer
    jz = jn_zeros(n, P)
    B = assemble(n, c, P=P, G=32)
    r = solve(B, "evr")
    lam, V = canonicalize(r["lam"], r["V"], n, jz)
    nm = min(nb, P)
    psiN = synthesize_nd(n, jz, V[:, :nm], xg)

    rows = []
    for m in range(nm):
        lr = lam_ref[m]
        pr = psi_ref[:, m]
        pn = psiN[:, m] * np.sign(np.sum(psiN[:, m] * pr * mu))
        l2 = float(np.sqrt(max(0.0, np.sum((pn - pr) ** 2 * mu))))
        reg = ("deep-plateau" if lr > 1 - 1e-6 else ("separated" if lr >= 1e-3 else "tail"))
        rows.append(dict(m=m, lam=float(lam[m]), eig_err=float(abs(lam[m] - lr)), l2=l2, regime=reg))

    Kcore = int((lam_ref > 1 - 1e-6).sum())                # degenerate plateau core
    return dict(M=M, P=P, nm=nm, Kcore=Kcore, rows=rows, mu_ok=True,
                sub_core=subspace_sin(psiN, psi_ref, mu, Kcore),
                sub_full=subspace_sin(psiN, psi_ref, mu, min(M, nm)))


def run(save=True):
    n = 0
    results = {}
    for c in [40.0, 80.0]:
        pr = m_profile(n, c)
        results[c] = pr
        print("=" * 92)
        print(f"TASK 13.2 -- m-sweep at n={n}, c={c:.0f}  (M={pr['M']}, P={pr['P']}, plateau core Kcore={pr['Kcore']})")
        print("=" * 92)
        print(f"  {'m':>3s} {'lam_m':>10s} {'eig err':>10s} {'indiv L2':>10s}  regime")
        for r in pr["rows"]:
            print(f"  {r['m']:3d} {r['lam']:10.6f} {r['eig_err']:10.2e} {r['l2']:10.2e}  {r['regime']}")
        print(f"  concentrated-subspace sin (K={pr['Kcore']}, lam>1-1e-6) = {pr['sub_core']:.2e}   "
              f"| full leading-M subspace sin = {pr['sub_full']:.2e} (plunge-inflated)")
        print()

    ok = True
    for c, pr in results.items():
        rows = pr["rows"]; Kc = pr["Kcore"]
        deep_eig = max(r["eig_err"] for r in rows if r["regime"] == "deep-plateau")
        # separated-band individual L2: the well-separated shoulder modes (just past the core) are small
        sep = [r for r in rows if r["regime"] == "separated"]
        sep_shoulder_small = min((r["l2"] for r in sep[:2]), default=1.0) < 5e-2   # first separated modes
        deep_indiv_large = max((r["l2"] for r in rows if r["regime"] == "deep-plateau"), default=0) > 0.3
        core_ok = pr["sub_core"] < 5e-2
        print(f"  c={c:.0f}: deep-plateau eig machine={deep_eig:.1e}(<1e-6)  "
              f"shoulder indiv L2 small={sep_shoulder_small}  deep indiv L2 large(rotation)={deep_indiv_large}  "
              f"core-subspace sin={pr['sub_core']:.1e}(<5e-2={core_ok})")
        ok = ok and (deep_eig < 1e-6) and sep_shoulder_small and deep_indiv_large and core_ok

    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(
            os.path.join(CAMPAIGN, "task_13_2_m_sweep.npz"),
            **{f"c{int(c)}_m": np.array([r["m"] for r in pr["rows"]]) for c, pr in results.items()},
            **{f"c{int(c)}_lam": np.array([r["lam"] for r in pr["rows"]]) for c, pr in results.items()},
            **{f"c{int(c)}_eigerr": np.array([r["eig_err"] for r in pr["rows"]]) for c, pr in results.items()},
            **{f"c{int(c)}_l2": np.array([r["l2"] for r in pr["rows"]]) for c, pr in results.items()},
            **{f"c{int(c)}_subcore": np.array(pr["sub_core"]) for c, pr in results.items()},
        )
        print(f"\n  saved: data/campaign/task_13_2_m_sweep.npz")

    print(f"\n  TASK 13.2 m-SWEEP VERIFIED: {ok}")
    return results


if __name__ == "__main__":
    run()
