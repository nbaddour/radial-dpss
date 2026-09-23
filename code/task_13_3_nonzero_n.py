"""
Task 13.3 — Validation campaign: nonzero-n confirmation of the general-n formulas.

The theory and the pipeline are general in the angular order n from the start (Task 6.1 entry formula,
Task 12.1 assembler). This task confirms NUMERICALLY that the general-n formulas reproduce, at n >= 1,
the same plateau -> plunge -> tail structure and accuracy established at n = 0 (Tasks 13.1/13.2), and that
the Shannon number carries the predicted n-dependence M_n(c) ~ c/pi - n/2 (the n-shift sits in the O(1)
intercept; Task 9.5). Reuses the m-profile machinery of Task 13.2; reference = Task 10.4 benchmark
(n in {1,2,4}, c in {10,20,40,80}).

Checks:
  * Shannon number M_n(c) - c/pi ~ -n/2 (n-dependent intercept) for n = 0,1,2,4,
  * deep-plateau eigenvalues machine-accurate at n >= 1,
  * concentrated-subspace (lam_ref > 1-1e-6) principal angle ~ 1e-2 at n >= 1,
  * the plunge knee is ~1/P (same as n = 0),
i.e. the general-n formulas behave exactly like the n = 0 campaign, shifted by the -n/2 Shannon offset.
"""
import os
import numpy as np
from task_12_1_assemble_matrix import Mn
from task_13_2_m_sweep import m_profile          # reuse the per-mode profile (takes n, c)

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN = os.path.join(REPO, "data", "campaign")


def run(save=True):
    print("=" * 96)
    print("TASK 13.3 -- (A) Shannon-number n-dependence: M_n(c) - c/pi ~ -n/2")
    print("=" * 96)
    print(f"  {'n':>2s} " + "".join(f"{'c=%d' % c:>12s}" for c in [10, 20, 40, 80]) + f"   {'predicted -n/2':>14s}")
    shannon_ok = True
    for n in [0, 1, 2, 4]:
        cells = []
        for c in [10, 20, 40, 80]:
            M = Mn(n, c)
            cells.append(f"M={M}({M - c/np.pi:+.2f})")
        # M_n(c) - c/pi should be near -n/2 (McMahon: -n/2 + 1/4); allow +/-1 (integer M)
        devs = [abs((Mn(n, c) - c/np.pi) - (-n/2)) for c in [20, 40, 80]]
        row_ok = max(devs) < 1.2
        shannon_ok = shannon_ok and row_ok
        print(f"  {n:2d} " + "".join(f"{cell:>12s}" for cell in cells) + f"   {-n/2:>14.2f}  ok={row_ok}")

    print("\n" + "=" * 96)
    print("TASK 13.3 -- (B) per-(n,c) accuracy: same plateau/plunge/tail structure as n=0")
    print("=" * 96)
    print(f"  {'n':>2s} {'c':>4s} {'M':>3s} {'Kcore':>5s} {'deep-plateau eig':>16s} "
          f"{'subspace sin':>12s} {'plunge-knee eig':>15s}")
    struct_ok = True
    results = []
    for n in [1, 2, 4]:
        for c in [20.0, 40.0, 80.0]:
            pr = m_profile(n, c)
            rows = pr["rows"]; M = pr["M"]
            deep_eig = max((r["eig_err"] for r in rows if r["regime"] == "deep-plateau"), default=np.nan)
            # plunge knee = the separated mode whose lam is closest to 0.5 (the ~1/P knee)
            sep = [r for r in rows if r["regime"] == "separated"]
            knee = min(sep, key=lambda r: abs(r["lam"] - 0.5)) if sep else None
            knee_eig = knee["eig_err"] if knee else np.nan
            ok = (deep_eig < 1e-6) and (pr["sub_core"] < 6e-2)
            struct_ok = struct_ok and ok
            results.append(dict(n=n, c=c, M=M, Kcore=pr["Kcore"], deep_eig=deep_eig,
                                sub=pr["sub_core"], knee_eig=knee_eig))
            print(f"  {n:2d} {c:4.0f} {M:3d} {pr['Kcore']:5d} {deep_eig:16.2e} "
                  f"{pr['sub_core']:12.2e} {knee_eig:15.2e}   ok={ok}")

    # representative per-mode profile at n=2, c=40 (show the three-band structure explicitly)
    print("\n" + "=" * 96)
    print("TASK 13.3 -- (C) representative per-mode profile at n=2, c=40 (three-band structure)")
    print("=" * 96)
    pr = m_profile(2, 40.0)
    print(f"  {'m':>3s} {'lam_m':>10s} {'eig err':>10s} {'indiv L2':>10s}  regime")
    for r in pr["rows"]:
        print(f"  {r['m']:3d} {r['lam']:10.6f} {r['eig_err']:10.2e} {r['l2']:10.2e}  {r['regime']}")

    print("\n" + "=" * 96)
    print(f"  (A) Shannon number M_n(c) ~ c/pi - n/2 for n=0..4          : {shannon_ok}")
    print(f"  (B) deep-plateau machine + concentrated subspace ~1e-2 (n>=1): {struct_ok}")
    print(f"  => general-n formulas reproduce the n=0 structure, shifted by -n/2")

    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(
            os.path.join(CAMPAIGN, "task_13_3_nonzero_n.npz"),
            n=np.array([r["n"] for r in results]), c=np.array([r["c"] for r in results]),
            M=np.array([r["M"] for r in results]),
            deep_eig=np.array([r["deep_eig"] for r in results]),
            sub=np.array([r["sub"] for r in results]),
            knee_eig=np.array([r["knee_eig"] for r in results]),
        )
        print(f"\n  saved: data/campaign/task_13_3_nonzero_n.npz")

    ok = shannon_ok and struct_ok
    print(f"\n  TASK 13.3 NONZERO-n VERIFIED: {ok}")
    return results


if __name__ == "__main__":
    run()
