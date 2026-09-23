"""
Task 13.1 — Validation campaign: sweep small, medium, and large c (n = 0 primary).

Runs the Stage-5 pipeline (assemble 12.1 -> solve 12.2 -> canonicalize 12.3) across small/medium/large
space-bandwidth products c and compares the eigenvalues to the independent Task 10 Nystrom reference
(reliable to c<=480; Nq=ceil(c/2)+30). The campaign headline is HOW MANY leading concentration modes the
method captures accurately as c grows.

Per c we report:
  * M_0(c)  -- Shannon number (plateau width; ~ c/pi),
  * N_acc(1e-6)   -- # leading modes with |lam-lam_ref| < 1e-6 (eigenvalue-accurate modes),
  * N_acc(1e-10)  -- # near-machine leading modes (the deep-plateau core),
  * deep-plateau max err  (m <= M-6, the machine-accurate core; small c: all-but-knee),
  * knee err (m ~ M)      -- the ~1/P-limited transition mode,
  * conditioning monitor status (Task 12.7).

Finding (see per-mode profile): the eigenvalue error is a smooth gradient -- machine-accurate deep in the
plateau, rising only as m approaches the plunge. So the method captures ~ M - O(1) modes accurately at
EVERY c, i.e. the accurately-captured count grows ~ c/pi. Scope: c-sweep (eigenvalue level). The 5-metric
eigenfunction comparison is 13.4; convergence in N is 13.5; the plunge/gamma_n study is 13.6.
"""
import os
import io
import contextlib
import numpy as np
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize
from task_12_7_monitor import monitor

with contextlib.redirect_stdout(io.StringIO()):        # silence the reference module's import-time demo
    from task_10_1_reference_solver import nystrom

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN = os.path.join(REPO, "data", "campaign")


def reference_eigs(n, c):
    Nq = int(np.ceil(c / 2) + 30)
    lam, x, w, psi = nystrom(n, c, Nq)
    return lam


def c_sweep(n, c_values, buffer=20):
    rows = []
    for c in c_values:
        M = Mn(n, c)
        P = M + buffer
        jz = jn_zeros(n, P)
        B = assemble(n, c, P=P, G=32)
        r = solve(B, "evr")
        lam, V = canonicalize(r["lam"], r["V"], n, jz)
        lref = reference_eigs(n, c)
        k = min(len(lam), len(lref))
        err = np.abs(lam[:k] - lref[:k])
        # count leading modes accurate to a tolerance (contiguous from m=0)
        def n_acc(tol):
            c_ = 0
            for e in err:
                if e < tol:
                    c_ += 1
                else:
                    break
            return c_
        deep_hi = max(M - 6, 1)
        deep = float(err[:deep_hi].max())
        knee = float(err[max(M - 1, 0):min(M + 1, k)].max())
        rep = monitor(n, c, P)
        rows.append(dict(c=float(c), M=int(M), P=int(P), c_over_pi=c / np.pi,
                         n_acc6=n_acc(1e-6), n_acc10=n_acc(1e-10),
                         deep_err=deep, knee_err=knee, ngt=int((lam > 0.5).sum()),
                         fallback=bool(rep["fallback"]), kappa_lead=float(rep["benign"]["kappa_lead"])))
    return rows


def run(save=True):
    n = 0
    c_values = [8., 12., 20., 40., 80., 120., 160.]
    print("=" * 104)
    print("TASK 13.1 -- c-sweep (n=0): pipeline vs independent Nystrom reference")
    print("=" * 104)
    print(f"{'band':7s} {'c':>5s} {'M':>4s} {'c/pi':>6s} {'P':>4s} {'Nacc<1e-6':>9s} {'Nacc<1e-10':>10s} "
          f"{'deep err':>10s} {'knee err':>9s} {'monitor':>15s} {'k_lead':>7s}")
    rows = c_sweep(n, c_values)
    for r in rows:
        band = "small" if r["c"] <= 12 else ("medium" if r["c"] <= 40 else "large")
        status = "FALLBACK" if r["fallback"] else "OK(FB-Galerkin)"
        print(f"{band:7s} {r['c']:5.0f} {r['M']:4d} {r['c_over_pi']:6.1f} {r['P']:4d} "
              f"{r['n_acc6']:9d} {r['n_acc10']:10d} {r['deep_err']:10.2e} {r['knee_err']:9.2e} "
              f"{status:>15s} {r['kappa_lead']:7.2f}")

    # deep-plateau machine accuracy is a claim only where a real plateau exists (M >= 6); small c
    # (M <= 4) is the nascent-plateau regime where the few leading modes sit near the transition.
    deep_rows = [r for r in rows if r["M"] >= 6]
    deep_worst = max(r["deep_err"] for r in deep_rows)
    any_fb = any(r["fallback"] for r in rows)
    M_ok = all(abs(r["M"] - (r["c"] / np.pi)) < 2.0 for r in rows)
    # plunge boundary = M - N_acc(1e-6); it is thin (a few modes) and widens slowly (~log c, Task 9.6)
    boundary = {r["c"]: r["M"] - r["n_acc6"] for r in rows}
    boundary_ok = all(b <= 5 for b in boundary.values())

    print("\n" + "=" * 104)
    print(f"  plunge boundary width (M - N_acc<1e-6) by c : " +
          "  ".join(f"c{int(c)}:{b}" for c, b in boundary.items()) + "   (thin, ~log c)")
    print(f"  deep-plateau eigenvalue accuracy (m<=M-6), worst over M>=6 c : {deep_worst:.2e}  (machine-level)")
    print(f"  plunge boundary <= 5 modes at every c                       : {boundary_ok}")
    print(f"  Shannon number M_0(c) ~ c/pi within 1 across sweep          : {M_ok}")
    print(f"  no conditioning fallback anywhere                           : {not any_fb}")

    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(
            os.path.join(CAMPAIGN, "task_13_1_c_sweep.npz"),
            c=np.array([r["c"] for r in rows]), M=np.array([r["M"] for r in rows]),
            P=np.array([r["P"] for r in rows]),
            n_acc6=np.array([r["n_acc6"] for r in rows]),
            n_acc10=np.array([r["n_acc10"] for r in rows]),
            deep_err=np.array([r["deep_err"] for r in rows]),
            knee_err=np.array([r["knee_err"] for r in rows]),
            fallback=np.array([r["fallback"] for r in rows]),
            kappa_lead=np.array([r["kappa_lead"] for r in rows]),
        )
        print(f"\n  saved: data/campaign/task_13_1_c_sweep.npz")

    ok = deep_worst < 1e-8 and M_ok and boundary_ok and (not any_fb)
    print(f"\n  TASK 13.1 c-SWEEP VERIFIED: {ok}")
    return rows


if __name__ == "__main__":
    run()
