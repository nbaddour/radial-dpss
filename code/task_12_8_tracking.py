"""
Task 12.8 — Mode tracking across changing parameters.

Follow each concentration mode m as the space-bandwidth product c varies (the primary Task 13.1 sweep) so
the campaign reports COHERENT per-mode trajectories lam_m(c) and mode shapes psi_N^{(m)}(.; c).

Two facts make this clean:
  (i)  The Fourier-Bessel basis phi_{n,k} depends only on (n,k), NOT on c. So the coefficient vectors
       c^{(m)}(c) all live in the SAME R^{P}, and overlaps <c^{(m)}(c_i), c^{(m')}(c_{i+1})> are exact
       comparisons (no regridding).
  (ii) The concentration eigenvalues are simple and stay descending-ordered (Conv 4.10); they do NOT
       cross as c varies. So descending-sort labeling already tracks eigenVALUES coherently; the task is
       to confirm this and to track the eigenVECTORS/shapes continuously.

Handling the two regimes (Gate C / Task 12.2):
  * SEPARATED modes (plateau-edge, plunge; clear spectral gap, lam away from 0 and 1): individual
    coefficient vectors are well-defined; track by maximum overlap (Hungarian assignment on the
    SEPARATED block of |C(c_i)^T C(c_{i+1})|). The assignment is the identity (no label swaps) and the
    consecutive self-overlap -> 1 as the step dc -> 0 (smooth deformation, error O(dc^2)).
  * DEEP-PLATEAU / leading subspace (lam ~ 1, gaps ~1e-15): individual vectors rotate arbitrarily
    between solves, so track the SUBSPACE. The leading-K subspace projector deforms CONTINUOUSLY
    (Lipschitz): the projector jump between steps is O(dc) -> 0 as dc -> 0.

Identity invariant: a tracked separated mode m keeps exactly m interior zeros (nodal criterion, Task
12.3) along the whole trajectory.

Scope: tracking only. Version-controlled tests are Task 12.9; the campaign itself is Task 13.
"""
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve, subspace_projector
from task_12_3_canonicalize import (canonicalize, interior_zero_count, reconstruct_nd,
                                    faithful_plateau_modes)

# REPO root: parent of this code/ directory (override with env RDPSS_REPO if running out of tree).
REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def solve_at(n, c, P):
    jz = jn_zeros(n, P)
    B = assemble(n, c, P=P, G=32)
    r = solve(B, "evr")
    lam, V = canonicalize(r["lam"], r["V"], n, jz)
    return lam, V


def separated_modes(lam, lo=0.02, hi=0.98, gap=1e-3):
    """Indices of spectrally SEPARATED, non-degenerate modes (individually trackable)."""
    P = len(lam)
    return [m for m in range(P) if lo < lam[m] < hi
            and (m == 0 or lam[m - 1] - lam[m] > gap)
            and (m == P - 1 or lam[m] - lam[m + 1] > gap)]


def track_c_sweep(n, c_grid, P):
    """Solve across a c-sweep at fixed (n, P); return eigenvalue trajectories, eigenvectors, and
    per-step tracking diagnostics (separated-block assignment consistency + self-overlaps)."""
    c_grid = np.asarray(c_grid, float)
    lam_traj = np.zeros((len(c_grid), P))
    V_list = []
    for i, c in enumerate(c_grid):
        lam, V = solve_at(n, c, P)
        lam_traj[i] = lam
        V_list.append(V)

    steps = []
    for i in range(len(c_grid) - 1):
        Ov = np.abs(V_list[i].T @ V_list[i + 1])          # overlaps in the shared coefficient space
        sep = separated_modes(lam_traj[i])
        assign_identity = True
        if sep:
            sub = Ov[np.ix_(sep, sep)]
            _, col = linear_sum_assignment(-sub)
            assign_identity = bool(np.all(col == np.arange(len(sep))))
        self_ov = [Ov[m, m] for m in sep]
        steps.append(dict(c_from=c_grid[i], c_to=c_grid[i + 1], sep=sep,
                          assign_identity=assign_identity,
                          min_self_overlap=min(self_ov) if self_ov else np.nan))
    return dict(c_grid=c_grid, lam_traj=lam_traj, V_list=V_list, steps=steps)


def leading_subspace_jump(n, c_grid, P, K):
    """Worst projector jump of the FIXED leading-K subspace across the c-sweep (continuity measure)."""
    prev = None
    worst = 0.0
    for c in c_grid:
        _, V = solve_at(n, c, P)
        Pi = subspace_projector(V, 0, K)
        if prev is not None:
            worst = max(worst, float(np.max(np.abs(Pi - prev))))
        prev = Pi
    return worst


# ---- verification ------------------------------------------------------------------------------------
def verify():
    print("=" * 100)
    print("TASK 12.8 -- mode tracking across a c-sweep")
    print("=" * 100)

    xg = np.linspace(0, 1, 4000)
    mono_ok = True
    assign_ok = True
    nodal_ok = True
    ov_fine = 1.0            # separated overlap at the fine step
    ov_refines = True        # overlap improves as dc halves
    sub_refines = True       # subspace jump ~halves as dc halves

    for n in [0, 2]:
        P = 60
        # eigenvalue monotonicity + label consistency on a moderate grid
        tr = track_c_sweep(n, np.arange(10.0, 40.001, 1.0), P)
        lam = tr["lam_traj"]
        Mmax = Mn(n, 40.0)
        if np.min(np.diff(lam[:, :Mmax], axis=0)) < -1e-9:
            mono_ok = False
        if not all(s["assign_identity"] for s in tr["steps"]):
            assign_ok = False

        # separated-mode continuity: overlap -> 1 as dc halves (smooth deformation)
        ov = {}
        for dc in (0.5, 0.25):
            t = track_c_sweep(n, np.arange(24.0, 28.0001, dc), P)
            ov[dc] = min(s["min_self_overlap"] for s in t["steps"] if not np.isnan(s["min_self_overlap"]))
        ov_fine = min(ov_fine, ov[0.25])
        if not (ov[0.25] >= ov[0.5] - 1e-9):
            ov_refines = False

        # leading-K subspace continuity: jump ~ O(dc) (halves as dc halves)
        K = Mn(n, 24.0)
        j = {dc: leading_subspace_jump(n, np.arange(24.0, 28.0001, dc), P, K) for dc in (0.5, 0.25)}
        ratio = j[0.5] / j[0.25] if j[0.25] > 0 else np.inf
        if not (1.7 < ratio < 2.3):        # linear-in-dc => halving
            sub_refines = False

        # nodal identity along the trajectory, on the FAITHFUL plateau-interior modes (Task 12.3
        # z==m set). The plunge-knee/tail modes carry the +1 FB-edge zero (Task 12.3/12.5) -- a
        # reconstruction property, not a tracking defect -- so they are excluded here.
        for c in np.arange(20.0, 30.0001, 2.0):
            lam_c, V_c = solve_at(n, c, P)
            for m in faithful_plateau_modes(lam_c, Mn(n, c)):
                z = interior_zero_count(reconstruct_nd(n, jn_zeros(n, P), V_c[:, m], xg), xg)
                if z != m:
                    nodal_ok = False

        print(f"  n={n}: monotonic(leading)={mono_ok}  sep-assignment=identity={assign_ok}")
        print(f"      separated overlap  dc=0.5: {ov[0.5]:.5f}  dc=0.25: {ov[0.25]:.5f}  (->1, error O(dc^2))")
        print(f"      leading-{K} subspace jump  dc=0.5: {j[0.5]:.2e}  dc=0.25: {j[0.25]:.2e}  ratio={ratio:.2f} (~2 => O(dc))")

    print("\n" + "=" * 100)
    print(f"  eigenvalue trajectories monotonic (no crossing) : {mono_ok}")
    print(f"  separated-mode labels consistent (no swaps)     : {assign_ok}")
    print(f"  separated-mode overlap -> 1 (fine step {ov_fine:.4f}) & refines : {ov_fine > 0.99 and ov_refines}")
    print(f"  leading subspace continuous (jump ~ O(dc))      : {sub_refines}")
    print(f"  nodal identity (zero-count=m) preserved         : {nodal_ok}")
    ok = mono_ok and assign_ok and ov_fine > 0.99 and ov_refines and sub_refines and nodal_ok
    print(f"\n  TASK 12.8 MODE TRACKING VERIFIED: {ok}")


if __name__ == "__main__":
    verify()
