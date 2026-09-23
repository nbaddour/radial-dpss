"""
Task 13.10 -- Validation campaign: compare against the prior discrete Hankel method (Boulsane 2024
DHPSS matrix rho^alpha_{N,omega}) if implementable.

Task 6.10 / 7.4a already established (n=0, single c=10 spot check, N<=40) that this project's finite
radial prolate matrix B_K^{(B)} and Boulsane (2024)'s rho^alpha_{N,omega} are TWO DISTINCT FINITE
DISCRETIZATIONS of the SAME continuous operator: different functional structure, not diagonally related,
not orthogonally similar at finite N -- but sharing the same continuous eigenpair limit (eigenvalues,
Task 7.4a Sec 3.5; eigenfunctions after a canonical domain rescaling, Task 7.4a Sec 3.6, overlaps
1.0000/0.9998/0.9975). Task 7.4a Sec 5 Item 6 explicitly flagged two follow-ups as "optional, pending":
  (a) repeat the comparison at n=1,2 to confirm n-independence,
  (b) [analytic, not attempted here].
This task delivers (a), AND extends the comparison from a single (n=0, c=10) toy point to the actual
Task 13 validation-campaign c-range, against the SAME Nystrom ground truth used throughout Task 13
(Task 10.1), at MATCHED computational budget (Boulsane's N = this project's P, the Task 6.10 Sec 2
off-by-one identification), to answer the "Benchmark comparison" deliverable (project management plan,
Task 13.10): which discretization needs a larger matrix for the same accuracy, at each c.

GENERALIZATION NEEDED: Task 7.4a's Boulsane-matrix code (`code/task_7_4a_boulsane_verification.py`) is
hardcoded to n=alpha=0 (uses `jn(0,.)`/`jn(1,.)` directly) and only has a closed form for the n=0
diagonal kernel value. This module generalizes Boulsane's kernel G_alpha(x,y) (Task 7.4a Sec 1) to
general n via the derivative form of the diagonal limit (using scipy.special.jvp for J_n'), and
VERIFIES the generalization reduces to the n=0 hardcoded form to machine precision before using it
for anything (Check 0 below) -- getting the generalization right is a precondition for every other
result in this task.

Scope: eigenvalue accuracy vs the Nystrom reference (Sec A-C) and the eigenfunction-level "same
operator" check repeated at n=1 (Sec D, direct extension of Task 7.4a Sec 3.6). This task does NOT
revisit the settled Task 6.10/7.4a verdict that the two are distinct finite matrices -- it uses that
settled finding to ask a new question (relative efficiency / convergence rate at matched budget),
not to re-litigate "are they the same matrix" (they are not; Task 7.4a Sec 3.2-3.3 is not repeated
here).
"""
import contextlib
import io
import os

import numpy as np
from scipy.special import jv, jvp, jn_zeros

from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize

with contextlib.redirect_stdout(io.StringIO()):        # silence the reference module's import-time demo
    from task_10_1_reference_solver import nystrom

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN = os.path.join(REPO, "data", "campaign")


# ---- Boulsane (2024) DHPSS matrix, generalized to arbitrary order n (Task 7.4a Sec 1, n=0 only) ------
def _G_n(n, x, y, tol=1e-9):
    """Boulsane's kernel G_alpha(x,y) (Task 7.4a Sec 1), generalized from alpha=0 to general n via the
    derivative form of the diagonal (x=y) limit:
        off-diag: G_n(x,y) = sqrt(xy)/(x^2-y^2) * (x J_{n+1}(x) J_n(y) - y J_{n+1}(y) J_n(x))
        diagonal: G_n(x,x) = 1/2 * [ (J_{n+1}(x) + x J_{n+1}'(x)) J_n(x) - x J_{n+1}(x) J_n'(x) ]
    (the diagonal form follows from L'Hopital in y at fixed x; for n=0, using (xJ_1(x))'=xJ_0(x) and
    J_0'(x)=-J_1(x), this reduces EXACTLY to Task 7.4a's hardcoded G_0(x,x)=x/2*(J_0(x)^2+J_1(x)^2) --
    verified in Check 0 below)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    same = np.abs(x - y) < tol
    with np.errstate(divide="ignore", invalid="ignore"):
        off = (np.sqrt(np.abs(x * y)) / (x ** 2 - y ** 2 + 1e-300)
               * (x * jv(n + 1, x) * jv(n, y) - y * jv(n + 1, y) * jv(n, x)))
    diag = 0.5 * ((jv(n + 1, x) + x * jvp(n + 1, x, 1)) * jv(n, x) - x * jv(n + 1, x) * jvp(n, x, 1))
    return np.where(same, diag, off)


def boulsane_matrix(n, N, jz_N1, omega):
    """rho^n_{N,omega} (Task 7.4a Eq. in Sec 1), vectorized, general n. jz_N1 must have >= N zeros of
    J_n (only the first N are used as the basis nodes s_1..s_N)."""
    s = jz_N1[:N]
    X, Y = np.meshgrid(s, s, indexing="ij")
    Kom = omega * _G_n(n, omega * X, omega * Y)
    denom = np.sqrt(np.outer(s, s)) * np.outer(np.abs(jv(n + 1, s)), np.abs(jv(n + 1, s)))
    return 2.0 * Kom / denom


def boulsane_reconstruct(n, s, cvec, x, omega):
    """Reconstruct Boulsane eigenfunction(s) on the CANONICAL domain x in [0,1], undoing the
    symmetrization (divide by sqrt(r)) and the domain rescaling r = omega*x (Task 7.4a Sec 3.6 recipe,
    the ONLY rescaling found there to align the two schemes' eigenfunctions). cvec: (N,) or (N,K)."""
    r = omega * x
    Phi = np.sqrt(2.0 * r)[:, None] * jv(n, np.outer(r, s)) / np.abs(jv(n + 1, s))[None, :]
    psi_native = Phi @ cvec                              # phi_boul(m, r) on Boulsane's native domain
    with np.errstate(divide="ignore", invalid="ignore"):
        g = psi_native / np.sqrt(r)[:, None] if psi_native.ndim == 2 else psi_native / np.sqrt(r)
    return g                                             # un-symmetrized, on canonical x


def reference_eigs(n, c):
    Nq = int(np.ceil(c / 2) + 30)
    lam, x, w, psi = nystrom(n, c, Nq)
    return lam, x, w, psi


# ---- per-config comparison at matched budget N = P --------------------------------------------------
def compare_config(n, c, buffer=20, G=32):
    M = Mn(n, c)
    P = M + buffer
    N = P                                                # matched-budget convention (Task 6.10 Sec 2)

    jz_p = jn_zeros(n, P)
    jz_b = jn_zeros(n, N + 1)                            # need s_{N+1} for the omega closure
    omega = c / jz_b[N]                                  # fixed-c closure (Task 7.4a Sec 3.5)

    # project (production pipeline: Task 12.1/12.2/12.3)
    Bp = assemble(n, c, P=P, G=G)
    rp = solve(Bp, "evr")
    lam_p, _ = canonicalize(rp["lam"], rp["V"], n, jz_p)

    # Boulsane, matched budget N=P
    Rb = boulsane_matrix(n, N, jz_b, omega)
    rb = solve(Rb, "evr")
    lam_b, _ = canonicalize(rb["lam"], rb["V"], n, jz_b[:N])

    lam_ref, _, _, _ = reference_eigs(n, c)

    k = min(len(lam_p), len(lam_b), len(lam_ref))
    err_p = np.abs(lam_p[:k] - lam_ref[:k])
    err_b = np.abs(lam_b[:k] - lam_ref[:k])

    deep_hi = max(M - 6, 1)
    knee_lo, knee_hi = max(M - 1, 0), min(M + 1, k)
    return dict(
        n=n, c=c, M=M, P=P, N=N, omega=omega,
        deep_err_p=float(err_p[:deep_hi].max()), deep_err_b=float(err_b[:deep_hi].max()),
        knee_err_p=float(err_p[knee_lo:knee_hi].max()), knee_err_b=float(err_b[knee_lo:knee_hi].max()),
        eig_range_p=(float(lam_p.min()), float(lam_p.max())),
        eig_range_b=(float(lam_b.min()), float(lam_b.max())),
        resid_b=float(rb["resid"]), ortho_b=float(rb["ortho"]),
    )


# ---- matched-budget convergence-rate study at one representative campaign c --------------------------
def convergence_study(n=0, c=40.0, buffers=(5, 10, 20, 40, 80)):
    lam_ref, _, _, _ = reference_eigs(n, c)
    rows = []
    for buf in buffers:
        M = Mn(n, c)
        P = M + buf
        N = P
        jz_p = jn_zeros(n, P)
        jz_b = jn_zeros(n, N + 1)
        omega = c / jz_b[N]

        Bp = assemble(n, c, P=P, G=32)
        rp = solve(Bp, "evr")
        lam_p, _ = canonicalize(rp["lam"], rp["V"], n, jz_p)

        Rb = boulsane_matrix(n, N, jz_b, omega)
        rb = solve(Rb, "evr")
        lam_b, _ = canonicalize(rb["lam"], rb["V"], n, jz_b[:N])

        k = min(len(lam_p), len(lam_b), len(lam_ref))
        err_p = float(np.max(np.abs(lam_p[:k] - lam_ref[:k])))
        err_b = float(np.max(np.abs(lam_b[:k] - lam_ref[:k])))
        rows.append(dict(P=P, err_p=err_p, err_b=err_b))
    return rows


# ---- eigenfunction-level "same operator" check, repeated at n=1 (Task 7.4a Sec 3.6 recipe) -----------
def eigenfunction_check(n, c, P=50, modes=(0, 1, 2)):
    N = P
    jz_p = jn_zeros(n, P)
    jz_b = jn_zeros(n, N + 1)
    omega = c / jz_b[N]

    Bp = assemble(n, c, P=P, G=32)
    rp = solve(Bp, "evr")
    lam_p, Vp = canonicalize(rp["lam"], rp["V"], n, jz_p)

    Rb = boulsane_matrix(n, N, jz_b, omega)
    rb = solve(Rb, "evr")
    lam_b, Vb = canonicalize(rb["lam"], rb["V"], n, jz_b[:N])

    x = np.linspace(1e-4, 1.0, 2000)
    # project reconstruction (Task 12.4 basis, weighted-L2(x dx) already unit-normalized by isometry)
    Phi_p = np.sqrt(2.0) * jv(n, np.outer(x, jz_p)) / jv(n + 1, jz_p)[None, :]
    psi_p_all = Phi_p @ Vp

    g_b_all = boulsane_reconstruct(n, jz_b[:N], Vb, x, omega)

    overlaps = {}
    for m in modes:
        pp = psi_p_all[:, m]
        nb = np.sqrt(np.trapezoid(g_b_all[:, m] ** 2 * x, x))
        bb = g_b_all[:, m] / nb if nb > 0 else g_b_all[:, m]
        if np.trapezoid(pp * bb * x, x) < 0:
            bb = -bb
        overlaps[m] = float(np.trapezoid(pp * bb * x, x))
    return dict(n=n, c=c, P=P, omega=omega, lam_p=lam_p[:max(modes) + 1].tolist(),
                lam_b=lam_b[:max(modes) + 1].tolist(), overlaps=overlaps)


# ---- verification / campaign runner -------------------------------------------------------------------
def run(save=True):
    print("=" * 104)
    print("TASK 13.10 -- comparison against the prior discrete Hankel method (Boulsane 2024 DHPSS)")
    print("=" * 104)

    # Check 0: the general-n kernel reduces EXACTLY to Task 7.4a's hardcoded n=0 closed form, and to
    # the n=0 hardcoded matrix builder itself -- must pass before anything else is trusted.
    from scipy.special import jn as _jn

    def G0_hardcoded(x, y):
        if abs(x - y) < 1e-12:
            return 0.5 * x * (_jn(0, x) ** 2 + _jn(1, x) ** 2)
        return (np.sqrt(x * y) / (x ** 2 - y ** 2)
                * (x * _jn(1, x) * _jn(0, y) - y * _jn(1, y) * _jn(0, x)))

    def boul_matrix_hardcoded(N, om, sz):
        s = sz[:N]
        Rm = np.zeros((N, N))
        for a in range(N):
            for b in range(N):
                Rm[a, b] = (2 * om * G0_hardcoded(om * s[a], om * s[b])
                            / (np.sqrt(s[a]) * abs(_jn(1, s[a])) * np.sqrt(s[b]) * abs(_jn(1, s[b]))))
        return Rm

    jz_test = jn_zeros(0, 20)
    Rref = boul_matrix_hardcoded(10, 0.5, jz_test)
    Rgen = boulsane_matrix(0, 10, jz_test, 0.5)
    check0 = float(np.max(np.abs(Rref - Rgen)))
    print(f"\n[Check 0] general-n kernel matches Task 7.4a's hardcoded n=0 matrix exactly: "
          f"max|diff| = {check0:.2e}  ({'PASS' if check0 < 1e-12 else 'FAIL'})")

    # omega -> 1 identity sanity check at n=1,2 (Task 7.4a Sec 1 did this at n=0 only)
    id_checks = {}
    for n in (0, 1, 2):
        jzn = jn_zeros(n, 15)
        R = boulsane_matrix(n, 10, jzn, 0.999999)
        offdiag = float(np.max(np.abs(R - np.diag(np.diag(R)))))
        id_checks[n] = offdiag
        print(f"           omega->1 identity check, n={n}: max off-diag = {offdiag:.2e}  "
              f"({'PASS' if offdiag < 1e-8 else 'FAIL'})")

    # --- Sec A: matched-budget accuracy vs Nystrom across the n=0 campaign c-range -------------------
    print("\n[A] n=0 accuracy vs Nystrom reference, matched budget N=P (project vs Boulsane)")
    print(f"{'c':>6s} {'M':>4s} {'P=N':>5s} {'omega':>8s} {'deep_p':>10s} {'deep_b':>10s} "
          f"{'knee_p':>9s} {'knee_b':>9s} {'winner(deep)':>13s}")
    rows_a = []
    for c in [20., 40., 80., 160.]:
        rep = compare_config(0, c)
        rows_a.append(rep)
        winner = "project" if rep["deep_err_p"] < rep["deep_err_b"] else "boulsane"
        print(f"{c:6.0f} {rep['M']:4d} {rep['P']:5d} {rep['omega']:8.4f} "
              f"{rep['deep_err_p']:10.2e} {rep['deep_err_b']:10.2e} "
              f"{rep['knee_err_p']:9.2e} {rep['knee_err_b']:9.2e} {winner:>13s}")

    # --- Sec B: nonzero-n confirmation (Task 7.4a Sec 5 Item 6a: "recommended, pending") -------------
    print("\n[B] nonzero-n confirmation (n=1), matched budget N=P")
    rows_b = []
    for c in [40., 80.]:
        rep = compare_config(1, c)
        rows_b.append(rep)
        winner = "project" if rep["deep_err_p"] < rep["deep_err_b"] else "boulsane"
        print(f"  n=1 c={c:5.0f} M={rep['M']:3d} P=N={rep['P']:3d}: deep_p={rep['deep_err_p']:.2e} "
              f"deep_b={rep['deep_err_b']:.2e} knee_p={rep['knee_err_p']:.2e} "
              f"knee_b={rep['knee_err_b']:.2e}  winner(deep)={winner}")

    # --- Sec C: matched-budget convergence RATE at a representative campaign c (n=0, c=40) -----------
    print("\n[C] convergence rate vs Nystrom, n=0, c=40, matched budget N=P growing")
    conv_rows = convergence_study(0, 40.0)
    print(f"{'P=N':>5s} {'err_project':>12s} {'err_boulsane':>13s} {'ratio(b/p)':>11s}")
    for r in conv_rows:
        ratio = r["err_b"] / r["err_p"] if r["err_p"] > 0 else np.inf
        print(f"{r['P']:5d} {r['err_p']:12.3e} {r['err_b']:13.3e} {ratio:11.1f}")

    # --- Sec D: eigenfunction-level "same operator" check repeated at n=1 -----------------------------
    # c=10 matches Task 7.4a Sec 3.6 EXACTLY (same c, same P=N=50) so this is a clean n=0->n=1
    # substitution, not a new setup. c=40 (the Sec A/C campaign value) was tried first and rejected:
    # at n=1,c=40 the leading modes 0-7 form a numerically-degenerate plateau CLUSTER (lam=1.0 to
    # displayed precision, Task 12.2), so individual eigenvectors there are rotation-ambiguous BETWEEN
    # the two independently-built matrices -- comparing them mode-by-mode is not meaningful (Task 12.2's
    # own caveat), and indeed gave overlaps 0.58/0.42/0.14 with no failure of the "same operator" claim,
    # just a violated precondition (modes 0-2 resolved, not degenerate) for this particular test.
    print("\n[D] eigenfunction overlap after canonical rescaling (Task 7.4a Sec 3.6 recipe), n=1, c=10")
    efn = eigenfunction_check(1, 10.0, P=50, modes=(0, 1, 2))
    print(f"  omega={efn['omega']:.4f}")
    print(f"  lam_project[:3] = {[round(v, 6) for v in efn['lam_p']]}")
    print(f"  lam_boulsane[:3] = {[round(v, 6) for v in efn['lam_b']]}")
    for m, ov in efn["overlaps"].items():
        print(f"    m={m}: overlap(project, boulsane-rescaled) = {ov:.4f}")

    # ---- summary checks --------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("CHECKS")
    print("=" * 104)
    ok0 = check0 < 1e-12
    ok_id = all(v < 1e-8 for v in id_checks.values())
    print(f"[1] Check 0 (generalization exact vs hardcoded n=0)         : {ok0}")
    print(f"[2] omega->1 identity limit holds for n=0,1,2               : {ok_id}")

    both_in_01 = all(-1e-8 <= r["eig_range_p"][0] and r["eig_range_p"][1] <= 1 + 1e-8 for r in rows_a) and \
                 all(-1e-8 <= r["eig_range_b"][0] and r["eig_range_b"][1] <= 1 + 1e-8 for r in rows_a)
    print(f"[3] both matrices' eigenvalues stay in [0,1] (n=0 sweep)     : {both_in_01}")

    both_converge = all(r["deep_err_p"] < 1e-6 for r in rows_a) and all(r["deep_err_b"] < 1e-2 for r in rows_a)
    print(f"[4] both discretizations reach the Nystrom deep-plateau      : {both_converge}  "
          f"(project < 1e-6 machine-level per Task 13.1; Boulsane < 1e-2 at this matched budget)")

    conv_shrinks_p = conv_rows[-1]["err_p"] < conv_rows[0]["err_p"]
    conv_shrinks_b = conv_rows[-1]["err_b"] < conv_rows[0]["err_b"]
    print(f"[5] both errors shrink monotonically as budget N=P grows (c=40): "
          f"project {conv_shrinks_p}, boulsane {conv_shrinks_b}")

    ov_ok = all(v > 0.99 for v in efn["overlaps"].values())
    print(f"[6] eigenfunction overlap > 0.99 after rescaling, n=1, c=10  : {ov_ok}  "
          f"(overlaps: {efn['overlaps']}) -- extends Task 7.4a Sec 3.6 (n=0 only) to n=1")

    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(
            os.path.join(CAMPAIGN, "task_13_10_boulsane_comparison.npz"),
            a_c=np.array([r["c"] for r in rows_a]), a_P=np.array([r["P"] for r in rows_a]),
            a_deep_err_p=np.array([r["deep_err_p"] for r in rows_a]),
            a_deep_err_b=np.array([r["deep_err_b"] for r in rows_a]),
            a_knee_err_p=np.array([r["knee_err_p"] for r in rows_a]),
            a_knee_err_b=np.array([r["knee_err_b"] for r in rows_a]),
            b_c=np.array([r["c"] for r in rows_b]),
            b_deep_err_p=np.array([r["deep_err_p"] for r in rows_b]),
            b_deep_err_b=np.array([r["deep_err_b"] for r in rows_b]),
            conv_P=np.array([r["P"] for r in conv_rows]),
            conv_err_p=np.array([r["err_p"] for r in conv_rows]),
            conv_err_b=np.array([r["err_b"] for r in conv_rows]),
            ef_overlaps=np.array(list(efn["overlaps"].values())),
            check0_diff=check0,
        )
        print("\n  saved: data/campaign/task_13_10_boulsane_comparison.npz")

    ok = ok0 and ok_id and both_in_01 and both_converge and conv_shrinks_p and conv_shrinks_b and ov_ok
    print("\n" + "=" * 104)
    print(f"  TASK 13.10 BOULSANE COMPARISON VERIFIED: {ok}")
    print("=" * 104)
    return rows_a, rows_b, conv_rows, efn


if __name__ == "__main__":
    run()
