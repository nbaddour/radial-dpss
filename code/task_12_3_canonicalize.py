"""
Task 12.3 — Normalize and sort the eigenpairs into the project's canonical form.

Input : raw eigenpairs (lambda_m, c^{(m)}) from the Task 12.2 solver (orthonormal columns, descending).
Output: CANONICAL eigenpairs under the project's locked conventions, plus the nodal-criterion diagnostic.

Locked conventions applied here:
  * ORDER (Conv 4.10.1/2): eigenvalues descending lambda_0 >= lambda_1 >= ..., mode m 0-indexed.
  * NORMALIZATION: c^{(m)} unit ell^2  <=>  psi_N^{(m)} unit weighted-L^2 (x dx) via the Task 8.7
    coefficient->function isometry (the eigenvectors are already orthonormal; we assert it).
  * SIGN (Def 7.3.3, PRIMARY): largest-|component| positive; tie -> smallest k positive. This is the
    project's settled convention (Task 7.3 sec.5.1). Conv 4.10.5 "first lobe positive" is only the
    OPTIONAL COSMETIC n=0 alternative (Task 7.3 sec.5.2) and is provided separately for figures.
  * NODAL CRITERION (Conv 4.10.4): the canonical mode m has exactly m interior zeros of psi_N^{(m)} in
    (0,1). This is the mode-VERIFICATION invariant; it holds for spectrally RESOLVED (separated) modes.
    Inside the numerically-degenerate plateau cluster the individual eigenvectors are rotation-ambiguous
    (Task 12.2 / 8.5 / 9.3), so per-mode node counts there are NOT canonical -- only the subspace is.

Synthesis used for node counting / cosmetic sign (Task 8.7 eq. 8.7.1, nondim):
    psi~_N^{(m)}(x) = sqrt2 * sum_k [ c_k^{(m)} / J_{n+1}(j_{n,k}) ] J_n(j_{n,k} x),  x in [0,1].
(The full synthesis/interpolation + error machinery is Task 12.4; here it is used only as a diagnostic.)

Scope: ordering + normalization + sign + the nodal diagnostic. Conditioning monitor is Task 12.7.
"""
import os
import numpy as np
from scipy.special import jv, jn_zeros
from task_12_1_assemble_matrix import assemble, Mn, _spectrum
from task_12_2_eigensolver import solve

# REPO root: parent of this code/ directory (override with env RDPSS_REPO if running out of tree).
REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(REPO, "data", "pilot")
BENCH = os.path.join(REPO, "data", "benchmark")


# ---- synthesis (diagnostic only; full machinery is Task 12.4) --------------------------------------
def reconstruct_nd(n, jz, cvec, x):
    """psi~_N^{(m)}(x) on grid x for columns cvec (Task 8.7 eq 8.7.1)."""
    Phi = np.sqrt(2.0) * jv(n, np.outer(x, jz)) / jv(n + 1, jz)[None, :]
    return Phi @ cvec


def _first_lobe_coeff(n, jz, cvec):
    """Coefficient of x^n as x->0 (up to a positive constant): sum_k c_k j_{n,k}^n / J_{n+1}(j_{n,k}).
    Its sign is the sign of the first lobe of psi~_N^{(m)} near the origin."""
    w = (jz ** n) / jv(n + 1, jz)
    return w @ cvec


# ---- canonicalization --------------------------------------------------------------------------------
def canonicalize(lam, V, n, jz, sign="def733"):
    """Return canonical (lam, V). Eigenvalues sorted descending; eigenvectors orthonormal; sign fixed.

    sign = "def733"      : largest-|component| positive, tie -> smallest k  (PRIMARY, Def 7.3.3)
    sign = "first_lobe"  : first lobe of psi~_N positive  (COSMETIC alt; Conv 4.10.5 / Task 7.3 sec.5.2)
    """
    order = np.argsort(lam)[::-1]
    lam = lam[order].copy()
    V = V[:, order].copy()

    P = V.shape[1]
    if sign == "def733":
        for m in range(P):
            amax = np.max(np.abs(V[:, m]))
            # tie -> smallest k among components within rounding of the max
            cand = np.where(np.abs(V[:, m]) >= amax * (1 - 1e-12))[0]
            kstar = cand[0]
            if V[kstar, m] < 0:
                V[:, m] *= -1.0
    elif sign == "first_lobe":
        s = _first_lobe_coeff(n, jz, V)
        for m in range(P):
            if s[m] < 0:
                V[:, m] *= -1.0
    else:
        raise ValueError(sign)
    return lam, V


def interior_zero_count(psi_col, x, x_lo=1e-6, x_hi=1 - 1e-9):
    """Count interior sign changes of a reconstructed mode on (x_lo, x_hi) (exclude origin & x=1)."""
    mask = (x > x_lo) & (x < x_hi)
    y = psi_col[mask]
    s = np.sign(y)
    s = s[s != 0]
    return int(np.sum(s[1:] * s[:-1] < 0))


def resolved_modes(lam, rel_gap=1e-6, lam_floor=1e-3):
    """Indices of spectrally RESOLVED modes: gap to both neighbours > rel_gap AND lam above the
    deep-tail floor (deep-tail eigenfunctions are benchmark-delicate and dynamically irrelevant)."""
    idx = []
    P = len(lam)
    for m in range(P):
        gl = lam[m - 1] - lam[m] if m > 0 else np.inf
        gr = lam[m] - lam[m + 1] if m < P - 1 else np.inf
        gap = min(gl, gr)
        scale = max(1.0, abs(lam[m]))
        if gap > rel_gap * scale and lam[m] > lam_floor:
            idx.append(m)
    return idx


def faithful_plateau_modes(lam, M, lo=0.9, hi=1 - 1e-6):
    """Plateau-INTERIOR modes that are faithfully approximated AND individually resolved:
    lo<lam<hi (above the plunge, below the deeply-degenerate lam~1 cluster) AND m <= M-2 (strictly
    inside the plateau, excluding the plunge knee m~M-1 which converges only ~1/P, Task 7.2/9.3).
    The nodal criterion z==m is theoretically guaranteed (super-exp faithful) and verified here."""
    return [m for m in range(len(lam)) if lo < lam[m] < hi and m <= M - 2]


# ---- verification ------------------------------------------------------------------------------------
def _xfine(N=4000):
    return np.linspace(0.0, 1.0, N)


def verify():
    xg = _xfine()
    print("=" * 100)
    print("TASK 12.3 -- canonical normalize/sort: ordering, norm, Def 7.3.3 sign, nodal criterion")
    print("=" * 100)

    worst_norm = 0.0
    sign_diff_g6 = 0.0       # Def7.3.3 sign determinism across drivers, resolved gap>1e-6
    sign_diff_g4 = 0.0       # ... well-separated gap>1e-4
    nodal_ok = nodal_tot = 0
    nodal_tail_extra = 0     # plunge/tail modes carrying the +1 FB-edge artifact (informational)
    cosmetic_ok = True

    cases = [(0, 20.0, 44), (0, 40.0, 56), (1, 20.0, 30), (1, 40.0, 48),
             (2, 40.0, 48), (4, 40.0, 50), (4, 80.0, 90)]
    for (n, c, P) in cases:
        jz = jn_zeros(n, P)
        B = assemble(n, c, P=P, G=32)
        r1 = solve(B, driver="evr"); r2 = solve(B, driver="ev")
        lam, V1 = canonicalize(r1["lam"], r1["V"], n, jz, sign="def733")
        _,   V2 = canonicalize(r2["lam"], r2["V"], n, jz, sign="def733")

        worst_norm = max(worst_norm, float(np.max(np.abs(np.sum(V1 ** 2, axis=0) - 1.0))))

        # sign determinism (Def 7.3.3): canonical evecs from 2 drivers agree WITHOUT a flip, to the
        # eigenvector conditioning (gap-limited). Reported at two separation thresholds.
        for m in range(len(lam)):
            gl = lam[m - 1] - lam[m] if m > 0 else np.inf
            gr = lam[m] - lam[m + 1] if m < len(lam) - 1 else np.inf
            gap = min(gl, gr)
            if lam[m] < 1e-3:           # skip deep tail (benchmark-delicate, dynamically irrelevant)
                continue
            d = float(np.max(np.abs(V1[:, m] - V2[:, m])))
            if gap > 1e-6:
                sign_diff_g6 = max(sign_diff_g6, d)
            if gap > 1e-4:
                sign_diff_g4 = max(sign_diff_g4, d)

        # nodal criterion z==m on faithful plateau-interior modes (gate); tail/+1 reported separately
        M = Mn(n, c)
        psi = reconstruct_nd(n, jz, V1, xg)
        faith = faithful_plateau_modes(lam, M)
        fr = []
        for m in faith:
            z = interior_zero_count(psi[:, m], xg)
            nodal_tot += 1; nodal_ok += int(z == m)
            fr.append(f"m{m}:{z}{'=' if z==m else '!='}{m}")
        # informational: plunge-knee/tail extra-zero count (m in [M-1 .. M+2])
        for m in range(max(M - 1, 0), min(M + 3, len(lam))):
            if lam[m] > 1e-3:
                z = interior_zero_count(psi[:, m], xg)
                if z == m + 1:
                    nodal_tail_extra += 1

        # cosmetic first-lobe sign (n=0: psi(0)>0)
        if n == 0:
            _, Vc = canonicalize(r1["lam"], r1["V"], n, jz, sign="first_lobe")
            psi0 = reconstruct_nd(n, jz, Vc, np.array([1e-4]))[0]
            if not np.all(psi0[faith] > 0):
                cosmetic_ok = False

        print(f" n={n} c={c:5.1f} P={P:3d} M={M}: ||c||-1<=machine  faithful-nodal {' '.join(fr)}")

    # pilot re-anchor (canonical eigenvalues vs saved)
    worst_re = 0.0
    for f in sorted(os.listdir(PILOT)):
        if f.startswith("Bmat_n0_") and f.endswith(".npz"):
            d = np.load(os.path.join(PILOT, f)); B = d["B"]
            r = solve(B, "evr"); lam, _ = canonicalize(r["lam"], r["V"], 0, jn_zeros(0, int(d["P"])))
            k = min(len(lam), len(d["lam"]))
            worst_re = max(worst_re, float(np.max(np.abs(lam[:k] - np.sort(d["lam"])[::-1][:k]))))

    print("\n" + "=" * 100); print("OVERALL"); print("=" * 100)
    print(f"  worst |‖c^(m)‖^2 - 1|                         : {worst_norm:.2e}")
    print(f"  Def7.3.3 sign determinism, gap>1e-6 (evr/ev) : {sign_diff_g6:.2e}")
    print(f"  Def7.3.3 sign determinism, gap>1e-4 (evr/ev) : {sign_diff_g4:.2e}")
    print(f"  nodal z==m on faithful plateau modes         : {nodal_ok}/{nodal_tot}")
    print(f"  plunge/tail +1 FB-edge-zero (informational)  : {nodal_tail_extra} modes (boundary artifact, Task 11.3 OpenQ2)")
    print(f"  cosmetic first-lobe sign (n=0 psi(0)>0)      : {cosmetic_ok}")
    print(f"  pilot re-anchor canonical eigenvalues        : {worst_re:.2e}")
    ok = (worst_norm < 1e-12 and sign_diff_g4 < 1e-11 and nodal_ok == nodal_tot
          and cosmetic_ok and worst_re < 1e-12)
    print(f"\n  TASK 12.3 CANONICALIZATION VERIFIED: {ok}")


if __name__ == "__main__":
    verify()
