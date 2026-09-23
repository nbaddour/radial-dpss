"""
Task 12.5 — Stabilization near r = 0.

Finding (established, not assumed): the DIRECT synthesis (Task 12.4) is already numerically stable near
the origin for every n and every mode, to machine precision (verified against 40-digit mpmath, worst
relative error ~2e-14 down to x = 1e-6, for n in {0,2,4,8} across plateau/plunge/tail modes). scipy's
J_n is accurate for small arguments, so there is no catastrophic cancellation or underflow in the
practical range x in (0, 1]. The ONLY genuinely singular quantity is the normalized near-origin profile
psi~_N(x)/x^n at EXACTLY x = 0 (a 0/0), which is resolved analytically by the leading coefficient.

Near-origin expansion (psi~_N in the nondim variable x):
    psi~_N^{(m)}(x) = zeta_m x^n + O(x^{n+2}),
    zeta_m = sqrt2 * sum_k [ c_k^{(m)} / J_{n+1}(j_{n,k}) ] (j_{n,k}/2)^n / Gamma(n+1),
using J_n(z) = (z/2)^n/Gamma(n+1) + O(z^{n+2}). Hence:
    * n = 0 :  psi~_N(0) = zeta_m  (finite; the origin value, n=0),
    * n >= 1:  psi~_N(0) = 0,  and  psi~_N(x)/x^n -> zeta_m  as x -> 0.
zeta_m is exactly the "first-lobe" coefficient whose SIGN Task 12.3's cosmetic convention uses; here it is
verified to be the exact x^n limit (psi~_N/x^n - zeta_m scales as x^2).

This module provides a uniformly stable evaluator (`synthesize_stable`) that is the plain direct
synthesis for x >= x_switch and the analytic leading term zeta_m x^n below it (the dropped O(x^{n+2}) term
is < x_switch^2 there), plus `origin_value` and `normalized_profile` (psi~_N/x^n, finite at 0). The
dimensional form follows the single rule psi_N(r) = (1/R) psi~_N(r/R); r = 0 maps to x = 0.

NOTE: zeta_m here is a SCALAR (one number per mode m) -- the origin/leading Taylor coefficient.
It is unrelated to the MATRIX A of nodal-sample vectors in Task 12.6 (different object, reused letter
avoided by using zeta here).

Scope: near-origin evaluation only. Weighted-orthogonality correction is Task 12.6; conditioning monitor
is 12.7.
"""
import os
import numpy as np
from scipy.special import jv, jn_zeros, gamma
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize
from task_12_4_synthesis import synthesize_nd

# REPO root: parent of this code/ directory (override with env RDPSS_REPO if running out of tree).
REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "data", "benchmark")


def origin_coeff(n, jz, cvec):
    """zeta_m: leading x^n coefficient of psi~_N^{(m)} (= limit of psi~_N/x^n as x->0). Columns of cvec."""
    w = (np.sqrt(2.0) / gamma(n + 1)) * (jz / 2.0) ** n / jv(n + 1, jz)   # per-k weight
    return w @ cvec


def origin_value(n, jz, cvec):
    """psi~_N^{(m)}(0): zeta_m for n=0 (finite), 0 for n>=1."""
    if n == 0:
        return origin_coeff(0, jz, cvec)
    return np.zeros(np.atleast_2d(cvec).shape[1] if cvec.ndim > 1 else 1)


def synthesize_stable(n, jz, cvec, x, x_switch=1e-6):
    """Uniformly stable synthesis including x=0. Direct for |x|>=x_switch; analytic zeta_m x^n below
    (dropped term O(x^{n+2}) < x_switch^2). Handles scalar or array x; cvec (P,) or (P,nm)."""
    x = np.atleast_1d(np.asarray(x, float))
    C = cvec[:, None] if cvec.ndim == 1 else cvec
    out = np.empty((x.size, C.shape[1]))
    near = np.abs(x) < x_switch
    if np.any(~near):
        out[~near] = synthesize_nd(n, jz, C, x[~near])
    if np.any(near):
        zeta = origin_coeff(n, jz, C)                    # (nm,)
        out[near] = (x[near] ** n)[:, None] * zeta[None, :]  # exact at x=0 (0 for n>=1, zeta_m for n=0)
    return out[:, 0] if cvec.ndim == 1 else out


def normalized_profile(n, jz, cvec, x, x_switch=1e-6):
    """psi~_N^{(m)}(x)/x^n, finite at x=0 (=zeta_m). The near-origin SHAPE, stable everywhere."""
    x = np.atleast_1d(np.asarray(x, float))
    C = cvec[:, None] if cvec.ndim == 1 else cvec
    out = np.empty((x.size, C.shape[1]))
    near = np.abs(x) < x_switch
    if np.any(~near):
        out[~near] = synthesize_nd(n, jz, C, x[~near]) / (x[~near] ** n)[:, None]
    if np.any(near):
        out[near] = origin_coeff(n, jz, C)[None, :]
    return out[:, 0] if cvec.ndim == 1 else out


# ---- verification ------------------------------------------------------------------------------------
def verify():
    print("=" * 100)
    print("TASK 12.5 -- near-origin stabilization: direct stability, zeta_m limit, x=0, stable evaluator")
    print("=" * 100)

    try:
        import mpmath as mp
        mp.mp.dps = 40
        have_mp = True
    except Exception:
        have_mp = False

    def psi_mp(n, jz, col, x):
        s = mp.mpf(0)
        for k in range(len(jz)):
            s += mp.mpf(float(col[k])) / mp.besselj(n + 1, mp.mpf(float(jz[k]))) \
                 * mp.besselj(n, mp.mpf(float(jz[k])) * mp.mpf(x))
        return mp.sqrt(2) * s

    cases = [(0, 20., 44), (1, 40., 48), (2, 40., 48), (4, 40., 50), (8, 40., 50), (4, 80., 90)]

    worst_mp = 0.0
    worst_x2 = 0.0            # zeta_m limit: |psi/x^n - zeta_m| should be ~x^2 (physical), shrinking
    worst_stab = 0.0         # synthesize_stable vs direct on [x_switch,1]
    x0_ok = True
    nan_ok = True
    for (n, c, P) in cases:
        jz = jn_zeros(n, P); B = assemble(n, c, P=P, G=32)
        r = solve(B, "evr"); lam, V = canonicalize(r["lam"], r["V"], n, jz)
        M = Mn(n, c); modes = range(min(M + 4, P))

        # (1) direct vs mpmath (ground truth), all modes, near origin
        if have_mp:
            for m in modes:
                for x in (1e-2, 1e-4, 1e-6):
                    d = float(synthesize_nd(n, jz, V[:, m][:, None], np.array([x]))[0, 0])
                    g = float(psi_mp(n, jz, V[:, m], x))
                    if g != 0:
                        worst_mp = max(worst_mp, abs(d - g) / abs(g))

        # (2) zeta_m is the exact limit: |psi/x^n - zeta_m| must SCALE as x^2 (physical Taylor term),
        #     i.e. r(1e-4)/r(1e-5) ~ 100. This proves zeta_m is the exact leading coefficient.
        zeta = origin_coeff(n, jz, V[:, :min(M + 4, P)])
        for m in modes:
            if zeta[m] == 0:
                continue
            r4 = abs(float(synthesize_nd(n, jz, V[:, m][:, None], np.array([1e-4]))[0, 0]) / (1e-4 ** n if n > 0 else 1) - zeta[m])
            r5 = abs(float(synthesize_nd(n, jz, V[:, m][:, None], np.array([1e-5]))[0, 0]) / (1e-5 ** n if n > 0 else 1) - zeta[m])
            if r5 > 1e-13:                      # ignore modes where the residual is already at noise
                worst_x2 = max(worst_x2, abs(r4 / r5 - 100.0))   # deviation from the x^2 law (100x/decade)

        # (3) stable evaluator matches direct on [x_switch, 1]
        xg = np.linspace(1e-6, 1.0, 500)
        ds = synthesize_stable(n, jz, V[:, :min(M + 2, P)], xg)
        dd = synthesize_nd(n, jz, V[:, :min(M + 2, P)], xg)
        worst_stab = max(worst_stab, float(np.max(np.abs(ds - dd))))

        # (4) x = 0 exact + finiteness
        p0 = synthesize_stable(n, jz, V[:, :min(M + 2, P)], np.array([0.0]))[0]
        if n >= 1 and not np.all(np.abs(p0) < 1e-14):
            x0_ok = False
        prof0 = normalized_profile(n, jz, V[:, :min(M + 2, P)], np.array([0.0]))[0]
        if not np.allclose(prof0, zeta[:min(M + 2, P)], atol=1e-12):
            x0_ok = False
        if not np.all(np.isfinite(synthesize_stable(n, jz, V, np.linspace(0, 1, 100)))):
            nan_ok = False

        print(f"  n={n} c={c:5.1f} P={P:3d} M={M}: zeta_m x^2-law dev(|r4/r5-100|)={worst_x2:.1e}"
              f"  stable-vs-direct={worst_stab:.1e}  x0_ok={x0_ok}  finite={nan_ok}")

    print("\n" + "=" * 100); print("OVERALL"); print("=" * 100)
    if have_mp:
        print(f"  direct vs mpmath, worst rel.err (all modes, x>=1e-6) : {worst_mp:.2e}")
    else:
        print("  (mpmath unavailable; skipped ground-truth check)")
    print(f"  zeta_m is exact x^n limit (x^2 law: |r(1e-4)/r(1e-5) - 100|): {worst_x2:.2e}")
    print(f"  synthesize_stable vs direct on [1e-6,1]               : {worst_stab:.2e}")
    print(f"  x=0 exact (n>=1 ->0; profile->zeta_m) & all finite        : {x0_ok and nan_ok}")
    ok = ((not have_mp or worst_mp < 1e-12) and worst_x2 < 5.0 and worst_stab < 1e-12
          and x0_ok and nan_ok)
    print(f"\n  TASK 12.5 NEAR-ORIGIN STABILIZATION VERIFIED: {ok}")


if __name__ == "__main__":
    verify()
