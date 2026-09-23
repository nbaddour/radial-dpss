"""
Task 12.1 — Assemble the finite radial prolate matrix B_K^{(B)}_{N-1}(n, c) for ARBITRARY (n, c, N).

This is the Stage-5 pipeline generalization of the Gate-C pilot assembler (Task 11.2), which ran only
n = 0. The entry formula (Task 6.1 eq. 3.1) is already general in the angular order n:

    B[m,k] = 2 j_{n,m} j_{n,k} * INT_0^c  u J_n(u)^2 / [ (u^2 - j_{n,m}^2)(u^2 - j_{n,k}^2) ] du ,
                                                                          m,k = 1 .. N-1 = P
with j_{n,p} the p-th positive zero of J_n, and c = KR the space-bandwidth product. In the F1 fixed-c
regime c is held fixed and independent of N (the regime of the Task 9 convergence theorems); in the F2
closure regime one sets c = j_{n,N}. This module takes (n, c) and the matrix dimension P = N-1 directly
and is agnostic to which regime supplied c.

Two independent assembly routes (mutual cross-check), both general-n:
  ROUTE B (primary, efficient): partial fraction (Task 6.1 eq. 3.3'/3.5):
     off-diag  B[m,k] = 2 j_m j_k (I_m - I_k)/(j_m^2 - j_k^2),  I_p = INT_0^c u J_n^2/(u^2 - j_p^2) du
     diagonal  B[m,m] = 2 j_m^2 D_m,                            D_p = INT_0^c u J_n^2/(u^2 - j_p^2)^2 du
  ROUTE A (independent check): the raw double-pole integrand of (3.1) per entry, same nodes.

Removable singularities at u -> j_{n,p} (Task 6.1 sec. 3.2, general n):
     u J_n^2/(u^2 - j_p^2)   -> 0
     u J_n^2/(u^2 - j_p^2)^2 -> J_{n+1}^2(j_{n,p}) / (4 j_{n,p})
using J_n'(j_{n,p}) = -J_{n+1}(j_{n,p}). These hold for every n >= 0.

Quadrature: composite Gauss-Legendre with panel breakpoints at {0, j_{n,1}, ..., j_{n,M_n(c)}, c} -- the
J_n zeros below c -- so panels align with the integrand's oscillation and removable-singularity points,
giving spectral accuracy per panel.

Run as a script to execute the Task 12.1 validation battery (n=0 re-anchor + n>=1 verification).
"""
import os
import numpy as np
from scipy.special import jv, jn_zeros

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(REPO, "data", "pilot")
BENCH = os.path.join(REPO, "data", "benchmark")


# ----------------------------------------------------------------------------------------------------
# Core assembly (general n)
# ----------------------------------------------------------------------------------------------------
def _panels_nodes(c, jzeros_below_c, G):
    """Composite Gauss-Legendre nodes/weights on [0,c] with breakpoints at the J_n zeros below c."""
    bps = np.concatenate(([0.0], jzeros_below_c, [c]))
    t, w = np.polynomial.legendre.leggauss(G)
    U, Wq = [], []
    for a, b in zip(bps[:-1], bps[1:]):
        if b <= a:
            continue
        U.append(0.5 * (b - a) * t + 0.5 * (a + b))
        Wq.append(0.5 * (b - a) * w)
    return np.concatenate(U), np.concatenate(Wq)


def assemble(n, c, P=None, N=None, G=24, return_cache=False):
    """Assemble the (P x P) finite radial prolate matrix for angular order n at space-bandwidth c.

    Provide either P (= matrix dimension = N-1) or N (= truncation; P is set to N-1).
    Route B (partial fraction) is the primary route. Returns B, or (B, cache) if return_cache.
    """
    if P is None:
        if N is None:
            raise ValueError("supply P (= N-1) or N")
        P = N - 1
    if P < 1:
        raise ValueError(f"P must be >= 1 (got {P})")

    jz = jn_zeros(n, P)                     # j_{n,1..P}
    j_below = jz[jz < c]
    U, Wq = _panels_nodes(c, j_below, G)
    J2 = jv(n, U) ** 2
    uJ2W = U * J2 * Wq                      # common factor u * J_n^2 * w
    j2 = jz ** 2
    Jnp1 = jv(n + 1, jz)                    # J_{n+1}(j_{n,p}) for the diagonal limit

    I = np.empty(P)
    D = np.empty(P)
    for p in range(P):
        den = U * U - j2[p]
        near = np.abs(U - jz[p]) < 1e-11
        ti = uJ2W / den
        if near.any():
            ti[near] = 0.0                  # removable: u J_n^2/(u^2 - j_p^2) -> 0
        I[p] = ti.sum()
        td = uJ2W / den ** 2
        if near.any():
            td[near] = (Jnp1[p] ** 2 / (4.0 * jz[p])) * Wq[near]   # -> J_{n+1}^2/(4 j_p)
        D[p] = td.sum()

    B = np.empty((P, P))
    for m in range(P):
        B[m, m] = 2.0 * j2[m] * D[m]
        for k in range(m + 1, P):
            B[m, k] = B[k, m] = 2.0 * jz[m] * jz[k] * (I[m] - I[k]) / (j2[m] - j2[k])

    if return_cache:
        return B, dict(U=U, Wq=Wq, J2=J2, jz=jz, j2=j2)
    return B


def assemble_routeA(n, c, P, cache):
    """Independent cross-check: raw double-pole integrand of (3.1), per entry, on the same nodes."""
    U, Wq, J2, jz, j2 = cache["U"], cache["Wq"], cache["J2"], cache["jz"], cache["j2"]
    uJ2W = U * J2 * Wq
    BA = np.empty((P, P))
    for m in range(P):
        for k in range(m, P):
            den = (U * U - j2[m]) * (U * U - j2[k])
            near = (np.abs(U - jz[m]) < 1e-11) | (np.abs(U - jz[k]) < 1e-11)
            t = uJ2W / den
            if near.any():
                t[near] = 0.0               # numerator double-zero dominates -> 0 at node==pole
            BA[m, k] = BA[k, m] = 2.0 * jz[m] * jz[k] * t.sum()
    return BA


def Mn(n, c):
    """Shannon number M_n(c) = #{ k >= 1 : j_{n,k} < c }."""
    return int(np.sum(jn_zeros(n, int(c / np.pi) + 30) < c))


# ----------------------------------------------------------------------------------------------------
# Validation battery (Task 12.1)
# ----------------------------------------------------------------------------------------------------
def _spectrum(B):
    lam = np.sort(np.linalg.eigvalsh(B))[::-1]
    return lam


def reanchor_n0():
    """Re-anchor the general-n assembler against the saved n=0 pilot matrices (Task 11.2)."""
    print("=" * 96)
    print("RE-ANCHOR: general-n assembler vs saved n=0 pilot matrices (Task 11.2)")
    print("=" * 96)
    worst = 0.0
    files = sorted(f for f in os.listdir(PILOT) if f.startswith("Bmat_n0_") and f.endswith(".npz"))
    for f in files:
        d = np.load(os.path.join(PILOT, f))
        c, P = float(d["c"]), int(d["P"])
        B = assemble(0, c, P=P, G=24)
        err = np.max(np.abs(B - d["B"]))
        worst = max(worst, err)
        print(f"  {f:28s} c={c:5.1f} P={P:3d}   max|B_new - B_saved| = {err:.2e}")
    print(f"\n  WORST over all pilot files: {worst:.2e}")
    return worst


def verify_general_n(cases):
    """Full verification for arbitrary n: route A/B, quadrature convergence, structure, benchmark."""
    print("\n" + "=" * 96)
    print("GENERAL-n VERIFICATION  (n >= 1: assembly never exercised before Task 12.1)")
    print("=" * 96)
    summary = []
    for (n, c, P) in cases:
        M = Mn(n, c)
        B, cache = assemble(n, c, P=P, G=24, return_cache=True)
        B40 = assemble(n, c, P=P, G=40)
        conv = np.max(np.abs(B - B40))                       # quadrature convergence
        BA = assemble_routeA(n, c, P, cache)
        ab = np.max(np.abs(B - BA))                          # route A vs B
        sym = np.max(np.abs(B - B.T))
        lam = _spectrum(B)
        in01 = bool(lam.min() > -1e-12 and lam.max() < 1 + 1e-12)
        ngt = int((lam > 0.5).sum())

        # Benchmark cross-check (Task 10.4 reference eigenvalues), split by spectral regime.
        # Per the locked theory (Task 7.2 plateau-only guarantee; Task 11.2/11.4) the finite matrix
        # tracks the continuous operator super-fast on the PLATEAU and only ~1/P on the PLUNGE KNEE.
        # The honest correctness metric is therefore the plateau match; the knee error is REPORTED
        # (and its ~1/P decay is the documented behavior), not used as a correctness gate.
        bench_plateau = np.nan       # max over deep+shoulder plateau modes m <= M-2
        bench_knee = np.nan          # max over the plunge knee m in {M-1, M}
        bf = os.path.join(BENCH, f"cpswf_n{n}_c{int(c)}.npz")
        if os.path.exists(bf):
            lref = np.load(bf)["lam"]
            lb = _spectrum(assemble(n, c, P=P, G=32))
            k = min(len(lref), len(lb), M + 3)
            err = np.abs(lb[:k] - lref[:k])
            p_hi = max(M - 2, 1)                              # plateau modes 0..M-2 (0-indexed)
            bench_plateau = float(err[:p_hi].max())
            bench_knee = float(err[p_hi:].max()) if k > p_hi else np.nan

        print(f"\n n={n} c={c:5.1f} P={P:3d}  M_n(c)={M}")
        print(f"   routeA-B               = {ab:.2e}")
        print(f"   quad(G24 vs G40)       = {conv:.2e}")
        print(f"   symmetry |B-B^T|       = {sym:.2e}")
        print(f"   eig in [0,1]           = {in01}   (lam_min={lam.min():.2e}, lam_max={lam.max():.6f})")
        print(f"   #lam>0.5               = {ngt}   (vs M_n(c)={M}; M+1 ok when the knee sits >1/2)")
        print(f"   leading lam            = " + ", ".join(f"{v:.6f}" for v in lam[:min(6, P)]))
        if not np.isnan(bench_plateau):
            print(f"   bench plateau (m<=M-2) = {bench_plateau:.2e}   <- correctness metric")
            print(f"   bench knee (m~M)       = {bench_knee:.2e}   <- ~1/P, documented (Task 11.2)")
        summary.append(dict(n=n, c=c, P=P, M=M, ab=ab, conv=conv, sym=sym, in01=in01,
                            ngt=ngt, bench_plateau=bench_plateau, bench_knee=bench_knee))
    return summary


if __name__ == "__main__":
    worst_reanchor = reanchor_n0()

    # n >= 1 cases: choose P comfortably above M_n(c)+plunge so the leading plateau is resolved.
    # M_n(c) ~ c/pi - n/2; pick P ~ c/pi + 12 .. 16.
    cases = [
        (1, 20.0, 30), (1, 40.0, 48),
        (2, 20.0, 30), (2, 40.0, 48),
        (4, 20.0, 32), (4, 40.0, 50), (4, 80.0, 90),   # n=4, c=80 stretch
    ]
    summary = verify_general_n(cases)

    print("\n" + "=" * 96)
    print("OVERALL")
    print("=" * 96)
    ab_max = max(s["ab"] for s in summary)
    conv_max = max(s["conv"] for s in summary)
    sym_max = max(s["sym"] for s in summary)
    all01 = all(s["in01"] for s in summary)
    count_ok = all(abs(s["ngt"] - s["M"]) <= 1 for s in summary)
    plateau_max = max((s["bench_plateau"] for s in summary if not np.isnan(s["bench_plateau"])),
                      default=float("nan"))
    knee_max = max((s["bench_knee"] for s in summary if not np.isnan(s["bench_knee"])), default=float("nan"))
    print(f"  n=0 re-anchor worst              : {worst_reanchor:.2e}   (exact match to pilot)")
    print(f"  route A-B  worst (n>=1)          : {ab_max:.2e}")
    print(f"  quad conv  worst (n>=1)          : {conv_max:.2e}")
    print(f"  symmetry   worst (n>=1)          : {sym_max:.2e}")
    print(f"  eig in [0,1] all cases           : {all01}")
    print(f"  #lam>0.5 within 1 of M_n(c)      : {count_ok}")
    print(f"  bench PLATEAU worst (correctness): {plateau_max:.2e}")
    print(f"  bench KNEE   worst (~1/P, info)  : {knee_max:.2e}  (documented Task 11.2; not a gate)")
    ok = (worst_reanchor < 1e-12 and ab_max < 1e-12 and conv_max < 1e-10
          and sym_max < 1e-14 and all01 and count_ok and plateau_max < 1e-4)
    print(f"\n  TASK 12.1 ASSEMBLY VERIFIED: {ok}")
