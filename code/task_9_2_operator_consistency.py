"""
Task 9.2 — numerical illustration of operator consistency (Level 1, NORM convergence).

Theorem: T_N := P_N T P_N -> T in operator norm as N -> inf (T = B_R B_K B_R compact, {phi_k}
complete ONB, P_N orthogonal projection onto V_{N-1}). The finite matrix B_K^{(B)}_{N-1} is the
matrix of T_N (Task 6.7/5.9, B_R^disc = I).

Illustration at FIXED c (the F1 family): approximate T by T_ref = the M_ref x M_ref Galerkin
matrix (M_ref large); for M < M_ref, the compression error ||T_ref - P_M T_ref P_M||_2 = the
operator norm of the part of T_ref outside the top-left M x M block. It -> 0 as M grows,
super-exponentially once M > M_n(c).
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def galerkin(n, c, M):
    """M x M Galerkin matrix <phi_m, T phi_k> at fixed bandwidth c (integral upper limit c)."""
    jz = jn_zeros(n, M); j = jz[:M]; pts = [z for z in j if 0 < z < c]
    B = np.zeros((M, M))
    for a in range(M):
        for b in range(a, M):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, c, points=pts, limit=400); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    return B

n = 0; c = 20.0
Mn = sum(1 for z in jn_zeros(n, 60) if z < c)
M_ref = 44
Tref = galerkin(n, c, M_ref)           # ~ T (the operator, in the FB basis, fixed c)
print(f"n={n}, c={c}, M_n(c)={Mn}, reference M_ref={M_ref}")
print("operator-consistency norm error  ||T_ref - P_M T_ref P_M||_2  (-> 0 as M grows):")
prev = None
for M in (6, 8, 12, 16, 20, 24, 28):
    comp = np.zeros_like(Tref); comp[:M, :M] = Tref[:M, :M]     # P_M T_ref P_M
    err = np.linalg.norm(Tref - comp, 2)
    rate = "" if prev is None else f"  (ratio {err/prev:.2e})"
    print(f"   M={M:3d} (M-1={M-1:3d} vs M_n={Mn}):  {err:.3e}{rate}")
    prev = err
print("\nReading: norm error decays as M grows, rapidly once M-1 exceeds M_n(c)=%d" % Mn)
print("-- confirming T_N -> T in operator norm (Level 1, the strongest operator-consistency form).")
