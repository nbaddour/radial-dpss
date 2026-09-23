"""
Task 8.5 — illustrate the main approximation claim at the nodal level:
the m-th finite radial prolate function's nodal values psi_M^{(m)}(r_k) approximate the sampled
values of the m-th CPSWF psi^{(m)}(r_k), super-exponentially for m < M_n(c).

Setup (fixed c, R=1 so r_k = j_{n,k}/c): a high-resolution reference (M_ref large) gives
psi_ref^{(m)} ~ CPSWF; smaller M is the finite approximation. Compare nodal values at the common
interior nodes r_k = j_{n,k}/c < 1 (k with j_{n,k} < c). Sign-aligned to the reference.

Expected: super-exp small nodal error for well-separated leading modes; O(1) for plunge/tail modes;
and (separately) clustered plateau modes (lambda ~ 1 to <1e-6) show rotation noise = the
mode-identification error of Task 8.4, NOT a failure of the approximation claim.
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def prolate_funcs(n, c, M, R=1.0):
    jz = jn_zeros(n, M); j = jz[:M]
    pts = [z for z in j if 0 < z < c]
    B = np.zeros((M, M))
    for a in range(M):
        for b in range(a, M):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, c, points=pts, limit=400); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    w, V = np.linalg.eigh(B); idx = np.argsort(w)[::-1]
    lam, C = w[idx], V[:, idx]
    phi = lambda l, r: (np.sqrt(2)/R)*jn(n, j[l]*r/R)/jn(n+1, j[l])
    def psi(m, r):   # m-th reconstructed function at r
        return sum(C[l, m]*phi(l, r) for l in range(M))
    return lam, psi, j

n = 0; c = 20.0; R = 1.0
nodes = [z/c for z in jn_zeros(n, 10) if z/c < 1.0]    # interior grid nodes r_k = j_k/c < 1
Mn = sum(1 for z in jn_zeros(n, 40) if z < c)
lam_ref, psi_ref, _ = prolate_funcs(n, c, 30)
print(f"=== n={n}, c={c}, M_n(c)={Mn};  {len(nodes)} interior nodes r_k=j_k/c ===")
print(f"reference lambda_m (M=30), m=0..7: " + ", ".join(f"{x:.5f}" for x in lam_ref[:8]))
print()
for M in (8, 12):
    lam, psi, _ = prolate_funcs(n, c, M)
    print(f"M={M}: per-mode max nodal error |psi_M^(m)(r_k) - psi_ref^(m)(r_k)| (sign-aligned):")
    for m in range(min(M, 8)):
        vM = np.array([psi(m, rk) for rk in nodes])
        vR = np.array([psi_ref(m, rk) for rk in nodes])
        if vM@vR < 0:
            vM = -vM
        err = np.max(np.abs(vM - vR))
        gap = min(abs(lam[m]-lam[m-1]) if m > 0 else 9, abs(lam[m]-lam[m+1]) if m+1 < M else 9)
        tag = "  <- clustered (mode-id, src 4)" if gap < 1e-5 else ("  <- plunge/tail" if lam[m] < 0.5 else "")
        print(f"    m={m}: err={err:.2e}   (lambda={lam[m]:.5f}, nbr-gap={gap:.1e}){tag}")
    print()
