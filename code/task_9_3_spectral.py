"""
Task 9.3 — numerical support for the spectral approximation statements (fixed c).

L2 (eigenvalue): |lambda_m^{(M)} - lambda_m^{ref}| -> 0, super-exp for m < M_n(c).
L3 (subspace):   top-K spectral projector ||P_K^{(M)} - P_K^{ref}|| -> 0 super-exp when K sits at a
                 spectral GAP (K = M_n(c): the plateau-plunge gap is O(1)).
Gap structure:   intra-plateau gaps are super-exp small (mode-id-hard); the plateau->plunge gap is O(1)
                 (so the leading GROUP up to that gap converges cleanly -- L3 robust, L4-per-mode not).
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def galerkin(n, c, M):
    jz = jn_zeros(n, M); j = jz[:M]; pts = [z for z in j if 0 < z < c]
    B = np.zeros((M, M))
    for a in range(M):
        for b in range(a, M):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, c, points=pts, limit=400); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    w, V = np.linalg.eigh(B); idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]

n = 0; c = 20.0; M_ref = 40
Mn = sum(1 for z in jn_zeros(n, 60) if z < c)
lam_ref, V_ref = galerkin(n, c, M_ref)
print(f"n={n}, c={c}, M_n(c)={Mn}, reference M_ref={M_ref}")
print(f"reference lambda_m, m=0..7: " + ", ".join(f"{x:.5f}" for x in lam_ref[:8]))
print(f"gap structure: intra-plateau lambda_0-lambda_1 = {lam_ref[0]-lam_ref[1]:.2e} (super-exp); "
      f"plateau->plunge lambda_{Mn-1}-lambda_{Mn} = {lam_ref[Mn-1]-lam_ref[Mn]:.3f} (O(1))\n")

print("L2 eigenvalue error |lambda_m^(M) - lambda_m^ref| (super-exp for m < M_n(c)=%d):" % Mn)
lamM = {}
for M in (8, 12, 16, 20):
    lam, V = galerkin(n, c, M); lamM[M] = (lam, V)
    errs = [abs(lam[m]-lam_ref[m]) for m in range(8)]
    print(f"  M={M:2d}: " + "  ".join(f"m{m}:{errs[m]:.1e}" for m in range(8)))

print("\nL3 leading-group subspace error ||P_K^(M) - P_K^ref||_2  (K at plateau-plunge gap, K=M_n=%d):" % Mn)
def topK_proj(V, K, dim):
    Q = np.zeros((dim, K)); Q[:V.shape[0], :] = V[:, :K]
    return Q@Q.T
Pref = topK_proj(V_ref, Mn, M_ref)
for M in (8, 12, 16, 20):
    P = topK_proj(lamM[M][1], Mn, M_ref)
    print(f"  M={M:2d}: ||P_{Mn}^(M) - P_{Mn}^ref|| = {np.linalg.norm(P-Pref,2):.2e}")
print("\nReading: L2 eigenvalue errors super-exp small for m<M_n; L3 leading-group subspace converges")
print("super-exp (the gap to the plunge is O(1)). Intra-plateau per-mode splitting is super-exp -> the")
print("per-mode L4 rate in the plateau is gap-limited (deferred to conjecture 9.4); the GROUP is clean.")
