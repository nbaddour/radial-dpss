"""
Task 8.4 — illustrate that the four error sources are distinct and separately identifiable.

(1) Discretization error: at FIXED c, the finite matrix B_M(c) (size M=N-1) is the Galerkin
    compression of the continuous operator to V_M; lambda_m^{(M)} -> lambda_m^{cont} as M grows.
    (Use the fixed-c family, NOT the closure c=j_{n,N}, since closure ties c to N.)
(2) Truncation error: the true eigenfunction's Fourier-Bessel tail (coeffs c_k for k > M) that a
    size-M reconstruction discards; super-exponentially small for m < M_n(c).
(3) Reconstruction error: ZERO via the coefficient route (S_N exact isometry); O(eps_N) only if the
    reconstruction passes through the sample/DHT domain (Baddour residual, C2).
(4) Mode-identification error: eigenvalue clustering (min gap) that threatens index<->mode matching,
    dominant in the plunge region.
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def B_fixed_c(n, c, M):
    """Finite radial prolate matrix of size M at FIXED bandwidth c (integral upper limit = c)."""
    jz = jn_zeros(n, M); j = jz[:M]
    pts = [z for z in j if 0 < z < c]
    B = np.zeros((M, M))
    for a in range(M):
        for b in range(a, M):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, c, points=pts, limit=400); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    w, V = np.linalg.eigh(B); idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx], j

n = 0; c = 20.0
print(f"=== fixed bandwidth c={c}, n={n};  M_n(c)=#{{j_n,k<c}} = {sum(1 for k in range(1,40) if jn_zeros(n,40)[k-1]<c)} ===\n")
# (1) DISCRETIZATION: eigenvalue convergence as matrix size M grows at fixed c
lams = {}
for M in (7, 12, 19):
    lam, V, j = B_fixed_c(n, c, M); lams[M] = (lam, V)
ref = lams[19][0]
print("(1) DISCRETIZATION error (eigenvalues converge as M grows at fixed c):")
for M in (7, 12):
    lam = lams[M][0]
    diff = [abs(lam[m]-ref[m]) for m in range(min(len(lam), 6))]
    print(f"    M={M:2d}: |lambda_m^(M) - lambda_m^(19)|, m=0..5 = " + ", ".join(f"{d:.2e}" for d in diff))
print(f"    lambda_m^(19) (reference), m=0..6: " + ", ".join(f"{x:.6f}" for x in ref[:7]))

# (2) TRUNCATION: FB-coefficient tail of the M=19 eigenvector beyond index 7 (what M=7 discards)
print("\n(2) TRUNCATION error (FB tail beyond k=7 in the M=19 eigenvector; what an M=7 cut discards):")
V19 = lams[19][1]
for m in (0, 3, 5, 6):
    tail = np.sqrt(np.sum(V19[7:, m]**2))   # energy in coeffs k=8..19 (0-indexed 7:)
    print(f"    m={m}: ||tail_{{k>7}}|| = {tail:.2e}   (lambda_m={ref[m]:.5f})")

# (3) RECONSTRUCTION: exact via coefficients; eps_N via samples (closure regime, N=8)
print("\n(3) RECONSTRUCTION error:")
N = 8; jz = jn_zeros(0, N); j = jz[:N-1]; jN = jz[N-1]; K = 1.0; R = jN/K; P = N-1
phi = lambda l, r: (np.sqrt(2)/R)*jn(0, j[l]*r/R)/jn(1, j[l])
Phi = np.array([[phi(l, j[k]/K) for l in range(P)] for k in range(P)])
Wd = np.diag(2/(K**2*jn(1, j)**2))
Bc = np.zeros((P, P)); ptsN = [z for z in jz if 0 < z < jN]
for a in range(P):
    for b in range(a, P):
        f = lambda u: u*jn(0, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
        v, _ = quad(f, 0, jN, points=ptsN, limit=400); Bc[a, b] = Bc[b, a] = 2*j[a]*j[b]*v
_, Vc = np.linalg.eigh(Bc); c0 = Vc[:, np.argsort(_)[::-1][0]]
g = lambda r: sum(c0[l]*phi(l, r) for l in range(P))
norm2, _ = quad(lambda r: g(r)**2*r, 0, R, limit=200)
A = Phi@Vc
print(f"    coefficient route (S_N exact isometry): | ||psi||^2 - 1 | = {abs(norm2-1):.2e}   [ZERO]")
print(f"    sample route (DHT/quadrature, C2):      ||A^T W A - I|| = {np.max(np.abs(A.T@Wd@A-np.eye(P))):.2e}   [eps_N]")

# (4) MODE-IDENTIFICATION: eigenvalue clustering (min gap) in the closure regime
print("\n(4) MODE-IDENTIFICATION error (eigenvalue gaps; small gap => index<->mode ambiguity):")
for Nc in (6, 8):
    jz2 = jn_zeros(0, Nc); jj = jz2[:Nc-1]; jN2 = jz2[Nc-1]; Pp = Nc-1
    Bm = np.zeros((Pp, Pp)); pp = [z for z in jz2 if 0 < z < jN2]
    for a in range(Pp):
        for b in range(a, Pp):
            f = lambda u: u*jn(0, u)**2/((u**2-jj[a]**2)*(u**2-jj[b]**2))
            v, _ = quad(f, 0, jN2, points=pp, limit=400); Bm[a, b] = Bm[b, a] = 2*jj[a]*jj[b]*v
    lam = np.sort(np.linalg.eigvalsh(Bm))[::-1]
    gaps = np.abs(np.diff(lam))
    print(f"    closure N={Nc} (c=j_n,N): min eigenvalue gap = {gaps.min():.2e}  (clustered => mode-id hard)")
