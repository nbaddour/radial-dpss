"""
Task 8.7 — verification of the nondimensional reconstruction formula.

Nondim radial variable x = r/R in [0,1], measure x dx; nondim basis
    phitil_{n,k}(x) = sqrt(2) J_n(j_{n,k} x) / J_{n+1}(j_{n,k}),   orthonormal in L^2([0,1], x dx).
Nondim reconstruction (SAME coefficients c_k^{(m)}, depend only on (n,c,N)):
    psitil_N^{(m)}(x) = sqrt(2) sum_k [c_k^{(m)}/J_{n+1}(j_{n,k})] J_n(j_{n,k} x),  x in [0,1].
Relations: psi_N^{(m)}(r) = (1/R) psitil_N^{(m)}(r/R);  nodes x_k = r_k/R = j_{n,k}/c;
||psitil||^2_{x dx} = 1; eigenvectors of B and c^2*Btilde are identical (coeffs scale-invariant).
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

for n in (0, 1):
    N = 6; jz = jn_zeros(n, N); j = jz[:N-1]; jN = jz[N-1]; c = jN; K = 1.0; R = jN/K; P = N-1
    # dimensional basis and nondim basis
    phi = lambda l, r: (np.sqrt(2)/R)*jn(n, j[l]*r/R)/jn(n+1, j[l])
    phitil = lambda l, x: np.sqrt(2)*jn(n, j[l]*x)/jn(n+1, j[l])
    # finite radial prolate matrix B^(B) (depends only on n,c) and nondim Btilde = B/c^2
    B = np.zeros((P, P)); pts = [z for z in jz if 0 < z < c]
    for a in range(P):
        for b in range(a, P):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, c, points=pts, limit=400); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    _, Vc = np.linalg.eigh(B); cm = Vc[:, np.argsort(_)[::-1][0]]        # c^{(0)}
    _, Vt = np.linalg.eigh(B/c**2)                                       # eigvecs of Btilde
    cm_til = Vt[:, np.argsort(_)[::-1][0]]
    print(f"--- n={n}, N={N}, c={c:.4f} ---")
    # 1. eigenvectors identical for B and Btilde=B/c^2 (coefficients scale-invariant)
    print("  eigvec(B) vs eigvec(B/c^2): |dot| =", abs(cm@cm_til), " (=1 => same coefficients)")
    # 2. nondim basis orthonormal in x dx
    onorm, _ = quad(lambda x: phitil(0, x)**2*x, 0, 1, limit=200)
    print("  ||phitil_{n,1}||^2_{x dx} (=1):", onorm)
    # 3. nondim reconstruction unit norm
    psitil = lambda x: sum(cm[l]*phitil(l, x) for l in range(P))
    ntil, _ = quad(lambda x: psitil(x)**2*x, 0, 1, limit=200)
    print("  ||psitil_N^{(0)}||^2_{x dx} (=1):", ntil)
    # 4. dim<->nondim relation: psi(r) = (1/R) psitil(r/R), i.e. psitil(x) = R psi(Rx)
    psi = lambda r: sum(cm[l]*phi(l, r) for l in range(P))
    xs = np.linspace(0.05, 0.95, 7)
    rel = max(abs(R*psi(R*x) - psitil(x)) for x in xs)
    print("  max|R*psi(Rx) - psitil(x)| over x:", rel)
    # 5. nodes x_k = j_k/c
    xk = j/c
    print("  x_k = j_k/c in (0,1)?", np.all((xk > 0) & (xk < 1)), " x_k:", np.array2string(xk, precision=4))
    print()
