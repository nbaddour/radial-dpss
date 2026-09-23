"""
Task 8.1 — numerical verification of the synthesis / interpolation map properties.

Maps (dimensional, primary):
  Synthesis     S_N : R^{N-1} -> V_{N-1},  c |-> sum_k c_k phi_{n,k}(r)        (exact isometry)
  Analysis      A_N = S_N^{-1} : g |-> (<g,phi_{n,k}>)_k
  Nodal sample  Nod : V_{N-1} -> R^{N-1}, g |-> (g(r_k))_k = Phi c, Phi=(K/R)S  (exact)
  Interpolation I_N : R^{N-1} -> V_{N-1}, a |-> S_N[Phi^{-1} a]                 (exact interpolation)

Checks: isometry of S_N; exact interpolation I_N[a](r_j)=a_j; Phi=(K/R)S and Phi invertible;
nodal samples of S_N[c^{(m)}] equal (K/R)(S c^{(m)}); W-quadrature Parseval to the Baddour residual;
and that the DHT-image vector S^{-1}c is NOT the nodal-sample vector (C1).
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def setup(n, N):
    jz = jn_zeros(n, N); j = jz[:N-1]; jN = jz[N-1]; K = 1.0; R = jN/K; P = N-1
    phi = lambda l, r: (np.sqrt(2)/R)*jn(n, j[l]*r/R)/jn(n+1, j[l])   # orthonormal FB basis, 0-indexed l
    r_nodes = j/K
    Phi = np.array([[phi(l, r_nodes[k]) for l in range(P)] for k in range(P)])
    D = np.diag(R*jn(n+1, j)/np.sqrt(2))
    T = np.array([[2/(jn(n+1, j[a])*jn(n+1, j[b])*jN)*jn(n, j[a]*j[b]/jN) for b in range(P)] for a in range(P)])
    S = D@T
    W = np.diag(2/(K**2*jn(n+1, j)**2))
    return dict(n=n, N=N, P=P, j=j, jN=jN, K=K, R=R, phi=phi, r=r_nodes, Phi=Phi, S=S, W=W)

def inner(g, h, R, n):
    f = lambda r: g(r)*h(r)*r
    v, _ = quad(f, 0, R, limit=200)
    return v

for n in (0, 1):
    e = setup(n, 5)
    P, R, K, S, Phi, W = e['P'], e['R'], e['K'], e['S'], e['Phi'], e['W']
    phi, r = e['phi'], e['r']
    # 1. Phi = (K/R) S, invertible
    print(f"--- n={n}, N={e['N']} ---")
    print("  Phi=(K/R)S max err:", np.max(np.abs(Phi-(K/R)*S)), " cond(Phi)=%.2e"%np.linalg.cond(Phi))
    # 2. Synthesis isometry: <S_N[c],S_N[c']> = c.c'  (test on basis-coefficient random vectors)
    c1, c2 = np.random.randn(P), np.random.randn(P)
    g1 = lambda r: sum(c1[l]*phi(l, r) for l in range(P))
    g2 = lambda r: sum(c2[l]*phi(l, r) for l in range(P))
    iso = inner(g1, g2, R, n) - c1@c2
    print("  synthesis isometry <S c1,S c2> - c1.c2:", iso)
    # 3. nodal samples of S_N[c] equal Phi c = (K/R) S c (exact)
    a_true = np.array([g1(rk) for rk in r])
    print("  nodal(S_N[c]) - (K/R)Sc  max err:", np.max(np.abs(a_true-(K/R)*S@c1)))
    # 4. exact interpolation: I_N[a](r_j)=a_j
    a = np.random.randn(P)
    cI = np.linalg.solve(Phi, a)             # Phi^{-1} a
    gI = lambda r: sum(cI[l]*phi(l, r) for l in range(P))
    interp_err = np.max(np.abs(np.array([gI(rk) for rk in r]) - a))
    print("  exact interpolation  max|I_N[a](r_j)-a_j|:", interp_err)
    # 5. analysis = Phi^{-1} on nodal samples recovers coefficients
    print("  analysis Phi^{-1}(nodal) - c  max err:", np.max(np.abs(np.linalg.solve(Phi, a_true)-c1)))
    # 6. W-quadrature Parseval to Baddour residual: sum_k w_k g(r_k)^2 vs ||g||^2
    quad_norm = a_true@W@a_true
    true_norm = inner(g1, g1, R, n)
    print("  W-quadrature Parseval  |sum w_k g(r_k)^2 - ||g||^2|:", abs(quad_norm-true_norm))
    # 7. C1: DHT-image vector S^{-1}c is NOT the nodal samples
    dht_image = np.linalg.solve(S, c1)
    print("  ||S^{-1}c - nodal||:", np.max(np.abs(dht_image-a_true)), " (C1: distinct)")
    print()
