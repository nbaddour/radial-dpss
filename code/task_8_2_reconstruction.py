"""
Task 8.2 — verification of the explicit reconstruction formula and its normalization constants.

Reconstruction (synthesis, Candidate B):
    psi_N^{(m)}(r) = sum_{k=1}^{N-1} c_k^{(m)} * (sqrt(2)/R) * J_n(j_{n,k} r/R) / J_{n+1}(j_{n,k})
Checks: ||psi||^2_{r dr} = 1 for true finite-radial-prolate eigenvectors; value at node
psi(r_k) = (K/R)(S c)_k; cardinal-function property L_k(r_j)=delta; reconstruction of an in-space
function is exact; dimensional sanity (phi has units 1/length, c dimensionless, psi units 1/length).
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def setup(n, N):
    jz = jn_zeros(n, N); j = jz[:N-1]; jN = jz[N-1]; K = 1.0; R = jN/K; P = N-1
    phi = lambda l, r: (np.sqrt(2)/R)*jn(n, j[l]*r/R)/jn(n+1, j[l])
    r_nodes = j/K
    Phi = np.array([[phi(l, r_nodes[k]) for l in range(P)] for k in range(P)])
    D = np.diag(R*jn(n+1, j)/np.sqrt(2))
    T = np.array([[2/(jn(n+1, j[a])*jn(n+1, j[b])*jN)*jn(n, j[a]*j[b]/jN) for b in range(P)] for a in range(P)])
    S = D@T
    # finite radial prolate matrix B^(B)
    B = np.zeros((P, P)); pts = [z for z in jz if 0 < z < jN]
    for a in range(P):
        for b in range(a, P):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, jN, points=pts, limit=400); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    w_, V = np.linalg.eigh(B); idx = np.argsort(w_)[::-1]; C = V[:, idx]
    return dict(n=n, N=N, P=P, j=j, jN=jN, K=K, R=R, phi=phi, r=r_nodes, Phi=Phi, S=S, C=C, lam=w_[idx])

for n in (0, 1):
    e = setup(n, 6); P, R, K, phi, r, Phi, S, C = e['P'], e['R'], e['K'], e['phi'], e['r'], e['Phi'], e['S'], e['C']
    print(f"--- n={n}, N={e['N']} ---")
    # 1. ||psi_N^{(m)}||^2 = 1 for true eigenvectors
    norms = []
    for m in range(P):
        c = C[:, m]
        g = lambda rr: sum(c[l]*phi(l, rr) for l in range(P))
        val, _ = quad(lambda rr: g(rr)**2*rr, 0, R, limit=200)
        norms.append(val)
    print("  ||psi_N^{(m)}||^2 (should all be 1):", np.array2string(np.array(norms), precision=8))
    # 2. value at node psi(r_k) = (K/R)(S c)_k  for m=0
    c0 = C[:, 0]
    nodal = np.array([sum(c0[l]*phi(l, rk) for l in range(P)) for rk in r])
    print("  max|psi(r_k) - (K/R)(S c)_k| (m=0):", np.max(np.abs(nodal-(K/R)*S@c0)))
    # 3. cardinal property: L_k(r_j)=delta, L_k(r)=sum_l (Phi^{-1})_{lk} phi_l(r)
    Phinv = np.linalg.inv(Phi)
    Lk_at_nodes = np.array([[sum(Phinv[l, k]*phi(l, r[jx]) for l in range(P)) for k in range(P)] for jx in range(P)])
    print("  cardinal L_k(r_j)=delta  max|Lk@nodes - I|:", np.max(np.abs(Lk_at_nodes-np.eye(P))))
    # 4. reconstruct in-space function exactly: take coeff cc, samples a=Phi cc, recover via Phi^{-1}
    cc = np.random.randn(P); a = Phi@cc
    rec = np.linalg.solve(Phi, a)
    print("  exact reconstruction of coeffs from samples:", np.max(np.abs(rec-cc)))
    # 5. boundary: psi(R)=0
    print("  psi_N^{(0)}(R) (should be ~0):", abs(sum(c0[l]*phi(l, R) for l in range(P))))
    # 6. normalization dictionary: c_k = (R J_{n+1}(j_k)/sqrt2) f_k  where f(r)=sum f_k J_n(j_k r/R)
    j = e['j']; fk = np.random.randn(P)
    ffun = lambda rr: sum(fk[l]*jn(n, j[l]*rr/R) for l in range(P))
    ck = np.array([quad(lambda rr: ffun(rr)*phi(k, rr)*rr, 0, R, limit=200)[0] for k in range(P)])
    Dkk = R*jn(n+1, j)/np.sqrt(2)
    print("  c_k = (R J_{n+1}/sqrt2) f_k  max err:", np.max(np.abs(ck - Dkk*fk)))
    print("  (5.7 §2.2 prose 'c_k=(sqrt2/(R J_{n+1})) f_k' would err by:",
          np.max(np.abs(ck - (np.sqrt(2)/(R*jn(n+1, j)))*fk)), ")")
    print()
