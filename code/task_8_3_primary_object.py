"""
Task 8.3 — evidence for the primary-computational-object decision.

Compares the three candidate "primary objects" of the finite radial prolate framework on the
properties that decide the choice:
  (A) coefficient eigenvector c^{(m)}  (Candidate B)
  (B) nodal-sample vector a^{(m)} = Phi c^{(m)} = (K/R) S c^{(m)}  (Candidate A nodal values)
  (C) reconstructed continuous function psi_N^{(m)} = S_N[c^{(m)}]

Facts checked:
  1. coefficient orthonormality C^T C = I  -- EXACT (machine)
  2. nodal-sample W-orthonormality a^T W a' -- APPROXIMATE (Baddour residual, C2)
  3. synthesis isometry ||psi||^2 = ||c||^2 -- EXACT
  4. off-grid evaluation requires reconstruction: psi(r*) for r* not a node is obtained from c
     (the 8.2 formula); from samples it requires the interpolation map I_N = S_N o Phi^{-1}
     (i.e. reconstruction). Confirm I_N[a](r*) = S_N[c](r*).
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def setup(n, N):
    jz = jn_zeros(n, N); j = jz[:N-1]; jN = jz[N-1]; K = 1.0; R = jN/K; P = N-1
    phi = lambda l, r: (np.sqrt(2)/R)*jn(n, j[l]*r/R)/jn(n+1, j[l])
    r = j/K
    Phi = np.array([[phi(l, r[k]) for l in range(P)] for k in range(P)])
    D = np.diag(R*jn(n+1, j)/np.sqrt(2)); W = np.diag(2/(K**2*jn(n+1, j)**2))
    T = np.array([[2/(jn(n+1, j[a])*jn(n+1, j[b])*jN)*jn(n, j[a]*j[b]/jN) for b in range(P)] for a in range(P)])
    S = D@T
    B = np.zeros((P, P)); pts = [z for z in jz if 0 < z < jN]
    for a in range(P):
        for b in range(a, P):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, jN, points=pts, limit=400); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    w_, V = np.linalg.eigh(B); idx = np.argsort(w_)[::-1]; C = V[:, idx]
    return dict(P=P, R=R, K=K, phi=phi, r=r, Phi=Phi, S=S, W=W, C=C)

for n in (0, 1):
    e = setup(n, 8); P, R, K, phi, C, Phi, S, W = e['P'], e['R'], e['K'], e['phi'], e['C'], e['Phi'], e['S'], e['W']
    # 1. coefficient orthonormality (exact)
    coeff_orth = np.max(np.abs(C.T@C - np.eye(P)))
    # 2. nodal-sample W-orthonormality (approximate, C2)
    A = Phi@C                          # nodal-sample vectors, columns
    GW = A.T@W@A
    nodal_orth = np.max(np.abs(GW - np.eye(P)))
    # 3. synthesis isometry (exact): ||psi_N^{(0)}||^2 vs ||c||^2=1
    c0 = C[:, 0]
    g = lambda r: sum(c0[l]*phi(l, r) for l in range(P))
    norm2, _ = quad(lambda r: g(r)**2*r, 0, R, limit=200)
    # 4. off-grid evaluation: pick r* between nodes; value from coeffs vs interpolation-from-samples
    rstar = 0.5*(e['r'][2] + e['r'][3])
    val_from_coeffs = sum(c0[l]*phi(l, rstar) for l in range(P))
    a0 = Phi@c0                        # nodal samples of psi_N^{(0)}
    c_from_samples = np.linalg.solve(Phi, a0)             # I_N: Phi^{-1} a  (= reconstruction)
    val_from_samples = sum(c_from_samples[l]*phi(l, rstar) for l in range(P))
    print(f"--- n={n}, N=8 ---")
    print(f"  (A) coeff orthonormality  ||C^T C - I||        = {coeff_orth:.2e}   [EXACT]")
    print(f"  (B) nodal W-orthonormality ||A^T W A - I||      = {nodal_orth:.2e}   [APPROX, Baddour resid C2]")
    print(f"  (C) synthesis isometry    | ||psi||^2 - 1 |     = {abs(norm2-1):.2e}   [EXACT]")
    print(f"  (4) off-grid r*={rstar:.4f}: value from coeffs   = {val_from_coeffs:.10f}")
    print(f"      same via interpolation-from-samples I_N[a]  = {val_from_samples:.10f}")
    print(f"      |diff| = {abs(val_from_coeffs-val_from_samples):.2e}  (off-grid value needs reconstruction)")
    print()
