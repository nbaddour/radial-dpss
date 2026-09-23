"""
Task 8.6 — exactness audit: separate (1) exact-at-every-finite-N identities from (2) the ε_N
(DHT/quadrature) family that is exact only as N->infinity.

Representative members (all computable from the grid/basis matrices alone, no eigendecomposition,
so reliable at large N despite eigenvalue clustering):
  EXACT  (category 1): Phi = (K/R)S  ;  D^2 W = (R/K)^2 I            -> ~machine, N-independent
  eps_N  (category 2): T^2 = I (Baddour Eq 47) ; Phi^T W Phi = I     -> shrink with N (Baddour resid)

(Truncation/discretization -- category 3 -- is mode-dependent and is treated in Tasks 8.4/8.5;
 not re-run here.)
"""
import numpy as np
from scipy.special import jn, jn_zeros

def matrices(n, N):
    jz = jn_zeros(n, N); j = jz[:N-1]; jN = jz[N-1]; K = 1.0; R = jN/K; P = N-1
    phi = lambda l, r: (np.sqrt(2)/R)*jn(n, j[l]*r/R)/jn(n+1, j[l])
    r = j/K
    Phi = np.array([[phi(l, r[k]) for l in range(P)] for k in range(P)])
    D = np.diag(R*jn(n+1, j)/np.sqrt(2)); W = np.diag(2/(K**2*jn(n+1, j)**2))
    T = np.array([[2/(jn(n+1, j[a])*jn(n+1, j[b])*jN)*jn(n, j[a]*j[b]/jN) for b in range(P)] for a in range(P)])
    S = D@T
    return dict(P=P, K=K, R=R, Phi=Phi, D=D, W=W, T=T, S=S)

n = 0
print("                       CATEGORY 1 (EXACT, finite N)        |     CATEGORY 2 (eps_N: exact as N->inf)")
print("  N    P    ||Phi-(K/R)S||   ||D^2 W-(R/K)^2 I||  |  ||T^2-I||      ||Phi^T W Phi - I||")
for N in (8, 16, 24, 32, 48):
    m = matrices(n, N); P, K, R, Phi, D, W, T, S = m['P'], m['K'], m['R'], m['Phi'], m['D'], m['W'], m['T'], m['S']
    e_phi = np.max(np.abs(Phi - (K/R)*S))
    e_d2w = np.max(np.abs(D@D@W - (R/K)**2*np.eye(P)))
    e_t2 = np.max(np.abs(T@T - np.eye(P)))
    e_pwp = np.max(np.abs(Phi.T@W@Phi - np.eye(P)))
    print(f"  {N:3d}  {P:3d}    {e_phi:.2e}        {e_d2w:.2e}      |  {e_t2:.2e}     {e_pwp:.2e}")
print()
print("Reading: category 1 stays at ~machine (N-independent, EXACT); category 2 shrinks toward 0")
print("as N grows (Baddour discrete-orthogonality residual eps_N: 10^-3 small N -> 10^-7 for N>30).")
