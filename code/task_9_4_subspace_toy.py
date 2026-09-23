"""
Task 9.4 (clarification) — toy showing why a near-degenerate (plateau-like) cluster gives
UNSTABLE individual eigenvectors but a STABLE subspace.

3x3 symmetric A with eigenvalues {1+eps, 1, 0.5}: the top two ({1+eps, 1}) form a near-degenerate
cluster (gap eps), well separated from the third (gap ~0.5). Perturb A by a tiny symmetric eta-sized
matrix (eta >> eps) and compare:
  (i) individual top eigenvector v1   -> rotates a lot (angle ~ eta/eps),
  (ii) top-2 subspace projector P2     -> barely moves (error ~ eta/0.5),
i.e. the *axis within the cluster* is undetermined, the *plane* is solid.
This is the finite-N plateau situation (eps = super-exp-small eigenvalue gap; eta = truncation defect).
"""
import numpy as np
np.random.seed(1)

def principal_angle_deg(u, v):
    c = abs(u @ v) / (np.linalg.norm(u)*np.linalg.norm(v))
    return np.degrees(np.arccos(min(1.0, c)))

# fixed orthonormal eigenbasis Q, eigenvalues with a near-degenerate top pair
X = np.random.randn(3, 3); Q, _ = np.linalg.qr(X)
eps = 1e-10
Lam = np.diag([1+eps, 1.0, 0.5])
A = Q @ Lam @ Q.T
A = 0.5*(A+A.T)

# tiny symmetric perturbation, eta >> eps (within-cluster) but << gap-to-third
eta = 1e-8
P = np.random.randn(3, 3); P = 0.5*(P+P.T); P = eta*P/np.linalg.norm(P, 2)
A2 = A + P

def top2_and_v1(M):
    w, V = np.linalg.eigh(M); idx = np.argsort(w)[::-1]
    V = V[:, idx]
    v1 = V[:, 0]
    P2 = V[:, :2] @ V[:, :2].T      # projector onto top-2 subspace
    return v1, P2

v1, P2 = top2_and_v1(A)
v1b, P2b = top2_and_v1(A2)

print(f"cluster gap eps = {eps:.0e}   perturbation eta = {eta:.0e}   (eta >> eps)")
print(f"(i)  individual top eigenvector rotation:  {principal_angle_deg(v1, v1b):8.3f} degrees   <- LARGE (unstable)")
print(f"(ii) top-2 subspace projector change ||P2-P2'||_2 = {np.linalg.norm(P2-P2b,2):.2e}   <- TINY (stable)")
print()
print("Interpretation: a perturbation far bigger than the intra-cluster gap (eta >> eps) swings the")
print("INDIVIDUAL top eigenvector by a large angle, yet the top-2 SUBSPACE is essentially unchanged.")
print("Finite-N plateau: eps = super-exp-small eigenvalue gap, eta = truncation defect delta_N.")
print("=> subspace converges fast (basis-free); per-mode locks on only once delta_N < eps (gap-limited).")
