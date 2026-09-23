"""
Task 7.5 — n >= 1 confirmation of the finite orthogonality residual sizes.

Generalization of task_7_5_orthogonality_verification.py (which fixed n=0) to arbitrary
Bessel order n, by replacing the hardcoded J_0/J_1 with J_n/J_{n+1} and j_{0,k} with j_{n,k}.
Loops over n = 0, 1, 2 at fixed N = 8 (matrix size N-1 = 7, closure c = j_{n,N}).

Purpose (Task 7.5 Open Q3): confirm that the C1/C2 picture and the Baddour residual sizes
carry over from n=0 to n>=1:
  - T^2 = I and S^T W S = (R/K)^2 I are approximate (Baddour residual; C2),
  - Phi = (K/R) S exact; S^{-1}c (DHT image) != nodal Phi c (C1),
  - S^{-1}c is exactly (S^T S)-orthonormal but NOT W-orthonormal,
  - nodal Phi c is W-orthonormal to the Baddour residual.
n=0 row reproduces the original script's numbers as a regression check.
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad
np.set_printoptions(precision=4, suppress=True, linewidth=140)


def run(n, N=8):
    jz = jn_zeros(n, N)        # j_{n,1} .. j_{n,N}
    j = jz[:N-1]               # interior nodes k=1..N-1
    jN = jz[N-1]               # closure j_{n,N}
    K = 1.0; R = jN/K; c = K*R
    P = N-1

    # B_K^{(B)} in the orthonormal Fourier-Bessel basis (Task 6.1 (3.1) entries)
    B = np.zeros((P, P))
    pts = [z for z in jz if 0 < z < c]
    for a in range(P):
        for b in range(a, P):
            jm, jk = j[a], j[b]
            f = lambda u: u*jn(n, u)**2/((u**2-jm**2)*(u**2-jk**2))
            v, _ = quad(f, 0, c, points=pts, limit=400)
            B[a, b] = B[b, a] = 2*jm*jk*v
    w_, V = np.linalg.eigh(B)
    idx = np.argsort(w_)[::-1]
    lam = w_[idx]; C = V[:, idx]      # columns c^{(m)}, orthonormal eigenvectors

    # core matrices (Tasks 5.7, 5.8; Baddour T)
    D = np.diag(R*jn(n+1, j)/np.sqrt(2))
    W = np.diag(2/(K**2*jn(n+1, j)**2))
    T = np.array([[2/(jn(n+1, j[a])*jn(n+1, j[b])*jN)*jn(n, j[a]*j[b]/jN)
                   for b in range(P)] for a in range(P)])
    S = D@T
    # nodal sample matrix Phi[k,l] = phi_{n,l}(r_k), r_k = j_{n,k}/K
    Phi = np.array([[(np.sqrt(2)/R)*jn(n, j[a]*j[b]/jN)/jn(n+1, j[b])
                     for b in range(P)] for a in range(P)])

    res = {}
    res['n'] = n
    res['R/K'] = R/K
    res['(R/K)^2'] = (R/K)**2
    res['T^2 - I'] = np.max(np.abs(T@T - np.eye(P)))
    res['D^2 W - (R/K)^2 I'] = np.max(np.abs(D@D@W - (R/K)**2*np.eye(P)))
    res['S^T W S - (R/K)^2 I'] = np.max(np.abs(S.T@W@S - (R/K)**2*np.eye(P)))
    res['Phi - (K/R)S'] = np.max(np.abs(Phi - (K/R)*S))
    res['C^T C - I'] = np.max(np.abs(C.T@C - np.eye(P)))

    A_S = np.linalg.solve(S, C)   # a = S^{-1} c (DHT image), columns
    A_Phi = Phi@C                 # nodal samples = (K/R) S c, columns
    res['||S^-1 c  -  nodal||'] = np.max(np.abs(A_S - A_Phi))

    GW_S = A_S.T@W@A_S
    GW_P = A_Phi.T@W@A_Phi
    GSS_S = A_S.T@(S.T@S)@A_S
    GSS_P = A_Phi.T@(S.T@S)@A_Phi
    res['S^-1c: W-Gram diag min/max'] = (np.diag(GW_S).min(), np.diag(GW_S).max())
    res['S^-1c: W-Gram offdiag'] = np.max(np.abs(GW_S - np.diag(np.diag(GW_S))))
    res['S^-1c: (S^TS)-Gram - I'] = np.max(np.abs(GSS_S - np.eye(P)))
    res['nodal: W-Gram - I'] = np.max(np.abs(GW_P - np.eye(P)))
    res['nodal: (S^TS)-Gram diag min/max'] = (np.diag(GSS_P).min(), np.diag(GSS_P).max())
    res['lambda (sorted)'] = np.round(lam, 6)
    return res


for n in (0, 1, 2):
    r = run(n)
    print(f"\n================  n = {n}   (N=8, P=7, c=j_n,N)  ================")
    print(f"  R/K = {r['R/K']:.4f},  (R/K)^2 = {r['(R/K)^2']:.2f}")
    print("  --- C2: approximate identities (Baddour residual, amplified by (R/K)^2) ---")
    print(f"  T^2 - I                  max abs : {r['T^2 - I']:.3e}")
    print(f"  D^2 W - (R/K)^2 I        max abs : {r['D^2 W - (R/K)^2 I']:.3e}   (exact, diagonal)")
    print(f"  S^T W S - (R/K)^2 I      max abs : {r['S^T W S - (R/K)^2 I']:.3e}   (= (R/K)^2 * (T^2-I))")
    print("  --- C1: nodal vs DHT-image vector ---")
    print(f"  Phi - (K/R)S             max abs : {r['Phi - (K/R)S']:.3e}   (exact nodal map)")
    print(f"  || S^-1 c  -  nodal ||   max abs : {r['||S^-1 c  -  nodal||']:.4f}   (distinct vectors)")
    print(f"  C^T C - I                max abs : {r['C^T C - I']:.3e}")
    print("  S^-1 c  (DHT image): exactly (S^T S)-orthonormal, NOT W-orthonormal")
    print(f"     (S^TS)-Gram - I       max abs : {r['S^-1c: (S^TS)-Gram - I']:.3e}")
    print(f"     W-Gram diag (min,max)         : {r['S^-1c: W-Gram diag min/max'][0]:.4f}, {r['S^-1c: W-Gram diag min/max'][1]:.4f}")
    print(f"     W-Gram offdiag        max abs : {r['S^-1c: W-Gram offdiag']:.4f}")
    print("  nodal Phi c: W-orthonormal to the Baddour residual")
    print(f"     W-Gram - I            max abs : {r['nodal: W-Gram - I']:.3e}")
    print(f"  lambda: {r['lambda (sorted)']}")
