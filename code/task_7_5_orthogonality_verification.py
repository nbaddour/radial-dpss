import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad
np.set_printoptions(precision=4, suppress=True, linewidth=140)
n=0
N=8                      # matrix size N-1 = 7
jz=jn_zeros(0, N)        # j_1..j_N
j=jz[:N-1]               # interior nodes k=1..N-1
jN=jz[N-1]               # closure j_{0,N}
K=1.0; R=jN/K; c=K*R     # closure c=jN
P=N-1

# orthonormal FB basis coefficient matrix B_K^{(B)}
def BKB():
    B=np.zeros((P,P)); pts=[z for z in jz if 0<z<c]
    for a in range(P):
        for b in range(a,P):
            jm,jk=j[a],j[b]
            f=lambda u:u*jn(0,u)**2/((u**2-jm**2)*(u**2-jk**2))
            v,_=quad(f,0,c,points=pts,limit=400); B[a,b]=B[b,a]=2*jm*jk*v
    return B
B=BKB()
w_,V=np.linalg.eigh(B); idx=np.argsort(w_)[::-1]; lam=w_[idx]; C=V[:,idx]  # columns c^{(m)}

# matrices
D=np.diag(R*jn(1,j)/np.sqrt(2))
W=np.diag(2/(K**2*jn(1,j)**2))
T=np.array([[2/(jn(1,j[a])*jn(1,j[b])*jN)*jn(0,j[a]*j[b]/jN) for b in range(P)] for a in range(P)])
S=D@T
# nodal sample matrix Phi: Phi[k,l]=phi_{0,l}(r_k)
Phi=np.array([[ (np.sqrt(2)/R)*jn(0,j[a]*j[b]/jN)/jn(1,j[b]) for b in range(P)] for a in range(P)])

print("=== basic identities ===")
print("T^2 - I  max abs:", np.max(np.abs(T@T-np.eye(P))))
print("D^2 W - (R/K)^2 I max abs:", np.max(np.abs(D@D@W-(R/K)**2*np.eye(P))))
print("S^T W S - (R/K)^2 I max abs:", np.max(np.abs(S.T@W@S-(R/K)**2*np.eye(P))))
print("Phi vs S^{-1} max abs:", np.max(np.abs(Phi-np.linalg.inv(S))))
print("Phi vs (K/R)*S max abs:", np.max(np.abs(Phi-(K/R)*S)))

# eigenvector orthogonality (coefficients)
print("\n=== Candidate B (coefficients) ===")
print("C^T C - I max abs:", np.max(np.abs(C.T@C-np.eye(P))))
print("C^T B C diag (should be lambda):", np.round(np.diag(C.T@B@C),6))
print("C^T B C offdiag max abs:", np.max(np.abs(C.T@B@C-np.diag(np.diag(C.T@B@C)))))

# samples via S^{-1} and via Phi
A_S=np.linalg.solve(S,C)     # a^{(m)} = S^{-1} c^{(m)}, columns
A_Phi=Phi@C                  # nodal samples (should equal reconstruction at nodes)
print("\n=== samples: S^{-1}c vs nodal Phi c ===")
print("A_S vs A_Phi max abs:", np.max(np.abs(A_S-A_Phi)))

for label,Amat in [("A_S=S^{-1}c",A_S),("A_Phi=nodal",A_Phi)]:
    GW=Amat.T@W@Amat
    GSS=Amat.T@(S.T@S)@Amat
    print(f"\n--- {label} ---")
    print("  (a^T W a) diag:", np.round(np.diag(GW),6))
    print("  (a^T W a) offdiag max abs:", np.max(np.abs(GW-np.diag(np.diag(GW)))))
    print("  (a^T S^TS a) - I max abs:", np.max(np.abs(GSS-np.eye(P))))
