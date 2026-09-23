import numpy as np, os
from scipy.special import jv, jvp, jn_zeros

# ---- my (3.1) partial-fraction matrix ----
def panels_nodes(c,jb,G):
    bps=np.concatenate(([0.0],jb,[c]));t,w=np.polynomial.legendre.leggauss(G);U=[];Wq=[]
    for a,b in zip(bps[:-1],bps[1:]):
        if b<=a: continue
        U.append(0.5*(b-a)*t+0.5*(a+b));Wq.append(0.5*(b-a)*w)
    return np.concatenate(U),np.concatenate(Wq)
def assemble_PF(n,c,P,G=28):
    jz=jn_zeros(n,P);jb=jz[jz<c];U,Wq=panels_nodes(c,jb,G)
    J=jv(n,U);J2=J*J;uJ2W=U*J2*Wq;j2=jz**2;J1z=jv(n+1,jz);I=np.empty(P);Dd=np.empty(P)
    for p in range(P):
        den=U*U-j2[p];near=np.abs(U-jz[p])<1e-11;ti=uJ2W/den
        if near.any(): ti[near]=0.0
        I[p]=ti.sum();td=uJ2W/den**2
        if near.any(): td[near]=(J1z[p]**2/(4*jz[p]))*Wq[near]
        Dd[p]=td.sum()
    B=np.empty((P,P))
    for m in range(P):
        B[m,m]=2*j2[m]*Dd[m]
        for k in range(m+1,P): B[m,k]=B[k,m]=2*jz[m]*jz[k]*(I[m]-I[k])/(j2[m]-j2[k])
    return B

# ---- INDEPENDENT Galerkin matrix via the 10.1 continuous finite-Hankel kernel ----
def K0(c,x):
    X=x[:,None];Y=x[None,:];cX,cY=c*X,c*Y
    JX,JY,JpX,JpY=jv(0,cX),jv(0,cY),jvp(0,cX),jvp(0,cY)
    with np.errstate(divide='ignore',invalid='ignore'):
        K=c*(X*JY*JpX-Y*JX*JpY)/(Y**2-X**2)
    cx=c*x; np.fill_diagonal(K,(c**2/2)*(jvp(0,cx)**2+jv(0,cx)**2)); return K
def phi(k_index, jz, x):   # nondim FB basis phi_{0,k}(x)=sqrt2 J0(j_k x)/J1(j_k), R=1
    return np.sqrt(2)*jv(0,jz[k_index]*x)/jv(1,jz[k_index])
def assemble_GAL(c,P,Nq=600):
    t,w=np.polynomial.legendre.leggauss(Nq);x=0.5*(t+1);w=0.5*w
    K=K0(c,x)
    jz=jn_zeros(0,P)
    Phi=np.column_stack([phi(k,jz,x) for k in range(P)])     # Nq x P, basis at nodes
    # T Phi = K @ (Phi * (x w)) ; then G = Phi^T diag(xw) (T Phi)
    TPhi = K @ (Phi*(x*w)[:,None])
    G = (Phi*(x*w)[:,None]).T @ TPhi
    G=0.5*(G+G.T)
    return G

c=20.0
print("Independent check: eigenvalues of MY (3.1) matrix vs an independent Galerkin matrix")
print("(Galerkin via the 10.1 continuous kernel + FB basis, 600-node quadrature)\n")
D="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
lref=np.load(os.path.join(D,"cpswf_n0_c20.npz"))['lam']
for P in [24,44,100]:
    lamPF=np.sort(np.linalg.eigvalsh(assemble_PF(0,c,P)))[::-1]
    lamGAL=np.sort(np.linalg.eigvalsh(assemble_GAL(c,P)))[::-1]
    print(f" P={P}: max|lam_PF - lam_GAL|(leading 8) = {np.max(np.abs(lamPF[:8]-lamGAL[:8])):.2e}")
    print(f"        lam_PF  4,5,6 = {lamPF[4]:.6f},{lamPF[5]:.6f},{lamPF[6]:.6f}")
    print(f"        lam_GAL 4,5,6 = {lamGAL[4]:.6f},{lamGAL[5]:.6f},{lamGAL[6]:.6f}   (ref {lref[5]:.5f},{lref[6]:.5f})")

# FB-coefficient decay of benchmark eigenfunctions (plateau m=3 vs plunge m=5,6)
print("\nFB-coefficient |<psi^{(m)}, phi_k>| decay (benchmark psi, c=20):")
d=np.load(os.path.join(D,"cpswf_n0_c20.npz")); xg=d['x_gl']; wg=d['w_gl']; psi=d['psi_gl']
jz=jn_zeros(0,60)
for m in [3,5,6]:
    coeffs=[abs(np.sum(psi[:,m]*(np.sqrt(2)*jv(0,jz[k]*xg)/jv(1,jz[k]))*xg*wg)) for k in range(60)]
    coeffs=np.array(coeffs)
    # report coeff at k=10,20,30,40,50
    print(f"  m={m} (lam={d['lam'][m]:.4f}): |c_k| at k=10,20,30,40,50 = "
          + ", ".join(f"{coeffs[k]:.2e}" for k in [10,20,30,40,50]))
