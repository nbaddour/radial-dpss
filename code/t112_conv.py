import numpy as np, os
from scipy.special import jv, jn_zeros
import importlib.util
spec=importlib.util.spec_from_file_location("asm","task_11_2_assemble_matrix.py")
# reuse assemble by re-import of functions
def panels_nodes(c, jb, G):
    bps=np.concatenate(([0.0],jb,[c])); t,w=np.polynomial.legendre.leggauss(G); U=[];Wq=[]
    for a,b in zip(bps[:-1],bps[1:]):
        if b<=a: continue
        U.append(0.5*(b-a)*t+0.5*(a+b)); Wq.append(0.5*(b-a)*w)
    return np.concatenate(U),np.concatenate(Wq)
def assemble(n,c,P,G=24):
    jz=jn_zeros(n,P); jb=jz[jz<c]; U,Wq=panels_nodes(c,jb,G)
    J=jv(n,U);J2=J*J;uJ2W=U*J2*Wq;j2=jz**2;J1z=jv(n+1,jz)
    I=np.empty(P);D=np.empty(P)
    for p in range(P):
        den=U*U-j2[p];near=np.abs(U-jz[p])<1e-11
        ti=uJ2W/den
        if near.any(): ti[near]=0.0
        I[p]=ti.sum()
        td=uJ2W/den**2
        if near.any(): td[near]=(J1z[p]**2/(4*jz[p]))*Wq[near]
        D[p]=td.sum()
    B=np.empty((P,P))
    for m in range(P):
        B[m,m]=2*j2[m]*D[m]
        for k in range(m+1,P):
            B[m,k]=B[k,m]=2*jz[m]*jz[k]*(I[m]-I[k])/(j2[m]-j2[k])
    return B
D="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
d=np.load(os.path.join(D,"cpswf_n0_c20.npz")); lref=d['lam']
print("c=20 benchmark lam4,5,6 =",", ".join(f"{lref[i]:.6f}" for i in (4,5,6)))
print("Galerkin matrix eigenvalues vs P (watch plunge modes 4,5,6 converge UP to benchmark):")
for P in [12,24,44,70,100,150,200,300]:
    lam=np.sort(np.linalg.eigvalsh(assemble(0,20.0,P,G=28)))[::-1]
    print(f"  P={P:3d}: lam4={lam[4]:.6f} lam5={lam[5]:.6f} lam6={lam[6]:.6f}  "
          f"max|lam-ref|(0..6)={np.max(np.abs(lam[:7]-lref[:7])):.2e}")
