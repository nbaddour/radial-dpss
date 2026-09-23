import numpy as np, os
from scipy.special import jv, jvp
def Kmat(n,c,x):
    X=x[:,None];Y=x[None,:];cX,cY=c*X,c*Y
    JX,JY,JpX,JpY=jv(n,cX),jv(n,cY),jvp(n,cX),jvp(n,cY)
    with np.errstate(divide='ignore',invalid='ignore'):
        K=c*(X*JY*JpX-Y*JX*JpY)/(Y**2-X**2)
    cx=c*x;np.fill_diagonal(K,(c**2/2)*(jvp(n,cx)**2+(1-(n/cx)**2)*jv(n,cx)**2));return K
def nystrom(n,c,Nq):
    t,w=np.polynomial.legendre.leggauss(Nq);x=0.5*(t+1);w=0.5*w
    s=np.sqrt(w*x);A=(s[:,None]*Kmat(n,c,x))*s[None,:];A=0.5*(A+A.T)
    lam,V=np.linalg.eigh(A);idx=np.argsort(lam)[::-1];return lam[idx],x,w,V[:,idx]/s[:,None]
def Knfun_vec(n,c,xv,x):
    cx=c*x;cxv=c*xv
    with np.errstate(divide='ignore',invalid='ignore'):
        row=c*(xv*jv(n,cx)*jvp(n,cxv)-x*jv(n,cxv)*jvp(n,cx))/(x**2-xv**2)
    near=np.abs(x-xv)<1e-13
    if near.any(): row[near]=(c**2/2)*(jvp(n,cxv)**2+(1-(n/cxv)**2)*jv(n,cxv)**2)
    return row
D="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
print("CHECK A (corrected) — psi_unif accuracy on SEPARATED, non-deep-tail modes (1e-6<=lam<=1-1e-9)")
print("   [clustered lam~1 modes are subspace-ambiguous and rotate between solves -> excluded by design]")
for fn,c,n in [("cpswf_n0_c20.npz",20.0,0),("cpswf_n0_c80.npz",80.0,0),
               ("cpswf_n2_c80.npz",80.0,2),("cpswf_n4_c40.npz",40.0,4)]:
    d=np.load(os.path.join(D,fn)); xu=d['x_unif']; psiu=d['psi_unif']; lam_s=d['lam']
    lam2,x2,w2,psi2=nystrom(n,c,360); nm=psiu.shape[1]
    rebuilt=np.zeros_like(psiu)
    for i,xv in enumerate(xu):
        row=Knfun_vec(n,c,xv,x2)*(x2*w2); rebuilt[i,:]=(row@psi2[:,:nm])/lam2[:nm]
    for m in range(nm):
        if np.sum(rebuilt[:,m]*psiu[:,m])<0: rebuilt[:,m]*=-1
    sep=(lam_s>=1e-6)&(lam_s<=1-1e-9)
    errs=np.max(np.abs(rebuilt[:,sep]-psiu[:,sep]),axis=0)
    print(f"  {fn}: {int(sep.sum())} separated non-tail modes, max|psi_unif-indep rebuild| = {errs.max():.1e}")

print("\nPer-mode psi_unif error vs lam (n=0,c=80) — is the ~1e-6 only on small-lam tail modes?")
d=np.load(os.path.join(D,"cpswf_n0_c80.npz")); xu=d['x_unif']; psiu=d['psi_unif']; lam_s=d['lam']
lam2,x2,w2,psi2=nystrom(0,80.0,360); nm=psiu.shape[1]
rebuilt=np.zeros_like(psiu)
for i,xv in enumerate(xu):
    row=Knfun_vec(0,80.0,xv,x2)*(x2*w2); rebuilt[i,:]=(row@psi2[:,:nm])/lam2[:nm]
for m in range(nm):
    if np.sum(rebuilt[:,m]*psiu[:,m])<0: rebuilt[:,m]*=-1
for m in range(nm):
    cl = "clustered(rotates)" if abs(lam_s[m]-1)<1e-9 else ""
    print(f"   m={m:2d}: lam={lam_s[m]:.3e}  max|psi_unif-rebuild|={np.max(np.abs(rebuilt[:,m]-psiu[:,m])):.1e}  {cl}")
