import numpy as np, os
from scipy.special import jv, jvp, jn_zeros
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

print("CHECK A — psi_unif is accurate: rebuild at uniform pts from an INDEPENDENT higher-Nq solve")
for fn,c,n in [("cpswf_n0_c20.npz",20.0,0),("cpswf_n2_c80.npz",80.0,2)]:
    d=np.load(os.path.join(D,fn)); xu=d['x_unif']; psiu=d['psi_unif']; reliab=d['reliab']; lam_s=d['lam']
    lam2,x2,w2,psi2=nystrom(n,c,360)                       # independent, higher Nq
    # interpolate psi2 to uniform grid
    rebuilt=np.zeros_like(psiu)
    nm=psiu.shape[1]
    for i,xv in enumerate(xu):
        row=Knfun_vec(n,c,xv,x2)*(x2*w2); rebuilt[i,:]=(row@psi2[:,:nm])/lam2[:nm]
    # sign-align rebuilt to stored
    for m in range(nm):
        if np.sum(rebuilt[:,m]*psiu[:,m])<0: rebuilt[:,m]*=-1
    # compare only modes with lam>1e-6 (informative); report deep-tail separately
    info=lam_s>1e-6
    err_info=np.max(np.abs(rebuilt[:,info]-psiu[:,info]))
    err_all=np.max(np.abs(rebuilt-psiu))
    print(f"  {fn}: max|psi_unif - independent rebuild|  informative(lam>1e-6)={err_info:.1e}  all-modes={err_all:.1e}")

print("\nCHECK B — trace deviation is the OMITTED TAIL, not solver error")
for fn,c,n in [("cpswf_n0_c80.npz",80.0,0)]:
    d=np.load(os.path.join(D,fn)); lam_s=d['lam']; x=d['x_gl']; w=d['w_gl']
    cx=c*x; diag=(c**2/2)*(jvp(n,cx)**2+(1-(n/cx)**2)*jv(n,cx)**2); trace_full=np.sum(diag*x*w)
    lam_all,_,_,_=nystrom(n,c,240)
    print(f"  {fn}: sum(stored {len(lam_s)} lam)={np.sum(lam_s):.8f}  full trace={trace_full:.8f}")
    print(f"     sum(ALL lam)={np.sum(lam_all):.8f} (matches full trace to {abs(np.sum(lam_all)-trace_full):.1e})")
    print(f"     omitted tail sum(lam[{len(lam_s)}:])={np.sum(lam_all[len(lam_s):]):.2e}  ~ the 'trace_err' seen = OK")

print("\nCHECK C — sign determinism: reload and confirm psi_gl>0 at its max-|.| node (all files)")
import glob
ok=True
for fp in sorted(glob.glob(os.path.join(D,"*.npz"))):
    d=np.load(fp); psi=d['psi_gl']
    for m in range(psi.shape[1]):
        j=np.argmax(np.abs(psi[:,m]))
        if psi[j,m]<=0: ok=False; print("  SIGN FAIL", os.path.basename(fp), m)
print("  sign convention holds for every stored mode:", ok)

print("\nCHECK D — deep-tail dual flag: count modes with lam<1e-8 (dual delicate) per case")
for fp in sorted(glob.glob(os.path.join(D,"*.npz"))):
    d=np.load(fp); lam=d['lam']; dn=d['dual_norm']
    deep=lam<1e-8
    print(f"  {os.path.basename(fp):20} modes lam<1e-8: {int(deep.sum())}  "
          f"max|dual_norm-1| on lam>=1e-8: {np.max(np.abs(dn[~deep]-1)) if (~deep).any() else 0:.1e}")
