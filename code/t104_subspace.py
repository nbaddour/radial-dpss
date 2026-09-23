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
def interp(n,c,lam,x,w,psi,cols,xc):                      # Nystrom-interpolate selected modes to xc
    out=np.zeros((len(xc),len(cols)))
    for i,xv in enumerate(xc):
        row=Knfun_vec(n,c,xv,x)*(x*w); out[i,:]=(row@psi[:,cols])/lam[cols]
    return out
D="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"

# common high-order GL grid for accurate inner products <.,.>_{x dx}
tc,wc=np.polynomial.legendre.leggauss(400); xc=0.5*(tc+1); wc=0.5*wc; Wc=xc*wc

print("CROSS-GRID validation of the SEPARATED modes (individual) and PLATEAU modes (subspace)")
print("  comparing the STORED solve (Nq) against an INDEPENDENT solve (Nq+160), both interpolated to a")
print("  common 400-node grid; inner product <.,.>_{x dx}.\n")
for fn in ["cpswf_n0_c20.npz","cpswf_n0_c80.npz","cpswf_n2_c80.npz","cpswf_n4_c40.npz"]:
    d=np.load(os.path.join(D,fn)); n=int(d['n']); c=float(d['c']); regime=d['regime']; Nq=int(d['Nq'])
    lamA,xA,wA,psiA=nystrom(n,c,Nq); lamB,xB,wB,psiB=nystrom(n,c,Nq+160)
    sep=np.where(regime==1)[0]; pla=np.where(regime==0)[0]
    # separated: individual overlap |<psiA_m, psiB_m>| should be ~1 (gap-limited)
    sepmin=1.0
    if len(sep):
        Ua=interp(n,c,lamA,xA,wA,psiA,sep,xc); Ub=interp(n,c,lamB,xB,wB,psiB,sep,xc)
        for k in range(len(sep)):
            a=Ua[:,k]; b=Ub[:,k]
            a/=np.sqrt(np.sum(a*a*Wc)); b/=np.sqrt(np.sum(b*b*Wc))
            ov=abs(np.sum(a*b*Wc)); sepmin=min(sepmin,ov)
    # plateau: principal angles between subspaces via SVD of cross-Gram of orthonormalized bases
    plamax=0.0
    if len(pla)>=1:
        Pa=interp(n,c,lamA,xA,wA,psiA,pla,xc); Pb=interp(n,c,lamB,xB,wB,psiB,pla,xc)
        def orthon(M):                                   # QR in the W inner product
            Q=np.zeros_like(M)
            for j in range(M.shape[1]):
                v=M[:,j].copy()
                for i in range(j):
                    v-=np.sum(Q[:,i]*v*Wc)*Q[:,i]
                v/=np.sqrt(np.sum(v*v*Wc)); Q[:,j]=v
            return Q
        Qa=orthon(Pa); Qb=orthon(Pb)
        Cgram=(Qa*Wc[:,None]).T@Qb                       # <Qa_i, Qb_j>_W
        sv=np.linalg.svd(Cgram,compute_uv=False)
        plamax=np.sqrt(max(0.0,1-min(sv)**2))            # largest principal angle sine = subspace distance
    print(f"  {fn:18} sep modes={len(sep):2d} min|<a,b>|={sepmin:.6f} (1-min={1-sepmin:.1e})   "
          f"plateau dim={len(pla):2d} subspace dist sin(theta_max)={plamax:.1e}")
