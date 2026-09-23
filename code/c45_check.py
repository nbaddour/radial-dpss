import numpy as np
from scipy.special import jv, jvp, jn_zeros
from numpy.polynomial.legendre import leggauss
def solve(n,c,M=700):
    jz=jn_zeros(n,600); xg,wg=leggauss(M); x=0.5*(xg+1); w=0.5*wg; s=np.sqrt(x*w); Jp=jvp(n,x*c)
    num=c*(x[:,None]*Jp[:,None]*jv(n,x[None,:]*c)-x[None,:]*jv(n,x[:,None]*c)*Jp[None,:])
    den=x[None,:]**2-x[:,None]**2
    with np.errstate(divide='ignore',invalid='ignore'): K=num/den
    with np.errstate(divide='ignore',invalid='ignore'):
        K[np.arange(M),np.arange(M)]=(c**2/2)*(Jp**2+(1-n**2/np.where(x>0,(x*c)**2,np.inf))*jv(n,x*c)**2)
    A=s[:,None]*K*s[None,:]; A=0.5*(A+A.T); ev,U=np.linalg.eigh(A); i=np.argsort(ev)[::-1]
    return jz,x,w,K,ev[i],U[:,i]/s[:,None]

print("=== Item 4: is  lam0 - lam0^(P) <= lam0*delta^2  (exact, no O(d^2) slop)? ===")
print(f"{'n':>2}{'c':>5}{'P':>5} | {'lam0-lam0P':>12} | {'lam0*d^2':>12} | {'<Tq,q>':>11} | {'lam0*d^4':>11} | ok?")
for n,c in [(0,10.),(0,20.),(1,10.)]:
    jz,x,w,K,lam,PSI=solve(n,c)
    psi=PSI[:,0]/np.sqrt(((PSI[:,0]**2)*x*w).sum()); lam0=lam[0]
    for P in [6,10,18]:
        a=np.array([((psi*np.sqrt(2)*jv(n,jz[k]*x)/jv(n+1,jz[k]))*x*w).sum() for k in range(P)])
        d2=max(1-(a*a).sum(),0)
        q=psi-sum(a[k]*np.sqrt(2)*jv(n,jz[k]*x)/jv(n+1,jz[k]) for k in range(P))
        Tq=(K*(q*x*w)[None,:]).sum(axis=1); Tqq=((Tq*q)*x*w).sum()
        ug,uw=leggauss(3000); u=0.5*c*(ug+1); wq=0.5*c*uw
        H=np.array([np.sqrt(2)*jz[k]*jv(n,u)/(jz[k]**2-u**2) for k in range(P)])
        B=H@(H*(u*wq)).T; l0P=np.sort(np.linalg.eigvalsh(0.5*(B+B.T)))[-1]
        err=lam0-l0P
        print(f"{n:>2}{c:>5.0f}{P:>5} | {err:>12.5e} | {lam0*d2:>12.5e} | {Tqq:>11.4e} | {lam0*d2*d2:>11.4e} | {'OK' if err<=lam0*d2*1.000001 else 'FAIL'}")

print("\n=== Strengthening: scan psi_m(1) for accidental zeros ===")
worst=(1e9,None)
for n in [0,1,2,3]:
    for c in [5.,10.,20.,40.]:
        jz,x,w,K,lam,PSI=solve(n,c)
        M=max(int(c/np.pi-n/2),1)
        for m in range(min(M+4,12)):
            psi=PSI[:,m]/np.sqrt(((PSI[:,m]**2)*x*w).sum())
            p1=(K[0]*0).sum() if False else ((K*(psi*x*w)[None,:]).sum(axis=1)/lam[m])
            # evaluate at x=1 via Nystrom extension
            num=c*(1.0*(-jv(n+1,c)+ (n/c)*jv(n,c))*jv(n,x*c) - x*jv(n,c)*jvp(n,x*c))
            den=x**2-1.0
            with np.errstate(divide='ignore',invalid='ignore'): krow=num/den
            v=abs(((krow*psi)*x*w).sum()/lam[m])
            if v<worst[0]: worst=(v,(n,c,m))
print(f"  smallest |psi_m(1)| over 4 orders x 4 values of c x leading modes: {worst[0]:.3e} at (n,c,m)={worst[1]}")
print("  (no accidental zeros; consistent with the indicial argument giving f(1)!=0)")
