import numpy as np
from scipy.special import jv, jvp, jn_zeros
from numpy.polynomial.legendre import leggauss
# Check the repaired Thm 7(b) bound:  ||psi_P - psi|| <= (1+C_m) d_P + d_P^2/2,  C_m = 4*lam0/delta_m
n,c=0,10.0; jz=jn_zeros(n,600)
M=600; xg,wg=leggauss(M); x=0.5*(xg+1); w=0.5*wg; s=np.sqrt(x*w); Jp=jvp(n,x*c)
num=c*(x[:,None]*Jp[:,None]*jv(n,x[None,:]*c)-x[None,:]*jv(n,x[:,None]*c)*Jp[None,:])
den=x[None,:]**2-x[:,None]**2
with np.errstate(divide='ignore',invalid='ignore'): K=num/den
K[np.arange(M),np.arange(M)]=(c**2/2)*(Jp**2+jv(n,x*c)**2)
A=s[:,None]*K*s[None,:]; A=0.5*(A+A.T); ev,U=np.linalg.eigh(A); i=np.argsort(ev)[::-1]
lam=ev[i]; PSI=U[:,i]/s[:,None]
def nrm(f): return f/np.sqrt(((f*f)*x*w).sum())
def galerkin(P,Q=3000):
    ug,uw=leggauss(Q); u=0.5*c*(ug+1); wq=0.5*c*uw
    H=np.array([np.sqrt(2)*jz[k]*jv(n,u)/(jz[k]**2-u**2) for k in range(P)])
    B=H@(H*(u*wq)).T; return 0.5*(B+B.T)
def synth(coef): return sum(coef[k]*np.sqrt(2)*jv(n,jz[k]*x)/jv(n+1,jz[k]) for k in range(len(coef)))
lam0=lam[0]
print(f"{'m':>2}{'P':>5} | {'delta_P':>9} {'delta_m':>9} | {'actual err':>11} | {'bound':>10} | ok?")
for m in [0,1,2]:
    dm=min(lam[m-1]-lam[m] if m>0 else 1e9, lam[m]-lam[m+1])
    for P in [8,14,24,40]:
        B=galerkin(P); ev2,V=np.linalg.eigh(B); j=np.argsort(ev2)[::-1]
        psiP=nrm(synth(V[:,j][:,m])); psi=nrm(PSI[:,m])
        a=np.array([((psi*np.sqrt(2)*jv(n,jz[k]*x)/jv(n+1,jz[k]))*x*w).sum() for k in range(P)])
        dP=np.sqrt(max(1-(a*a).sum(),0))
        err=min(np.sqrt(((psiP-psi)**2*x*w).sum()), np.sqrt(((psiP+psi)**2*x*w).sum()))
        bound=(1+4*lam0/dm)*dP + dP**2/2
        print(f"{m:>2}{P:>5} | {dP:>9.3e} {dm:>9.3e} | {err:>11.3e} | {bound:>10.3e} | {'OK' if err<=bound else 'FAIL'}")
