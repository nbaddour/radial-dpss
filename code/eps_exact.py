import numpy as np
from scipy.special import jv, jvp, jn_zeros
from numpy.polynomial.legendre import leggauss
def nystrom(n,c,M=500):
    xg,wg=leggauss(M); x=0.5*(xg+1); w=0.5*wg; s=np.sqrt(x*w); Jp=jvp(n,x*c)
    num=c*(x[:,None]*Jp[:,None]*jv(n,x[None,:]*c)-x[None,:]*jv(n,x[:,None]*c)*Jp[None,:])
    den=x[None,:]**2-x[:,None]**2
    with np.errstate(divide='ignore',invalid='ignore'): K=num/den
    with np.errstate(divide='ignore',invalid='ignore'):
        K[np.arange(M),np.arange(M)]=(c**2/2)*(Jp**2+(1-n**2/np.where(x>0,(x*c)**2,np.inf))*jv(n,x*c)**2)
    A=s[:,None]*K*s[None,:]; return np.sort(np.linalg.eigvalsh(0.5*(A+A.T)))[::-1]
def ours(n,c,P,jz,Q=3000):
    ug,uw=leggauss(Q); u=0.5*c*(ug+1); w=0.5*c*uw
    H=np.array([np.sqrt(2)*jz[m]*jv(n,u)/(jz[m]**2-u**2) for m in range(P)])
    B=H@(H*(u*w)).T; return np.sort(np.linalg.eigvalsh(0.5*(B+B.T)))[::-1]
def boul(n,N,om,jz,Q=3000):
    tg,tw=leggauss(Q); t=0.5*om*(tg+1); w=0.5*om*tw; s=jz[:N]
    Phi=jv(n,np.outer(s,t))/np.abs(jv(n+1,s))[:,None]
    R=2*(Phi*(t*w)[None,:])@Phi.T; return np.sort(np.linalg.eigvalsh(0.5*(R+R.T)))[::-1]
print(f"{'n':>2}{'c':>5}{'P':>5} | {'ours knee':>10} | {'Boul eps=pi/2':>13} {'ratio':>6} | {'Boul eps*_N exact':>17} {'ratio':>6}")
for n,c in [(0,20.),(0,40.),(0,80.),(1,40.),(2,40.)]:
    jz=jn_zeros(n,400); ref=nystrom(n,c); M=int(c/np.pi-n/2); P=M+20
    o=ours(n,c,P,jz); ok=abs(o[M]-ref[M]); row=f"{n:>2}{c:>5.0f}{P:>5} | {ok:>10.2e} |"
    for eps in [np.pi/2, (P+n/2+0.25)*np.pi-jz[P-1]]:
        b=boul(n,P,c/(jz[P-1]+eps),jz); bk=abs(b[M]-ref[M]); row+=f" {bk:>13.2e} {bk/ok:>6.2f} |"
    print(row)
