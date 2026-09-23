import numpy as np
from scipy.special import jv, jvp, jn_zeros
from numpy.polynomial.legendre import leggauss

def ref(n,c,M=700):
    jz=jn_zeros(n,500); xg,wg=leggauss(M); x=0.5*(xg+1); w=0.5*wg; s=np.sqrt(x*w)
    Jp=jvp(n,x*c)
    num=c*(x[:,None]*Jp[:,None]*jv(n,x[None,:]*c)-x[None,:]*jv(n,x[:,None]*c)*Jp[None,:])
    den=x[None,:]**2-x[:,None]**2
    with np.errstate(divide='ignore',invalid='ignore'): K=num/den
    with np.errstate(divide='ignore',invalid='ignore'):
        K[np.arange(M),np.arange(M)]=(c**2/2)*(Jp**2+(1-n**2/np.where(x>0,(x*c)**2,np.inf))*jv(n,x*c)**2)
    A=s[:,None]*K*s[None,:]; A=0.5*(A+A.T); ev,U=np.linalg.eigh(A); i=np.argsort(ev)[::-1]
    return jz,x,w,ev[i],U[:,i]/s[:,None]
def ours(n,c,P,jz,Q=3000):
    ug,uw=leggauss(Q); u=0.5*c*(ug+1); wq=0.5*c*uw
    H=np.array([np.sqrt(2)*jz[m]*jv(n,u)/(jz[m]**2-u**2) for m in range(P)])
    B=0.5*((H@(H*(u*wq)).T)+(H@(H*(u*wq)).T).T); ev,V=np.linalg.eigh(B); i=np.argsort(ev)[::-1]
    return ev[i],V[:,i]
def boul(n,N,omega,jz,Q=3000):
    tg,tw=leggauss(Q); t=0.5*omega*(tg+1); wq=0.5*omega*tw
    s=jz[:N]; Phi=jv(n,np.outer(s,t))/np.abs(jv(n+1,s))[:,None]
    R=2*(Phi*(t*wq)[None,:])@Phi.T; R=0.5*(R+R.T); ev,V=np.linalg.eigh(R); i=np.argsort(ev)[::-1]
    return ev[i],V[:,i]
def synth(n,coef,xs,jz):
    return sum(coef[k]*np.sqrt(2)*jv(n,jz[k]*xs)/jv(n+1,jz[k]) for k in range(len(coef)))
def nrm(f,x,w): return f/np.sqrt(((f*f)*x*w).sum())

for (n,c) in [(1,40.0),(0,10.0),(1,10.0)]:
    jz,x,w,lam,Uref = ref(n,c); M=max(int(c/np.pi-n/2),1); P=M+20
    lo,Vo = ours(n,c,P,jz)
    sgn = np.sign(jv(n+1,jz[:P]))                      # |J| -> J convention fix
    print(f"\n=== n={n}, c={c}, P={P}, M={M} ===   lam_ref[0:4]={np.array2string(lam[:4],precision=6)}")
    out={}
    for tag,eps in [('pi/2',np.pi/2),('pi',jz[P]-jz[P-1])]:
        om=c/(jz[P-1]+eps); lb,Vb=boul(n,P,om,jz); out[tag]=(om,Vb)
    hdr=f"{'m':>2} {'lam_ref':>10} | {'ours/ref':>9} | {'Boul(pi/2)/ref':>14} | {'Boul(pi)/ref':>12}"
    print(hdr)
    for m in range(min(4,M+1)):
        fr=nrm(Uref[:,m],x,w); fo=nrm(synth(n,Vo[:,m],x,jz),x,w)
        r=f"{m:>2} {lam[m]:>10.6f} | {abs(((fo*fr)*x*w).sum()):>9.6f} |"
        for tag in ['pi/2','pi']:
            om,Vb=out[tag]
            g=nrm(synth(n,Vb[:,m]*sgn, om*x, jz),x,w)
            r+=f" {abs(((g*fr)*x*w).sum()):>14.6f} |" if tag=='pi/2' else f" {abs(((g*fr)*x*w).sum()):>12.6f}"
        print(r)
    # leading-subspace principal angle (K = number of lam>0.5)
    K=int((lam>0.5).sum())
    Fr=np.array([nrm(Uref[:,j],x,w) for j in range(K)]).T*np.sqrt(x*w)[:,None]
    Fo=np.array([nrm(synth(n,Vo[:,j],x,jz),x,w) for j in range(K)]).T*np.sqrt(x*w)[:,None]
    qo=np.linalg.qr(Fo)[0]; qr=np.linalg.qr(Fr)[0]
    sv=np.linalg.svd(qo.T@qr,compute_uv=False); ang_o=np.arccos(np.clip(sv.min(),-1,1))
    line=f"  leading subspace K={K}: principal angle  ours/ref {ang_o:.3e}"
    for tag in ['pi/2','pi']:
        om,Vb=out[tag]
        Fb=np.array([nrm(synth(n,Vb[:,j]*sgn,om*x,jz),x,w) for j in range(K)]).T*np.sqrt(x*w)[:,None]
        qb=np.linalg.qr(Fb)[0]
        s2=np.linalg.svd(qb.T@qr,compute_uv=False); line+=f" | Boul({tag})/ref {np.arccos(np.clip(s2.min(),-1,1)):.3e}"
    print(line)

# --- values needed for the manuscript text ---
print("\n=== ours vs Boulsane directly (Appendix D claim, c=10, n=0) ===")
n,c=0,10.0
jz,x,w,lam,Uref=ref(n,c); M=3; P=M+20
lo,Vo=ours(n,c,P,jz); sgn=np.sign(jv(n+1,jz[:P]))
for tag,eps in [('pi/2',np.pi/2),('pi',jz[P]-jz[P-1])]:
    om=c/(jz[P-1]+eps); lb,Vb=boul(n,P,om,jz)
    vals=[]
    for m in range(3):
        fo=nrm(synth(n,Vo[:,m],x,jz),x,w); g=nrm(synth(n,Vb[:,m]*sgn,om*x,jz),x,w)
        vals.append(abs(((fo*g)*x*w).sum()))
    print(f"  eps={tag}: overlaps m=0,1,2 = " + ", ".join(f"{v:.4f}" for v in vals))
