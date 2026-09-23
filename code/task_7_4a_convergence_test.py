import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad
n=0
jz=jn_zeros(0,60)

def proj_matrix(P,c):
    j=jz[:P]; B=np.zeros((P,P))
    pts=[z for z in jz if 0<z<c]
    for a in range(P):
        for b in range(a,P):
            jm,jk=j[a],j[b]
            f=lambda u: u*jn(0,u)**2/((u**2-jm**2)*(u**2-jk**2))
            v,_=quad(f,0,c,points=pts,limit=400)
            B[a,b]=B[b,a]=2*jm*jk*v
    return B
def G0(x,y):
    if abs(x-y)<1e-12: return 0.5*x*(jn(0,x)**2+jn(1,x)**2)
    return np.sqrt(x*y)/(x**2-y**2)*(x*jn(1,x)*jn(0,y)-y*jn(1,y)*jn(0,x))
def boul_matrix(N,om):
    s=jz[:N]; R=np.zeros((N,N))
    for a in range(N):
        for b in range(N):
            R[a,b]=2*om*G0(om*s[a],om*s[b])/(np.sqrt(s[a])*abs(jn(1,s[a]))*np.sqrt(s[b])*abs(jn(1,s[b])))
    return R
def topk(M,k): return np.sort(np.linalg.eigvalsh(M))[::-1][:k]

C=10.0
print("GROUND TRUTH: project matrix, increasing P at fixed c=10 (top 6 eigs)")
gt=None
for P in [10,20,40]:
    ev=topk(proj_matrix(P,C),6); 
    if gt is not None: print(f"  P={P}: ",np.round(ev,8),"  |dvs prev|max=",f"{np.max(np.abs(ev-gt)):.2e}")
    else: print(f"  P={P}: ",np.round(ev,8))
    gt=ev
truth=gt  # project P=40

print("\nBOULSANE: increasing N, omega=c/s_(N+1) to HOLD c_eff=10 fixed (top 6 eigs)")
for N in [6,10,15,20,30,40]:
    om=C/jz[N]        # s_(N+1)=jz[N]
    if om>=1: print(f"  N={N}: omega>=1 skip"); continue
    ev=topk(boul_matrix(N,om),6)
    print(f"  N={N}, om={om:.4f}: ",np.round(ev,8),"  max|dvs proj P40|=",f"{np.max(np.abs(ev-truth)):.2e}")
