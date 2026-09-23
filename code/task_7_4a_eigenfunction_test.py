import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad
jz=jn_zeros(0,80)

def proj_matrix(P,c):
    j=jz[:P]; B=np.zeros((P,P)); pts=[z for z in jz if 0<z<c]
    for a in range(P):
        for b in range(a,P):
            jm,jk=j[a],j[b]
            f=lambda u:u*jn(0,u)**2/((u**2-jm**2)*(u**2-jk**2))
            v,_=quad(f,0,c,points=pts,limit=400); B[a,b]=B[b,a]=2*jm*jk*v
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
def eigpairs(M):
    w,V=np.linalg.eigh(M); idx=np.argsort(w)[::-1]; return w[idx],V[:,idx]

c=10.0
# Project reference eigenfunctions on x in [0,1] (R=1,K=c)
P=50; wB,VB=eigpairs(proj_matrix(P,c)); jP=jz[:P]
def psi_proj(m,x):
    return sum(VB[k,m]*np.sqrt(2)*jn(0,jP[k]*x)/jn(1,jP[k]) for k in range(P))
# Boulsane eigenfunctions, N large, omega=c/s_{N+1}
N=50; om=c/jz[N]; wb,Vb=eigpairs(boul_matrix(N,om)); s=jz[:N]
def phi_boul(m,r):
    return sum(Vb[k,m]*np.sqrt(2*r)*jn(0,s[k]*r)/abs(jn(1,s[k])) for k in range(N))

print(f"c={c}, omega(Boulsane)={om:.4f}")
print("proj eigs[:5] :",np.round(wB[:5],6))
print("boul eigs[:5] :",np.round(wb[:5],6))

xs=np.linspace(1e-4,1,400)
for m in [0,1,2]:
    pp=np.array([psi_proj(m,x) for x in xs]); pp/=np.sqrt(np.trapz(pp**2*xs,xs))  # L2(x dx) normalize
    # Boulsane phi is in L2(dr) (symmetrized). Un-symmetrize to compare in x dx: g=phi/sqrt(r)
    bb_sym=np.array([phi_boul(m,x) for x in xs])
    # try direct (un-symmetrized) on same [0,1]
    bb=bb_sym/np.sqrt(xs); 
    nb=np.sqrt(np.trapz(bb**2*xs,xs)); bb/=nb
    # fix signs
    if np.trapz(pp*bb*xs,xs)<0: bb=-bb
    ov_direct=np.trapz(pp*bb*xs,xs)
    # try rescaled r->x*om (Boulsane concentrated in (0,om))
    bb2=np.array([phi_boul(m,x*om) for x in xs]); bb2=bb2/np.sqrt(xs*om)
    n2=np.sqrt(np.trapz(bb2**2*xs,xs)); 
    if n2>0: bb2/=n2
    if np.trapz(pp*bb2*xs,xs)<0: bb2=-bb2
    ov_rescaled=np.trapz(pp*bb2*xs,xs)
    print(f" m={m}: overlap(proj, boul direct)={ov_direct:.4f} | overlap(proj, boul rescaled r->om x)={ov_rescaled:.4f}")
