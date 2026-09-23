import numpy as np
from scipy.special import jv, jn_zeros
from numpy.polynomial.legendre import leggauss

c = 8.0
def kern(r, rp):
    K = c
    num = K*(r*(-jv(1,r*K))*jv(0,rp*K) - rp*jv(0,r*K)*(-jv(1,rp*K)))
    den = rp**2 - r**2
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num/den
    diag = (K**2/2)*(jv(1,r*K)**2 + jv(0,r*K)**2)
    return np.where(np.isclose(r,rp,rtol=0,atol=1e-13), diag, out)

# Reference (Nystrom M=800)
M = 800
xg, wg = leggauss(M); x = 0.5*(xg+1); w = 0.5*wg
s = np.sqrt(x*w)
A = s[:,None]*kern(x[:,None], x[None,:])*s[None,:]; A = 0.5*(A+A.T)
ev, U = np.linalg.eigh(A)
lam0, lam1 = ev[-1], ev[-2]
psi = U[:,-1]/s
if psi[0] < 0: psi = -psi
psi1 = (kern(1.0,x)*psi*x*w).sum()/lam0
print(f"reference: lam0={lam0:.14f} lam1={lam1:.14f} psi(1)={psi1:.6e}")

# Galerkin matrix N_max=300 via kernel quadrature Q=3000
Nmax = 300
Q = 3000
xq_, wq_ = leggauss(Q); xq = 0.5*(xq_+1); wq = 0.5*wq_
sq = np.sqrt(xq*wq)
jz = jn_zeros(0, 2000)
Phi = np.sqrt(2)*jv(0, np.outer(xq, jz[:Nmax]))/jv(1, jz[:Nmax])[None,:]   # Q x Nmax
Phis = Phi*sq[:,None]
Kq = sq[:,None]*kern(xq[:,None], xq[None,:])*sq[None,:]
B = Phis.T @ (Kq @ Phis)
B = 0.5*(B+B.T)
print("B symm ok; B[0,0]=%.12f  max|B|=%.3f" % (B[0,0], np.abs(B).max()))

# cross-check entries by band-integral route (u-integral), few (m,k)
Qu = 2000
ug_, uw_ = leggauss(Qu); u = 0.5*c*(ug_+1); uw = 0.5*c*uw_
def H(jk):  # sqrt2 * jk * J0(u)/(jk^2-u^2), removable at u=jk
    den = jk**2 - u**2
    return np.sqrt(2)*jk*jv(0,u)/den
for (m,k) in [(0,0),(0,1),(1,2),(4,7),(50,60),(150,200)]:
    val = (H(jz[m])*H(jz[k])*u*uw).sum()
    print(f"  entry({m},{k}): kernel-route {B[m,k]: .12e}  band-route {val: .12e}  diff {abs(B[m,k]-val):.1e}")

# eigenvalue error vs N (principal truncations)
Ns = [2,3,4,5,6,8,10,12,16,20,28,40,56,80,110,150,210,300]
errs = []
for N in Ns:
    evN = np.linalg.eigvalsh(B[:N-1,:N-1])   # size N-1
    errs.append(lam0 - evN[-1])
# coefficients a_k to k=600 for delta_N^2 (fine grid Q2=8000)
Q2 = 8000
x2_, w2_ = leggauss(Q2); x2 = 0.5*(x2_+1); w2 = 0.5*w2_
psi2 = (kern(x2[:,None], x[None,:])@(psi*x*w))/lam0
Kc = 600
a = np.array([ (psi2*np.sqrt(2)*jv(0,jz[k]*x2)/jv(1,jz[k])*x2*w2).sum() for k in range(Kc) ])
tail_beyond = 2*psi1**2*np.sum(1.0/jz[Kc:]**2)
d2 = np.cumsum((a[::-1]**2))[::-1] + tail_beyond
print("\n N-1   err=lam0-lam0N     lower=(gap)*d2     upper=lam0*d2")
for N,e in zip(Ns,errs):
    dd = d2[N-1] if N-1 < Kc else np.nan
    print(f"{N-1:4d}  {e: .6e}   {(lam0-lam1)*dd: .6e}   {lam0*dd: .6e}")
np.savez("t195_data.npz", c=c, lam0=lam0, lam1=lam1, psi1=psi1, Ns=np.array(Ns), errs=np.array(errs), a=a, d2=d2, jz=jz[:Kc])
