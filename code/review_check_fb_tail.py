import numpy as np
from scipy.special import jv, jn_zeros
from numpy.polynomial.legendre import leggauss

n_ord, c = 0, 20.0
def kern(r, rp):
    r, rp = np.asarray(r,float), np.asarray(rp,float)
    K = c
    num = K*(r*(-jv(1,r*K))*jv(0,rp*K) - rp*jv(0,r*K)*(-jv(1,rp*K)))
    den = rp**2 - r**2
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num/den
    diag = (K**2/2)*(jv(1,r*K)**2 + jv(0,r*K)**2)
    return np.where(np.isclose(r,rp,rtol=0,atol=1e-13), diag, out)

M = 600
xg, wg = leggauss(M)
x = 0.5*(xg+1); w = 0.5*wg
s = np.sqrt(x*w)
A = s[:,None]*kern(x[:,None], x[None,:])*s[None,:]
A = 0.5*(A+A.T)
ev, U = np.linalg.eigh(A)
lam0 = ev[-1]
psi = U[:,-1]/s
if psi[0] < 0: psi = -psi
print("lambda0 =", lam0)
psi1 = (kern(1.0, x)*psi*x*w).sum()/lam0
print("psi(1) =", psi1)

Kq = 4000
xq_, wq_ = leggauss(Kq)
xq = 0.5*(xq_+1); wq = 0.5*wq_
psi_q = (kern(xq[:,None], x[None,:])@(psi*x*w))/lam0

nk = 400
jz = jn_zeros(0, nk)
res = np.empty(nk)
for k in range(1, nk+1):
    jk = jz[k-1]
    phik = np.sqrt(2)*jv(0, jk*xq)/jv(1, jk)
    res[k-1] = (psi_q*phik*xq*wq).sum()
print("\n k     j_k        a_k            j_k*a_k     sqrt2*psi(1)=%.6e" % (np.sqrt(2)*psi1))
for k in [1,2,3,5,8,12,20,40,80,150,250,350,400]:
    jk = jz[k-1]
    print(f"{k:4d} {jk:9.3f} {res[k-1]: .6e}  {jk*res[k-1]: .6e}")
tails = np.cumsum(res[::-1]**2)[::-1]
print("\n tail sum_(k>=N) a_k^2  and  2*psi1^2 * sum_(k>=N) 1/j_k^2:")
for N in [8,10,15,20,40,80,160,320]:
    pred = 2*psi1**2*np.sum(1.0/jz[N-1:]**2)  # truncated pred (same range)
    print(f" N={N:4d}  actual {tails[N-1]:.6e}   pred {pred:.6e}")
