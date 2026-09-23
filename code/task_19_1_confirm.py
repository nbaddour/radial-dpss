import numpy as np
from scipy.special import jv, jvp, jn_zeros
from numpy.polynomial.legendre import leggauss

c = 20.0
def kern(n, r, rp):
    r, rp = np.asarray(r,float), np.asarray(rp,float)
    K = c
    Jp  = jvp(n, r*K); Jpp = jvp(n, rp*K)
    num = K*(r*Jp*jv(n,rp*K) - rp*jv(n,r*K)*Jpp)
    den = rp**2 - r**2
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num/den
    with np.errstate(divide='ignore', invalid='ignore'):
        diag = (K**2/2)*(jvp(n,r*K)**2 + (1 - n**2/np.where(r>0,(r*K)**2,np.inf))*jv(n,r*K)**2)
    return np.where(np.isclose(r,rp,rtol=0,atol=1e-13), diag, out)

M = 600
xg, wg = leggauss(M); x = 0.5*(xg+1); w = 0.5*wg
s = np.sqrt(x*w)
Kq = 4000
xq_, wq_ = leggauss(Kq); xq = 0.5*(xq_+1); wq = 0.5*wq_
nk = 400

print(f"c={c}.  Columns: ratio_k = j_k*a_k/(sqrt2*psi_m(1)) at k=100,200,300,400;")
print("tailratio_N = [sum_(k>=N) a_k^2] / [2 psi_m(1)^2 sum_(k>=N) j_k^-2] at N=160,320 (both sums truncated at k=400)\n")
for n in [0, 1]:
    jz = jn_zeros(n, nk)
    A = s[:,None]*kern(n, x[:,None], x[None,:])*s[None,:]
    A = 0.5*(A+A.T)
    ev, U = np.linalg.eigh(A)
    for m in [0, 1, 2]:
        lam = ev[-1-m]
        psi = U[:,-1-m]/s
        # Nystrom extension to fine grid and to x=1
        wpsi = psi*x*w
        psi_q = (kern(n, xq[:,None], x[None,:])@wpsi)/lam
        psi1 = (kern(n, 1.0, x)*wpsi).sum()/lam
        a = np.empty(nk)
        for k in range(nk):
            a[k] = (psi_q*np.sqrt(2)*jv(n, jz[k]*xq)/jv(n+1, jz[k])*xq*wq).sum()
        ratios = [jz[k-1]*a[k-1]/(np.sqrt(2)*psi1) for k in (100,200,300,400)]
        tails = np.cumsum(a[::-1]**2)[::-1]
        trat = []
        for N in (160,320):
            pred = 2*psi1**2*np.sum(1.0/jz[N-1:]**2)
            trat.append(tails[N-1]/pred)
        print(f"n={n} m={m}: lam={lam:.12f}  psi_m(1)={psi1: .4e}  "
              f"ratio(k=100..400)=[{ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f},{ratios[3]:.3f}]  "
              f"tailratio(N=160,320)=[{trat[0]:.3f},{trat[1]:.3f}]")
