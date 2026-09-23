import numpy as np
from scipy.special import jv, jn_zeros
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def solve(c, M=800):
    def kern(r, rp):
        num = c*(r*(-jv(1,r*c))*jv(0,rp*c) - rp*jv(0,r*c)*(-jv(1,rp*c)))
        den = rp**2 - r**2
        with np.errstate(divide='ignore', invalid='ignore'):
            out = num/den
        diag = (c**2/2)*(jv(1,r*c)**2 + jv(0,r*c)**2)
        return np.where(np.isclose(r,rp,rtol=0,atol=1e-13), diag, out), None
    xg, wg = leggauss(M); x = 0.5*(xg+1); w = 0.5*wg
    s = np.sqrt(x*w)
    Kk,_ = kern(x[:,None], x[None,:])
    A = s[:,None]*Kk*s[None,:]; A = 0.5*(A+A.T)
    ev, U = np.linalg.eigh(A)
    psi = U[:,-1]/s
    if psi[0] < 0: psi = -psi
    Kb,_ = kern(np.array([1.0])[:,None], x[None,:])
    psi1 = (Kb[0]*psi*x*w).sum()/ev[-1]
    return ev[-1], ev[-2], psi, psi1, x, w, kern

def coeffs(c, psi, x, w, kern, lam0, Kc=400, Q=8000):
    x2_, w2_ = leggauss(Q); x2 = 0.5*(x2_+1); w2 = 0.5*w2_
    Kk,_ = kern(x2[:,None], x[None,:])
    psi2 = (Kk@(psi*x*w))/lam0
    jz = jn_zeros(0, Kc)
    a = np.array([ (psi2*np.sqrt(2)*jv(0,jz[k]*x2)/jv(1,jz[k])*x2*w2).sum() for k in range(Kc) ])
    return jz, a

# panel (a) data
res = {}
for c in [8.0, 20.0]:
    lam0, lam1, psi, psi1, x, w, kern = solve(c)
    jz, a = coeffs(c, psi, x, w, kern, lam0)
    res[c] = (jz, a, psi1, lam0, lam1)

# panel (b) data (c=8): reuse saved B-run errors + Parseval delta2
d = np.load("t195_data.npz")
Ns, errs, a8 = d['Ns'], d['errs'], d['a']
lam0, lam1 = float(d['lam0']), float(d['lam1'])
csum = np.cumsum(a8**2)
P = Ns - 1
delta2 = np.array([1.0 - csum[p-1] for p in P])
low  = (lam0-lam1)*delta2
high = lam0*delta2

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax = axs[0]
for c, col in [(8.0,'tab:blue'), (20.0,'tab:red')]:
    jz, a, psi1, l0, l1 = res[c]
    k = np.arange(1, len(a)+1)
    ax.loglog(k, np.abs(a), '.', ms=3, color=col, label=f"$|a_k|$, $c={int(c)}$")
    ax.loglog(k, np.sqrt(2)*abs(psi1)/jz, '--', color=col, lw=1,
              label=rf"$\sqrt{{2}}\,|\tilde\psi_0(1)|/j_{{0,k}}$, $c={int(c)}$")
ax.set_xlabel("$k$"); ax.set_ylabel("coefficient magnitude")
ax.set_title("(a) two-regime coefficient decay, $m=0$")
ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(1e-12, 2)
ax = axs[1]
ax.loglog(P, errs, 'o-', ms=4, color='k', label=r"$\lambda_0-\lambda_0^{(P)}$ (measured)")
ax.fill_between(P, low, high, color='tab:orange', alpha=0.3,
                label=r"Theorem 6 band $[(\lambda_0-\lambda_1)\delta_P^2,\ \lambda_0\delta_P^2]$")
ax.loglog(P, high[np.searchsorted(P,79)]*(P[np.searchsorted(P,79)]/P.astype(float)), ':', color='gray', lw=1, label=r"$\propto 1/P$")
ax.set_xlabel("matrix size $P$"); ax.set_ylabel("eigenvalue error")
ax.set_title("(b) eigenvalue-error crossover, $n=0$, $c=8$")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("task_19_5_crossover.pdf")
print("saturation of upper bound:", (errs/high)[[np.searchsorted(P,x) for x in (79,149,299)]])
print("figure written")
