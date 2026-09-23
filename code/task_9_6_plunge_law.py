"""
Task 9.6 — empirical plunge-region law for the radial concentration eigenvalues.

The plunge is the transition band where lambda_m drops from ~1 (plateau) to ~0 (tail).
Measure, at fixed c:
  - plunge WIDTH  W_eps(n,c) = #{m : eps < lambda_m < 1-eps}   (eps = 0.1 and 0.01),
  - plunge CENTER = the (interpolated) index where lambda crosses 1/2,
and test:
  (a) center ~ M_n(c) ~ c/pi - n/2  (the Shannon number, Task 9.5),
  (b) width W_eps ~ alpha(eps) * log(c) + beta  (the universal log-c plunge growth),
with alpha identified empirically (not assumed), and its n-dependence.
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def eigsB(n, c, Msize):
    jz = jn_zeros(n, Msize); j = jz[:Msize]; pts = [z for z in j if 0 < z < c]
    B = np.zeros((Msize, Msize))
    for a in range(Msize):
        for b in range(a, Msize):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, c, points=pts, limit=250); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    return np.sort(np.linalg.eigvalsh(B))[::-1]

def Mn(n, c):
    return int(np.sum(jn_zeros(n, int(c/np.pi)+30) < c))

def plunge(lam, eps):
    return int(np.sum((lam > eps) & (lam < 1-eps)))

def center(lam):
    # interpolated index m where lambda crosses 1/2 (0-indexed)
    above = np.where(lam >= 0.5)[0]
    if len(above) == 0 or above[-1]+1 >= len(lam):
        return np.nan
    m = above[-1]
    return m + (lam[m]-0.5)/(lam[m]-lam[m+1])

def Vsmooth(lam):
    return float(np.sum(lam*(1-lam)))         # smooth plunge width Sum lambda(1-lambda)

cs_by_n = {0: [20.0, 40.0, 80.0, 120.0], 1: [20.0, 40.0, 80.0], 2: [20.0, 40.0, 80.0]}
print("SMOOTH plunge width V=sum lambda(1-lambda); center(1/2) vs M_n(c)~c/pi-n/2:")
for n in (0, 1, 2):
    print(f"\n n={n}:    c   M_n   center(1/2)   V=Σλ(1-λ)   W_0.01")
    rows = []
    for c in cs_by_n[n]:
        M = Mn(n, c); Msize = M + 14
        lam = eigsB(n, c, Msize)
        rows.append((c, M, center(lam), Vsmooth(lam), plunge(lam, 0.01)))
        print(f"        {int(c):3d}  {M:3d}     {rows[-1][2]:6.2f}      {rows[-1][3]:6.3f}      {rows[-1][4]:3d}   (c/pi-n/2={c/np.pi-n/2:.2f})")
    lc = np.log([r[0] for r in rows]); V = np.array([r[3] for r in rows])
    A = np.vstack([lc, np.ones_like(lc)]).T
    (al, be), *_ = np.linalg.lstsq(A, V, rcond=None)
    print("     fit V = %.4f*log(c) + %.3f   [1D ref slope 1/(2 pi^2) = %.4f]" % (al, be, 1/(2*3.141592653589793**2)))
