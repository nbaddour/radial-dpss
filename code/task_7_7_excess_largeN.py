"""
Task 7.7 — strengthen the all-N evidence: commuting-excess at larger sizes (high precision).

Extends task_7_6_excess.py / _hp_confirm.py to N = 7, 8 (P = 6, 7) at mpmath high precision.
Prediction (no bispectral accident): measured commutant dim by bandwidth p equals the GENERIC
baseline dim_p = max(1, P - C(P-p,2)), so excess e_p = 0 for all p. A notable consequence at
these sizes: the minimum bandwidth carrying ANY nontrivial commuting operator is p* (≈ P-sqrt(2P)):
  P=6 -> p*=3 (no tri-, no penta-diagonal commutes),
  P=7 -> p*=4 (nothing below a 9-diagonal commutes).
"""
import mpmath as mp
from math import comb

mp.mp.dps = 60
THRESH = mp.mpf(10)**(-22)   # true zeros land ~1e-(dps-23); non-commuting >= eigen-gap (>=~1e-18 here)


def besselzeros(n, N):
    return [mp.besseljzero(mp.mpf(n), k) for k in range(1, N+1)]


def radial_BKB(n, N):
    jz = besselzeros(n, N); j = jz[:N-1]; jN = jz[N-1]; c = jN; P = N-1
    pts = [mp.mpf(0)] + list(j) + [c]
    B = mp.zeros(P, P)
    for a in range(P):
        for b in range(a, P):
            jm, jk = j[a], j[b]
            f = lambda u: u*mp.besselj(n, u)**2/((u**2-jm**2)*(u**2-jk**2))
            v = mp.quad(f, pts)
            B[a, b] = B[b, a] = 2*jm*jk*v
    return B, j, jN


def banded_basis(P, p):
    basis = []
    for i in range(P):
        for k in range(i, min(i+p+1, P)):
            E = mp.zeros(P, P)
            if i == k:
                E[i, i] = 1
            else:
                E[i, k] = E[k, i] = 1
            basis.append(E)
    return basis


def commutant_dim(M, p):
    P = M.rows
    basis = banded_basis(P, p)
    K = len(basis)
    cols = []
    for E in basis:
        Cm = M*E - E*M
        cols.append([Cm[i, k] for i in range(P) for k in range(P)])
    G = mp.zeros(K, K)
    for a in range(K):
        for b in range(a, K):
            s = mp.fsum(cols[a][t]*cols[b][t] for t in range(P*P))
            G[a, b] = G[b, a] = s
    ev = mp.eigsy(G, eigvals_only=True)
    Mn = max(abs(M[i, k]) for i in range(P) for k in range(P))
    sv = sorted(mp.sqrt(abs(e))/Mn for e in ev)
    dim = sum(1 for s in sv if s < THRESH)
    return dim, sv


def generic_dim(P, p):
    if p >= P-1:
        return P
    return max(1, P - comb(P-p, 2))


for (n, N) in [(0, 7), (0, 8)]:
    P = N-1
    B, j, jN = radial_BKB(n, N)
    ev = sorted(mp.eigsy(B, eigvals_only=True), reverse=True)
    mingap = min(abs(ev[i]-ev[i+1]) for i in range(P-1))
    meas = {}; smallest_tri = None
    for p in range(P):
        d, sv = commutant_dim(B, p)
        meas[p] = d
        if p == 1:
            smallest_tri = sv[:5]
    gen = {p: generic_dim(P, p) for p in range(P)}
    exc = {p: meas[p]-gen[p] for p in range(P)}
    pstar = min(p for p in range(P) if gen[p] > 1)
    print("=== n=" + str(n) + " N=" + str(N) + " (P=" + str(P) + ", dps=" + str(mp.mp.dps) + ") ===")
    print("   min eigen-gap = " + mp.nstr(mingap, 4))
    print("   measured dim by bandwidth p: " + str(meas))
    print("   generic  dim by bandwidth p: " + str(gen))
    print("   EXCESS   e_p               : " + str(exc))
    print("   tridiagonal singular values (rel): " + str([mp.nstr(s, 3) for s in smallest_tri]))
    print("   minimum nontrivial commuting bandwidth p* = " + str(pstar)
          + "   (tridiagonal p=1 commutant dim = " + str(meas[1]) + ")")
    print()
