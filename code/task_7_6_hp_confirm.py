"""
Task 7.6 — high-precision confirmation (mpmath) that the finite radial prolate matrix has NO
nontrivial commuting tridiagonal operator, and that its commutant contains no sparse element
(dim of bandwidth-p commutant stays 1 until full bandwidth p = P-1).

High precision removes two double-precision doubts:
  (a) eigenvalue clustering (gaps ~1e-7..1e-14) inflating threshold-based null counts;
  (b) the ill-conditioned S = D T similarity used to form the Candidate-C sample matrix.

We build B^(B) (coefficient domain) and B^(C) (symmetric sample domain) at dps digits, form the
commutator map J |-> [M,J] on symmetric bandwidth-p matrices, and read its singular values via
the symmetric eigenproblem of L^T L (mp.eigsy). True commuting directions sit at ~10^-dps;
clustering near-misses sit at ~eigen-gap and are correctly counted as NONzero.
"""
import mpmath as mp

mp.mp.dps = 50
# True commuting directions land at ~1e-27 (precision loss in forming the Gram L^T L); the next
# (non-commuting / clustering) singular values sit at >= ~1e-9 (eigen-gap scale). Threshold 1e-18
# sits in the ~18-order gap between them.
THRESH = mp.mpf(10)**(-18)


def besselzeros(n, N):
    return [mp.besseljzero(mp.mpf(n), k) for k in range(1, N+1)]


def Jn(n, x):
    return mp.besselj(n, x)


def radial_BKB(n, N):
    jz = besselzeros(n, N); j = jz[:N-1]; jN = jz[N-1]; c = jN; P = N-1
    pts = [mp.mpf(0)] + list(j) + [c]
    B = mp.zeros(P, P)
    for a in range(P):
        for b in range(a, P):
            jm, jk = j[a], j[b]
            f = lambda u: u*Jn(n, u)**2/((u**2-jm**2)*(u**2-jk**2))
            v = mp.quad(f, pts)
            B[a, b] = B[b, a] = 2*jm*jk*v
    return B, j, jN


def radial_BKC(n, N):
    B, j, jN = radial_BKB(n, N); P = N-1; K = mp.mpf(1); R = jN/K
    D = mp.diag([R*Jn(n+1, j[k])/mp.sqrt(2) for k in range(P)])
    wdiag = [2/(K**2*Jn(n+1, j[k])**2) for k in range(P)]
    T = mp.matrix(P, P)
    for a in range(P):
        for b in range(P):
            T[a, b] = 2/(Jn(n+1, j[a])*Jn(n+1, j[b])*jN)*Jn(n, j[a]*j[b]/jN)
    S = D*T
    BA = S**-1 * B * S
    Wh = mp.diag([mp.sqrt(w) for w in wdiag])
    Wih = mp.diag([1/mp.sqrt(w) for w in wdiag])
    BC = Wh*BA*Wih
    return (BC + BC.T)/2


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


def commutant_singvals(M, p):
    P = M.rows
    basis = banded_basis(P, p)
    K = len(basis)
    cols = []
    for E in basis:
        Cm = M*E - E*M
        cols.append([Cm[i, k] for i in range(P) for k in range(P)])
    # L is (P^2 x K); form Gram G = L^T L  (K x K)
    G = mp.zeros(K, K)
    for a in range(K):
        for b in range(a, K):
            s = mp.fsum(cols[a][t]*cols[b][t] for t in range(P*P))
            G[a, b] = G[b, a] = s
    ev = mp.eigsy(G, eigvals_only=True)
    nrm = mp.sqrt(max(abs(e) for e in ev)) if K else mp.mpf(1)
    sv = sorted(mp.sqrt(abs(e)) for e in ev)
    Mn = mp.mpf(0)
    for i in range(P):
        for k in range(P):
            Mn = max(Mn, abs(M[i, k]))
    sv_rel = [s/Mn for s in sv]
    dim = sum(1 for s in sv_rel if s < THRESH)
    return dim, sv_rel


print("mpmath dps = " + str(mp.mp.dps) + ", threshold = 1e-18 "
      "(true commuting dirs land ~1e-27; non-commuting >= ~1e-9)\n")
for n in (0, 1, 2):
    for N in (5, 6):
        B, j, jN = radial_BKB(n, N); P = N-1
        ev = sorted(mp.eigsy(B, eigvals_only=True), reverse=True)
        gaps = [abs(ev[i]-ev[i+1]) for i in range(P-1)]
        mingap = min(gaps)
        print(f"=== n={n} N={N}  (size {P}x{P}, c=j_n,N={mp.nstr(jN,8)}) ===")
        print(f"    1-lambda: {[mp.nstr(1-e,4) for e in ev]}")
        print(f"    min eigen-gap = {mp.nstr(mingap,4)}")
        sweep = {}; smallest = {}
        for p in range(P):
            dim, sv = commutant_singvals(B, p)
            sweep[p] = dim
            smallest[p] = sv
            if p == 1:
                tri_sv = sv
        tri_list = [mp.nstr(s, 3) for s in tri_sv[:6]]
        full_list = [mp.nstr(s, 3) for s in smallest[P-1][:P+1]]
        print("    Candidate B  dim(commutant) by bandwidth p: " + str(sweep) +
              "  (p=1 tridiagonal; full p=" + str(P-1) + " -> " + str(P) + ")")
        print("      tridiagonal (p=1) singular values (rel): " + str(tri_list))
        print("      full-band (p=" + str(P-1) + ") smallest svs (rel):   " + str(full_list))
        BC = radial_BKC(n, N)
        dimC, svC = commutant_singvals(BC, 1)
        print("    Candidate C  dim(TRIDIAGONAL commutant) = " + str(dimC) +
              "   smallest sv (rel): " + str([mp.nstr(s, 3) for s in svC[:4]]))
        print()
