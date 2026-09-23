"""
Task 7.6 — Does a sparse (tridiagonal) operator commute with the finite radial prolate matrix?

Strategy: the symmetric matrices that commute with a real symmetric matrix M with SIMPLE
spectrum form exactly the algebra of polynomials in M (dimension P = size of M). A *tridiagonal*
(or banded, bandwidth p) symmetric matrix commutes with M iff some such polynomial happens to be
banded. We test this directly by the null space of the linear "commutator map"

    L_p :  {symmetric matrices of bandwidth <= p}  ->  matrices,   J |-> M J - J M.

dim ker L_p = number of independent symmetric bandwidth-p matrices commuting with M.
  * p = 0 (diagonal): ker always contains span{I} (dim >= 1; = 1 iff no diagonal != I commutes).
  * p = 1 (tridiagonal): dim ker = 1  <=>  ONLY multiples of I  <=>  NO nontrivial commuting
    tridiagonal operator. dim ker >= 2 <=> a nontrivial one exists (the 1D DPSS situation).

Validation: 1D DPSS prolate (sinc) matrix has Slepian's exact commuting tridiagonal (Slepian
1978) -> the method MUST report dim ker L_1 >= 2 there. Then we apply it to the project's finite
radial prolate matrix B^(B) (Candidate B, coefficient domain) and B^(C) (Candidate C, symmetric
sample domain) for n = 0,1,2.

Numerical care: project eigenvalues cluster super-exponentially (closure c = j_{n,N} => all
plateau). We keep N small (min eigen-gap >> machine eps) and READ the singular-value spectrum of
L_p: true commuting directions sit at ~machine zero; clustering-induced near-misses sit at ~gap
size, well above. A separate mpmath high-precision run (task_7_6_hp_confirm.py) confirms n=0.
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad
np.set_printoptions(precision=3, linewidth=160, suppress=False)


# ---------- matrices ----------
def radial_BKB(n, N):
    """Finite radial prolate matrix B^(B) in the orthonormal Fourier-Bessel basis (Task 6.1)."""
    jz = jn_zeros(n, N); j = jz[:N-1]; jN = jz[N-1]; c = jN; P = N-1
    B = np.zeros((P, P)); pts = [z for z in jz if 0 < z < c]
    for a in range(P):
        for b in range(a, P):
            jm, jk = j[a], j[b]
            f = lambda u: u*jn(n, u)**2/((u**2-jm**2)*(u**2-jk**2))
            v, _ = quad(f, 0, c, points=pts, limit=400)
            B[a, b] = B[b, a] = 2*jm*jk*v
    return B, j, jN


def radial_BKC(n, N):
    """Candidate-C symmetric SAMPLE-domain matrix B^(C) = W^{1/2} B^(A) W^{-1/2},
       B^(A) = S^{-1} B^(B) S, S = D T.  Same spectrum as B^(B); different eigenvectors."""
    B, j, jN = radial_BKB(n, N)
    P = N-1; K = 1.0; R = jN/K
    D = np.diag(R*jn(n+1, j)/np.sqrt(2))
    W = np.diag(2/(K**2*jn(n+1, j)**2))
    T = np.array([[2/(jn(n+1, j[a])*jn(n+1, j[b])*jN)*jn(n, j[a]*j[b]/jN)
                   for b in range(P)] for a in range(P)])
    S = D@T
    BA = np.linalg.solve(S, B@S)          # S^{-1} B S
    Wh = np.diag(np.sqrt(np.diag(W)))
    Wih = np.diag(1/np.sqrt(np.diag(W)))
    BC = Wh@BA@Wih
    return 0.5*(BC+BC.T)                   # symmetrize (numerical)


def dpss_prolate(M, Wb):
    """1D DPSS prolate (sinc-kernel) matrix, M x M, half-bandwidth Wb."""
    P = np.empty((M, M))
    for a in range(M):
        for b in range(M):
            P[a, b] = 2*Wb if a == b else np.sin(2*np.pi*Wb*(a-b))/(np.pi*(a-b))
    return P


def slepian_tridiagonal(M, Wb):
    """Slepian's tridiagonal that commutes EXACTLY with the DPSS prolate matrix (Slepian 1978)."""
    d = np.array([((M-1)/2.0 - k)**2*np.cos(2*np.pi*Wb) for k in range(M)])
    e = np.array([0.5*(k+1)*(M-1-k) for k in range(M-1)])
    return np.diag(d) + np.diag(e, 1) + np.diag(e, -1)


# ---------- commutator null-space over banded symmetric matrices ----------
def banded_sym_basis(P, p):
    """List of symmetric elementary matrices with bandwidth <= p."""
    basis = []
    for i in range(P):
        for jj in range(i, min(i+p+1, P)):
            E = np.zeros((P, P))
            if i == jj:
                E[i, i] = 1.0
            else:
                E[i, jj] = E[jj, i] = 1.0
            basis.append(E)
    return basis


def commutant_dim(M, p, tol):
    """Singular values of J|->[M,J] on symmetric bandwidth-p J; null dim at threshold tol."""
    P = M.shape[0]
    basis = banded_sym_basis(P, p)
    cols = [(M@E - E@M).ravel() for E in basis]
    L = np.array(cols).T                       # P^2 x K
    sv = np.linalg.svd(L, compute_uv=False)
    scale = np.linalg.norm(M, 2)
    sv_rel = sv/scale
    dim = int(np.sum(sv_rel < tol))
    return dim, np.sort(sv_rel), basis, L


def commuting_matrices(M, p, tol):
    """Return the actual symmetric banded matrices spanning ker (for inspection)."""
    P = M.shape[0]
    basis = banded_sym_basis(P, p)
    L = np.array([(M@E - E@M).ravel() for E in basis]).T
    U, sv, Vt = np.linalg.svd(L, full_matrices=True)
    scale = np.linalg.norm(M, 2)
    nullvecs = [Vt[i] for i in range(len(basis)) if (sv[i] if i < len(sv) else 0)/scale < tol]
    mats = []
    for coef in nullvecs:
        Jm = sum(c*E for c, E in zip(coef, basis))
        mats.append(Jm)
    return mats


TOL = 1e-10   # threshold: true zeros ~1e-15; clustering near-misses ~ eigen-gap (>=1e-9 here)

print("="*78)
print("PART 1 — METHOD VALIDATION ON 1D DPSS (commuting tridiagonal KNOWN to exist)")
print("="*78)
for (M, Wb) in [(6, 0.25), (7, 0.2), (9, 0.25)]:
    P = dpss_prolate(M, Wb); Tsl = slepian_tridiagonal(M, Wb)
    comm = Tsl@P - P@Tsl
    lam = np.sort(np.linalg.eigvalsh(P))[::-1]
    dim1, sv1, _, _ = commutant_dim(P, 1, TOL)
    dim0, sv0, _, _ = commutant_dim(P, 0, TOL)
    print(f"\n1D M={M}, W={Wb}: ||[T_Slepian, P]||/||P|| = {np.linalg.norm(comm,2)/np.linalg.norm(P,2):.2e}"
          f"   (exact-commute check)")
    print(f"   eigenvalues of P: {np.array2string(lam, precision=4)}")
    print(f"   dim(diagonal commutant,  p=0) = {dim0}")
    print(f"   dim(TRIDIAGONAL commutant, p=1) = {dim1}   <-- expect >= 2 (I + Slepian)")
    print(f"   smallest 4 singular values (rel): {sv1[:4]}")

print("\n"+"="*78)
print("PART 2 — FINITE RADIAL PROLATE MATRIX  (Candidate B, coefficient domain)")
print("="*78)
for n in (0, 1, 2):
    for N in (5, 6):
        B, j, jN = radial_BKB(n, N); P = N-1
        lam = np.sort(np.linalg.eigvalsh(B))[::-1]
        mingap = np.abs(np.diff(lam)).min()
        dims = {p: commutant_dim(B, p, TOL)[0] for p in range(P)}
        _, sv1, _, _ = commutant_dim(B, 1, TOL)
        print(f"\nn={n} N={N} (size {P}x{P}): min eigen-gap={mingap:.2e}")
        print(f"   dim(commutant) by bandwidth p: {dims}   [p=1 is tridiagonal; full p={P-1} -> {P}]")
        print(f"   tridiagonal singular values (rel), smallest {min(5,len(sv1))}: {sv1[:5]}")

print("\n"+"="*78)
print("PART 3 — CANDIDATE C symmetric SAMPLE-domain matrix B^(C) (closest to 1D DPSS setting)")
print("="*78)
for n in (0, 1, 2):
    for N in (5, 6):
        BC = radial_BKC(n, N); P = N-1
        lam = np.sort(np.linalg.eigvalsh(BC))[::-1]
        mingap = np.abs(np.diff(lam)).min()
        dim1 = commutant_dim(BC, 1, TOL)[0]
        dim0 = commutant_dim(BC, 0, TOL)[0]
        print(f"n={n} N={N}: min gap={mingap:.2e}  dim(diag p=0)={dim0}  dim(TRIDIAG p=1)={dim1}")

print("\n"+"="*78)
print("PART 4 — sanity: in 1D the p=1 null space really is span{I, T_Slepian}")
print("="*78)
M, Wb = 6, 0.25
P = dpss_prolate(M, Wb); Tsl = slepian_tridiagonal(M, Wb)
mats = commuting_matrices(P, 1, TOL)
print(f"1D M={M}: found {len(mats)} commuting tridiagonal directions.")
# express: is Tsl in their span? project Tsl onto the span
if len(mats) >= 2:
    basisflat = np.array([m.ravel() for m in mats]).T
    coef, *_ = np.linalg.lstsq(basisflat, Tsl.ravel(), rcond=None)
    resid = np.linalg.norm(basisflat@coef - Tsl.ravel())/np.linalg.norm(Tsl.ravel())
    print(f"   Slepian T reconstructed from null space, rel resid = {resid:.2e}  (≈0 => T is in ker)")
