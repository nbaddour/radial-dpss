"""
Task 9.4 — high-precision test of the individual Level-4 rate for CLUSTERED (plateau) modes.

Question: does psi_M^{(m)} -> psi^{(m)} super-exponentially for clustered m (deep plateau), or is it
gap-starved? In Task 8.5 (double precision) clustered modes showed ~1e-2 rotation noise. Is that an
exact-arithmetic effect (=> conjecture A false) or a finite-precision artifact (=> conjecture A true,
matching the 1D DPSS->PSWF super-exp behavior, eigenfunctions pinned by oscillation order)?

At FIXED c, (B_M)_{mk} = (B_ref)_{mk} for m,k <= M (entries independent of M), so build B_ref ONCE in
high precision; B_M = top-left M x M block. Eigendecompose each block (mpmath eigsy resolves the
super-exp-clustered eigenvectors), reconstruct mode m at interior points, compare to the M_ref result
(sign-aligned). Super-exp decrease with M for clustered m => conjecture A supported.
"""
import mpmath as mp
mp.mp.dps = 30
n = 0; c = mp.mpf(20)

def besselzeros(N):
    return [mp.besseljzero(mp.mpf(n), k) for k in range(1, N+1)]

M_ref = 16
j = besselzeros(M_ref)
pts = [mp.mpf(0)] + [z for z in j if 0 < z < c] + [c]
# Build B_ref once (top-left blocks are the fixed-c matrices B_M)
print(f"building B_ref ({M_ref}x{M_ref}) at dps={mp.mp.dps}, fixed c={mp.nstr(c,4)} ...")
B = mp.zeros(M_ref, M_ref)
for a in range(M_ref):
    for b in range(a, M_ref):
        jm, jk = j[a], j[b]
        f = lambda u: u*mp.besselj(n, u)**2/((u**2-jm**2)*(u**2-jk**2))
        v = mp.quad(f, pts)
        B[a, b] = B[b, a] = 2*jm*jk*v

def eigpairs(M):
    Bm = B[:M, :M]
    E, V = mp.eigsy(Bm)               # ascending eigenvalues
    order = sorted(range(M), key=lambda i: -E[i])   # descending
    return [E[i] for i in order], [[V[r, i] for r in range(M)] for i in order]

def psi_at(coef, M, x):
    return mp.sqrt(2)*mp.fsum(coef[k]*mp.besselj(n, j[k]*x)/mp.besselj(n+1, j[k]) for k in range(M))

xs = [mp.mpf('0.15'), mp.mpf('0.4'), mp.mpf('0.65'), mp.mpf('0.85')]
lam_ref, V_ref = eigpairs(M_ref)
print("reference lambda_m, m=0..6: " + ", ".join(mp.nstr(lam_ref[m], 6) for m in range(7)))
print("\nindividual L4 nodal error max_x |psi_M^(m)(x) - psi_ref^(m)(x)| (sign-aligned), HIGH PRECISION:")
for M in (8, 10, 12, 14):
    lam, V = eigpairs(M)
    row = []
    for m in (0, 1, 2, 4, 5):
        vM = [psi_at(V[m], M, x) for x in xs]
        vR = [psi_at(V_ref[m], M_ref, x) for x in xs]
        dot = mp.fsum(vM[i]*vR[i] for i in range(len(xs)))
        if dot < 0:
            vM = [-t for t in vM]
        err = max(abs(vM[i]-vR[i]) for i in range(len(xs)))
        row.append(f"m{m}:{mp.nstr(err,2)}")
    print(f"  M={M:2d}: " + "  ".join(row))
print("\nIf clustered modes (m=0,1,2) decrease super-exp with M -> conjecture A supported (the 8.5")
print("double-precision ~1e-2 was a finite-precision artifact, not exact-arithmetic gap-starvation).")
