"""
Task 12.4 — Synthesis / interpolation: the reconstruction map from canonical eigenpairs to continuous
radial functions, plus the four-source error separation (Task 8.4).

Maps (Task 8.1 / 8.2 / 8.7), all on V_{N-1} = span{phi_{n,k}}_{k=1..N-1}:

  SYNTHESIS (primary, EXACT isometry R^{N-1} ~= V_{N-1}):
     nondim:  psi~_N^{(m)}(x) = sqrt2 * sum_k [c_k^{(m)}/J_{n+1}(j_{n,k})] J_n(j_{n,k} x),  x in [0,1]
     dim:     psi_N^{(m)}(r)  = (1/R) psi~_N^{(m)}(r/R),  r in [0,R]   (nodes r_k = j_{n,k}/K = R x_k)
     properties: unit weighted-L^2 norm (= c^T c), orthonormal family, psi_N(R)=0, O(r^n) at origin.

  NODAL SAMPLING / INTERPOLATION:
     sampling matrix (nondim) Phi~[j,k] = phi~_{n,k}(x_j) at nodes x_j = j_{n,j}/c ;
     analysis-from-samples  c = Phi~^{-1} a   (EXACT for data in V_{N-1});
     interpolation I_N[a](x) = synthesis(Phi~^{-1} a)(x) with cardinal property L_k(x_j)=delta_{jk}.

FOUR-SOURCE ERROR SEPARATION (Task 8.4), per mode m vs the Task 10.4 benchmark:
  1. DISCRETIZATION : |lam_m^{(N)} - lam_m^{ref}|                 (eigenvalue/operator; super-exp m<M)
  2. TRUNCATION     : FB tail sum_{k>P} |<psi^ref_m, phi~_{n,k}>|^2 (eigenfunction; super-exp m<M)
  3. RECONSTRUCTION : synthesis defect == 0 in the coefficient route (isometry; verified ~machine)
  4. MODE-ID        : eigenvalue gap min(|lam_m - lam_{m±1}|)      (binding only where gaps shrink)

Consumes the Task 12.3 canonical eigenpairs. Scope: synthesis/interpolation + the error split (the tool
Task 13 runs across the full sweep). r=0 stabilization is Task 12.5; weighted-orthogonality correction is
12.6; the conditioning monitor is 12.7.
"""
import os
import numpy as np
from scipy.special import jv, jn_zeros
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize

# REPO root: parent of this code/ directory (override with env RDPSS_REPO if running out of tree).
REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(REPO, "data", "pilot")
BENCH = os.path.join(REPO, "data", "benchmark")


# ---- synthesis ---------------------------------------------------------------------------------------
def basis_nd(n, jz, x):
    """Phi[i,k] = phi~_{n,k}(x_i) = sqrt2 J_n(j_{n,k} x_i)/J_{n+1}(j_{n,k})  (orthonormal in x dx)."""
    return np.sqrt(2.0) * jv(n, np.outer(x, jz)) / jv(n + 1, jz)[None, :]


def synthesize_nd(n, jz, cvec, x):
    """psi~_N(x) for coefficient columns cvec."""
    return basis_nd(n, jz, x) @ cvec


def synthesize_dim(n, jz, cvec, r, R):
    """psi_N(r) = (1/R) psi~_N(r/R) on [0,R]."""
    return (1.0 / R) * synthesize_nd(n, jz, cvec, r / R)


# ---- nodal sampling / interpolation ------------------------------------------------------------------
def nodes_nd(jz, c):
    """Nondim Bessel-zero nodes x_k = j_{n,k}/c (Task 8.7)."""
    return jz / c


def sampling_matrix_nd(n, jz, c):
    """Phi~[j,k] = phi~_{n,k}(x_j), x_j = j_{n,j}/c."""
    return basis_nd(n, jz, nodes_nd(jz, c))


def interpolate_nd(n, jz, a, x, c):
    """Exact interpolant of nodal data a: recover coefficients c=Phi~^{-1} a, then synthesize at x."""
    Phi = sampling_matrix_nd(n, jz, c)
    coef = np.linalg.solve(Phi, a)
    return synthesize_nd(n, jz, coef, x), coef


# ---- four-source error separation --------------------------------------------------------------------
def fb_coefficients(n, jz, x_gl, w_gl, psi_ref):
    """FB coefficients b[k,m] = <psi^ref_m, phi~_{n,k}> via GL quadrature on the benchmark grid."""
    Phi = basis_nd(n, jz, x_gl)                 # (Ngl, K)
    return (Phi * (x_gl * w_gl)[:, None]).T @ psi_ref     # (K, nmodes)


def error_sources(n, c, P):
    """Return per-mode four-source errors vs the Task 10.4 benchmark, plus the reconstruction defects."""
    jz = jn_zeros(n, P)
    B = assemble(n, c, P=P, G=32)
    r = solve(B, "evr")
    lam, V = canonicalize(r["lam"], r["V"], n, jz)
    M = Mn(n, c)

    d = np.load(os.path.join(BENCH, f"cpswf_n{n}_c{int(c)}.npz"))
    x_gl, w_gl, psi_ref = d["x_gl"], d["w_gl"], d["psi_gl"]
    lam_ref = d["lam"]; nb = int(d["nmodes"])
    nm = min(nb, P)

    # (1) discretization: eigenvalue error
    disc = np.abs(lam[:nm] - lam_ref[:nm])

    # (2) truncation: FB tail of the TRUE eigenfunction beyond k=P
    Kmax = P + 80
    jzK = jn_zeros(n, Kmax)
    b = fb_coefficients(n, jzK, x_gl, w_gl, psi_ref[:, :nm])   # (Kmax, nm)
    trunc = np.array([float(np.sum(b[P:, m] ** 2)) for m in range(nm)])

    # (3) reconstruction defect (coefficient route): unit norm / orthonormality / boundary -> ~0
    psiN = synthesize_nd(n, jz, V[:, :nm], x_gl)
    norms = np.sum(psiN ** 2 * (x_gl * w_gl)[:, None], axis=0)
    e_norm = float(np.max(np.abs(norms - 1)))
    G = (psiN * (x_gl * w_gl)[:, None]).T @ psiN
    e_ortho = float(np.max(np.abs(G - np.eye(nm))))
    e_bdry = float(np.max(np.abs(synthesize_nd(n, jz, V[:, :nm], np.array([1.0]))[0])))
    recon = max(e_norm, e_ortho, e_bdry)

    # (4) mode-id: eigenvalue gaps
    gap = np.empty(nm)
    for m in range(nm):
        gl = lam[m - 1] - lam[m] if m > 0 else np.inf
        gr = lam[m] - lam[m + 1] if m < len(lam) - 1 else np.inf
        gap[m] = min(gl, gr)

    return dict(M=M, lam=lam[:nm], disc=disc, trunc=trunc, recon=recon, gap=gap,
                e_norm=e_norm, e_ortho=e_ortho, e_bdry=e_bdry)


# ---- verification ------------------------------------------------------------------------------------
def verify():
    print("=" * 100)
    print("TASK 12.4 -- synthesis/interpolation + four-source error separation")
    print("=" * 100)

    # A. synthesis machinery: norm/ortho/boundary + dim<->nondim, general (n,c)
    print("\n[A] Synthesis: unit norm, orthonormality, boundary psi_N(1)=0, dim<->nondim rescale")
    wn = wo = wb = wdim = 0.0
    for (n, c, P) in [(0,20.,44),(0,40.,56),(1,20.,30),(1,40.,48),(2,40.,48),(4,40.,50),(4,80.,90)]:
        jz = jn_zeros(n, P); B = assemble(n,c,P=P,G=32); r = solve(B,"evr")
        lam, V = canonicalize(r["lam"], r["V"], n, jz)
        x = np.linspace(0,1,300)[1:]   # avoid x=0 exact for the n>=1 O(x^n)
        xg, wg = np.polynomial.legendre.leggauss(400); xg = 0.5*(xg+1); wg *= 0.5
        psi = synthesize_nd(n, jz, V, xg)
        norms = np.sum(psi**2*(xg*wg)[:,None],axis=0); wn=max(wn,float(np.max(np.abs(norms-1))))
        Gm=(psi*(xg*wg)[:,None]).T@psi; wo=max(wo,float(np.max(np.abs(Gm-np.eye(P)))))
        wb=max(wb,float(np.max(np.abs(synthesize_nd(n,jz,V,np.array([1.0]))[0]))))
        R=2.7; rr=R*x
        dim=synthesize_dim(n,jz,V[:,:3],rr,R); nd=synthesize_nd(n,jz,V[:,:3],x)
        wdim=max(wdim,float(np.max(np.abs(R*dim-nd))))
    print(f"  worst |‖psi‖^2-1| = {wn:.2e}   ortho = {wo:.2e}   psi(1) = {wb:.2e}   dim<->nondim = {wdim:.2e}")

    # B. interpolation: cardinal property L_k(x_j)=delta + exact round-trip from samples
    print("\n[B] Interpolation: cardinal property + exact analysis-from-samples (data in V_{N-1})")
    wc = wrt = 0.0
    for (n, c, P) in [(0,20.,30),(1,40.,40),(4,40.,40)]:
        jz = jn_zeros(n, P); Phi = sampling_matrix_nd(n, jz, c)
        card = Phi @ np.linalg.inv(Phi)
        wc = max(wc, float(np.max(np.abs(card - np.eye(P)))))
        rng = np.random.default_rng(0); cc = rng.standard_normal(P); cc/=np.linalg.norm(cc)
        a = Phi @ cc
        _, crec = interpolate_nd(n, jz, a, np.array([0.5]), c)
        wrt = max(wrt, float(np.max(np.abs(crec - cc))))
        condPhi = np.linalg.cond(Phi)
        print(f"  n={n} c={c} P={P}: cardinal |Phi Phi^-1 - I|={float(np.max(np.abs(card-np.eye(P)))):.1e}"
              f"  round-trip |c_rec-c|={float(np.max(np.abs(crec-cc))):.1e}  cond(Phi)={condPhi:.1e}")

    # C. four-source error separation (n=0,c=20 reproduces Task 8.4 signatures; generalizes to n>=1)
    print("\n[C] Four-source error separation vs benchmark (Task 8.4)")
    for (n, c, P) in [(0,20.,44),(2,40.,48),(4,40.,50)]:
        es = error_sources(n, c, P); M = es["M"]
        print(f"\n  n={n} c={c} P={P} M={M}: reconstruction(coeff route) = {es['recon']:.2e} "
              f"(norm {es['e_norm']:.1e}/ortho {es['e_ortho']:.1e}/bdry {es['e_bdry']:.1e})")
        print(f"   m :   lam       discretization   truncation(FBtail)   mode-id gap   regime")
        for m in range(min(M+2, len(es['lam']))):
            reg = "plateau" if m < M-1 else ("plunge" if m <= M else "tail")
            print(f"   {m:2d}: {es['lam'][m]:.6f}   {es['disc'][m]:.3e}        {es['trunc'][m]:.3e}"
                  f"          {es['gap'][m]:.2e}   {reg}")

    ok = (wn<1e-12 and wo<1e-11 and wb<1e-12 and wdim<1e-12 and wc<1e-9 and wrt<1e-9)
    print("\n" + "=" * 100)
    print(f"  synthesis: norm {wn:.1e} ortho {wo:.1e} bdry {wb:.1e} dim<->nondim {wdim:.1e}")
    print(f"  interpolation: cardinal {wc:.1e} round-trip {wrt:.1e}")
    print(f"\n  TASK 12.4 SYNTHESIS/INTERPOLATION VERIFIED: {ok}")


if __name__ == "__main__":
    verify()
