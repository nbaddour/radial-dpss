"""
Task 10.4 — Build the benchmark CPSWF dataset for selected (n, m, c).

For each selected (n, c) the reference solver (Task 10.1, Solver A) is run at a generous node count
(well above the Task 10.3 spectral threshold) and the following are stored for the leading/informative
modes m = 0 .. M_n(c)+4 (one file per case, data/benchmark/cpswf_n{n}_c{c}.npz):

  lam[m]              concentration eigenvalue (in [0,1], descending)
  x_gl, w_gl          Gauss-Legendre nodes/weights on [0,1] (measure x dx) -> exact quadrature
  psi_gl[:, m]        space-limited eigenfunction psi^{(m)} in SL(R) at the GL nodes (||.||_{x dx}=1)
  x_unif, psi_unif    psi^{(m)} on a portable dense uniform grid (Nystrom-interpolated; see note)
  rho_gl (=c*x_gl)    frequency nodes on [0,K]=[0,c] for the band-limited dual
  phitilde[:, m]      nondim band-limited dual phitilde(xi)=c*phi^{(m)}(c*xi), xi=rho/c (||phi||_{rho drho}=1)
  gap[m]              spectral gap to nearest neighbor (individual-eigenfunction conditioning ~ eps/gap)
  regime[m]           0=plateau (clustered at lam~1, subspace-ambiguous individual mode),
                      1=separated (individually reliable, eigenfunction ~ machine-accurate),
                      2=deep-tail (lam<TAIL_TOL: eigenvalue accurate, dual/interp 1/lam-amplified)
  reliab[m]           1 iff regime==1 (convenience alias)
  dual_norm[m], selfdual[m]   per-mode quality of the band-limited dual (||phi||^2 and R<->K self-duality)

Conventions (Task 1a / 7.3 / 10.2 sec.4): nondim x in [0,1], measure x dx, R=1, K=c=KR; eigenvalues
descending. SIGN fixed deterministically: psi^{(m)} (and its dual, tied) scaled so psi^{(m)}>0 at its
max-|.| node. (Task 11 may re-align sign at comparison time; this only makes the stored file reproducible.)

Reliability (Task 10.3 + this task): individual-eigenfunction error ~ eps_solve/gap (textbook conditioning).
PLATEAU modes (1-lam<PLATEAU_TOL) have gap~eps and are subspace-ambiguous: their *individual* labels
rotate, but the *subspace* they span is exact (validate at projector level). DEEP-TAIL modes (lam<TAIL_TOL)
have accurate eigenvalues but their dual/interpolation is amplified by 1/lam. SEPARATED modes (plunge +
transition) are individually machine-accurate. psi_gl is the raw eigenvector (no amplification); psi_unif
is a plotting/pointwise convenience whose tail-mode accuracy is 1/lam-limited -- use psi_gl for precision.

Node count: Nq_bench = max(200, 3*c) (generous over the 10.3 rule Nq~ceil(c/2)+30, also recorded).
"""
import numpy as np, os, json, datetime
from scipy.special import jv, jvp, jn_zeros

def Kmat(n, c, x):
    X = x[:, None]; Y = x[None, :]; cX, cY = c*X, c*Y
    JX, JY, JpX, JpY = jv(n, cX), jv(n, cY), jvp(n, cX), jvp(n, cY)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = c*(X*JY*JpX - Y*JX*JpY)/(Y**2 - X**2)
    cx = c*x
    np.fill_diagonal(K, (c**2/2)*(jvp(n, cx)**2 + (1-(n/cx)**2)*jv(n, cx)**2))
    return K

def Knfun_vec(n, c, xv, x):
    cx = c*x; cxv = c*xv
    with np.errstate(divide='ignore', invalid='ignore'):
        row = c*(xv*jv(n, cx)*jvp(n, cxv) - x*jv(n, cxv)*jvp(n, cx))/(x**2 - xv**2)
    near = np.abs(x - xv) < 1e-13
    if near.any():
        row[near] = (c**2/2)*(jvp(n, cxv)**2 + (1-(n/cxv)**2)*jv(n, cxv)**2)
    return row

def nystrom(n, c, Nq):
    t, w = np.polynomial.legendre.leggauss(Nq); x = 0.5*(t+1); w = 0.5*w
    s = np.sqrt(w*x); A = (s[:, None]*Kmat(n, c, x))*s[None, :]; A = 0.5*(A+A.T)
    lam, V = np.linalg.eigh(A); idx = np.argsort(lam)[::-1]
    return lam[idx], x, w, V[:, idx]/s[:, None]

def Mn(n, c):
    return int(np.sum(jn_zeros(n, int(c/np.pi)+30) < c))

CASES_N = [0, 1, 2, 4]
CASES_C = [10.0, 20.0, 40.0, 80.0]
N_UNIF  = 401
PLATEAU_TOL = 1e-6     # 1-lam < this  => plateau (subspace-ambiguous individual mode)
TAIL_TOL    = 1e-6     # lam   < this  => deep-tail (dual/interp 1/lam-amplified)

OUT = "/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
os.makedirs(OUT, exist_ok=True)

manifest = {"created": datetime.date.today().isoformat(),
            "convention": "nondim x in [0,1], measure x dx, R=1, K=c=KR; eigenvalues descending; "
                          "sign fixed so psi^{(m)}>0 at its max-|.| node (dual tied).",
            "regimes": {"0": "plateau (clustered lam~1, subspace-ambiguous)",
                        "1": "separated (individually reliable)",
                        "2": "deep-tail (lam<%.0e: dual/interp 1/lam-amplified)" % TAIL_TOL},
            "plateau_tol": PLATEAU_TOL, "tail_tol": TAIL_TOL, "n_unif": N_UNIF, "cases": []}

print("Building benchmark dataset -> data/benchmark/")
x_unif = np.linspace(0.0, 1.0, N_UNIF)
for n in CASES_N:
    for c in CASES_C:
        M = Mn(n, c); nmodes = M + 5
        Nq = int(max(200, 3*c)); Nq_min = int(np.ceil(c/2)) + 30
        lam, x, w, psi = nystrom(n, c, Nq)
        psi = psi[:, :nmodes].copy(); lam_m = lam[:nmodes].copy()

        sgn = np.ones(nmodes)
        for m in range(nmodes):
            j = np.argmax(np.abs(psi[:, m]))
            if psi[j, m] < 0: sgn[m] = -1.0
        psi *= sgn[None, :]

        xi = x.copy(); phit = np.zeros((Nq, nmodes)); dual_norm = np.zeros(nmodes); selfdual = np.zeros(nmodes)
        for m in range(nmodes):
            G = (jv(n, c*np.outer(xi, x)) * (psi[:, m]*x*w)[None, :]).sum(axis=1)
            phit[:, m] = c*G/np.sqrt(lam_m[m])
            dual_norm[m] = np.sum(phit[:, m]**2 * xi * w)
            s2 = 1.0 if (phit[:, m] @ psi[:, m]) >= 0 else -1.0
            selfdual[m] = np.max(np.abs(s2*phit[:, m] - psi[:, m]))

        psi_unif = np.zeros((N_UNIF, nmodes))
        for i, xv in enumerate(x_unif):
            row = Knfun_vec(n, c, xv, x) * (x*w)
            psi_unif[i, :] = (row @ psi) / lam_m
        for m in range(nmodes):
            j = np.argmax(np.abs(psi_unif[:, m]))
            if psi_unif[j, m] < 0: psi_unif[:, m] *= -1.0

        # spectral gap to nearest neighbor (use full lam for neighbors beyond the stored block)
        gap = np.empty(nmodes)
        for m in range(nmodes):
            nb = []
            if m-1 >= 0: nb.append(abs(lam_m[m]-lam_m[m-1]))
            nb.append(abs(lam_m[m]-lam[m+1]))     # next eigenvalue (exists; lam has Nq entries)
            gap[m] = min(nb)

        regime = np.ones(nmodes, dtype=int)                       # default separated
        regime[(1.0 - lam_m) < PLATEAU_TOL] = 0                    # plateau
        regime[lam_m < TAIL_TOL] = 2                              # deep-tail
        reliab = (regime == 1).astype(int)
        rho_gl = c*x

        fname = f"cpswf_n{n}_c{int(c)}.npz"
        np.savez_compressed(os.path.join(OUT, fname),
            n=n, c=c, Nq=Nq, Nq_min_recommended=Nq_min, Mn=M, nmodes=nmodes,
            lam=lam_m, x_gl=x, w_gl=w, psi_gl=psi, x_unif=x_unif, psi_unif=psi_unif,
            rho_gl=rho_gl, phitilde=phit, gap=gap, regime=regime, reliab=reliab,
            dual_norm=dual_norm, selfdual=selfdual)
        npl=int((regime==0).sum()); nse=int((regime==1).sum()); nta=int((regime==2).sum())
        manifest["cases"].append(dict(file=fname, n=n, c=c, Mn=M, nmodes=nmodes, Nq=Nq,
            n_plateau=npl, n_separated=nse, n_tail=nta))
        print(f"  n={n} c={c:5.1f}: M_n={M}, {nmodes} modes, Nq={Nq}  -> "
              f"plateau={npl} separated={nse} tail={nta}")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nWrote {len(manifest['cases'])} cases + manifest.json to {OUT}")
