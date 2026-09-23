"""
Task 12.2 — Robust eigensolver workflow for the finite radial prolate matrix.

Consumes the assembled matrix B = B_K^{(B)}_{N-1}(n,c) (Task 12.1) and returns its eigenpairs
(lambda_m, c^{(m)}) robustly and reproducibly. The matrix is real-symmetric PSD with 0<=lambda<=1
(Task 6.6); its leading "plateau" eigenvalues cluster at 1 to within ~1e-11 or tighter, so the workflow
must be correct in the presence of NUMERICAL DEGENERACY (where individual eigenvectors are defined only
up to rotation within a cluster, but the cluster SUBSPACE is well-conditioned -- the Gate C posture).

Scope (Task 12.2 only): the eigendecomposition + robustness/diagnostics. The canonical sign/sort
CONVENTION (Def 7.3.3 largest-|component| positive; Conv 4.10 ordering + interior-zero tie-break) is
Task 12.3; synthesis/reconstruction is Task 12.4; the conditioning monitor / fallback trigger is 12.7.
Here we apply only a DETERMINISTIC provisional sign for reproducible storage and flag the clusters.

Design choices:
  * Solver: scipy.linalg.eigh (LAPACK symmetric). The matrix is DENSE (no commuting tridiagonal
    operator exists -- Task 7.7), so a dense symmetric solver is the right tool. Driver 'evr' (MRRR)
    is the default; 'ev' (QR) is used as an independent cross-check for reproducibility.
  * Spectrum is sorted DESCENDING (Conv 4.10.1: lambda_0 >= lambda_1 >= ...).
  * Eigenvalues are reported RAW; tiny out-of-[0,1] excursions (~1e-16) are reported, and a clamped
    copy in [0,1] is provided for downstream stability (clamp tolerance documented, not silent).
  * Degenerate clusters are identified by a relative gap test. The deep-plateau fine splitting is
    ill-conditioned BY DESIGN (gaps ~1e-15..1e-12); the reproducible, physically meaningful object is
    the LEADING CONCENTRATION SUBSPACE (leading K=#{lam>1/2} modes, O(1) plateau->plunge gap), which is
    verified stable across LAPACK drivers to machine precision.
"""
import os
import numpy as np
import scipy.linalg as sla
from task_12_1_assemble_matrix import assemble, Mn, _spectrum

# REPO root: parent of this code/ directory (override with env RDPSS_REPO if running out of tree).
REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(REPO, "data", "pilot")
BENCH = os.path.join(REPO, "data", "benchmark")


def solve(B, driver="evr", clamp_tol=1e-12):
    """Robust symmetric eigensolve. Returns a dict with raw/clamped eigenvalues, eigenvectors, and
    diagnostics (residual, orthonormality, [0,1] excursions). Eigenpairs sorted DESCENDING."""
    B = 0.5 * (B + B.T)                       # symmetrize defensively (B is symmetric by construction)
    w, V = sla.eigh(B, driver=driver)         # ascending
    w = w[::-1]
    V = V[:, ::-1]                            # descending; columns orthonormal

    # diagnostics
    resid = float(np.max(np.abs(B @ V - V * w)))           # max_m ||B v_m - lam_m v_m||_inf
    ortho = float(np.max(np.abs(V.T @ V - np.eye(V.shape[1]))))
    below = float(min(w.min(), 0.0))                       # how far below 0
    above = float(max(w.max() - 1.0, 0.0))                 # how far above 1

    w_clamped = np.clip(w, 0.0, 1.0)
    # provisional deterministic sign: largest-|component| positive (Def 7.3.3 form; convention=12.3)
    for m in range(V.shape[1]):
        j = int(np.argmax(np.abs(V[:, m])))
        if V[j, m] < 0:
            V[:, m] *= -1.0

    return dict(lam=w, lam_clamped=w_clamped, V=V, resid=resid, ortho=ortho,
                below0=below, above1=above, driver=driver)


def degenerate_clusters(lam, rel_tol=1e-8):
    """Group eigenvalues into numerically-degenerate clusters: consecutive lam_i whose successive
    gap is < rel_tol * max(1,|lam|). Returns list of index ranges (start, stop) (stop exclusive)."""
    clusters = []
    i = 0
    n = len(lam)
    while i < n:
        j = i + 1
        while j < n and abs(lam[j - 1] - lam[j]) < rel_tol * max(1.0, abs(lam[i])):
            j += 1
        clusters.append((i, j))
        i = j
    return clusters


def subspace_projector(V, lo, hi):
    """Orthogonal projector onto the eigenvector columns [lo:hi)."""
    Vs = V[:, lo:hi]
    return Vs @ Vs.T


# ----------------------------------------------------------------------------------------------------
# Verification battery
# ----------------------------------------------------------------------------------------------------
def reanchor_pilot():
    """Solve the saved n=0 pilot matrices; compare eigenvalues to the pilot's stored lam."""
    print("=" * 96)
    print("RE-ANCHOR: eigensolver vs saved n=0 pilot eigenvalues (Task 11.2/11.3)")
    print("=" * 96)
    worst_lam = 0.0
    worst_res = 0.0
    worst_ortho = 0.0
    files = sorted(f for f in os.listdir(PILOT) if f.startswith("Bmat_n0_") and f.endswith(".npz"))
    for f in files:
        d = np.load(os.path.join(PILOT, f))
        B = d["B"]
        lam_saved = d["lam"]                 # pilot stored sorted-descending eigenvalues
        r = solve(B, driver="evr")
        k = min(len(lam_saved), len(r["lam"]))
        e = float(np.max(np.abs(np.sort(r["lam"])[::-1][:k] - np.sort(lam_saved)[::-1][:k])))
        worst_lam = max(worst_lam, e)
        worst_res = max(worst_res, r["resid"])
        worst_ortho = max(worst_ortho, r["ortho"])
    print(f"  files: {len(files)}")
    print(f"  worst |lam_solve - lam_saved|      : {worst_lam:.2e}")
    print(f"  worst residual ||Bv - lam v||_inf  : {worst_res:.2e}")
    print(f"  worst orthonormality |V^T V - I|   : {worst_ortho:.2e}")
    return worst_lam, worst_res, worst_ortho


def verify_general(cases):
    print("\n" + "=" * 96)
    print("GENERAL (n,c) EIGENSOLVE: robustness + degeneracy + driver reproducibility")
    print("=" * 96)
    res_max = ortho_max = exc_max = 0.0
    drv_lam_max = drv_proj_max = 0.0
    bench_plateau_max = 0.0
    for (n, c, P) in cases:
        M = Mn(n, c)
        B = assemble(n, c, P=P, G=32)
        r = solve(B, driver="evr")
        r2 = solve(B, driver="ev")           # independent driver

        res_max = max(res_max, r["resid"])
        ortho_max = max(ortho_max, r["ortho"])
        exc = max(r["below0"] * -1, r["above1"])
        exc_max = max(exc_max, exc)

        # Driver reproducibility. Eigenvalues must agree to machine precision. Individual eigenvectors
        # in the deeply-degenerate plateau (successive gaps ~1e-15..1e-12) are defined only up to
        # rotation, so they are NOT expected to agree -- but the LEADING CONCENTRATION SUBSPACE
        # (leading K = #{lam>1/2} modes; O(1) plateau->plunge boundary gap; the Gate-C / Task 9.3
        # Level-3 object) is well-conditioned and MUST agree to machine precision.
        kk = min(len(r["lam"]), len(r2["lam"]))
        drv_lam = float(np.max(np.abs(np.sort(r["lam"])[::-1][:kk] - np.sort(r2["lam"])[::-1][:kk])))
        drv_lam_max = max(drv_lam_max, drv_lam)
        K = int((r["lam"] > 0.5).sum())
        Pi1 = subspace_projector(r["V"], 0, K)
        Pi2 = subspace_projector(r2["V"], 0, K)
        proj_err = float(np.max(np.abs(Pi1 - Pi2)))
        drv_proj_max = max(drv_proj_max, proj_err)
        clusters = degenerate_clusters(r["lam"], rel_tol=1e-8)

        # plateau eigenvalue match vs benchmark (correctness, regime-split as in 12.1)
        bp = np.nan
        bf = os.path.join(BENCH, f"cpswf_n{n}_c{int(c)}.npz")
        if os.path.exists(bf):
            lref = np.load(bf)["lam"]
            k = min(len(lref), len(r["lam"]), M + 3)
            err = np.abs(r["lam"][:k] - lref[:k])
            p_hi = max(M - 2, 1)
            bp = float(err[:p_hi].max())
            bench_plateau_max = max(bench_plateau_max, bp)

        biggest = max((hi - lo) for (lo, hi) in clusters)
        print(f"\n n={n} c={c:5.1f} P={P:3d} M={M}: resid={r['resid']:.1e} ortho={r['ortho']:.1e}"
              f" [0,1]-excursion={exc:.1e}")
        print(f"   #clusters={len(clusters)} largest-cluster-size={biggest}  "
              f"(plateau lam~1 degeneracy)")
        print(f"   driver evr-vs-ev: |dlam|={drv_lam:.1e}  leading-subspace(K={K}) diff={proj_err:.1e}")
        if not np.isnan(bp):
            print(f"   bench plateau (m<=M-2) |dlam| = {bp:.2e}")
    return dict(res=res_max, ortho=ortho_max, exc=exc_max,
                drv_lam=drv_lam_max, drv_proj=drv_proj_max, bench_plateau=bench_plateau_max)


def determinism_check(n, c, P, reps=3):
    """Same matrix solved repeatedly must give identical eigenvalues and identical subspace projectors."""
    B = assemble(n, c, P=P, G=32)
    base = solve(B, driver="evr")
    lam0 = base["lam"]
    clusters = degenerate_clusters(lam0, rel_tol=1e-8)
    dlam = 0.0
    dproj = 0.0
    for _ in range(reps):
        r = solve(B, driver="evr")
        dlam = max(dlam, float(np.max(np.abs(r["lam"] - lam0))))
        for (lo, hi) in clusters:
            dproj = max(dproj, float(np.max(np.abs(subspace_projector(r["V"], lo, hi)
                                                  - subspace_projector(base["V"], lo, hi)))))
    return dlam, dproj


if __name__ == "__main__":
    wl, wr, wo = reanchor_pilot()

    cases = [
        (0, 20.0, 44), (0, 40.0, 56),
        (1, 20.0, 30), (1, 40.0, 48),
        (2, 40.0, 48),
        (4, 40.0, 50), (4, 80.0, 90),
    ]
    s = verify_general(cases)

    dlam, dproj = determinism_check(2, 40.0, 48, reps=3)

    print("\n" + "=" * 96)
    print("OVERALL")
    print("=" * 96)
    print(f"  pilot re-anchor |dlam|            : {wl:.2e}")
    print(f"  worst residual ||Bv-lam v||       : {max(wr, s['res']):.2e}")
    print(f"  worst orthonormality |V^TV-I|     : {max(wo, s['ortho']):.2e}")
    print(f"  worst [0,1] excursion (raw lam)   : {s['exc']:.2e}")
    print(f"  driver evr-vs-ev eigenvalues      : {s['drv_lam']:.2e}")
    print(f"  driver evr-vs-ev leading-subspace : {s['drv_proj']:.2e}  (deep-plateau vecs rotate; leading subspace stable)")
    print(f"  re-run determinism |dlam|         : {dlam:.2e}")
    print(f"  re-run determinism subspace       : {dproj:.2e}")
    print(f"  bench plateau worst (correctness) : {s['bench_plateau']:.2e}")
    ok = (wl < 1e-12 and max(wr, s['res']) < 1e-12 and max(wo, s['ortho']) < 1e-12
          and s['exc'] < 1e-10 and s['drv_lam'] < 1e-12 and s['drv_proj'] < 1e-10
          and dlam == 0.0 and dproj < 1e-10 and s['bench_plateau'] < 1e-4)
    print(f"\n  TASK 12.2 EIGENSOLVER VERIFIED: {ok}")
