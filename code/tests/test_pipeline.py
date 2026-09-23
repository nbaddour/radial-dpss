"""
Task 12.9 — Automated test suite for the Stage-5 computational pipeline (Tasks 12.1-12.8).

Each test asserts the CORE numerical invariant of one pipeline module, at modest sizes so the whole
suite runs quickly (< 1 s). Tests that need the benchmark/pilot datasets skip cleanly when data/ is
absent, so the math tests run on a fresh checkout. Determinism: the only randomness (Task 12.4 round-trip)
uses a fixed seed.

Run:  cd code && python -m pytest tests -q
"""
import os
import numpy as np
import pytest
from scipy.special import jn_zeros

from task_12_1_assemble_matrix import assemble, assemble_routeA, Mn
from task_12_2_eigensolver import solve, subspace_projector
from task_12_3_canonicalize import canonicalize, interior_zero_count, reconstruct_nd, faithful_plateau_modes
from task_12_4_synthesis import (synthesize_nd, synthesize_dim, sampling_matrix_nd,
                                 interpolate_nd, error_sources)
from task_12_5_origin import origin_coeff, synthesize_stable, normalized_profile
from task_12_6_worthogonality import assess as w_assess
from task_12_7_monitor import monitor
from task_12_8_tracking import track_c_sweep, leading_subspace_jump, separated_modes

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HAVE_PILOT = os.path.isdir(os.path.join(REPO, "data", "pilot"))
HAVE_BENCH = os.path.isdir(os.path.join(REPO, "data", "benchmark"))


# ---- 12.1 assembly ----------------------------------------------------------------------------------
@pytest.mark.parametrize("n,c,P", [(0, 20., 24), (2, 30., 30), (4, 40., 40)])
def test_assembly_structure(n, c, P):
    B, cache = assemble(n, c, P=P, G=24, return_cache=True)
    assert np.max(np.abs(B - B.T)) == 0.0                          # exact symmetry
    lam = np.linalg.eigvalsh(B)
    assert lam.min() > -1e-12 and lam.max() < 1 + 1e-12            # PSD, in [0,1]
    assert np.max(np.abs(B - assemble_routeA(n, c, P, cache))) < 1e-14   # route A vs B
    assert abs(int((lam > 0.5).sum()) - Mn(n, c)) <= 1            # Shannon count

def test_assembly_quadrature_converged():
    B24 = assemble(0, 20., P=30, G=24)
    B40 = assemble(0, 20., P=30, G=40)
    assert np.max(np.abs(B24 - B40)) < 1e-12

@pytest.mark.skipif(not HAVE_PILOT, reason="pilot data absent")
def test_assembly_reanchor_pilot():
    import glob
    f = sorted(glob.glob(os.path.join(REPO, "data", "pilot", "Bmat_n0_*.npz")))[0]
    d = np.load(f)
    assert np.max(np.abs(assemble(0, float(d["c"]), P=int(d["P"]), G=24) - d["B"])) == 0.0


# ---- 12.2 eigensolver -------------------------------------------------------------------------------
@pytest.mark.parametrize("n,c,P", [(0, 20., 30), (2, 40., 40)])
def test_eigensolver_health(n, c, P):
    B = assemble(n, c, P=P, G=32)
    r = solve(B, "evr")
    assert r["resid"] < 1e-12 and r["ortho"] < 1e-12
    assert max(-r["below0"], r["above1"]) < 1e-10                 # eigenvalues in [0,1]

def test_eigensolver_determinism_and_driver_subspace():
    B = assemble(2, 40., P=40, G=32)
    r1 = solve(B, "evr"); r2 = solve(B, "evr")
    assert np.max(np.abs(r1["lam"] - r2["lam"])) == 0.0          # bit-identical re-run
    rev = solve(B, "ev")
    K = int((r1["lam"] > 0.5).sum())
    P1 = subspace_projector(r1["V"], 0, K); P2 = subspace_projector(rev["V"], 0, K)
    assert np.max(np.abs(P1 - P2)) < 1e-10                        # leading subspace driver-stable


# ---- 12.3 canonicalize ------------------------------------------------------------------------------
def test_canonical_norm_and_sign():
    n, c, P = 0, 20., 44
    jz = jn_zeros(n, P); B = assemble(n, c, P=P, G=32)
    lam1, V1 = canonicalize(*(lambda r: (r["lam"], r["V"]))(solve(B, "evr")), n, jz)
    lam2, V2 = canonicalize(*(lambda r: (r["lam"], r["V"]))(solve(B, "ev")), n, jz)
    assert np.max(np.abs(np.sum(V1 ** 2, axis=0) - 1)) < 1e-12    # unit norm
    # well-separated modes: canonical sign reproducible across drivers
    for m in range(len(lam1)):
        gap = min(lam1[m-1]-lam1[m] if m>0 else 9, lam1[m]-lam1[m+1] if m<len(lam1)-1 else 9)
        if gap > 1e-4 and lam1[m] > 1e-3:
            assert np.max(np.abs(V1[:, m] - V2[:, m])) < 1e-11

def test_canonical_nodal_identity():
    n, c, P = 2, 40., 48
    jz = jn_zeros(n, P); B = assemble(n, c, P=P, G=32)
    lam, V = canonicalize(*(lambda r: (r["lam"], r["V"]))(solve(B, "evr")), n, jz)
    xg = np.linspace(0, 1, 4000)
    for m in faithful_plateau_modes(lam, Mn(n, c)):
        assert interior_zero_count(reconstruct_nd(n, jz, V[:, m], xg), xg) == m


# ---- 12.4 synthesis / interpolation -----------------------------------------------------------------
def test_synthesis_isometry():
    n, c, P = 2, 40., 48
    jz = jn_zeros(n, P); B = assemble(n, c, P=P, G=32)
    lam, V = canonicalize(*(lambda r: (r["lam"], r["V"]))(solve(B, "evr")), n, jz)
    xg, wg = np.polynomial.legendre.leggauss(400); xg = 0.5*(xg+1); wg *= 0.5
    psi = synthesize_nd(n, jz, V, xg)
    assert np.max(np.abs(np.sum(psi**2*(xg*wg)[:,None], axis=0) - 1)) < 1e-11   # unit norm
    G = (psi*(xg*wg)[:,None]).T @ psi
    assert np.max(np.abs(G - np.eye(P))) < 1e-10                                # orthonormal
    assert np.max(np.abs(synthesize_nd(n, jz, V, np.array([1.0]))[0])) < 1e-11  # psi(1)=0

def test_interpolation_exact():
    n, c, P = 1, 40., 40
    jz = jn_zeros(n, P); Phi = sampling_matrix_nd(n, jz, c)
    assert np.max(np.abs(Phi @ np.linalg.inv(Phi) - np.eye(P))) < 1e-9          # cardinal
    rng = np.random.default_rng(0); cc = rng.standard_normal(P); cc /= np.linalg.norm(cc)
    _, crec = interpolate_nd(n, jz, Phi @ cc, np.array([0.5]), c)
    assert np.max(np.abs(crec - cc)) < 1e-9                                     # round-trip

@pytest.mark.skipif(not HAVE_BENCH, reason="benchmark data absent")
def test_error_sources_reconstruction_zero():
    es = error_sources(0, 20., 44)
    assert es["recon"] < 1e-12                                                  # coeff-route recon = 0
    assert es["disc"][0] < 1e-10 and es["trunc"][0] < 1e-6                      # leading mode tiny


# ---- 12.5 near-origin -------------------------------------------------------------------------------
def test_origin_coeff_is_exact_limit():
    n, c, P = 4, 40., 50
    jz = jn_zeros(n, P); B = assemble(n, c, P=P, G=32)
    lam, V = canonicalize(*(lambda r: (r["lam"], r["V"]))(solve(B, "evr")), n, jz)
    zeta = origin_coeff(n, jz, V[:, :5])
    m = 3
    r4 = abs(float(synthesize_nd(n, jz, V[:, m][:, None], np.array([1e-4]))[0,0])/(1e-4**n) - zeta[m])
    r5 = abs(float(synthesize_nd(n, jz, V[:, m][:, None], np.array([1e-5]))[0,0])/(1e-5**n) - zeta[m])
    assert abs(r4/r5 - 100.0) < 5.0                                             # x^2 law => exact limit

def test_synthesize_stable_matches_direct_and_finite():
    n, c, P = 4, 40., 50
    jz = jn_zeros(n, P); B = assemble(n, c, P=P, G=32)
    lam, V = canonicalize(*(lambda r: (r["lam"], r["V"]))(solve(B, "evr")), n, jz)
    xg = np.linspace(1e-6, 1., 300)
    assert np.max(np.abs(synthesize_stable(n, jz, V[:, :5], xg) - synthesize_nd(n, jz, V[:, :5], xg))) < 1e-12
    p0 = synthesize_stable(n, jz, V[:, :5], np.array([0.0]))[0]
    assert np.all(np.abs(p0) < 1e-12) and np.all(np.isfinite(p0))               # psi(0)=0 for n>=1


# ---- 12.6 weighted orthogonality --------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 2])
def test_worthogonality_exact_layers_and_correction(n):
    a = w_assess(n, 24)
    assert a["coeff_I"] < 1e-12 and a["coeff_band"] < 1e-12       # exact coefficient layer
    assert a["nodal_defect"] > 1e-9                               # nodal W-defect is real (eps_N)
    assert a["nodal_corr_defect"] < 1e-10                         # Lowdin correction is exact


# ---- 12.7 monitor -----------------------------------------------------------------------------------
def test_monitor_no_fallback_healthy():
    rep = monitor(0, 40., 56)
    assert rep["fallback"] is False
    assert max(rep["malign"].values()) < 1e-10

def test_monitor_fires_on_broken_matrix():
    B = assemble(0, 20., P=44, G=32); Bbad = B.copy(); Bbad[0, 0] += 0.5
    rep = monitor(0, 20., 44, B_override=Bbad)
    assert rep["fallback"] is True and "eig_excursion" in rep["fired"]


# ---- 12.8 tracking ----------------------------------------------------------------------------------
def test_tracking_monotone_and_labels():
    tr = track_c_sweep(0, np.arange(20., 30.001, 2.0), 44)
    lam = tr["lam_traj"]; M = Mn(0, 30.)
    assert np.min(np.diff(lam[:, :M], axis=0)) > -1e-9            # monotone, no crossing
    assert all(s["assign_identity"] for s in tr["steps"])        # separated labels consistent

def test_tracking_subspace_continuity():
    j05 = leading_subspace_jump(0, np.arange(24., 26.001, 0.5), 44, Mn(0, 24.))
    j025 = leading_subspace_jump(0, np.arange(24., 26.001, 0.25), 44, Mn(0, 24.))
    assert 1.6 < j05 / j025 < 2.4                                 # projector jump ~ O(dc)
