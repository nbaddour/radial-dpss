# Radial DPSS — computational pipeline

Reproducible software for the **finite radial prolate framework**: assembling the finite
radial prolate matrix for angular order `n` and space-bandwidth product `c = KR`, solving
its eigenproblem, and reconstructing the continuous approximants of the circular/radial
prolate spheroidal wave functions (CPSWFs).

`K` is the Hankel bandwidth, `R` the disc radius, `n` the angular (Bessel) order, and `P`
the Fourier-Bessel trial dimension.

## Pipeline modules (Task 12)

| Module | Provides |
|--------|----------|
| `task_12_1_assemble_matrix.py` | `assemble(n, c, P/N)`, the matrix from its closed-form entries by panel-Gauss quadrature; `Mn(n, c)`, the Shannon number |
| `task_12_2_eigensolver.py` | `solve(B)`, the symmetric eigensolve with diagnostics; `degenerate_clusters`, `subspace_projector` |
| `task_12_3_canonicalize.py` | `canonicalize(lam, V, n, jz)`, canonical order, normalization and sign; `interior_zero_count`, `faithful_plateau_modes` |
| `task_12_4_synthesis.py` | `synthesize_nd`/`synthesize_dim` (exact isometry), `sampling_matrix_nd`/`interpolate_nd` (exact interpolation), `error_sources` (the four-source split) |
| `task_12_5_origin.py` | near-`r=0` stabilization: `origin_coeff`, `synthesize_stable`, `normalized_profile` |
| `task_12_6_worthogonality.py` | `assess(n, N)`, the weighted-orthogonality defect; `lowdin_correct(A, W)` |
| `task_12_7_monitor.py` | `monitor(n, c, P)`, the conditioning report and the Nyström `fallback` trigger |
| `task_12_8_tracking.py` | `track_c_sweep`, coherent eigenvalue trajectories through a sweep in `c` |

The reference solver and the ground-truth dataset builder are the `task_10_*` group. The
Gate-C pilot is `task_11_*`. The validation campaign is `task_13_*`. The publication
figures are `task_17_*`.

## Phase 3 experiments

Every Phase 3 measurement is taken against a frozen reference spectrum, so these run in
order. Outputs land in `../data/task_25_4_phase3/`.

| Script | Writes |
|---|---|
| `task_25_4_phase3_freeze_manifest.py` | `reference_mode_manifest.json` and its `.sha256` |
| `task_25_4_phase3_gate0_interval.py` | `gate0_interval_stack.json` |
| `task_25_4_phase3_certification.py` | `track_A_certification.json`, `track_A_cases/` |
| `task_25_4_phase3_certification_audit.py` | `track_A_reference_audit.json` |
| `task_25_4_phase3_structure.py` | `track_B_structure.json` |
| `task_25_4_phase3_oracle.py` | `track_C_oracle.json` |
| `task_25_4_phase3_mechanism.py` | `track_C_fixed_offsets.json` |
| `task_25_4_phase3_boundary.py` | `track_D_boundary.json` |
| `task_25_4_phase3_timing.py` | `track_E_timing.json` |
| `task_25_4_phase3.py` | `manifest.json`, the six CSVs, `reference.npz`, `failure_log.json`, `figures/`, `artifact_inventory.json` |
| `task_25_4_phase3_boundary_figure.py` | `../figures/phase3_boundary.png` and `.pdf` |

`task_25_4_phase3_interval_ritz.py` is the interval-arithmetic Ritz module the
certification track calls. `task_25_4_phase3.py` recomputes nothing: it verifies and
flattens the seven track files and plots only from saved data.

`task_27_4_augmented_conditioning.py` writes `task_27_4_conditioning.json` here in `code/`,
which is the conditioning table of the augmented matrix.

The short `t1*`, `t2*`, `a*_check`, `c*_check`, `eps_*`, `dht_check` and `ref_check`
scripts are single-claim verifications written while the manuscript was checked. Each one
prints the quantity named in the passage it belongs to.

## Setup

```bash
pip install -r requirements.txt                    # numpy, scipy, mpmath, pytest, matplotlib
pip install -r requirements_phase3_certified.txt   # adds python-flint, for the certification track
```

## Run the tests

```bash
cd code
python -m pytest tests -q
```

The suite (`tests/test_pipeline.py`) asserts the core numerical invariant of each module:
symmetry and positive semidefiniteness, eigensolver residual and determinism, the synthesis
isometry, exact interpolation, the origin limit, the weighted-orthogonality correction, the
monitor trigger, and tracking continuity. It runs in under a second. Tests that need
`data/pilot` or `data/benchmark` skip cleanly when those are absent.

## Run a module's own verification

Every pipeline module has a `verify()` and a `__main__` that prints its full diagnostic
table:

```bash
cd code
python task_12_2_eigensolver.py        # or any task_12_*.py
```

Data paths resolve relative to the repository root, which each module computes from its own
location. Override with the `RDPSS_REPO` environment variable when running from outside the
tree.

## Determinism

`scipy.linalg.eigh` is deterministic, and the eigensolver re-run test asserts bit-identical
eigenvalues. The only randomness is the interpolation round-trip, which uses a fixed seed
(`np.random.default_rng(0)`). `data/benchmark` is the trusted CPSWF ground truth and ships
with the code; `task_10_4_build_dataset.py` rebuilds it from scratch.
