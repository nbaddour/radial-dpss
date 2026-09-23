# CPSWF Benchmark Dataset (Task 10.4)

Ground-truth continuous radial CPSWF eigenpairs from the Task 10.1 reference solver (Solver A: Nyström
on 𝒯 = B_R B_K B_R, closed Lommel-form finite-Hankel kernel), validated in Tasks 10.2/10.3/10.4.

**Convention:** nondim x ∈ [0,1], measure x dx, R = 1, K = c = KR; eigenvalues λ descending.
Sign fixed so ψ^{(m)} > 0 at its max-|·| node (the band-limited dual is tied to the same sign).

## Files
`cpswf_n{n}_c{c}.npz` for n ∈ {0,1,2,4}, c ∈ {10,20,40,80}; plus `manifest.json`. One file per (n,c),
storing the leading/informative modes m = 0 … M_n(c)+4.

## Arrays in each .npz
| key | shape | meaning |
|-----|-------|---------|
| `lam` | (nmodes,) | concentration eigenvalues λ_m ∈ [0,1] |
| `x_gl`, `w_gl` | (Nq,) | Gauss–Legendre nodes/weights on [0,1] for ∫·x dx (exact quadrature) |
| `psi_gl` | (Nq, nmodes) | space-limited ψ^{(m)} at GL nodes; ‖·‖_{x dx}=1. **Use for precision.** |
| `x_unif`, `psi_unif` | (401,), (401,nmodes) | ψ^{(m)} on a uniform grid (Nyström-interp; plotting convenience) |
| `rho_gl` | (Nq,) | frequency nodes ρ = c·x_gl on [0,K]=[0,c] for the dual |
| `phitilde` | (Nq, nmodes) | nondim band-limited dual φ̃(ξ)=c·φ^{(m)}(cξ), ξ=ρ/c; ‖φ‖_{ρ dρ}=1 |
| `gap` | (nmodes,) | spectral gap to nearest neighbour (eigfn conditioning ≈ ε/gap) |
| `regime` | (nmodes,) | 0=plateau (clustered λ≈1, subspace-ambiguous), 1=separated (individually reliable), 2=deep-tail (λ<1e-6) |
| `reliab` | (nmodes,) | 1 iff regime==1 |
| `dual_norm`, `selfdual` | (nmodes,) | per-mode dual quality: ‖φ‖² and R↔K self-duality residual |
| `n,c,Nq,Nq_min_recommended,Mn,nmodes` | scalars | metadata |

## How to compare against your method (Task 11)
- **Separated** modes (regime==1): compare individual eigenfunctions/eigenvalues directly (validated to
  machine precision across independent solvers).
- **Plateau** modes (regime==0): λ_m = 1 (exact); individual eigenfunctions are NOT well-defined
  (degenerate) — compare the **subspace** they span (principal angles / projector), not individual modes.
- **Deep-tail** modes (regime==2): λ_m accurate; the dual/uniform-grid ψ are 1/λ-amplified — use `psi_gl`.

## Loader
```python
import numpy as np
d = np.load("cpswf_n0_c20.npz")
lam, x, w, psi = d["lam"], d["x_gl"], d["w_gl"], d["psi_gl"]
# inner product <f,g>_{x dx} ≈ sum_i f_i g_i x_i w_i   (orthonormal: psi_m·psi_m'·x·w summed = δ)
sep = d["regime"] == 1            # individually reliable modes
```
Rebuild from scratch: `code/task_10_4_build_dataset.py`. Validate: `code/task_10_4_validate_dataset.py`.
