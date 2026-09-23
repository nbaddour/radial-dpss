"""
Task 12.6 — Weighted-orthogonality correction (when needed).

The finite radial prolate objects carry THREE orthogonality layers (Task 7.5):
  (1) COEFFICIENT  : c^{(m)T} c^{(m')} = delta,  c^{(m)T} B c^{(m')} = lam_m delta   -- EXACT at finite N
  (3) FUNCTION     : <psi_N^{(m)}, psi_N^{(m')}>_{r dr} = delta, band form = lam_m delta -- EXACT
  (2) NODAL SAMPLE : (a^{(m)})^T W a^{(m')} = delta + O(eps_N),  a^{(m)}=Phi c^{(m)}   -- EXACT only to eps_N
with W = diag(2/(K^2 J_{n+1}^2(j_{n,k}))) the Baddour quadrature weights and eps_N the Baddour Eq.37
residual (~1e-7 for N>30, -> 0 as N->inf). Layer 2 is the ONLY one with a defect (C2, Task 7.5): the
discrete weighted (W) inner product is a QUADRATURE approximation to the L^2(r dr) product.

"When needed": the pipeline's PRIMARY objects are the coefficients (layer 1) and the reconstructed
functions (layer 3) -- BOTH EXACT -- so NO correction is applied by default. The correction is provided
for any downstream use of the discrete nodal W-inner-product (nodal Parseval energy, sample-domain
concentration), where exact discrete W-orthonormality is wanted:

  LOWDIN symmetric W-orthogonalization:  given nodal vectors A (columns a^{(m)}) with Gram G = A^T W A,
  replace A -> A_corr = A G^{-1/2}.  Then A_corr^T W A_corr = I exactly, with ||A_corr - A|| = O(eps_N)
  (a minimal, symmetric correction that does not re-order or re-sign the modes).

Nodal W-orthonormality is a CLOSURE-grid (F2, c=j_{n,N}) construction (the Baddour DHT grid); this module
builds it in the closure regime where it is defined. The exact layers (1,3) hold in BOTH F1 and F2.

Scope: assessment + correction of the weighted orthogonality only. Conditioning monitor is Task 12.7.
"""
import os
import numpy as np
from scipy.linalg import sqrtm
from scipy.special import jv, jn_zeros
from task_12_1_assemble_matrix import assemble
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize

# REPO root: parent of this code/ directory (override with env RDPSS_REPO if running out of tree).
REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def closure_config(n, N, K=1.0):
    """Closure (F2) grid for order n, matrix size P=N-1: nodes j_{n,1..N-1}, c=j_{n,N}=KR."""
    jz = jn_zeros(n, N)
    j = jz[:N - 1]
    jN = jz[N - 1]
    R = jN / K
    c = K * R                      # = jN (closure)
    return j, jN, R, K, c


def nodal_weight(n, j, K):
    """W diagonal: w_k = 2/(K^2 J_{n+1}^2(j_{n,k}))  (Task 5.8)."""
    return 2.0 / (K ** 2 * jv(n + 1, j) ** 2)


def nodal_matrix(n, j, jN, R):
    """Phi[k,l] = phi_{n,l}(r_k) = (sqrt2/R) J_n(j_k j_l / jN) / J_{n+1}(j_l)  (Task 7.5)."""
    return (np.sqrt(2.0) / R) * jv(n, np.outer(j, j) / jN) / jv(n + 1, j)[None, :]


def lowdin_correct(A, W):
    """Symmetric W-orthogonalization: A -> A G^{-1/2}, G = A^T W A. Exactly W-orthonormal output."""
    G = A.T @ (W[:, None] * A)
    Ginvhalf = np.linalg.inv(sqrtm(G).real)
    return A @ Ginvhalf


def assess(n, N):
    """Return the three-layer orthogonality defects and the corrected nodal W-Gram, closure grid."""
    j, jN, R, K, c = closure_config(n, N)
    P = N - 1
    B = assemble(n, c, P=P, G=32)
    r = solve(B, "evr")
    lam, C = canonicalize(r["lam"], r["V"], n, jn_zeros(n, P))

    # (1) coefficient layer (exact)
    coeff_I = float(np.max(np.abs(C.T @ C - np.eye(P))))
    CBC = C.T @ B @ C
    coeff_band = float(np.max(np.abs(CBC - np.diag(np.diag(CBC)))))

    # (2) nodal sample layer (defect eps_N)
    w = nodal_weight(n, j, K)
    Phi = nodal_matrix(n, j, jN, R)
    A = Phi @ C                                  # nodal vectors a^{(m)}
    GW = A.T @ (w[:, None] * A)
    nodal_defect = float(np.max(np.abs(GW - np.eye(P))))

    # correction (Lowdin) -> exact W-orthonormality
    Acorr = lowdin_correct(A, w)
    GWc = Acorr.T @ (w[:, None] * Acorr)
    nodal_corr_defect = float(np.max(np.abs(GWc - np.eye(P))))
    corr_size = float(np.max(np.abs(Acorr - A)))

    return dict(P=P, coeff_I=coeff_I, coeff_band=coeff_band,
                nodal_defect=nodal_defect, nodal_corr_defect=nodal_corr_defect, corr_size=corr_size)


# ---- verification ------------------------------------------------------------------------------------
def verify():
    print("=" * 100)
    print("TASK 12.6 -- weighted-orthogonality: exact layers, nodal eps_N defect + Lowdin correction")
    print("=" * 100)

    print("\n[A] Exact layers (coefficient) hold to machine; nodal W-defect ~ eps_N; correction -> exact")
    print("  n   N   P |  coeff C^TC-I  coeff band  | nodal W-defect  ->  corrected  (corr size)")
    worst_coeff = 0.0
    worst_corr = 0.0
    decay = {}
    for n in [0, 1, 2, 4]:
        row_defects = []
        for N in [8, 16, 24, 32, 40]:
            a = assess(n, N)
            worst_coeff = max(worst_coeff, a["coeff_I"], a["coeff_band"])
            worst_corr = max(worst_corr, a["nodal_corr_defect"])
            row_defects.append((N, a["nodal_defect"]))
            print(f"  {n:2d} {N:3d} {a['P']:3d} |  {a['coeff_I']:.2e}    {a['coeff_band']:.2e}  |"
                  f"  {a['nodal_defect']:.2e}   ->  {a['nodal_corr_defect']:.2e}   ({a['corr_size']:.1e})")
        decay[n] = row_defects
        print()

    print("[B] Nodal W-defect decays with N (Baddour eps_N -> 0):")
    for n in [0, 1, 2, 4]:
        seq = "  ".join(f"N{N}:{d:.1e}" for N, d in decay[n])
        print(f"  n={n}: {seq}")

    ok = (worst_coeff < 1e-12 and worst_corr < 1e-10)
    print("\n" + "=" * 100)
    print(f"  worst exact-layer defect (coefficient ortho + band) : {worst_coeff:.2e}")
    print(f"  worst nodal W-defect AFTER Lowdin correction        : {worst_corr:.2e}  (was O(eps_N) before)")
    print(f"\n  TASK 12.6 WEIGHTED-ORTHOGONALITY VERIFIED: {ok}")


if __name__ == "__main__":
    verify()
