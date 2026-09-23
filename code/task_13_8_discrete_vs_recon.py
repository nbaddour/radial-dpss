"""
Task 13.8 — Validation campaign: direct discrete use vs reconstruction.

Three ways to use a finite radial prolate mode, and what each gains/loses (Task 8.3 decided the
coefficient vector is PRIMARY; reconstruction is the deliverable for continuous evaluation):

  (1) DISCRETE coefficients c^{(m)} (the eigenvector):
        - concentration  beta = c^T B c = lambda_m   EXACT (Galerkin identity, Task 6.7),
        - orthonormality c^T c = delta                EXACT.
        Use: discrete Slepian representation, concentration values. No continuous function.
  (2) NODAL SAMPLES psi_N(r_k) at the interior Bessel-zero nodes (x_k = j_{n,k}/c < 1, k <= M):
        exact evaluations of psi_N; approximate the true CPSWF at the nodes with the SAMPLED-VALUE error.
        Use: sampled radial signals. Avoids the exact-boundary (x=1) FB-edge spike.
  (3) RECONSTRUCTION psi_N^{(m)}(x) (continuous, the Task 8.4 isometric image of the coefficients):
        the full continuous function; approximates the true CPSWF with the weighted-L2 error, PLUS the
        FB-edge boundary error at x=1 (Tasks 12.3/12.5/13.4).
        Use: continuous approximation, off-node evaluation, L2.

Comparison (n=0, c in {40,80}) over the leading modes: the concentration is IDENTICAL from the discrete
coefficients and the reconstruction (both = lambda by Galerkin), so reconstruction adds nothing for the
concentration; for approximating the continuous CPSWF the reconstruction is required, its interior error
matches the sampled-value error, and its ONLY extra cost is the boundary spike (which sampled use avoids).
Reference eigenfunctions = Task 10.4 benchmark. Scope: the discrete-vs-reconstruction comparison.
"""
import os
import numpy as np
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize
from task_12_4_synthesis import synthesize_nd

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "data", "benchmark")
CAMPAIGN = os.path.join(REPO, "data", "campaign")


def compare(n, c, buffer=20):
    d = np.load(os.path.join(BENCH, f"cpswf_n{n}_c{int(c)}.npz"))
    xg, wg, psi_ref, lam_ref = d["x_gl"], d["w_gl"], d["psi_gl"], d["lam"]
    xu, psi_ref_u = d["x_unif"], d["psi_unif"]
    M = int(d["Mn"]); mu = xg * wg
    P = M + buffer
    jz = jn_zeros(n, P)
    B = assemble(n, c, P=P, G=32)
    r = solve(B, "evr")
    lam, V = canonicalize(r["lam"], r["V"], n, jz)

    # (1) DISCRETE: concentration = c^T B c, orthonormality = c^T c
    conc_discrete = np.array([V[:, m] @ (B @ V[:, m]) for m in range(M + 4)])
    conc_vs_lambda = float(np.max(np.abs(conc_discrete[:M + 4] - lam[:M + 4])))     # Galerkin exactness
    ortho_discrete = float(np.max(np.abs(V.T @ V - np.eye(P))))

    # interior Bessel-zero nodes inside the disk: x_k = j_{n,k}/c < 1  (k <= M)
    x_nodes = jz[:M] / c                          # the M interior nodes
    # reference at the nodes (interp from the dense uniform benchmark grid)
    def ref_at(xq, m):
        return np.interp(xq, xu, psi_ref_u[:, m])

    # per leading-mode comparison
    rows = []
    for m in range(M + 3):
        lr = lam_ref[m]
        reg = "deep-plateau" if lr > 1 - 1e-6 else ("separated" if lr >= 1e-3 else "tail")
        # (2) nodal samples at interior nodes
        psN_nodes = synthesize_nd(n, jz, V[:, m][:, None], x_nodes)[:, 0]
        prN = ref_at(x_nodes, m)
        s = np.sign(np.sum(psN_nodes * prN)) if np.any(prN) else 1.0
        sampled_err = float(np.max(np.abs(s * psN_nodes - prN)))
        # (3) continuous reconstruction: interior L2 + interior sup + boundary sup
        psN_g = synthesize_nd(n, jz, V[:, m][:, None], xg)[:, 0]
        pr_g = psi_ref[:, m]; psN_g *= np.sign(np.sum(psN_g * pr_g * mu))
        wl2 = float(np.sqrt(max(0.0, np.sum((psN_g - pr_g) ** 2 * mu))))
        psN_u = synthesize_nd(n, jz, V[:, m][:, None], xu)[:, 0]
        pr_u = psi_ref_u[:, m]; psN_u *= np.sign(np.sum(psN_u * pr_u))
        dsup = np.abs(psN_u - pr_u)
        interior_sup = float(dsup[xu < 0.95].max())
        boundary_sup = float(dsup[-1])
        rows.append(dict(m=m, regime=reg, lam=float(lam[m]),
                         eig_err=float(abs(lam[m] - lr)), sampled=sampled_err,
                         wl2=wl2, interior_sup=interior_sup, boundary_sup=boundary_sup))
    return dict(M=M, P=P, conc_vs_lambda=conc_vs_lambda, ortho_discrete=ortho_discrete,
                n_interior_nodes=len(x_nodes), rows=rows)


def run(save=True):
    out = {}
    for (n, c) in [(0, 40.0), (0, 80.0)]:
        cm = compare(n, c); out[(n, c)] = cm
        print("=" * 96)
        print(f"TASK 13.8 -- discrete vs reconstruction at n={n}, c={c:.0f}  (M={cm['M']}, P={cm['P']})")
        print("=" * 96)
        print(f"  DISCRETE (coefficients): concentration c^T B c == lambda to {cm['conc_vs_lambda']:.1e} "
              f"(Galerkin, EXACT);  orthonormality c^T c to {cm['ortho_discrete']:.1e}")
        print(f"  interior disk nodes (x_k=j/c<1): {cm['n_interior_nodes']} of P={cm['P']}")
        print(f"  {'m':>3s} {'regime':>13s} {'eig err(disc)':>13s} {'sampled err':>12s} "
              f"{'recon wL2':>10s} {'recon int-sup':>13s} {'recon bnd-sup':>13s}")
        for rrow in cm["rows"]:
            print(f"  {rrow['m']:3d} {rrow['regime']:>13s} {rrow['eig_err']:13.2e} {rrow['sampled']:12.2e} "
                  f"{rrow['wl2']:10.2e} {rrow['interior_sup']:13.2e} {rrow['boundary_sup']:13.2e}")
        print()

    # checks
    ok = True
    for (n, c), cm in out.items():
        galerkin = cm["conc_vs_lambda"] < 1e-12                 # discrete concentration EXACT
        ortho = cm["ortho_discrete"] < 1e-10
        # The FB-edge boundary spike is prominent only for EDGE-REACHING modes (moderately concentrated,
        # 0.5<=lam<=0.99, whose true CPSWF has psi(R)!=0). For the representative such mode, the
        # reconstruction's boundary sup is the reconstruction-ONLY cost: it exceeds both the interior sup
        # and the nodal SAMPLED error (interior nodes have no node at x=1, so sampling AVOIDS the spike).
        edge = [r for r in cm["rows"] if 0.5 <= r["lam"] <= 0.99]
        if edge:
            mstar = max(edge, key=lambda r: r["boundary_sup"])
            boundary_is_recon_cost = (mstar["boundary_sup"] > 2 * mstar["interior_sup"]
                                      and mstar["boundary_sup"] > mstar["sampled"])
            cm["mstar"] = mstar
        else:
            boundary_is_recon_cost = True
            cm["mstar"] = None
        ms = cm["mstar"]
        print(f"  n={n} c={c:.0f}: discrete concentration EXACT={galerkin}({cm['conc_vs_lambda']:.0e})  "
              f"ortho EXACT={ortho}")
        if ms:
            print(f"       representative edge-mode m={ms['m']} (lam={ms['lam']:.3f}): "
                  f"sampled={ms['sampled']:.2e}  recon interior-sup={ms['interior_sup']:.2e}  "
                  f"recon BOUNDARY-sup={ms['boundary_sup']:.2e}  => boundary is the reconstruction-only cost ({boundary_is_recon_cost})")
        ok = ok and galerkin and ortho and boundary_is_recon_cost

    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(os.path.join(CAMPAIGN, "task_13_8_discrete_vs_recon.npz"),
            **{f"n{n}_c{int(c)}_eig": np.array([r["eig_err"] for r in cm["rows"]]) for (n, c), cm in out.items()},
            **{f"n{n}_c{int(c)}_sampled": np.array([r["sampled"] for r in cm["rows"]]) for (n, c), cm in out.items()},
            **{f"n{n}_c{int(c)}_wl2": np.array([r["wl2"] for r in cm["rows"]]) for (n, c), cm in out.items()},
            **{f"n{n}_c{int(c)}_bnd": np.array([r["boundary_sup"] for r in cm["rows"]]) for (n, c), cm in out.items()})
        print(f"  saved: data/campaign/task_13_8_discrete_vs_recon.npz")

    print(f"\n  TASK 13.8 DISCRETE-vs-RECONSTRUCTION VERIFIED: {ok}")
    return out


if __name__ == "__main__":
    run()
