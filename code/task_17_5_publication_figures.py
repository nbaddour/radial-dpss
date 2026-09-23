"""
Task 17.5 — Publication figures: mode plots, error curves, plunge-region plots, efficiency plots.

Every figure is rendered from SAVED campaign / benchmark / pilot arrays only. No quantity is
recomputed, refit for appearance, or smoothed; the only plotting-time transform is a per-mode sign
flip when overlaying two independently-built eigenvector families (a legitimate alignment, applied
only to separated modes and noted in the caption). Degenerate plateau modes are NEVER shown as
individual method-vs-reference agreement (they are subspace-ambiguous — data/benchmark/README.md,
draft §7.3); only spectrally separated modes (regime==1) are overlaid individually.

Provenance:
  Fig A mode gallery      <- data/benchmark/cpswf_n{0,1,2,4}_c40.npz ; data/pilot/sol_n0_c40_P40.npz
  Fig B error curves      <- data/campaign/task_13_5_N_convergence.npz
  Fig C spectrum/plunge   <- data/campaign/task_13_2_m_sweep.npz ; task_13_6_plunge_gamma.npz
  Fig D efficiency        <- data/campaign/task_13_10_boulsane_comparison.npz ; task_13_9_cost_profile.npz

Outputs: figures/task_17_5_{mode_gallery,error_curves,spectrum_plunge,efficiency}.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "data", "campaign")
BENCH = os.path.join(REPO, "data", "benchmark")
PILOT = os.path.join(REPO, "data", "pilot")
FIGS = os.path.join(REPO, "figures")
os.makedirs(FIGS, exist_ok=True)

# Colourblind-safe palette (Wong 2011)
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#7f7f7f")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 140, "savefig.dpi": 140,
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.6,
})


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def align_sign(ref, other):
    """Flip sign of `other` to match `ref` (two independently-built eigenvectors)."""
    return other * np.sign(np.dot(ref, other) or 1.0)


# ----------------------------------------------------------------------------- Fig A: mode plots
def fig_mode_gallery():
    b40 = np.load(os.path.join(BENCH, "cpswf_n0_c40.npz"))
    s40 = np.load(os.path.join(PILOT, "sol_n0_c40_P40.npz"))
    x = b40["x_unif"]
    regime = b40["regime"]
    lam = b40["lam"]

    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.0), layout="constrained")

    # (a) reference mode-shape gallery across m (n=0, c=40): increasing oscillation count
    gal = [0, 4, 8, 11]
    cols = [C["blue"], C["orange"], C["green"], C["red"]]
    for m, col in zip(gal, cols):
        ax[0].plot(x, b40["psi_unif"][:, m], color=col, label=fr"$m={m}$")
    ax[0].set_title(r"(a) Radial prolate functions  ($n{=}0,\,c{=}40$)")
    ax[0].set_xlabel(r"$x=r/R$"); ax[0].set_ylabel(r"$\psi^{(m)}$")
    ax[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[0].legend(ncol=2, loc="lower center")

    # (b) order-n effect: first mode for n=0,1,2,4 at c=40 (origin behaviour ~ r^n)
    for n, col in zip((0, 1, 2, 4), (C["blue"], C["orange"], C["green"], C["red"])):
        bb = np.load(os.path.join(BENCH, f"cpswf_n{n}_c40.npz"))
        f = bb["psi_unif"][:, 0]
        f = f * np.sign(f[len(f) // 4] or 1.0)  # consistent sign for display
        ax[1].plot(bb["x_unif"], f, color=col, label=fr"$n={n}$")
    ax[1].set_title(r"(b) Leading mode vs. order $n$  ($c{=}40$)")
    ax[1].set_xlabel(r"$x=r/R$"); ax[1].set_ylabel(r"$\psi^{(0)}$")
    ax[1].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[1].legend(loc="upper right")

    # (c) method vs reference, SEPARATED modes only (regime==1); boundary layer annotated
    sep = [m for m in range(len(lam)) if regime[m] == 1][:3]  # e.g. 9,10,11
    cols = [C["blue"], C["green"], C["red"]]
    for m, col in zip(sep, cols):
        ref = b40["psi_unif"][:, m]
        meth = align_sign(ref, s40["psi_unif"][:, m])
        ax[2].plot(x, ref, color=col, lw=2.2, alpha=0.55,
                   label=fr"ref. $m={m}$ ($\lambda{{=}}{lam[m]:.3f}$)")
        ax[2].plot(x, meth, color=col, lw=1.0, ls="--")
    ax[2].axvspan(0.95, 1.0, color=C["grey"], alpha=0.15)
    ax[2].text(0.905, ax[2].get_ylim()[1]*0.86, "FB edge\nlayer", fontsize=7.5,
               ha="right", color=C["grey"])
    ax[2].set_title(r"(c) Reconstruction (– –) vs. reference (—)")
    ax[2].set_xlabel(r"$x=r/R$"); ax[2].set_ylabel(r"$\psi^{(m)}$")
    ax[2].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[2].legend(loc="lower left", fontsize=7.5)
    save(fig, "task_17_5_mode_gallery")
    return dict(gallery_modes=gal, separated_modes=sep,
                note="plateau modes (regime 0) are subspace-ambiguous; not shown as individual overlays")


# ----------------------------------------------------------------------------- Fig B: error curves
def fig_error_curves():
    d = np.load(os.path.join(CAMP, "task_13_5_N_convergence.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # (a) two-rate: plateau (super-exp) + plunge knee (~1/N), c=40 and c=80
    for c, col in ((40, C["blue"]), (80, C["orange"])):
        P = d[f"n0_c{c}_P"].astype(float)
        pl = d[f"n0_c{c}_eig_plateau"]
        kn = d[f"n0_c{c}_eig_knee"]
        ax[0].semilogy(P, pl, "o-", color=col, label=fr"plateau, $c={c}$")
        mk = np.isfinite(kn)
        ax[0].semilogy(P[mk], kn[mk], "s--", color=col, alpha=0.7,
                       label=fr"plunge knee, $c={c}$")
    # 1/P guide
    Pg = np.array([15, 102], float)
    ax[0].semilogy(Pg, 1.9 * Pg**-1.0, ":", color="k", alpha=0.6, label=r"$\propto 1/P$ guide")
    ax[0].set_title("(a) Two-rate eigenvalue convergence")
    ax[0].set_xlabel(r"matrix size $P$"); ax[0].set_ylabel("eigenvalue error vs. Nyström ref.")
    ax[0].legend(fontsize=7.8)

    # (b) boundary layer: interior sup (x<0.95) shrinks; boundary-region sup is N-independent
    P = d["n0_c40_P"].astype(float)
    isup = d["n0_c40_interior_sup"]; bsup = d["n0_c40_boundary_sup"]
    mi = np.isfinite(isup)
    ax[1].semilogy(P[mi], isup[mi], "o-", color=C["green"], label=r"interior sup ($x<0.95$)")
    mb = np.isfinite(bsup)
    ax[1].semilogy(P[mb], bsup[mb], "s--", color=C["red"],
                   label="boundary-region sup ($P$-indep.)")
    ax[1].set_title(r"(b) Reconstruction sup-norm: interior vs. FB edge  ($n{=}0,c{=}40$)")
    ax[1].set_xlabel(r"matrix size $P$"); ax[1].set_ylabel(r"$\sup|\psi_P-\psi|$")
    ax[1].legend(fontsize=8)
    save(fig, "task_17_5_error_curves")
    return dict(c_curves=[40, 80], boundary_layer="interior shrinks, edge N-independent (§7.3)")


# ----------------------------------------------------------------------------- Fig C: spectrum/plunge
def fig_spectrum_plunge():
    d2 = np.load(os.path.join(CAMP, "task_13_2_m_sweep.npz"))
    d6 = np.load(os.path.join(CAMP, "task_13_6_plunge_gamma.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # (a) spectrum lambda_m vs m: plateau -> plunge -> tail, with M_n(c) = c/pi - n/2 marked
    for c, col in ((40, C["blue"]), (80, C["orange"])):
        m = d2[f"c{c}_m"]; lam = d2[f"c{c}_lam"]
        ax[0].plot(m, lam, "o-", color=col, ms=4, label=fr"$c={c}$")
        Mn = c / np.pi  # n=0
        ax[0].axvline(Mn, color=col, ls=":", alpha=0.7)
    ax[0].text(40/np.pi, 0.5, r"$M_n(c)=c/\pi$", rotation=90, va="center",
               ha="right", fontsize=8, color=C["blue"])
    ax[0].set_title(r"(a) Concentration spectrum $\lambda_m$: plateau–plunge–tail  ($n{=}0$)")
    ax[0].set_xlabel(r"mode index $m$"); ax[0].set_ylabel(r"$\lambda_m$")
    ax[0].set_ylim(-0.03, 1.05)
    ax[0].legend()

    # (b) plunge width V = sum lambda(1-lambda) vs c for n=0,1,2,4 (logarithmically thin)
    c = d6["c"]
    for n, col in zip((0, 1, 2, 4), (C["blue"], C["orange"], C["green"], C["red"])):
        ax[1].semilogx(c, d6[f"n{n}_V"], "o-", color=col, ms=4, label=fr"$n={n}$")
    ax[1].set_title(r"(b) Plunge width $V=\sum_m\lambda_m(1-\lambda_m)$ vs. $c$")
    ax[1].set_xlabel(r"$c=KR$ (log scale)"); ax[1].set_ylabel(r"$V$")
    ax[1].legend(ncol=2, fontsize=8)
    save(fig, "task_17_5_spectrum_plunge")
    return dict(spectra_c=[40, 80], plunge_orders=[0, 1, 2, 4])


# ----------------------------------------------------------------------------- Fig D: efficiency
def fig_efficiency():
    d10 = np.load(os.path.join(CAMP, "task_13_10_boulsane_comparison.npz"))
    d9 = np.load(os.path.join(CAMP, "task_13_9_cost_profile.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # (a) accuracy at matched budget: this method vs Boulsane, deep-plateau error vs P (n0,c40)
    P = d10["conv_P"].astype(float)
    ep = d10["conv_err_p"]; eb = d10["conv_err_b"]
    ax[0].loglog(P, ep, "o-", color=C["blue"], label="this paper (operator-image)")
    ax[0].loglog(P, eb, "s--", color=C["red"], label=r"Boulsane $\rho^{n}_{N,\omega}$")
    ratio = np.round(np.mean(eb / ep), 2)
    ax[0].text(0.05, 0.05, fr"stabilized ratio $\approx{ratio:.1f}\times$"
               "\n(empirical; cause not derived)",
               transform=ax[0].transAxes, fontsize=8,
               bbox=dict(boxstyle="round", fc="white", ec=C["grey"], alpha=0.85))
    ax[0].set_title(r"(a) Accuracy at matched budget  ($n{=}0,c{=}40$)")
    ax[0].set_xlabel(r"matrix size $P$"); ax[0].set_ylabel("deep-plateau error vs. shared ref.")
    ax[0].legend(fontsize=8)

    # (b) compute cost: assemble ~ P^~1.8, eigensolve ~ P^~2.7 (fitted slopes from data)
    P = d9["c0_P"].astype(float)
    ta = d9["c0_t_assemble"]; te = d9["c0_t_eigensolve"]
    sa = float(d9["slope_assemble"]); se = float(d9["slope_eigensolve"])
    ax[1].loglog(P, ta, "o-", color=C["blue"], label=fr"assemble  (slope {sa:.1f})")
    ax[1].loglog(P, te, "s-", color=C["orange"], label=fr"eigensolve (slope {se:.1f})")
    ax[1].set_title("(b) Cost profile vs. matrix size")
    ax[1].set_xlabel(r"matrix size $P$ (log)"); ax[1].set_ylabel("wall time (s)")
    ax[1].legend(fontsize=8)
    save(fig, "task_17_5_efficiency")
    return dict(matched_ratio=float(ratio), cost_slopes=[sa, se])


if __name__ == "__main__":
    info = {}
    info["A_mode_gallery"] = fig_mode_gallery()
    info["B_error_curves"] = fig_error_curves()
    info["C_spectrum_plunge"] = fig_spectrum_plunge()
    info["D_efficiency"] = fig_efficiency()
    print("WROTE 4 figures to figures/task_17_5_*.{png,pdf}")
    for k, v in info.items():
        print(f"  {k}: {v}")
