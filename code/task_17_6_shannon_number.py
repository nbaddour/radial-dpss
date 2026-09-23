"""
Task 17.6 — Shannon-number figure: effective dimension N_eff vs. c for each tested order n.

Renders the headline spectral law of the campaign (Task 13.6, draft §7.5):
    N_eff(n, c) = trace(B) = sum_m lambda_m  ~  (1/pi) c - n/2,
with slope gamma_n = 1/pi (n-independent; measured, not assumed — the theoretical value is
Boulsane 2021, confirmed here) and intercept b_n ~ -n/2.

All plotted N_eff values are the SAVED measured pipeline traces from task_13_6_plunge_gamma.npz.
No value is recomputed or smoothed. The straight lines are (i) the theoretical law (1/pi)c - n/2
and (ii) the least-squares fit of the saved points; both are labeled as such. Panel (b) shows the
saved fitted slopes gamma_n against 1/pi and the fitted intercepts b_n against -n/2.

Output: figures/task_17_6_shannon_number.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "data", "campaign")
FIGS = os.path.join(REPO, "figures")
os.makedirs(FIGS, exist_ok=True)

C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", grey="#7f7f7f")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "figure.dpi": 140, "savefig.dpi": 140, "axes.grid": True, "grid.alpha": 0.25,
    "lines.linewidth": 1.6,
})


def main():
    d = np.load(os.path.join(CAMP, "task_13_6_plunge_gamma.npz"))
    c = d["c"]
    ns = [0, 1, 2, 4]
    cols = {0: C["blue"], 1: C["orange"], 2: C["green"], 4: C["red"]}
    inv_pi = 1.0 / np.pi

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.1), layout="constrained",
                           gridspec_kw={"width_ratios": [1.55, 1.0]})

    # --- (a) N_eff vs c, one series per n, with the theoretical law (1/pi)c - n/2 overlaid
    gammas, intercepts = [], []
    cc = np.linspace(0, c.max() * 1.03, 100)
    for n in ns:
        y = d[f"n{n}_Neff"]
        g, b = np.polyfit(c, y, 1)          # least-squares fit of the SAVED points
        gammas.append(g); intercepts.append(b)
        ax[0].plot(c, y, "o", color=cols[n], ms=6, label=fr"$n={n}$ (measured)")
        ax[0].plot(cc, inv_pi * cc - n / 2, "--", color=cols[n], lw=1.3, alpha=0.9)
    ax[0].plot([], [], "--", color=C["grey"],
              label=r"asymptotic $\;N_{\mathrm{eff}}\sim\frac{1}{\pi}c-\frac{n}{2}+O(1/c)$")
    ax[0].set_title(r"(a) Effective dimension $N_{\mathrm{eff}}(n,c)=\sum_m\lambda_m$")
    ax[0].set_xlabel(r"$c=KR$"); ax[0].set_ylabel(r"$N_{\mathrm{eff}}$")
    ax[0].set_xlim(0, c.max() * 1.03); ax[0].set_ylim(0, None)
    ax[0].legend(loc="upper left")

    # --- (b) fitted slope gamma_n vs 1/pi, and fitted intercept b_n vs -n/2
    gammas = np.array(gammas); intercepts = np.array(intercepts)
    saved_g = d["gammas"]                    # slopes as saved by Task 13.6
    axb = ax[1]
    axb.plot(ns, saved_g, "D", color=C["blue"], ms=7, label=r"measured $\gamma_n$")
    axb.axhline(inv_pi, color=C["blue"], ls=":", lw=1.4, label=r"$1/\pi$ (theory)")
    axb.set_ylabel(r"slope $\gamma_n$", color=C["blue"])
    axb.tick_params(axis="y", labelcolor=C["blue"])
    axb.set_ylim(inv_pi - 0.01, inv_pi + 0.004)
    axb.set_xlabel(r"order $n$"); axb.set_xticks(ns)
    axb.set_title(r"(b) Fitted slope and intercept vs. $n$")

    axr = axb.twinx()
    axr.plot(ns, intercepts, "s", color=C["red"], ms=7, label=r"measured $b_n$")
    axr.plot(ns, [-n / 2 for n in ns], "x--", color=C["red"], lw=1.2, alpha=0.8,
             label=r"$-n/2$ (theory)")
    axr.set_ylabel(r"intercept $b_n$", color=C["red"])
    axr.tick_params(axis="y", labelcolor=C["red"])
    axr.grid(False)

    # merged legend
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7.8)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"task_17_6_shannon_number.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # diagnostics
    print("c =", c)
    for n, g, b, sg in zip(ns, gammas, intercepts, saved_g):
        print(f"  n={n}: fit slope={g:.5f}  saved gamma={sg:.5f}  1/pi={inv_pi:.5f} | "
              f"intercept b_n={b:.4f}  -n/2={-n/2}")
    print("1D reference slope 2/pi =", 2 / np.pi, " (disc value is exactly half)")
    print("WROTE figures/task_17_6_shannon_number.{png,pdf}")


if __name__ == "__main__":
    main()
