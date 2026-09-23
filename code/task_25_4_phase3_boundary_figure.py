"""Figure: boundary-augmented Ritz control on the fixed-c path (Track D).

Reads data/task_25_4_phase3/track_D_boundary.json and writes
figures/phase3_boundary.{png,pdf}.

Panel (a) shows, for each of the sixteen fixed-c cases, the median over the last four
dimensions of the augmented plunge-knee eigenvalue error at Fourier-Bessel dimension P,
divided by the unaugmented error at matched total dimension P+1. This is the conservative
comparison: the augmented space has P+1 dimensions, so it is charged for the one extra
function it was given.

Panel (b) shows the quotient of the same-dimension ratio and the matched ratio. That
quotient is |e_proj(P)| / |e_proj(P+1)|, the augmented error having cancelled, so panel (b)
measures what one additional ORDINARY basis function is worth on its own. Setting it beside
panel (a) is what decides whether the improvement is an artifact of counting one extra
basis function.

Only one series is plotted in panel (a). The same-dimension and matched-dimension ratios
differ by under two percent (panel b), which is under half a pixel on a seven-decade log
axis, so plotting both put two markers on top of each other and carried no information.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = json.load(open(os.path.join(REPO, "data", "task_25_4_phase3", "track_D_boundary.json")))
FIGS = os.path.join(REPO, "figures"); os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 140, "savefig.dpi": 140})

lab, caus, matc = [], [], []
for case in D["cases"]:
    a = case["last_four_assessment"]
    if a["causal_ratio_median_or_null"] is None: continue
    lab.append(rf"$({case['n']},{case['c']})$")
    caus.append(a["causal_ratio_median_or_null"])
    matc.append(a["matched_ratio_median_or_null"])
caus = np.array(caus); matc = np.array(matc); x = np.arange(len(lab))

fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained",
                       gridspec_kw={"width_ratios": [1.7, 1.0]})

# (a) the matched-dimension comparison. Single series, so no legend: the y label names it.
ax[0].semilogy(x, matc, "o", ms=7, mfc="none", mew=1.6, color="tab:blue")
ax[0].axhline(1.0, color="0.4", lw=1, ls="--")
ax[0].text(len(lab)-0.4, 1.25, "no improvement", ha="right", va="bottom", fontsize=8, color="0.35")
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=45, ha="right")
ax[0].set_xlabel(r"case $(n,c)$")
ax[0].set_ylabel(r"median $|e_{\mathrm{aug}}(P)|\,/\,|e_{\mathrm{proj}}(P{+}1)|$")
ax[0].set_title("(a) One boundary-aware function: three to seven orders")

# (b) the same quantity divided instead by e_proj(P); the augmented error cancels.
q = matc / caus
ax[1].plot(x, q, "s", ms=6, color="tab:purple")
ax[1].axhline(1.0, color="0.4", lw=1, ls="--")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, rotation=45, ha="right")
ax[1].set_xlabel(r"case $(n,c)$")
ax[1].set_ylabel(r"$|e_{\mathrm{proj}}(P)|\,/\,|e_{\mathrm{proj}}(P{+}1)|$")
ax[1].set_title("(b) One extra plain function: under two percent")
ax[1].set_ylim(0.9, max(1.35, q.max()*1.06))

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGS, f"phase3_boundary.{ext}"), bbox_inches="tight")
plt.close(fig)

print(f"cases: {len(lab)}")
print(f"same-dimension median ratio  : {caus.min():.3e} to {caus.max():.3e}, median {np.median(caus):.3e}")
print(f"matched-dimension median ratio: {matc.min():.3e} to {matc.max():.3e}, median {np.median(matc):.3e}")
print(f"quotient matched/same        : {q.min():.4f} to {q.max():.4f}")

# Consistency check against the Theta(P^-1) law: the quotient should be the P -> P+1 step,
# median of 1 + 1/P over each case's four dimensions.
pred = np.array([np.median(1.0 + 1.0/np.array(c["last_four_assessment"]["dimensions_P"], float))
                 for c in D["cases"] if c["last_four_assessment"]["causal_ratio_median_or_null"] is not None])
print(f"predicted 1+1/P              : {pred.min():.4f} to {pred.max():.4f}"
      f"  (max relative discrepancy {100*np.abs(q/pred-1).max():.2f}%)")
print("WROTE figures/phase3_boundary.{png,pdf}")
