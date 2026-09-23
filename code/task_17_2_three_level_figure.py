"""
Task 17.2 — Three-level operator diagram.

Draws the three-level relationship (Task 6.11, as corrected 2026-07-08): continuous CPSWF
problem <-> infinite discrete operator (EXACT unitary equivalence, Task 6.7 Theorem 6.7.1)
-> finite radial prolate matrix (Galerkin truncation, the sole approximation step). NO new
mathematics: every statement is transcribed from existing task files:

  Level 1   ....................... task_6_11 Sec. 1 (citing 4.7, 4.3, 4.11)
  Arrow 1 (EXACT)  ................ task_6_7 Theorem 6.7.1 / task_6_11 Sec. 3 Arrow 1
  Level 2   ....................... task_6_11 Sec. 1 Level 2 (citing 6.3)
  Arrow 2 (GALERKIN) .............. task_6_11 Sec. 1/3 Arrow 2 with the interlacing direction
                                    as CORRECTED 2026-07-08 (lam^(N) <= lam, increasing;
                                    Tasks 9.2, 9.3 Thm 9.3.1, 14.2 Thm 6)
  Level 3   ....................... task_6_11 Sec. 1 Level 3; naming per Task 7.4 (locked)
  Return arrow .................... Theorems 5-7 (task_14_2)

Visual style matches code/task_17_1_pipeline_figure.py (same figure family).

Output: figures/task_17_2_three_level.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(REPO, "figures")
os.makedirs(FIGS, exist_ok=True)

# ----------------------------------------------------------------------------- content
LEVELS = [
    (  # Level 1
        "Level 1 — Continuous:  the CPSWF problem",
        [r"space $L^2([0,R],\,r\,dr)$;   $\mathcal{T}=B_R B_K B_R$  compact, self-adjoint, positive",
         r"$\mathcal{T}\,\psi^{(m)}=\lambda_m\,\psi^{(m)}$;   CPSWFs $\psi^{(m)}$;   $\lambda_0\geq\lambda_1\geq\cdots>0$"],
        "#eaf1f8",
    ),
    (  # Level 2
        "Level 2 — Infinite discrete:  the operator in Fourier–Bessel coordinates",
        [r"space $\ell^2(\mathbb{N})$;   $B_K^{(B,\infty)}$  bounded, self-adjoint, PSD;   same eigenvalues $\lambda_m$ (exact)",
         r"$B_K^{(B,\infty)}\,v^{(m)}=\lambda_m\,v^{(m)}$;   $v^{(m)}=\Omega\,\psi^{(m)}$ = FB coefficient sequence of the CPSWF"],
        "#eaf1f8",
    ),
    (  # Level 3
        "Level 3 — Finite:  the finite radial prolate matrix",
        [r"space $\mathbb{R}^{P}$;   $B_K^{(B)}$  real symmetric, PSD, eigenvalues in $[0,1]$",
         r"$B_K^{(B)}\,c^{(m)}=\lambda_m^{(P)}\,c^{(m)}$;   finite radial prolate sequences $c^{(m)}$",
         r"synthesis:  $\psi_P^{(m)}=\sum_k c_k^{(m)}\varphi_{n,k}$   (finite radial prolate functions)"],
        "#e7f0e7",
    ),
]

ARROW_1 = [  # Level 1 <-> Level 2 : EXACT
    r"$\bf{EXACT}$ — unitary equivalence, no approximation",
    r"$\Omega f=\left(\langle f,\varphi_{n,k}\rangle\right)_{k\geq 1}$:  isometric isomorphism $L^2\to\ell^2$",
    r"$\Omega\,\mathcal{T}\,\Omega^{-1}=B_K^{(B,\infty)}$;   identical spectrum",
]
ARROW_2 = [  # Level 2 -> Level 3 : GALERKIN (sole approximation step)
    r"$\bf{GALERKIN}$ — truncation: the sole approximation step",
    r"$\left(B_K^{(B)}\right)_{mk}=\left(B_K^{(B,\infty)}\right)_{mk}$,  $m,k\leq P$   (upper-left block)",
    r"$\lambda_m^{(P)}\leq\lambda_m$ (interlacing);   $\lambda_m^{(P)}\nearrow\lambda_m$ as $P$ grows",
]

# ----------------------------------------------------------------------------- layout
BW, BH3 = 8.6, 1.42          # box width; Level-3 box is taller (3 lines)
BH = 1.12                    # Level-1/2 box height (2 lines)
GAP = 1.30                   # vertical gap between boxes (arrow zone)
X0 = 0.30
Y3 = 0.18                    # bottom of Level-3 box
Y2 = Y3 + BH3 + GAP
Y1 = Y2 + BH + GAP

EDGE = "#4a4a4a"
AX_W = X0 + BW + 2.15        # extra right margin for the return arrow

fig, ax = plt.subplots(figsize=(11.4, 7.0))
ax.set_xlim(0, AX_W)
ax.set_ylim(-0.05, Y1 + BH + 0.15)
ax.axis("off")

def box(x, y, w, h, fc, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                fc=fc, ec=EDGE, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h - 0.11, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", zorder=3)
    for j, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.46 - 0.34 * j, ln, ha="center", va="top",
                fontsize=8.8, zorder=3)

box(X0, Y1, BW, BH,  LEVELS[0][2], LEVELS[0][0], LEVELS[0][1])
box(X0, Y2, BW, BH,  LEVELS[1][2], LEVELS[1][0], LEVELS[1][1])
box(X0, Y3, BW, BH3, LEVELS[2][2], LEVELS[2][0], LEVELS[2][1])

# arrow 1: double-headed (exact equivalence), left third of the gap
xa = X0 + BW * 0.16
ax.add_patch(FancyArrowPatch((xa, Y1 - 0.07), (xa, Y2 + BH + 0.07),
                             arrowstyle="<|-|>", mutation_scale=15, lw=1.4,
                             color="#2f2f2f", zorder=1))
for j, ln in enumerate(ARROW_1):
    ax.text(xa + 0.25, (Y1 + Y2 + BH) / 2 + 0.30 - 0.30 * j, ln,
            ha="left", va="center", fontsize=8.6, color="#2f2f2f")

# arrow 2: one-way down (Galerkin truncation)
ax.add_patch(FancyArrowPatch((xa, Y2 - 0.07), (xa, Y3 + BH3 + 0.07),
                             arrowstyle="-|>", mutation_scale=15, lw=1.4,
                             color="#2f2f2f", zorder=1))
for j, ln in enumerate(ARROW_2):
    ax.text(xa + 0.25, (Y2 + Y3 + BH3) / 2 + 0.30 - 0.30 * j, ln,
            ha="left", va="center", fontsize=8.6, color="#2f2f2f")

# return arrow (right side, dashed): Level 3 -> Level 1, "approximates (Thms 5-7)"
xr = X0 + BW + 0.45
ax.add_patch(FancyArrowPatch((X0 + BW + 0.07, Y3 + BH3 * 0.55), (xr, Y3 + BH3 * 0.55),
                             arrowstyle="-", lw=1.3, ls="--", color="#8a5a00", zorder=1))
ax.add_patch(FancyArrowPatch((xr, Y3 + BH3 * 0.55), (xr, Y1 + BH * 0.5),
                             arrowstyle="-", lw=1.3, ls="--", color="#8a5a00", zorder=1))
ax.add_patch(FancyArrowPatch((xr, Y1 + BH * 0.5), (X0 + BW + 0.07, Y1 + BH * 0.5),
                             arrowstyle="-|>", mutation_scale=15, lw=1.3, ls="--",
                             color="#8a5a00", zorder=1))
ax.text(xr + 0.16, (Y3 + Y1 + BH) / 2,
        "$\\psi_P^{(m)}$ approximates $\\psi^{(m)}$\n(Theorems 5–7)",
        ha="left", va="center", fontsize=8.6, style="italic", color="#8a5a00",
        rotation=90, linespacing=1.3)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGS, f"task_17_2_three_level.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
print("wrote figures/task_17_2_three_level.png and .pdf")
