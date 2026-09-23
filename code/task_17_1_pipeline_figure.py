"""
Task 17.1 — Conceptual pipeline figure.

Draws the paper's conceptual pipeline (continuous problem -> Fourier-Bessel bridge -> finite
radial prolate matrix -> eigendecomposition -> synthesis -> approximation of CPSWFs) as a
six-box serpentine diagram. NO new mathematics: every formula shown is transcribed verbatim
from existing task files:

  Box 1  T = B_R B_K B_R eigenproblem, CPSWFs, c = KR ......... task_16_2/16_3 (Sec. 3), task_4_*
  Box 2  FB basis phi_{n,k}, V_P ..... task_16_3 Sec. 4.1 (Tasks 5, 6.1)
  Box 3  matrix entries + structure ........................... task_16_3 Sec. 4.2-4.3 (Tasks 6.1, 6.5, 6.6; Thm 1)
  Box 4  eigenproblem, plateau-plunge-tail .................... stage_3_summary (Tasks 6.9, 7.3)
  Box 5  reconstruction formula (exact isometry) .............. stage_3_summary Task 8.1/8.2
  Box 6  convergence statements at Task 14.4 tiers ............ task_14_2 (Theorems 5-7)

Output: figures/task_17_1_pipeline.{png,pdf}
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
BOXES = [
    (  # 1
        "1. Continuous concentration problem  (§3)",
        [r"$\mathcal{T}\,\psi=\lambda\,\psi,\quad \mathcal{T}=B_R B_K B_R$",
         r"on $L^2([0,R],\,r\,dr)$, fixed angular order $n$",
         r"CPSWFs $\psi^{(m)}$;  eigenvalues $\lambda^n_m(c)$,  $c=KR$"],
    ),
    (  # 2
        "2. Fourier–Bessel bridge  (§4.1)",
        [r"$\varphi_{n,k}(r)=\frac{\sqrt{2}}{R}\,J_n(j_{n,k}r/R)\,/\,J_{n+1}(j_{n,k})$",
         r"$k=1,\dots,P$  (closure sizing $c=j_{n,P+1}$ optional)",
         r"$V_P=\mathrm{span}\{\varphi_{n,1},\dots,\varphi_{n,P}\}$"],
    ),
    (  # 3
        "3. Finite radial prolate matrix  (§4.2–4.3)",
        [r"$\left(B_K^{(B)}\right)_{mk}=2\,j_{n,m}j_{n,k}\int_0^c \frac{u\,J_n(u)^2\;du}{(u^2-j_{n,m}^2)(u^2-j_{n,k}^2)}$",
         r"real symmetric, PSD, eigenvalues in $[0,1]$",
         r"depends only on $(n,c,P)$"],
    ),
    (  # 4
        "4. Finite radial prolate sequences  (§4.4)",
        [r"$B_K^{(B)}\,c^{(m)}=\lambda_m^{(P)}\,c^{(m)}$",
         r"orthonormal eigenvectors $c^{(m)}$",
         r"$\lambda_0^{(P)}\geq\lambda_1^{(P)}\geq\cdots$ :  plateau – plunge – tail"],
    ),
    (  # 5
        "5. Finite radial prolate functions  (§4.5)",
        [r"$\psi_P^{(m)}(r)=\frac{\sqrt{2}}{R}\sum_{k=1}^{P}\frac{c_k^{(m)}}{J_{n+1}(j_{n,k})}\,J_n(j_{n,k}r/R)$",
         r"synthesis $\mathcal{S}_P$: exact isometry $\mathbb{R}^{P}\cong V_P$",
         r"doubly orthogonal family"],
    ),
    (  # 6
        "6. Approximation of CPSWFs  (§5, Thms 5–7)",
        [r"$0\leq\lambda_m-\lambda_m^{(P)}\leq 2\lambda_0\,\delta_P(E_{m+1})^2$  (Thm 6)",
         r"leading-mode subspace $\to$ CPSWFs in weighted $L^2$  (Thm 7a)",
         r"$\|\mathcal{T}_P-\mathcal{T}\|\to 0$  (Thm 5)"],
    ),
]

ARROWS = [  # (from-box, to-box, label)
    (0, 1, "FB expansion"),
    (1, 2, "Galerkin\ncompression"),
    (2, 3, "eigendecomposition"),
    (3, 4, r"synthesis $\mathcal{S}_P$"),
    (4, 5, r"$P\to\infty$"),
]

# ----------------------------------------------------------------------------- layout
BW, BH = 3.80, 1.62          # box width/height
GX = 0.95                    # horizontal gap between boxes
X0, Y_BOT = 0.25, 0.18
Y_TOP = Y_BOT + BH + 1.10

pos = {
    0: (X0 + 0 * (BW + GX), Y_TOP),
    1: (X0 + 1 * (BW + GX), Y_TOP),
    2: (X0 + 2 * (BW + GX), Y_TOP),
    3: (X0 + 2 * (BW + GX), Y_BOT),
    4: (X0 + 1 * (BW + GX), Y_BOT),
    5: (X0 + 0 * (BW + GX), Y_BOT),
}

FACE = ["#eaf1f8", "#eaf1f8", "#e7f0e7", "#e7f0e7", "#fdf3e3", "#fdf3e3"]
EDGE = "#4a4a4a"

fig, ax = plt.subplots(figsize=(14.4, 5.4))
ax.set_xlim(0, X0 + 3 * BW + 2 * GX + 0.25)
ax.set_ylim(-0.10, Y_TOP + BH + 0.15)
ax.axis("off")

for i, (title, lines) in enumerate(BOXES):
    x, y = pos[i]
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.06,rounding_size=0.10",
                                fc=FACE[i], ec=EDGE, lw=1.1, zorder=2))
    ax.text(x + BW / 2, y + BH - 0.13, title, ha="center", va="top",
            fontsize=10.0, fontweight="bold", zorder=3)
    for j, ln in enumerate(lines):
        ax.text(x + BW / 2, y + BH - 0.50 - 0.40 * j, ln, ha="center", va="top",
                fontsize=8.4, zorder=3)

def edge_pts(a, b):
    xa, ya = pos[a]; xb, yb = pos[b]
    if ya == yb:
        if xb > xa:   return (xa + BW + 0.07, ya + BH / 2), (xb - 0.07, yb + BH / 2)
        else:         return (xa - 0.07, ya + BH / 2), (xb + BW + 0.07, yb + BH / 2)
    else:
        return (xa + BW / 2, ya - 0.07), (xb + BW / 2, yb + BH + 0.07)

for (a, b, lab) in ARROWS:
    p, q = edge_pts(a, b)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                                 lw=1.3, color="#2f2f2f", zorder=1))
    if pos[a][1] == pos[b][1]:  # horizontal: label above the arrow, inside the gap
        ax.text((p[0] + q[0]) / 2, p[1] + 0.14, lab, ha="center", va="bottom", fontsize=7.8,
                style="italic", color="#2f2f2f", linespacing=1.1)
    else:                       # vertical: label to the left of the arrow (inside canvas)
        ax.text(p[0] - 0.14, (p[1] + q[1]) / 2, lab, ha="right", va="center", fontsize=7.8,
                style="italic", color="#2f2f2f")

# closing dashed arrow: box 6 -> box 1
x6, y6 = pos[5]; x1, y1 = pos[0]
ax.add_patch(FancyArrowPatch((x6 + BW * 0.10, y6 + BH + 0.07), (x1 + BW * 0.10, y1 - 0.07),
                             arrowstyle="-|>", mutation_scale=15, lw=1.3, ls="--",
                             color="#8a5a00", zorder=1))
ax.text(x6 + BW * 0.10 + 0.10, (y6 + BH + y1) / 2, "approximates",
        ha="left", va="center", fontsize=7.8, style="italic", color="#8a5a00")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGS, f"task_17_1_pipeline.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
print("wrote figures/task_17_1_pipeline.png and .pdf")
