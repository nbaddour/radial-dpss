"""
Task 13.7 — Validation campaign: the steady-state N-vs-error plot (analogous to the 2023 paper).

Renders the Task 13.5 N-convergence result as the campaign's headline figure: the eigenvalue error vs the
matrix size P = N-1, showing the TWO rates that define the method's steady state:
  * PLATEAU modes: super-EXPONENTIAL convergence to a floor (steady state) -- the leading modes are
    resolved to ~machine once P exceeds M_n(c) by a few,
  * PLUNGE-KNEE mode: ALGEBRAIC ~1/P.
Two panels, several c curves. Reference eigenvalues = high-Nq (=300) Nystrom (machine-accurate).

Outputs: figures/task_13_7_N_vs_error.{png,pdf} and the underlying data to data/campaign/.
"""
import os
import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize
with contextlib.redirect_stdout(io.StringIO()):
    from task_10_1_reference_solver import nystrom

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN = os.path.join(REPO, "data", "campaign")
FIGS = os.path.join(REPO, "figures")


def err_vs_P(n, c, m, P_list):
    lam_hi, *_ = nystrom(n, c, 300)
    out = []
    for P in P_list:
        if P <= m:
            continue
        jz = jn_zeros(n, P)
        B = assemble(n, c, P=P, G=32)
        r = solve(B, "evr")
        lam, _ = canonicalize(r["lam"], r["V"], n, jz)
        out.append((P, abs(lam[m] - lam_hi[m])))
    return np.array(out)


def build(n=0, c_values=(40, 80, 120)):
    data = {}
    for c in c_values:
        M = Mn(n, float(c))
        m_se = M - 6                                   # deep plateau mode (super-exp)
        m_knee = M                                     # plunge knee (~1/P)
        P_plateau = list(range(M - 5, M + 4))          # onset range (catch the super-exp drop)
        P_knee = [M + 2, M + 3, M + 5, M + 8, M + 12, M + 20, M + 32, M + 50, M + 80]
        data[c] = dict(M=M, m_se=m_se, m_knee=m_knee,
                       plateau=err_vs_P(n, float(c), m_se, P_plateau),
                       knee=err_vs_P(n, float(c), m_knee, P_knee))
    return data


def plot(n, data):
    os.makedirs(FIGS, exist_ok=True)
    colors = {40: "#1f77b4", 80: "#d62728", 120: "#2ca02c"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    # LEFT: plateau super-exponential convergence (semilog-y: straight/steepening line => super-exp)
    for c, d in data.items():
        P, e = d["plateau"][:, 0], np.clip(d["plateau"][:, 1], 1e-16, None)
        axL.semilogy(P - d["M"], e, "o-", color=colors.get(c, None), lw=1.6, ms=5,
                     label=f"c={c} (M={d['M']})")
    axL.set_xlabel(r"$N - M_n(c)$  (matrix size above the Shannon number)")
    axL.set_ylabel(r"eigenvalue error  $|\lambda_m^{(N)}-\lambda_m|$")
    axL.set_title(f"Plateau mode (m = M−6): super-exponential\n(n={n})")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend(fontsize=9, loc="upper right")

    # RIGHT: plunge-knee algebraic ~1/P (log-log: slope -1 reference line)
    for c, d in data.items():
        P, e = d["knee"][:, 0], np.clip(d["knee"][:, 1], 1e-16, None)
        axR.loglog(P, e, "s-", color=colors.get(c, None), lw=1.6, ms=5, label=f"c={c}")
    Pref = np.array([data[max(data)]["knee"][0, 0], data[max(data)]["knee"][-1, 0]], float)
    e0 = data[max(data)]["knee"][0, 1]
    axR.loglog(Pref, e0 * (Pref[0] / Pref), "k--", lw=1.2, alpha=0.7, label=r"$\propto 1/N$ (ref)")
    axR.set_xlabel(r"matrix size  $N$")
    axR.set_ylabel(r"eigenvalue error  $|\lambda_M^{(N)}-\lambda_M|$")
    axR.set_title(f"Plunge knee (m = M): algebraic ~ 1/N\n(n={n})")
    axR.grid(True, which="both", alpha=0.3)
    axR.legend(fontsize=9, loc="upper right")

    fig.suptitle("Radial DPSS — N-vs-error convergence (steady state): plateau super-exp vs plunge ~1/N",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    png = os.path.join(FIGS, "task_13_7_N_vs_error.png")
    pdf = os.path.join(FIGS, "task_13_7_N_vs_error.pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run(save=True):
    n = 0
    data = build(n)
    png, pdf = plot(n, data)
    print("=" * 92)
    print("TASK 13.7 -- steady-state N-vs-error plot (n=0)")
    print("=" * 92)
    for c, d in data.items():
        p_first, p_last = d["plateau"][0], d["plateau"][-1]
        k_first, k_last = d["knee"][0], d["knee"][-1]
        pdrop = np.log10(max(p_first[1], 1e-16)) - np.log10(max(p_last[1], 1e-16))
        kslope = np.polyfit(np.log(d["knee"][:, 0]), np.log(np.clip(d["knee"][:, 1], 1e-16, None)), 1)[0]
        print(f"  c={c:3d} M={d['M']:2d}: plateau {p_first[1]:.1e}(N={int(p_first[0])}) -> "
              f"{p_last[1]:.1e}(N={int(p_last[0])})  [{pdrop:.1f} decades, super-exp];  "
              f"knee ~ N^{kslope:.2f}")
    # sanity checks on the rendered data
    ok = True
    for c, d in data.items():
        pdrop = np.log10(max(d["plateau"][0, 1], 1e-16)) - np.log10(max(d["plateau"][-1, 1], 1e-16))
        kslope = np.polyfit(np.log(d["knee"][:, 0]), np.log(np.clip(d["knee"][:, 1], 1e-16, None)), 1)[0]
        ok = ok and (pdrop >= 4.0) and (-1.6 < kslope < -0.6)
    print(f"\n  figure written: figures/task_13_7_N_vs_error.png / .pdf")
    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(os.path.join(CAMPAIGN, "task_13_7_N_vs_error.npz"),
            **{f"c{c}_plateau_P": data[c]["plateau"][:, 0] for c in data},
            **{f"c{c}_plateau_err": data[c]["plateau"][:, 1] for c in data},
            **{f"c{c}_knee_P": data[c]["knee"][:, 0] for c in data},
            **{f"c{c}_knee_err": data[c]["knee"][:, 1] for c in data})
        print(f"  data written: data/campaign/task_13_7_N_vs_error.npz")
    print(f"\n  TASK 13.7 STEADY-STATE PLOT VERIFIED: {ok}")
    return data


if __name__ == "__main__":
    run()
