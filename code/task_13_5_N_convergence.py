"""
Task 13.5 — Validation campaign: convergence with respect to N (matrix size P = N-1).

Fix (n, c) and increase P. The locked theory (Tasks 7.2, 9.3, 11.2) predicts two DISTINCT rates:
  * PLATEAU modes (m < M_n(c) - plunge buffer): SUPER-EXPONENTIAL convergence in N,
  * PLUNGE-KNEE mode (m ~ M_n(c)):              ALGEBRAIC ~ 1/P.
And: larger P RESTORES the large-c deep-plateau accuracy to ~machine; the interior eigenfunction error
SHRINKS with N while the FB-edge boundary spike PERSISTS (psi_N(R)=0 forced for every P; Tasks 12.3/12.5).

To expose super-exponential convergence, a plateau mode is tracked from P just above its index (before it
saturates at the machine floor). Reference EIGENVALUES = a high-Nq (=300) Nystrom solve (machine-accurate,
NOT benchmark-floored); reference EIGENFUNCTIONS = the Task 10.4 benchmark (for interior/boundary sup).
Scope: the N-convergence study; the steady-state N-vs-error plot is 13.7.
"""
import os
import io
import contextlib
import numpy as np
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, Mn
from task_12_2_eigensolver import solve
from task_12_3_canonicalize import canonicalize
from task_12_4_synthesis import synthesize_nd
with contextlib.redirect_stdout(io.StringIO()):
    from task_10_1_reference_solver import nystrom

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "data", "benchmark")
CAMPAIGN = os.path.join(REPO, "data", "campaign")


def N_sweep(n, c):
    d = np.load(os.path.join(BENCH, f"cpswf_n{n}_c{int(c)}.npz"))
    xg, wg, psi_ref, lam_ref = d["x_gl"], d["w_gl"], d["psi_gl"], d["lam"]
    xu, psi_ref_u = d["x_unif"], d["psi_unif"]
    M = int(d["Mn"]); mu = xg * wg
    lam_hi, *_ = nystrom(n, c, 300)          # machine-accurate reference eigenvalues
    m_se = M - 6                             # deep plateau mode -> super-exp (tracked from its onset)
    m_shoulder = M - 2                       # eigenfunction interior/boundary tracking
    m_knee = M                               # the plunge knee -> ~1/P
    # low-P points (M-5..M) catch the super-exp onset of m_se; high-P points (M+3..M+90) give the knee 1/P
    P_list = [M - 5, M - 4, M - 3, M - 2, M - 1, M, M + 3, M + 8, M + 20, M + 45, M + 90]
    P_list = sorted(set(P for P in P_list if P > m_se))
    rows = []
    for P in P_list:
        jz = jn_zeros(n, P)
        B = assemble(n, c, P=P, G=32)
        r = solve(B, "evr")
        lam, V = canonicalize(r["lam"], r["V"], n, jz)
        eig_se = abs(lam[m_se] - lam_hi[m_se])                       # valid for all P > m_se
        eig_knee = abs(lam[m_knee] - lam_hi[m_knee]) if P > m_knee else np.nan
        if P > m_shoulder:
            pn = synthesize_nd(n, jz, V[:, m_shoulder][:, None], xu)[:, 0]
            pr = psi_ref_u[:, m_shoulder]; pn *= np.sign(np.sum(pn * pr))
            dsup = np.abs(pn - pr)
            isup, bsup = float(dsup[xu < 0.95].max()), float(dsup[-1])
        else:
            isup = bsup = np.nan
        rows.append(dict(P=P, eig_se=float(eig_se), eig_knee=float(eig_knee),
                         interior_sup=isup, boundary_sup=bsup))
    return dict(M=M, m_se=m_se, m_shoulder=m_shoulder, m_knee=m_knee, rows=rows)


def run(save=True):
    out = {}
    for (n, c) in [(0, 40.0), (0, 80.0)]:
        sw = N_sweep(n, c); rows = sw["rows"]; out[(n, c)] = sw
        print("=" * 96)
        print(f"TASK 13.5 -- N-convergence at n={n}, c={c:.0f}  (M={sw['M']}; "
              f"super-exp mode m={sw['m_se']}=M-6, knee m={sw['m_knee']}=M, shoulder m={sw['m_shoulder']})")
        print("=" * 96)
        print(f"  {'P':>4s} {'eig(plateau M-6)':>17s} {'eig(knee M)':>13s} {'interior sup':>13s} {'boundary sup':>13s}")
        for r in rows:
            kn = f"{r['eig_knee']:.2e}" if np.isfinite(r['eig_knee']) else "    --   "
            isup = f"{r['interior_sup']:.2e}" if np.isfinite(r['interior_sup']) else "   --  "
            bsup = f"{r['boundary_sup']:.2e}" if np.isfinite(r['boundary_sup']) else "   --  "
            print(f"  {r['P']:4d} {r['eig_se']:17.2e} {kn:>13s} {isup:>13s} {bsup:>13s}")
        se = [r["eig_se"] for r in rows]
        se_orders = np.log10(max(se)) - np.log10(max(min(se), 1e-16))     # decades the plateau mode drops
        se_floor = min(se)
        krows = [r for r in rows if np.isfinite(r["eig_knee"])]
        Pk = [r["P"] for r in krows]; kn = [r["eig_knee"] for r in krows]
        knee_slope = float(np.polyfit(np.log(Pk), np.log(kn), 1)[0])
        pxerr = [P * e for P, e in zip(Pk[-4:], kn[-4:])]
        irows = [r for r in rows if np.isfinite(r["interior_sup"])]
        i_ratio = irows[-1]["interior_sup"] / irows[0]["interior_sup"]
        b_ratio = irows[-1]["boundary_sup"] / irows[0]["boundary_sup"]
        print(f"  RATES: plateau(M-6) drops {se_orders:.1f} decades to floor {se_floor:.1e}  =>  SUPER-EXPONENTIAL")
        print(f"         knee(M) ~ P^{knee_slope:.2f} (algebraic ~1/P; P*err last4 = [{min(pxerr):.2f},{max(pxerr):.2f}])")
        print(f"         interior sup shrink (last/first) = {i_ratio:.2e}   boundary sup ratio = {b_ratio:.2f} (persists)")
        print()
        out[(n, c)].update(se_orders=se_orders, se_floor=se_floor, knee_slope=knee_slope,
                           i_ratio=i_ratio, b_ratio=b_ratio)

    ok = True
    for (n, c), sw in out.items():
        superexp = sw["se_orders"] >= 4.0 and sw["se_floor"] < 1e-9       # many orders down to ~machine
        knee_1oP = -1.5 < sw["knee_slope"] < -0.7                         # algebraic ~1/P
        interior_shrinks = sw["i_ratio"] < 0.8
        boundary_persists = sw["b_ratio"] > 0.5
        print(f"  n={n} c={c:.0f}: plateau super-exp={superexp}({sw['se_orders']:.1f} dec -> {sw['se_floor']:.0e})  "
              f"knee~1/P={knee_1oP}({sw['knee_slope']:.2f})  interior shrinks={interior_shrinks}  "
              f"boundary persists={boundary_persists}")
        ok = ok and superexp and knee_1oP and interior_shrinks and boundary_persists

    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(os.path.join(CAMPAIGN, "task_13_5_N_convergence.npz"),
            **{f"n{n}_c{int(c)}_P": np.array([r["P"] for r in sw["rows"]]) for (n, c), sw in out.items()},
            **{f"n{n}_c{int(c)}_eig_plateau": np.array([r["eig_se"] for r in sw["rows"]]) for (n, c), sw in out.items()},
            **{f"n{n}_c{int(c)}_eig_knee": np.array([r["eig_knee"] for r in sw["rows"]]) for (n, c), sw in out.items()},
            **{f"n{n}_c{int(c)}_interior_sup": np.array([r["interior_sup"] for r in sw["rows"]]) for (n, c), sw in out.items()},
            **{f"n{n}_c{int(c)}_boundary_sup": np.array([r["boundary_sup"] for r in sw["rows"]]) for (n, c), sw in out.items()})
        print(f"  saved: data/campaign/task_13_5_N_convergence.npz")

    print(f"\n  TASK 13.5 N-CONVERGENCE VERIFIED: {ok}")
    return out


if __name__ == "__main__":
    run()
