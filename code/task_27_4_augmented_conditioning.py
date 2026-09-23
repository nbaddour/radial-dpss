"""Task 27.4 - Conditioning of the boundary-augmented trial basis.

Referee item 5.  The augmented basis is (phi_1,...,phi_P,u_n) with u_n(x)=x^n.  Because
u_n lies in the closure of span{phi_k}, the component of u_n orthogonal to V_P has

    r_P^2 = nu_n - a^T a = sum_{k>P} 2/j_{n,k}^2 ~ 2/(pi^2 P),

so the basis becomes linearly dependent as P grows and both assembly routes of
Appendix F.2 divide by that shrinking quantity.  This script measures, against P:

  1. r_P^2 by three routes (Gram determinant, tail sum, asymptote);
  2. kappa_2 of the Gram matrix, measured and in closed form;
  3. the cancellation in the naive corner entry (d - 2 a^T b + a^T M a), whose three
     terms are O(1) while their sum is O(r_P^2);
  4. a cancellation-free assembly of the same border, obtained by forming the Hankel
     transform of the residual direction once and integrating its square;
  5. the eigenvalues of the two routes of Appendix F.2 against the stable assembly;
  6. the augmented and unaugmented Ritz eigenvalues at the knee mode against an
     independent Nystrom reference, which is what the conditioning is spent on.

Nothing here is smoothed or fitted.
"""
import math, json
import numpy as np
import scipy.linalg as sla
from scipy.special import jn_zeros, jv

from task_12_1_assemble_matrix import _panels_nodes, assemble

def hankel_nodes(n, c, zeros, G=64):
    below = zeros[zeros < c]
    u, wq = _panels_nodes(float(c), below, G)
    return u, u*wq                       # nodes and the measure u du

def transforms(n, u, zeros):
    """H phi_k and H u_n on the band nodes, signed convention."""
    Hphi = -math.sqrt(2)*zeros[None, :]*jv(n, u)[:, None]/(u[:, None]**2 - zeros[None, :]**2)
    Hu   = jv(n+1, u)/u
    return Hphi, Hu

def run(n, c, Ps, G=64):
    nu = 1.0/(2*n+2)
    Pmax = max(Ps)
    zeros = jn_zeros(n, Pmax)
    zbig  = jn_zeros(n, 200000)                       # for the tail sum
    u, meas = hankel_nodes(n, c, zeros, G)
    Hphi, Hu = transforms(n, u, zeros)
    M = assemble(n, float(c), P=Pmax, G=48)           # concentration block, signed basis
    M32 = assemble(n, float(c), P=Pmax, G=32)         # control: same block, coarser quadrature
    b = Hphi.T @ (meas*Hu)
    d = float(np.sum(meas*Hu*Hu))
    rows = []
    for P in Ps:
        a = math.sqrt(2)/zeros[:P]
        MP, bP = M[:P, :P], b[:P]

        # --- 1. the residual norm, three ways -------------------------------------
        r2_gram = nu - a @ a
        # exact identity, truncated at K0=2e5 zeros with the analytic remainder
        # sum_{k>K0} 2/j_{n,k}^2 ~ 2/(pi^2 K0) restored
        K0 = len(zbig)
        r2_tail = float(np.sum(2.0/zbig[P:]**2)) + 2.0/(math.pi**2*K0)
        r2_asym = 2.0/(math.pi**2*P)

        # --- 2. Gram conditioning --------------------------------------------------
        Gram = np.block([[np.eye(P), a[:, None]], [a[None, :], np.array([[nu]])]])
        kap = float(np.linalg.cond(Gram))
        s = math.sqrt((1-nu)**2 + 4*(a@a))
        kap_cf = ((1+nu+s)/2)/((1+nu-s)/2)

        # --- 3. naive border, with its cancellation --------------------------------
        t1, t2, t3 = d, -2*(a@bP), a @ (MP @ a)
        num_naive = t1 + t2 + t3
        cancel = max(abs(t1), abs(t2), abs(t3))/abs(num_naive)
        corner_naive = num_naive/r2_gram
        cross_naive = (bP - MP @ a)/math.sqrt(r2_gram)

        # --- 4. cancellation-free border ------------------------------------------
        # g = H(u_n - Pi_P u_n) formed once, then integrated.  No difference of
        # O(1) quantities appears anywhere.
        g = Hu - Hphi[:, :P] @ a
        num_stable = float(np.sum(meas*g*g))
        corner_stable = num_stable/r2_gram
        cross_stable = (Hphi[:, :P].T @ (meas*g))/math.sqrt(r2_gram)

        # --- 5. eigenvalues, three assemblies --------------------------------------
        A_naive = np.block([[MP, cross_naive[:, None]],
                            [cross_naive[None, :], np.array([[corner_naive]])]])
        A_stab  = np.block([[MP, cross_stable[:, None]],
                            [cross_stable[None, :], np.array([[corner_stable]])]])
        A_naive = 0.5*(A_naive+A_naive.T); A_stab = 0.5*(A_stab+A_stab.T)
        Mg = np.block([[MP, bP[:, None]], [bP[None, :], np.array([[d]])]])
        Mg = 0.5*(Mg+Mg.T)
        ev_n = np.sort(np.linalg.eigvalsh(A_naive))[::-1]
        ev_s = np.sort(np.linalg.eigvalsh(A_stab))[::-1]
        ev_g = np.sort(sla.eigvalsh(Mg, Gram, check_finite=False, driver="gvd"))[::-1]

        # --- control: the intrinsic floor of the UNaugmented block, measured the same
        # way, by reassembling it at a different quadrature order
        ep48 = np.sort(np.linalg.eigvalsh(MP))[::-1]
        ep32 = np.sort(np.linalg.eigvalsh(M32[:P, :P]))[::-1]
        d_plain = float(np.max(np.abs(ep48-ep32)))
        floor_plain = int(np.argmax(np.abs(ep48) < d_plain))

        rows.append(dict(P=P, d_plain=d_plain, floor_plain=floor_plain,
                         lam6_plain=float(ep48[6]), r2=r2_gram, r2_tail=r2_tail, r2_asym=r2_asym,
                         kappa=kap, kappa_cf=kap_cf, cancel=cancel,
                         corner_naive=corner_naive, corner_stable=corner_stable,
                         d_corner=abs(corner_naive-corner_stable),
                         d_naive=float(np.max(np.abs(ev_n-ev_s))),
                         d_gen=float(np.max(np.abs(ev_g-ev_s))),
                         lam_top=float(ev_s[0]), lam_knee=float(ev_s[6]),
                         lam_min=float(ev_s[-1]),
                         d_knee=abs(ev_n[6]-ev_s[6]),
                         rel_knee=abs(ev_n[6]-ev_s[6])/abs(ev_s[6]),
                         floor_index=int(np.argmax(np.abs(ev_s) < np.max(np.abs(ev_n-ev_s))))))
    return rows

def nystrom_reference(n, c, m, nq=2400):
    """Independent reference: Nystrom discretization of the finite Hankel kernel."""
    t, w = np.polynomial.legendre.leggauss(nq)
    x, w = 0.5*(t+1), 0.5*w
    X, Y = np.meshgrid(x, x, indexing="ij")
    A = c*np.sqrt(X*Y)*jv(n, c*X*Y)*np.sqrt(np.outer(w, w))
    A = 0.5*(A+A.T)
    g = np.linalg.eigvalsh(A)
    return float(g[np.argsort(-abs(g))][m]**2)

if __name__ == "__main__":
    n, c = 0, 20
    Ps = [20, 30, 45, 65, 95, 140, 200, 290, 400]
    rows = run(n, c, Ps)
    print(f"n={n} c={c}   nu_n={1/(2*n+2):.6f}")
    kap_pred = ((1+1/(2*n+2))**2)*(np.pi**2)/2
    print(f"predicted kappa_2/P -> (1+nu_n)^2 pi^2/2 = {kap_pred:.4f}")
    print(f"\n{'P':>5} {'r_P^2':>11} {'tail/gram':>10} {'asym/gram':>10} "
          f"{'kappa2':>10} {'kappa/P':>8} {'cancel':>9} {'corner':>10} "
          f"{'|corner err|':>12}")
    for r in rows:
        print(f"{r['P']:>5} {r['r2']:11.4e} {r['r2_tail']/r['r2']:10.7f} "
              f"{r['r2_asym']/r['r2']:10.6f} {r['kappa']:10.3e} {r['kappa']/r['P']:8.3f} "
              f"{r['cancel']:9.2e} {r['corner_stable']:10.3e} {r['d_corner']:12.2e}")
    print(f"\n{'P':>5} {'lam_0':>10} {'lam_6 (knee)':>13} {'lam_min':>11} "
          f"{'max|dlam| nv':>13} {'max|dlam| gen':>13} {'rel err lam_6':>14} "
          f"{'modes above':>12}")
    for r in rows:
        print(f"{r['P']:>5} {r['lam_top']:10.6f} {r['lam_knee']:13.6f} {r['lam_min']:11.3e} "
              f"{r['d_naive']:13.2e} {r['d_gen']:13.2e} {r['rel_knee']:14.2e} "
              f"{r['floor_index']:12d}")
    print("\nControl: the unaugmented block B_P, same measurements, floor set by")
    print("reassembling at G=32 instead of G=48.")
    print(f"\n{'P':>5} {'lam_6 plain':>13} {'lam_6 aug':>13} {'aug - plain':>13} "
          f"{'floor plain':>12} {'modes above':>12} {'modes above':>12}")
    print(f"{'':>5} {'':>13} {'':>13} {'':>13} {'':>12} {'(plain)':>12} {'(aug)':>12}")
    for r in rows:
        print(f"{r['P']:>5} {r['lam6_plain']:13.9f} {r['lam_knee']:13.9f} "
              f"{r['lam_knee']-r['lam6_plain']:13.2e} {r['d_plain']:12.2e} "
              f"{r['floor_plain']:12d} {r['floor_index']:12d}")

    ref = nystrom_reference(n, c, 6)
    print(f"\nIndependent Nystrom reference, lambda_6 = {ref:.15f}")
    print(f"{'P':>5} {'lam6 augmented':>20} {'aug - ref':>12} "
          f"{'lam6 plain':>20} {'plain - ref':>12}")
    for r in rows:
        print(f"{r['P']:>5} {r['lam_knee']:20.15f} {r['lam_knee']-ref:12.2e} "
              f"{r['lam6_plain']:20.15f} {r['lam6_plain']-ref:12.2e}")
    ea = [abs(r['lam_knee']-ref) for r in rows]
    ep = [abs(r['lam6_plain']-ref) for r in rows]
    Pv = np.log([r['P'] for r in rows])
    print(f"\nlog-log slope, augmented eigenvalue error : "
          f"{np.polyfit(Pv, np.log(ea), 1)[0]:+.2f}   (theory -5)")
    print(f"log-log slope, unaugmented eigenvalue error: "
          f"{np.polyfit(Pv, np.log(ep), 1)[0]:+.2f}   (theory -1)")
    json.dump(rows, open("task_27_4_conditioning.json", "w"), indent=1)
