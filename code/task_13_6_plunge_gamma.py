"""
Task 13.6 — Validation campaign: the plunge region and the Shannon-number slope gamma_n.

Confirm, from the FULL PIPELINE eigenvalues (finite radial prolate matrix; not just the operator/reference
of Tasks 9.5/9.6), the two Shannon/plunge laws:
  * EFFECTIVE DIMENSION  N_eff(n,c) = trace(B) = sum_m lambda_m  ~  gamma_n * c + b_n,  with
      gamma_n = 1/pi  (measured, NOT assumed; n-INDEPENDENT slope) and intercept b_n ~ -n/2 (Task 9.5),
  * PLUNGE WIDTH  V(n,c) = sum_m lambda_m (1 - lambda_m)  ~  alpha_n * log c + beta_n,  with
      alpha_n small and n-INDEPENDENT (~log c growth; Task 9.6); the plunge is CENTERED at M_n(c) ~ c/pi - n/2.

N_eff = trace(B) is exact for the assembled matrix and converges as P grows past M (tail diagonal entries
-> 0); V is likewise a trace functional (sum lambda(1-lambda) = trace(B) - trace(B^2)). Uses P = M + 40.
Scope: gamma_n and the plunge law from the pipeline. This is the headline numerical insight of Task 13.
"""
import os
import numpy as np
from scipy.special import jn_zeros
from task_12_1_assemble_matrix import assemble, Mn

REPO = os.environ.get("RDPSS_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN = os.path.join(REPO, "data", "campaign")


def neff_and_width(n, c, buffer=40):
    P = Mn(n, c) + buffer
    B = assemble(n, c, P=P, G=32)
    lam = np.sort(np.linalg.eigvalsh(B))[::-1]
    lam = np.clip(lam, 0.0, 1.0)
    N_eff = float(lam.sum())                       # trace(B) = sum lambda  (effective dimension)
    V = float(np.sum(lam * (1.0 - lam)))           # plunge width (smooth)
    return N_eff, V, P


def run(save=True):
    c_values = np.array([20., 30., 40., 60., 80., 110., 140., 170.])
    ns = [0, 1, 2, 4]
    data = {}
    print("=" * 96)
    print("TASK 13.6 -- (A) effective dimension N_eff(n,c) = trace(B) = sum lambda,  fit ~ gamma_n c + b_n")
    print("=" * 96)
    print(f"  {'n':>2s} " + "".join(f"{('c=%d'%c):>9s}" for c in c_values) + f" | {'gamma_n':>8s} {'1/pi?':>7s} {'b_n':>7s} {'-n/2':>6s}")
    gammas = {}
    for n in ns:
        Neff = np.array([neff_and_width(n, c)[0] for c in c_values])
        V = np.array([neff_and_width(n, c)[1] for c in c_values])
        data[n] = dict(Neff=Neff, V=V)
        # linear fit N_eff = gamma*c + b
        gamma, b = np.polyfit(c_values, Neff, 1)
        gammas[n] = gamma
        print(f"  {n:2d} " + "".join(f"{ne:9.3f}" for ne in Neff) +
              f" | {gamma:8.4f} {1/np.pi:7.4f} {b:7.3f} {-n/2:6.1f}")
    print(f"\n  (1/pi = {1/np.pi:.5f})")

    print("\n" + "=" * 96)
    print("TASK 13.6 -- (B) plunge width V(n,c) = sum lambda(1-lambda),  fit ~ alpha_n log c + beta_n")
    print("=" * 96)
    print(f"  {'n':>2s} " + "".join(f"{('c=%d'%c):>9s}" for c in c_values) + f" | {'alpha_n':>8s} {'beta_n':>7s}")
    alphas = {}
    for n in ns:
        V = data[n]["V"]
        alpha, beta = np.polyfit(np.log(c_values), V, 1)
        alphas[n] = alpha
        print(f"  {n:2d} " + "".join(f"{v:9.4f}" for v in V) + f" | {alpha:8.4f} {beta:7.4f}")

    # checks
    gvals = np.array([gammas[n] for n in ns])
    gamma_is_1opi = bool(np.all(np.abs(gvals - 1/np.pi) < 0.02))       # measured ~ 1/pi
    gamma_n_indep = bool((gvals.max() - gvals.min()) < 0.01)           # n-independent slope
    avals = np.array([alphas[n] for n in ns])
    alpha_small = bool(np.all(np.abs(avals) < 0.15))                   # small (~log c, gentle)
    alpha_n_indep = bool((avals.max() - avals.min()) < 0.05)          # n-independent width slope
    plunge_positive = bool(np.all(avals > 0))                          # width GROWS with c
    print("\n" + "=" * 96)
    print(f"  gamma_n ~ 1/pi (measured, not assumed)   : {gamma_is_1opi}  (gammas = {np.array2string(gvals,precision=4)})")
    print(f"  gamma_n n-INDEPENDENT (spread < 0.01)    : {gamma_n_indep}  (spread = {gvals.max()-gvals.min():.4f})")
    print(f"  plunge width grows ~ log c (alpha_n > 0) : {plunge_positive}  (alphas = {np.array2string(avals,precision=4)})")
    print(f"  plunge-width slope alpha_n n-independent  : {alpha_n_indep}  (spread = {avals.max()-avals.min():.4f})")

    if save:
        os.makedirs(CAMPAIGN, exist_ok=True)
        np.savez_compressed(os.path.join(CAMPAIGN, "task_13_6_plunge_gamma.npz"),
            c=c_values, ns=np.array(ns),
            **{f"n{n}_Neff": data[n]["Neff"] for n in ns},
            **{f"n{n}_V": data[n]["V"] for n in ns},
            gammas=gvals, alphas=avals)
        print(f"\n  saved: data/campaign/task_13_6_plunge_gamma.npz")

    ok = gamma_is_1opi and gamma_n_indep and alpha_small and alpha_n_indep and plunge_positive
    print(f"\n  TASK 13.6 PLUNGE/gamma_n VERIFIED: {ok}")
    return data, gammas, alphas


if __name__ == "__main__":
    run()
