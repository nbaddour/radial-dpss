"""
Task 9.5 — empirically identify the Shannon-number law N_eff(n,c) ~ gamma_n c.
gamma_n is treated as UNKNOWN (not assumed 1/pi).

N_eff (primary) = trace(T) = sum_m lambda_m = sum_m (B^{(B)})_{mm}  (basis-independent effective
dimension). The m-th diagonal entry is the concentration of the m-th Fourier-Bessel basis function:
    d(n,m,c) = 2 j_{n,m}^2 * integral_0^c u J_n(u)^2 / (u^2 - j_{n,m}^2)^2 du,
~1 for j_{n,m} << c, ~0 for j_{n,m} >> c. Sum over m (until j_{n,m} > c + margin) gives the trace
without forming the full matrix. Cross-check: N_eff_half = #{lambda_m > 1/2} from the full matrix.
Fit N_eff vs c per n -> slope gamma_n, intercept; compare to 1/pi=0.3183 and 2/pi=0.6366.
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.integrate import quad

def diag_entry(n, jm, c):
    f = lambda u: u*jn(n, u)**2/((u**2-jm**2)**2)
    pts = [jm] if 0 < jm < c else []
    v, _ = quad(f, 0, c, points=pts, limit=400)
    return 2*jm**2*v

def trace_Neff(n, c):
    # sum diagonal entries until j_{n,m} exceeds c + margin (terms ~0 there)
    Nz = int(c/np.pi) + 40
    jz = jn_zeros(n, Nz)
    s = 0.0
    for jm in jz:
        if jm > c + 25:
            break
        s += diag_entry(n, jm, c)
    return s

def half_Neff(n, c):
    Nz = int(c/np.pi) + 25; jz = jn_zeros(n, Nz); M = len(jz)
    j = jz; B = np.zeros((M, M)); pts = [z for z in j if 0 < z < c]
    for a in range(M):
        for b in range(a, M):
            f = lambda u: u*jn(n, u)**2/((u**2-j[a]**2)*(u**2-j[b]**2))
            v, _ = quad(f, 0, c, points=pts, limit=300); B[a, b] = B[b, a] = 2*j[a]*j[b]*v
    lam = np.linalg.eigvalsh(B)
    return np.sum(lam > 0.5)

cs = np.array([10, 20, 30, 40, 50, 60], float)
print("N_eff = trace(T) = sum lambda_m   vs   c   (gamma_n = slope):")
print("  n  |  " + "  ".join(f"c={int(c)}" for c in cs) + "   |  gamma_n (slope)  intercept")
gammas = {}
for n in (0, 1, 2, 3):
    vals = np.array([trace_Neff(n, c) for c in cs])
    A = np.vstack([cs, np.ones_like(cs)]).T
    (g, b), *_ = np.linalg.lstsq(A, vals, rcond=None)
    gammas[n] = (g, b)
    print(f"  {n}  |  " + "  ".join(f"{v:5.2f}" for v in vals) + f"   |  gamma={g:.4f}   b={b:+.3f}")
print(f"\n  reference: 1/pi = {1/np.pi:.4f},  2/pi = {2/np.pi:.4f}")
print("  gamma_n (trace) for n=0..3: " + ", ".join(f"{gammas[n][0]:.4f}" for n in range(4)))

print("\ncross-check N_eff_half = #{lambda_m > 1/2} (full matrix), n=0,1,2:")
for n in (0, 1, 2):
    vals = [half_Neff(n, c) for c in (20.0, 40.0)]
    print(f"  n={n}: c=20 -> {vals[0]},  c=40 -> {vals[1]}   (slope ~ {(vals[1]-vals[0])/20:.4f})")
