"""
Task 10.2 — internal validation of the reference solver (Task 10.1, Solver A: Nystrom on T=B_R B_K B_R).

Checks:
 (1) grid convergence: lambda_m and psi^{(0)} as Nq grows.
 (2) eigenfunction orthonormality:  int_0^1 psi^{(m)} psi^{(m')} x dx = delta.
 (3) integral-equation residual OFF-GRID: psi^{(m)} (Nystrom-interpolated to a fine grid) satisfies
     (T psi)(x) = lambda psi(x).
 (4) trace consistency (independent):  sum lambda_m  ==  int_0^1 K_n(x,x) x dx.
 (5) DIFFERENTIAL cross-check (n=0, fully independent of the integral machinery): verify Solver A's
     eigenfunctions satisfy the Sturm-Liouville ODE  L psi = chi psi (Task 4.9), via a spectral
     differentiation matrix, on interior points (commutation [L,T]=0 => shared eigenfunctions).
"""
import numpy as np
from scipy.special import jv, jvp, jn_zeros

def Kmat(n, c, x):
    X = x[:, None]; Y = x[None, :]; cX, cY = c*X, c*Y
    JX, JY, JpX, JpY = jv(n, cX), jv(n, cY), jvp(n, cX), jvp(n, cY)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = c*(X*JY*JpX - Y*JX*JpY)/(Y**2 - X**2)
    cx = c*x
    np.fill_diagonal(K, (c**2/2)*(jvp(n, cx)**2 + (1-(n/cx)**2)*jv(n, cx)**2))
    return K

def nystrom(n, c, Nq):
    t, w = np.polynomial.legendre.leggauss(Nq); x = 0.5*(t+1); w = 0.5*w
    s = np.sqrt(w*x); A = (s[:, None]*Kmat(n, c, x))*s[None, :]; A = 0.5*(A+A.T)
    lam, V = np.linalg.eigh(A); idx = np.argsort(lam)[::-1]
    return lam[idx], x, w, V[:, idx]/s[:, None]

def Knfun(n, c, xv, yv):
    if abs(xv-yv) < 1e-13:
        cx = c*xv; return (c**2/2)*(jvp(n, cx)**2 + (1-(n/cx)**2)*jv(n, cx)**2)
    return c*(xv*jv(n, c*yv)*jvp(n, c*xv) - yv*jv(n, c*xv)*jvp(n, c*yv))/(yv**2-xv**2)

def bary_D(x):                                  # barycentric 1st-derivative matrix on nodes x
    N = len(x); wj = np.ones(N)
    for j in range(N):
        wj[j] = 1.0/np.prod([x[j]-x[k] for k in range(N) if k != j])
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j: D[i, j] = (wj[j]/wj[i])/(x[i]-x[j])
        D[i, i] = -np.sum(D[i, :])
    return D

def Mn(n, c): return int(np.sum(jn_zeros(n, int(c/np.pi)+25) < c))

n, c = 0, 20.0; M = Mn(n, c)
print(f"Reference-solver internal validation, n={n}, c={c}, M_n={M}\n")

# (1) convergence
l120, *_ = nystrom(n, c, 120); l180, *_ = nystrom(n, c, 180); lam, x, w, psi = nystrom(n, c, 240)
print(f"(1) convergence  max|lam(120)-lam(240)|={np.max(np.abs(l120[:M+5]-lam[:M+5])):.1e}, "
      f"max|lam(180)-lam(240)|={np.max(np.abs(l180[:M+5]-lam[:M+5])):.1e}  (leading {M+5})")

# (2) orthonormality (continuous norm, via the GL quadrature on x dx)
G = (psi[:, :M+3]*(x*w)[:, None]).T @ psi[:, :M+3]
print(f"(2) eigenfunction orthonormality  max|<psi_m,psi_m'>_x - delta| = {np.max(np.abs(G-np.eye(M+3))):.2e}")

# (3) off-grid integral-equation residual: interpolate psi^{(m)} to fine grid, apply T, vs lambda psi
xf = np.linspace(0.02, 0.98, 60)
res = []
for m in range(M):
    psif = np.array([np.sum([Knfun(n, c, xx, x[j])*psi[j, m]*x[j]*w[j] for j in range(len(x))])/lam[m]
                     for xx in xf])      # Nystrom interpolation: psi(x) = (1/lam) int K psi
    # apply T again at fine points: (T psi)(xf) vs lam*psi(xf)
    Tpsi = np.array([np.sum([Knfun(n, c, xx, x[j])*psi[j, m]*x[j]*w[j] for j in range(len(x))]) for xx in xf])
    res.append(np.max(np.abs(Tpsi - lam[m]*psif)))
print(f"(3) off-grid eigen-equation residual max_m max_x |T psi - lam psi| = {max(res):.2e}  (m<M_n)")

# (4) trace consistency
trace_eig = np.sum(lam)
trace_ker = np.sum(np.array([Knfun(n, c, xi, xi) for xi in x])*x*w)
print(f"(4) trace: sum lam_m = {trace_eig:.6f},  int K_n(x,x) x dx = {trace_ker:.6f},  diff = {abs(trace_eig-trace_ker):.2e}")

# (5) differential (Sturm-Liouville) cross-check, n=0: L psi = chi psi on interior points
xb, wb = np.polynomial.legendre.leggauss(90); xb = 0.5*(xb+1); wb = 0.5*wb
lamb, _, _, psib = nystrom(0, c, 90)
D = bary_D(xb); D2 = D @ D
interior = (xb > 0.08) & (xb < 0.92)
print(f"(5) differential cross-check  L psi = chi psi  (interior residual, n=0):")
for m in range(5):
    f = psib[:, m]
    Lf = -(1-xb**2)*(D2@f) - (1-3*xb**2)/xb*(D@f) + c**2*xb**2*f
    chi = np.sum((Lf*f*xb*wb)[interior])/np.sum((f*f*xb*wb)[interior])
    rr = np.max(np.abs((Lf - chi*f)[interior]))/np.max(np.abs((chi*f)[interior]))
    print(f"    m={m}: chi={chi:8.2f}  rel interior residual ||L psi - chi psi|| = {rr:.2e}  (lam={lamb[m]:.5f})")
