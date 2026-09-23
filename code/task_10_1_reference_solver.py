"""
Task 10.1 — independent reference solver(s) for the continuous radial CPSWFs, BOTH SIDES of the
concentration pair: the space-limited psi^{(m)} in SL(R) and the band-limited dual phi^{(m)} in BL(K).

Nondim x in [0,1], measure x dx, R=1, K=c=KR.

SOLVER A (PRIMARY) — Nystrom on the integral operator T = B_R B_K B_R (space-limited side):
  int_0^1 K_n(x,y) psi(y) y dy = lambda psi(x),  finite-Hankel kernel
    K_n(x,y) = c[x J_n(cy)J_n'(cx) - y J_n(cx)J_n'(cy)]/(y^2-x^2)  (Lommel, off-diag),
    K_n(x,x) = (c^2/2)[J_n'(cx)^2 + (1-n^2/(cx)^2)J_n(cx)^2]        (diagonal).
  -> (lambda_m, psi^{(m)}).  Independent of the project's Fourier-Bessel/Bessel-zero matrix.

SOLVER B (CROSS-CHECK) — second Nystrom on the u=x^2 grid (Karoui-Moumni-style); -> lambda_m.

BAND-LIMITED DUAL phi^{(m)} in BL(K): phi = lambda^{-1/2} B_K psi, frequency profile
  G^{(m)}(rho) = H_n[psi^{(m)}](rho) = int_0^1 J_n(rho x) psi(x) x dx, phi(rho)=G(rho)/sqrt(lambda),
  rho in [0,K]. Nondim phitilde(xi)=c*phi(c*xi), xi=rho/K; by R<->K self-duality phitilde(xi) ~ psi(x).
"""
import numpy as np
from scipy.special import jv, jvp, jn_zeros

def Kmat(n, c, x):                                 # vectorized kernel matrix
    X = x[:, None]; Y = x[None, :]
    cX, cY = c*X, c*Y
    JX, JY, JpX, JpY = jv(n, cX), jv(n, cY), jvp(n, cX), jvp(n, cY)
    denom = Y**2 - X**2
    with np.errstate(divide='ignore', invalid='ignore'):
        K = c*(X*JY*JpX - Y*JX*JpY)/denom          # off-diagonal
    cx = c*x
    diagv = (c**2/2)*(jvp(n, cx)**2 + (1 - (n/cx)**2)*jv(n, cx)**2)
    np.fill_diagonal(K, diagv)
    return K

def nystrom(n, c, Nq):
    t, w = np.polynomial.legendre.leggauss(Nq); x = 0.5*(t+1); w = 0.5*w
    s = np.sqrt(w*x)
    A = (s[:, None]*Kmat(n, c, x))*s[None, :]; A = 0.5*(A+A.T)
    lam, V = np.linalg.eigh(A); idx = np.argsort(lam)[::-1]
    return lam[idx], x, w, V[:, idx]/s[:, None]

def nystrom_usub(n, c, Nq):                        # second independent grid u=x^2
    t, om = np.polynomial.legendre.leggauss(Nq); u = 0.5*(t+1); om = 0.5*om
    x = np.sqrt(u); wq = om/2.0; s = np.sqrt(wq)
    A = (s[:, None]*Kmat(n, c, x))*s[None, :]; A = 0.5*(A+A.T)
    return np.sort(np.linalg.eigvalsh(A))[::-1]

def bandlimited_dual(n, c, lam, x, w, psi, mmax):  # phi^{(m)} frequency profile, nondim phitilde(xi)
    xi = x.copy(); phit = np.zeros((len(xi), mmax))
    for m in range(mmax):
        G = (jv(n, c*np.outer(xi, x)) * (psi[:, m]*x*w)[None, :]).sum(axis=1)   # H_n[psi](c*xi)
        phit[:, m] = c*G/np.sqrt(lam[m])
    return xi, phit

def Mn(n, c):
    return int(np.sum(jn_zeros(n, int(c/np.pi)+25) < c))

print("="*72); print("SOLVER A (space-limited psi) — validation"); print("="*72)
for (n, c) in [(0, 20.0), (1, 20.0), (0, 40.0)]:
    lam1, *_ = nystrom(n, c, 160); lam2, x, w, psi = nystrom(n, c, 240)
    M = Mn(n, c)
    print(f" n={n}, c={c}: M_n={M}, #lam>1/2={int((lam2>0.5).sum())}, in[0,1]={lam2.min()>-1e-8 and lam2.max()<1+1e-8}, "
          f"conv(160vs240)={np.max(np.abs(lam1[:M+5]-lam2[:M+5])):.1e}")
    print(f"    lambda_0..6: " + ", ".join(f"{v:.6f}" for v in lam2[:7]))

print("\n"+"="*72); print("SOLVER B (u=x^2 grid) vs SOLVER A — independent eigenvalue cross-check"); print("="*72)
for (n, c) in [(0, 20.0), (1, 20.0), (0, 40.0)]:
    lamA, *_ = nystrom(n, c, 200); lamB = nystrom_usub(n, c, 200); M = Mn(n, c)
    print(f" n={n}, c={c}: max|lamA-lamB| (leading {M+5}) = {np.max(np.abs(lamA[:M+5]-lamB[:M+5])):.2e}")

print("\n"+"="*72); print("BAND-LIMITED DUAL phi^{(m)} (BL(K)) — norm, shared lambda, R<->K self-duality"); print("="*72)
for (n, c) in [(0, 20.0), (1, 20.0)]:
    lam, x, w, psi = nystrom(n, c, 200); mmax = Mn(n, c)
    xi, phit = bandlimited_dual(n, c, lam, x, w, psi, mmax)
    print(f" n={n}, c={c}:")
    for m in range(min(mmax, 5)):
        norm = np.sum(phit[:, m]**2 * xi * w)
        sgn = 1.0 if (phit[:, m] @ psi[:, m]) >= 0 else -1.0
        derr = np.max(np.abs(sgn*phit[:, m]-psi[:, m]))
        print(f"   m={m}: ||phi||^2_(rho drho)={norm:.6f}  self-dual max|phitilde(xi)-psi(x)|={derr:.2e}  lam={lam[m]:.5f}")
print("\n=> reference now carries BOTH sides: space-limited psi^{(m)} and band-limited dual phi^{(m)}.")
