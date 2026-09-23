import numpy as np
from scipy.special import jv, jn_zeros
from numpy.polynomial.legendre import leggauss

n=0; jz = jn_zeros(n, 400)
print("Boulsane's stated constraints on the matching eps (alpha=0):")
print(f"  eps < s_1 = j_(0,1)         = {jz[0]:.4f}")
gaps = np.diff(jz[:120])
print(f"  eps < zeta = 0.5*min gap    = {0.5*gaps.min():.4f}   (min gap {gaps.min():.4f})")
print(f"  project's eps = j_(N+1)-j_N ~ {gaps[-1]:.4f}  (-> pi = {np.pi:.4f})")
print(f"  Boulsane's own c=(N+a/2+1/4)*pi*w  =>  eps = pi/2 = {np.pi/2:.4f}")

# Nystrom reference for lambda_m(c)
def nystrom(c, M=700):
    xg,wg = leggauss(M); x=0.5*(xg+1); w=0.5*wg; s=np.sqrt(x*w)
    num = c*(x[:,None]*(-jv(1,x[:,None]*c))*jv(0,x[None,:]*c) - x[None,:]*jv(0,x[:,None]*c)*(-jv(1,x[None,:]*c)))
    den = x[None,:]**2 - x[:,None]**2
    with np.errstate(divide='ignore',invalid='ignore'): K = num/den
    diag = (c**2/2)*(jv(1,x*c)**2+jv(0,x*c)**2)
    K[np.arange(M),np.arange(M)] = diag
    A = s[:,None]*K*s[None,:]; A=0.5*(A+A.T)
    return np.sort(np.linalg.eigvalsh(A))[::-1]

def ours(c,P,Q=4000):
    ug,uw = leggauss(Q); u=0.5*c*(ug+1); w=0.5*c*uw
    H = np.array([np.sqrt(2)*jz[m]*jv(0,u)/(jz[m]**2-u**2) for m in range(P)])
    B = H@(H*(u*w)).T
    return np.sort(np.linalg.eigvalsh(0.5*(B+B.T)))[::-1]

def boulsane(N, omega, Q=4000):
    tg,tw = leggauss(Q); t=0.5*omega*(tg+1); w=0.5*omega*tw
    s = jz[:N]
    Phi = jv(0, np.outer(s,t))/np.abs(jv(1,s))[:,None]
    R = 2*(Phi*(t*w)[None,:])@Phi.T
    return np.sort(np.linalg.eigvalsh(0.5*(R+R.T)))[::-1]

c=40.0; M=12
ref = nystrom(c)
print(f"\n=== n=0, c={c}, deep-plateau m=M-6={M-6}, knee m=M={M} ===")
print(f"{'P=N':>4} {'eps':>8} {'omega':>7} | {'ours deep':>11} {'ours knee':>10} | {'Boul deep':>11} {'Boul knee':>10} | ratio deep/knee")
for P in [26,32,45]:
    o = ours(c,P)
    od, ok = abs(o[M-6]-ref[M-6]), abs(o[M]-ref[M])
    for label,eps in [("proj ~pi", jz[P]-jz[P-1]), ("Boul pi/2", np.pi/2)]:
        omega = c/(jz[P-1]+eps)
        b = boulsane(P, omega)
        bd, bk = abs(b[M-6]-ref[M-6]), abs(b[M]-ref[M])
        print(f"{P:>4} {eps:>8.4f} {omega:>7.4f} | {od:>11.2e} {ok:>10.2e} | {bd:>11.2e} {bk:>10.2e} | {bd/od:>6.2f} {bk/ok:>6.2f}")
