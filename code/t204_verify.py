import numpy as np
from scipy.special import jv, jn_zeros
from numpy.polynomial.legendre import leggauss
jz = jn_zeros(0, 400)

def boulsane(N, omega, Q=3000):
    tg,tw = leggauss(Q); t=0.5*omega*(tg+1); w=0.5*omega*tw
    s = jz[:N]
    Phi = jv(0, np.outer(s,t))/np.abs(jv(1,s))[:,None]
    R = 2*(Phi*(t*w)[None,:])@Phi.T
    return 0.5*(R+R.T)

# CHECK A: omega -> 1 must give the identity
R1 = boulsane(10, 1.0)
print("Check A  omega=1 -> I :  max|R-I| =", np.abs(R1-np.eye(10)).max())

# CHECK B: full spectrum comparison at both eps, c=40
def nystrom(c, M=500):
    xg,wg = leggauss(M); x=0.5*(xg+1); w=0.5*wg; s=np.sqrt(x*w)
    num = c*(x[:,None]*(-jv(1,x[:,None]*c))*jv(0,x[None,:]*c) - x[None,:]*jv(0,x[:,None]*c)*(-jv(1,x[None,:]*c)))
    den = x[None,:]**2 - x[:,None]**2
    with np.errstate(divide='ignore',invalid='ignore'): K=num/den
    K[np.arange(M),np.arange(M)] = (c**2/2)*(jv(1,x*c)**2+jv(0,x*c)**2)
    A = s[:,None]*K*s[None,:]
    return np.sort(np.linalg.eigvalsh(0.5*(A+A.T)))[::-1]
c=40.0; P=32; ref=nystrom(c)
print(f"\nCheck B  full spectrum, c={c}, N=P={P}: |lam_m(Boulsane) - lam_m(ref)|")
print(f"{'m':>3} {'ref':>12} | {'eps=pi (proj)':>14} | {'eps=pi/2 (Boul)':>16}")
for eps,tag in [(jz[P]-jz[P-1],'pi'),(np.pi/2,'pi/2')]:
    pass
b_pi   = np.sort(np.linalg.eigvalsh(boulsane(P, c/(jz[P-1]+(jz[P]-jz[P-1])))))[::-1]
b_half = np.sort(np.linalg.eigvalsh(boulsane(P, c/(jz[P-1]+np.pi/2))))[::-1]
for m in [0,4,8,10,11,12,13,14,16]:
    print(f"{m:>3} {ref[m]:>12.8f} | {abs(b_pi[m]-ref[m]):>14.3e} | {abs(b_half[m]-ref[m]):>16.3e}")
tot_pi   = np.abs(b_pi[:20]-ref[:20]).sum(); tot_h = np.abs(b_half[:20]-ref[:20]).sum()
print(f"sum of |err| over m<20:  eps=pi {tot_pi:.4e}   eps=pi/2 {tot_h:.4e}")

# CHECK C: scan eps to find what the project's choice costs
print("\nCheck C  knee (m=12) error vs eps, c=40, N=32:")
for eps in [0.5,1.0,np.pi/2,2.0,2.4,3.0,np.pi]:
    b = np.sort(np.linalg.eigvalsh(boulsane(P, c/(jz[P-1]+eps))))[::-1]
    flag = "" if eps < 1.5576 else ("  [> zeta]" if eps < 2.4048 else "  [> zeta, > s_1]")
    print(f"   eps={eps:.4f}  omega={c/(jz[P-1]+eps):.4f}  knee_err={abs(b[12]-ref[12]):.3e}{flag}")
