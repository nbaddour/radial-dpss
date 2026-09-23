import numpy as np
from scipy.special import jv, jn_zeros
from numpy.polynomial.legendre import leggauss

jz = jn_zeros(0, 400)
def galerkin(c, P, Q=4000):
    # B_{mk} = 2 j_m j_k int_0^c u J0(u)^2 /((u^2-j_m^2)(u^2-j_k^2)) du
    ug, uw = leggauss(Q); u = 0.5*c*(ug+1); w = 0.5*c*uw
    J = jv(0,u)
    H = np.empty((P,len(u)))
    for m in range(P):
        jm = jz[m]
        den = jm**2 - u**2
        H[m] = np.sqrt(2)*jm*J/den
    return H @ (H*(u*w)).T

# Referee's check: n=0, N=13, c=j_{0,13}, P=12
N = 13; c_cl = jz[N-1]; P = N-1
B = galerkin(c_cl, P)
ev = np.sort(np.linalg.eigvalsh(0.5*(B+B.T)))[::-1]
print(f"CLOSURE: N={N}, c=j_(0,{N})={c_cl:.4f}, P={P}")
print("  eigenvalues:", " ".join(f"{v:.4f}" for v in ev))
print(f"  count>0.5: {np.sum(ev>0.5)} of {P};  smallest = {ev[-1]:.4f}")
print(f"  Shannon c/pi = {c_cl/np.pi:.3f};  P+3/4 = {P+0.75}")

# Fixed c=40, P=20
B2 = galerkin(40.0, 20)
ev2 = np.sort(np.linalg.eigvalsh(0.5*(B2+B2.T)))[::-1]
print(f"\nFIXED c=40, P=20: count>0.5 = {np.sum(ev2>0.5)}")
print("  eigenvalues:", " ".join(f"{v:.4f}" for v in ev2))
