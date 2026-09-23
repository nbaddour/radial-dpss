import numpy as np
from scipy.special import jv, jn_zeros
from numpy.polynomial.legendre import leggauss
jz = jn_zeros(0, 50); c = 8.0; P = 6

# Route 1: manuscript body entry formula (dimensionless u-integral) = B
ug, uw = leggauss(4000); u = 0.5*c*(ug+1); w = 0.5*c*uw
H = np.array([np.sqrt(2)*jz[m]*jv(0,u)/(jz[m]**2-u**2) for m in range(P)])
B = H @ (H*(u*w)).T

# Route 2: matrix of Ttilde = c^2 K_n in unit-disc ONB phi_k(x)=sqrt2 J0(j_k x)/J1(j_k), measure x dx
xg, xw = leggauss(1200); x = 0.5*(xg+1); wx = 0.5*xw
xi, wi = leggauss(1200); xi = 0.5*(xi+1); wi = 0.5*wi
# k_n(x,x') = int_0^1 J0(c xi x) J0(c xi x') xi dxi
Kmat = (jv(0, np.outer(x, c*xi)) * (xi*wi)[None,:]) @ jv(0, np.outer(x, c*xi)).T
Phi = np.array([np.sqrt(2)*jv(0,jz[k]*x)/jv(1,jz[k]) for k in range(P)])
Btil = c**2 * (Phi*(x*wx)[None,:]) @ Kmat @ (Phi*(x*wx)[None,:]).T
print("B (body formula) diag:      ", " ".join(f"{B[i,i]:.8f}" for i in range(P)))
print("matrix of Ttilde=c^2 K_n:   ", " ".join(f"{Btil[i,i]:.8f}" for i in range(P)))
print("matrix of K_n alone (=B/c^2) diag:", " ".join(f"{Btil[i,i]/c**2:.8f}" for i in range(P)))
print("\nmax|B - Btilde|      =", np.abs(B-Btil).max())
print("max|B - c^2*Btilde|  =", np.abs(B-c**2*Btil).max())
print("max|B/c^2 - Btilde|  =", np.abs(B/c**2-Btil).max())
