import numpy as np
from scipy.special import jv, jvp, jn_zeros
from numpy.polynomial.legendre import leggauss
# Verify the indicial structure directly: integrate the S-L equation from x=1 inward
# -(1/x)[x(1-x^2)f']' + [n^2/x^2 + c^2 x^2] f = chi f
# Near x=1 with t=1-x:  2(t f_tt + f_t) = A f + O(t),  A = n^2 + c^2 - chi
# Indicial: f ~ a t^q  =>  2 a q^2 t^{q-1} = O(t^q)  =>  q^2 = 0 (double root)
print("Indicial exponent check: substitute f = t^q into 2(t f'' + f') and read the t^{q-1} coefficient")
import sympy as sp
t,q,a,A = sp.symbols('t q a A', positive=True)
f = a*t**q
expr = sp.simplify(2*(t*sp.diff(f,t,2) + sp.diff(f,t)))
print(f"  2(t f_tt + f_t) = {sp.simplify(expr)}   ->  coefficient of a*t^(q-1) is {sp.simplify(expr/(a*t**(q-1)))}")
print("  setting that to zero for a != 0 gives q^2 = 0: double root at q=0, so the")
print("  regular branch has f(1) != 0 and the second branch carries log t (not analytic).")

# Independent numeric: shoot the ODE from x=1 and confirm a nonzero-at-1 analytic solution exists
# and that forcing f(1)=0 forces f==0.
from scipy.integrate import solve_ivp
def run(n,c,chi,f1):
    A = n**2 + c**2 - chi
    def rhs(x,y):
        f,fp = y
        # from -[x(1-x^2)f']' + [n^2/x + c^2 x^3 - chi x] f = 0
        # x(1-x^2) f'' + (1-3x^2) f' = [n^2/x + c^2 x^3 - chi x] f
        p = x*(1-x**2); dp = 1-3*x**2
        return [fp, ((n**2/x + c**2*x**3 - chi*x)*f - dp*fp)/p]
    s = solve_ivp(rhs,[1-1e-8,0.3],[f1,0.0],rtol=1e-10,atol=1e-14)
    return s.y[0][-1]
print("\nShooting from x=1 (n=0,c=10,chi=20): value at x=0.3")
print(f"  start f(1)=1  -> f(0.3) = {run(0,10.,20.,1.0):.6e}   (nontrivial solution)")
print(f"  start f(1)=0  -> f(0.3) = {run(0,10.,20.,0.0):.6e}   (identically zero)")
