import numpy as np
from scipy.special import jn, jn_zeros, jvp
from scipy.integrate import quad

# ---- order n = alpha = 0 ----
n = 0

def zeros(n, M):
    return jn_zeros(n, M)  # first M positive zeros of J_n

# ---------- THIS PROJECT'S MATRIX (band-Gram of space basis, free band c) ----------
# B[m,k] = 2 j_m j_k * \int_0^c u J_n(u)^2 / ((u^2 - j_m^2)(u^2 - j_k^2)) du
def proj_matrix(P, c, jz):
    j = jz[:P]
    B = np.zeros((P,P))
    for a in range(P):
        for b in range(a, P):
            jm, jk = j[a], j[b]
            def integrand(u):
                return u*jn(n,u)**2/((u**2-jm**2)*(u**2-jk**2))
            # split points at the zeros inside [0,c] to help quad near removable sing.
            pts = [z for z in jz if 0 < z < c]
            val,_ = quad(integrand, 0, c, points=pts, limit=400)
            B[a,b]=B[b,a]=2*jm*jk*val
    return B

# ---------- BOULSANE'S MATRIX rho^alpha_{N,omega} ----------
def G0(x,y):
    if abs(x-y)<1e-12:
        return 0.5*x*(jn(0,x)**2+jn(1,x)**2)
    return np.sqrt(x*y)/(x**2-y**2)*(x*jn(1,x)*jn(0,y)-y*jn(1,y)*jn(0,x))
def Kom(x,y,om):
    return om*G0(om*x,om*y)
def boul_matrix(N, om, sz):
    s=sz[:N]
    R=np.zeros((N,N))
    for a in range(N):
        for b in range(N):
            R[a,b]=2*Kom(s[a],s[b],om)/(np.sqrt(s[a])*abs(jn(1,s[a]))*np.sqrt(s[b])*abs(jn(1,s[b])))
    return R

jz = zeros(0, 40)

# Sanity: at omega->1, Boulsane rho should be ~ Identity (basis orthonormal on [0,1])
Rtest=boul_matrix(6,0.999999,jz)
print("Boulsane rho at omega~1 (should be ~I): diag=",np.round(np.diag(Rtest),4))
print("  max offdiag abs =",np.max(np.abs(Rtest-np.diag(np.diag(Rtest)))))

print("\n=========== SPECTRAL COMPARISON ===========")
# Boulsane effective c (his Prop 1):  c_eff = omega * s_{N+1}
for (N,om) in [(6,0.5),(8,0.4),(10,0.6)]:
    R = boul_matrix(N, om, jz)
    ev_b = np.sort(np.linalg.eigvalsh(R))[::-1]
    c_eff = om*jz[N]   # s_{N+1} = jz[N] (0-indexed -> (N+1)th zero)
    B = proj_matrix(N, c_eff, jz)
    ev_p = np.sort(np.linalg.eigvalsh(B))[::-1]
    print(f"\nN={N}, omega={om}, c_eff=omega*j_(0,{N+1})={c_eff:.4f}")
    print("  Boulsane eigs:", np.round(ev_b,6))
    print("  Project  eigs:", np.round(ev_p,6))
    print("  max |dlambda| =", np.max(np.abs(ev_b-ev_p)))

print("\n=========== SCAN c FOR EXACT SPECTRAL MATCH (Boulsane N=6, omega=0.5) ===========")
N,om=6,0.5
Rb=boul_matrix(N,om,jz); ev_b=np.sort(np.linalg.eigvalsh(Rb))[::-1]
best=None
for c in np.linspace(8.0,16.0,161):
    Bp=proj_matrix(N,c,jz); ev_p=np.sort(np.linalg.eigvalsh(Bp))[::-1]
    d=np.max(np.abs(ev_b-ev_p))
    if best is None or d<best[1]: best=(c,d,ev_p)
print("Boulsane eigs:", np.round(ev_b,6))
print(f"best-match c={best[0]:.3f}, max|dlambda|={best[1]:.3e}")
print("project eigs :", np.round(best[2],6))

print("\n=========== ENTRY RELATIONSHIP at the asymptotic c_eff ===========")
c_eff=om*jz[N]
Bp=proj_matrix(N,c_eff,jz)
# ratio matrix B_proj / rho_boul
ratio=Bp/Rb
print("ratio B_proj[m,k]/rho[m,k]:")
np.set_printoptions(precision=4,suppress=True)
print(ratio)
# test if ratio is separable: ratio[m,k] ?= g_m*g_k  -> ratio[m,k]^2 ?= ratio[m,m]*ratio[k,k]
gg=np.outer(np.sqrt(np.abs(np.diag(ratio))),np.sqrt(np.abs(np.diag(ratio))))
print("separable test max|ratio - sqrt(diag)outer|:", np.max(np.abs(np.abs(ratio)-gg)))

print("\n=========== REFINED c-scan + N-dependence of best spectral gap ===========")
def best_gap(N, om, crange):
    Rb=boul_matrix(N,om,jz); ev_b=np.sort(np.linalg.eigvalsh(Rb))[::-1]
    best=(None,1e9)
    for c in crange:
        Bp=proj_matrix(N,c,jz); ev_p=np.sort(np.linalg.eigvalsh(Bp))[::-1]
        d=np.max(np.abs(ev_b-ev_p))
        if d<best[1]: best=(c,d)
    return ev_b,best
for (N,om) in [(6,0.5),(10,0.5),(14,0.5)]:
    ev_b,(c,d)=best_gap(N,om,np.linspace(8,40,321))
    # refine
    _,(c2,d2)=best_gap(N,om,np.linspace(c-0.3,c+0.3,121))
    print(f"N={N}, omega={om}: best c={c2:.4f}, min max|dlambda|={d2:.3e}  (#plunge eigs in (1e-4,1-1e-4): {np.sum((ev_b>1e-4)&(ev_b<1-1e-4))})")

print("\n=========== SANITY: project matrix c->large should -> Identity ===========")
for c in [60, 120, 240]:
    Bp=proj_matrix(4,c,jz)
    print(f" c={c}: diag={np.round(np.diag(Bp),4)}, max offdiag={np.max(np.abs(Bp-np.diag(np.diag(Bp)))):.2e}")

print("\n=========== AIRTIGHT: ratio separability at BEST-FIT c (N=6) ===========")
N,om=6,0.5
Rb=boul_matrix(N,om,jz)
for c in [10.17]:
    Bp=proj_matrix(N,c,jz)
    ratio=Bp/Rb
    gg=np.outer(np.sqrt(np.abs(np.diag(ratio))),np.sqrt(np.abs(np.diag(ratio))))
    print(f" c={c}: separability residual max|ratio|-sqrt(diag)outer = {np.max(np.abs(np.abs(ratio)-gg)):.3f}")
    print("   (≈0 would mean B=D rho D for diagonal D; large => NO diagonal normalization)")

print("\n=========== STRUCTURAL: confirm dual forms (off-diagonal integrand) ===========")
print(" Project off-diag integrand: u*J0(u)^2/((u^2-jm^2)(u^2-jk^2))  [single J0 squared, frequency-side]")
print(" Boulsane via Lommel:        2/(|J1(sj)||J1(sk)|) * INT_0^w t J0(sj t)J0(sk t) dt  [cross product, space-side]")
# numerically confirm Boulsane entry equals the cross-integral form
from scipy.integrate import quad as q
N,om=6,0.5; s=jz[:N]
j,k=0,2
lhs=2*Kom(s[j],s[k],om)/(np.sqrt(s[j])*abs(jn(1,s[j]))*np.sqrt(s[k])*abs(jn(1,s[k])))
rhs=2/(abs(jn(1,s[j]))*abs(jn(1,s[k])))*q(lambda t: t*jn(0,s[j]*t)*jn(0,s[k]*t),0,om)[0]
print(f" check Boulsane rho[1,3]: kernel-form={lhs:.6f}, cross-integral-form={rhs:.6f}, diff={abs(lhs-rhs):.2e}")
