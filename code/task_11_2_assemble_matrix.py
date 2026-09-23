"""
Task 11.2 — Assemble the finite radial prolate matrix B_K^{(B)}_{N-1}(n=0, c) in the F1 fixed-c regime.

Entry formula (Task 6.1 eq. 3.1):
    B[m,k] = 2 j_m j_k INT_0^c  u J_n(u)^2 / [ (u^2 - j_m^2)(u^2 - j_k^2) ] du,    m,k = 1..N-1
with j_p = j_{n,p} the Bessel zeros, c FIXED (F1; not the closure j_{n,N}). Indices p run 1..N-1; some
j_p may exceed c (those basis functions are high-frequency / tail — their poles lie outside [0,c]).

Two independent assembly routes (cross-check):
  ROUTE B (primary, efficient): partial fraction (Task 6.1 eq. 3.3'):
     off-diag  B[m,k] = 2 j_m j_k (I_m - I_k)/(j_m^2 - j_k^2),  I_p = INT_0^c u J_n^2/(u^2 - j_p^2) du
     diagonal  B[m,m] = 2 j_m^2 D_m,                            D_p = INT_0^c u J_n^2/(u^2 - j_p^2)^2 du
  ROUTE A (independent check): the raw double-pole integrand per entry, same quadrature nodes.

Quadrature: composite Gauss-Legendre with panel breakpoints at {0, j_1,...,j_{M_n(c)}, c} (the J_n zeros
below c). Panels align with the integrand's oscillation/removable-singularity structure -> spectral
accuracy. Removable singularities (u -> j_p) handled by analytic limits (guard); GL nodes are interior.
"""
import numpy as np, os
from scipy.special import jv, jn_zeros

def panels_nodes(c, jzeros_below_c, G):
    bps = np.concatenate(([0.0], jzeros_below_c, [c]))
    t, w = np.polynomial.legendre.leggauss(G)
    U=[]; Wq=[]
    for a,b in zip(bps[:-1], bps[1:]):
        if b<=a: continue
        U.append(0.5*(b-a)*t + 0.5*(a+b)); Wq.append(0.5*(b-a)*w)
    return np.concatenate(U), np.concatenate(Wq)

def assemble(n, c, P, G=24):
    # Bessel zeros j_{n,1..P}; and those below c for breakpoints
    jz = jn_zeros(n, P)                      # j_{n,1..P}
    j_below = jz[jz < c]
    U, Wq = panels_nodes(c, j_below, G)
    J = jv(n, U); J2 = J*J
    uJ2W = U*J2*Wq                           # common weight*u*J^2
    j2 = jz**2
    # I_p and D_p with removable-singularity guards (term=0 for I at u=j_p; analytic for D)
    I = np.empty(P); D = np.empty(P)
    J1z = jv(n+1, jz)                         # J_{n+1}(j_{n,p}) for the limits
    for p in range(P):
        den = U*U - j2[p]
        near = np.abs(U - jz[p]) < 1e-11
        ti = uJ2W/den
        if near.any(): ti[near] = 0.0
        I[p] = ti.sum()
        td = uJ2W/den**2
        if near.any(): td[near] = (J1z[p]**2/(4*jz[p]))*Wq[near]
        D[p] = td.sum()
    # Route B
    B = np.empty((P,P))
    for m in range(P):
        B[m,m] = 2*j2[m]*D[m]
        for k in range(m+1,P):
            val = 2*jz[m]*jz[k]*(I[m]-I[k])/(j2[m]-j2[k])
            B[m,k]=B[k,m]=val
    return B, (U,Wq,J2,jz,j2)

def assemble_routeA(n,c,P,cache):
    U,Wq,J2,jz,j2 = cache; uJ2W=U*J2*Wq; P=len(jz); BA=np.empty((P,P))
    for m in range(P):
        for k in range(m,P):
            den=(U*U-j2[m])*(U*U-j2[k])
            near=(np.abs(U-jz[m])<1e-11)|(np.abs(U-jz[k])<1e-11)
            t=uJ2W/den
            if near.any(): t[near]=0.0   # removable (numerator double-zero dominates) -> 0 at node==pole
            BA[m,k]=BA[k,m]=2*jz[m]*jz[k]*t.sum()
    return BA

def Mn(n,c): return int(np.sum(jn_zeros(n,int(c/np.pi)+30)<c))

OUT="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/pilot"; os.makedirs(OUT,exist_ok=True)
SWEEP={10.0:[8,12,16,24,32], 20.0:[12,18,24,32,44], 40.0:[20,28,40,56]}

print("="*92)
print("TASK 11.2 — assemble B_K^{(B)} (n=0, F1 fixed-c). Route B (partial-fraction) primary.")
print("="*92)
for c in [10.0,20.0,40.0]:
    M=Mn(0,c)
    print(f"\n c={c}  M_0(c)={M}")
    for P in SWEEP[c]:
        B,cache = assemble(0,c,P,G=24)
        # quadrature convergence: G=24 vs G=40
        B40,_ = assemble(0,c,P,G=40)
        conv=np.max(np.abs(B-B40))
        # route A vs B
        BA = assemble_routeA(0,c,P,cache)
        ab=np.max(np.abs(B-BA))
        # structure
        sym=np.max(np.abs(B-B.T))
        lam=np.sort(np.linalg.eigvalsh(B))[::-1]
        in01 = lam.min()>-1e-12 and lam.max()<1+1e-12
        ngt=int((lam>0.5).sum())
        np.savez_compressed(os.path.join(OUT,f"Bmat_n0_c{int(c)}_P{P}.npz"),
                            n=0,c=c,P=P,B=B,lam=lam,Mn=M)
        print(f"   P={P:3d}: quad(G24vsG40)={conv:.1e} routeA-B={ab:.1e} sym={sym:.1e} "
              f"lam in[0,1]={in01} #lam>.5={ngt}(M={M}) lam0..4="
              + ",".join(f"{v:.5f}" for v in lam[:5]))

# benchmark eigenvalue spot-check (informal; formal comparison is Task 11.4)
print("\n"+"="*92); print("Spot-check vs Task 10.4 benchmark eigenvalues (informal)"); print("="*92)
D="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
for c,P in [(20.0,44),(10.0,32),(40.0,56)]:
    d=np.load(os.path.join(D,f"cpswf_n0_c{int(c)}.npz")); lref=d['lam']
    B,_=assemble(0,c,P,G=32); lam=np.sort(np.linalg.eigvalsh(B))[::-1]
    k=min(len(lref),7)
    print(f" c={c} P={P}: B-matrix lam0..{k-1}: "+", ".join(f"{v:.6f}" for v in lam[:k]))
    print(f"          benchmark   lam0..{k-1}: "+", ".join(f"{v:.6f}" for v in lref[:k]))
    print(f"          max|lam_B - lam_ref| (leading {k}) = {np.max(np.abs(lam[:k]-lref[:k])):.2e}")
