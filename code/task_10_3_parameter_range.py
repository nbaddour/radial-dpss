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
    return np.sort(np.linalg.eigvalsh(A))[::-1], x, w

def nystrom_usub(n, c, Nq):
    t, om = np.polynomial.legendre.leggauss(Nq); u = 0.5*(t+1); om = 0.5*om
    x = np.sqrt(u); wq = om/2.0; s = np.sqrt(wq)
    A = (s[:, None]*Kmat(n, c, x))*s[None, :]; A = 0.5*(A+A.T)
    return np.sort(np.linalg.eigvalsh(A))[::-1]

def Mn(n, c):
    return int(np.sum(jn_zeros(n, int(c/np.pi)+30) < c))

def Kdiag(n, c, x):
    cx = c*x
    return (c**2/2)*(jvp(n, cx)**2 + (1-(n/cx)**2)*jv(n, cx)**2)

N_LIST = [0, 1, 2, 4, 8]
C_LIST = [10.0, 20.0, 40.0, 80.0, 160.0]
TAUS = [1e-8, 1e-10, 1e-12]

def ladder_for(c):
    return list(range(30, 1001, 10))

def ref_Nq(c):
    return int(min(1400, max(600, 6.0*c + 200)))

print("="*100)
print("TASK 10.3  —  RELIABLE PARAMETER RANGE OF THE REFERENCE SOLVER")
print("Leading modes = m=0..M_n(c)+4.  e_conv vs Nq_ref;  e_AB = SolverA vs SolverB;  trace identity.")
print("="*100)

rows = []
for n in N_LIST:
    for c in C_LIST:
        M = Mn(n, c); k = M + 5
        Nref = ref_Nq(c)
        lam_ref, xr, wr = nystrom(n, c, Nref)
        lam_ref2, _, _ = nystrom(n, c, Nref-120)
        ref_stab = np.max(np.abs(lam_ref[:k] - lam_ref2[:k]))
        minNq = {tau: None for tau in TAUS}
        econv_at = {}
        for Nq in ladder_for(c):
            if Nq >= Nref: break
            if Nq < k: continue
            lam, _, _ = nystrom(n, c, Nq)
            e = np.max(np.abs(lam[:k] - lam_ref[:k]))
            econv_at[Nq] = e
            for tau in TAUS:
                if minNq[tau] is None and e < tau:
                    minNq[tau] = Nq
        Nwork = minNq[1e-10] or minNq[1e-8] or (Nref-120)
        lamA, _, _ = nystrom(n, c, Nwork); lamB = nystrom_usub(n, c, Nwork)
        e_AB = np.max(np.abs(lamA[:k] - lamB[:k]))
        trace_eig = np.sum(lam_ref)
        trace_ker = np.sum(Kdiag(n, c, xr)*xr*wr)
        e_trace = abs(trace_eig - trace_ker)
        n_clust = int(np.sum(np.abs(lam_ref - 1.0) < 1e-12))
        rows.append(dict(n=n, c=c, M=M, k=k, Nref=Nref, ref_stab=ref_stab,
                         minNq=minNq, e_AB=e_AB, Nwork=Nwork, e_trace=e_trace,
                         n_clust=n_clust))
        print(f"\n n={n}  c={c:6.1f}  M_n={M}  (score leading {k} modes)   Nq_ref={Nref}  ref-stability={ref_stab:.1e}")
        print(f"    min Nq:  tau=1e-8 -> {minNq[1e-8]}   tau=1e-10 -> {minNq[1e-10]}   tau=1e-12 -> {minNq[1e-12]}")
        thr = minNq[1e-10]
        near = {Nq: e for Nq, e in econv_at.items() if (thr is None) or (abs(Nq-thr) <= 40)}
        print(f"    e_conv near threshold: " + ", ".join(f"{Nq}:{e:.0e}" for Nq, e in sorted(near.items())[:10]))
        print(f"    Solver A vs B (Nq={Nwork}): e_AB={e_AB:.1e}    trace identity err={e_trace:.1e}    "
              f"clustered@1 (<1e-12): {n_clust} of leading")

print("\n"+"="*100)
print("SUMMARY  —  minimum Nq to reach tolerance on leading M_n+5 modes")
print("="*100)
print(f"{'n':>3} {'c':>7} {'M_n':>5} | {'Nq(1e-8)':>9} {'Nq(1e-10)':>10} {'Nq(1e-12)':>10} | "
      f"{'e_AB':>9} {'e_trace':>9} {'clus@1':>7}")
for r in rows:
    f = lambda v: ("%d" % v) if v is not None else "  >ladder"
    print(f"{r['n']:>3} {r['c']:>7.1f} {r['M']:>5} | {f(r['minNq'][1e-8]):>9} {f(r['minNq'][1e-10]):>10} "
          f"{f(r['minNq'][1e-12]):>10} | {r['e_AB']:>9.1e} {r['e_trace']:>9.1e} {r['n_clust']:>7}")

print("\n"+"="*100)
print("PRACTICAL RULE  —  least-squares Nq(1e-10) ~ a + b*c   (per n)")
print("="*100)
for n in N_LIST:
    cs = []; Nq10 = []
    for r in rows:
        if r['n'] == n and r['minNq'][1e-10] is not None:
            cs.append(r['c']); Nq10.append(r['minNq'][1e-10])
    if len(cs) >= 2:
        A = np.vstack([np.ones(len(cs)), cs]).T
        coef, *_ = np.linalg.lstsq(A, np.array(Nq10, float), rcond=None)
        pred = A @ coef
        print(f" n={n}: Nq ~ {coef[0]:.0f} + {coef[1]:.2f}*c   (c={cs}, Nq={Nq10}, max|resid|={np.max(np.abs(pred-np.array(Nq10))):.0f})")
    else:
        print(f" n={n}: insufficient points reaching 1e-10 on the ladder")
