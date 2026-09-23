"""
Task 11.4 — Formal comparison of the pilot finite radial prolate method against the Task 10.4 benchmark.

Pilot PSI_N (Task 11.3) and benchmark PSI_ref are stored on the SAME grid x_gl, so inner products use the
benchmark GL quadrature <f,g> = sum f g x w directly (no interpolation).

Regime-aware metrics (Task 11.3 flags):
  - eigenvalue error |lam_N - lam_ref|, reported per region (plateau / plunge-knee / informative max).
  - PLATEAU modes (benchmark regime 0, lam~1, degenerate): SUBSPACE principal angle (max sin theta).
  - SEPARATED informative modes (regime 1): individual sign-aligned L2 error ||psi_N - psi_ref||_{x dx}.
  - DEEP-TAIL modes (regime 2, or lam_ref < TAIL): eigenvalue only (eigenfunction is truncation/benchmark
    limited and dynamically irrelevant; not held against the method).
Convergence vs P, plus a power-law fit of the plunge-knee eigenvalue error.
"""
import numpy as np, os, glob
from scipy.special import jn_zeros

BEN="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
PIL="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/pilot"
def Mn(n,c): return int(np.sum(jn_zeros(n,int(c/np.pi)+30)<c))

def subspace_angle(A,B,Wc):     # max principal-angle sine between span(A),span(B); columns weighted by sqrt(Wc)
    Aw=A*np.sqrt(Wc)[:,None]; Bw=B*np.sqrt(Wc)[:,None]
    Qa,_=np.linalg.qr(Aw); Qb,_=np.linalg.qr(Bw)
    sv=np.linalg.svd(Qa.T@Qb,compute_uv=False)
    return np.sqrt(np.clip(1-sv.min()**2,0,1))

for c in [10.0,20.0,40.0]:
    M=Mn(0,c)
    ben=np.load(os.path.join(BEN,f"cpswf_n0_c{int(c)}.npz"))
    xg=ben['x_gl']; wg=ben['w_gl']; Wc=xg*wg
    psiref=ben['psi_gl']; lref=ben['lam']; reg=ben['regime']; nb=int(ben['nmodes'])
    pla=np.where(reg==0)[0]; sep=np.where(reg==1)[0]
    knee=M     # 0-indexed plunge-knee mode (m=M_0)
    files=sorted(glob.glob(os.path.join(PIL,f"sol_n0_c{int(c)}_P*.npz")),
                 key=lambda s:int(s.split('_P')[1].split('.')[0]))
    print("="*100)
    print(f" c={c}  M_0={M}  | benchmark modes m=0..{nb-1} (plateau {len(pla)}, separated {len(sep)}); "
          f"plunge-knee m={knee} (lam_ref={lref[knee]:.4f})")
    print("="*100)
    print(f"{'P':>4} {'plateau dlam':>12} {'plat subangle':>13} {'knee dlam':>10} {'knee L2err':>10} "
          f"{'sep-info L2max':>13}")
    Ps=[]; knee_dl=[]
    for fp in files:
        s=np.load(fp); P=int(s['P']); psiN=s['psi_gl']; lamN=s['lam']; nm=psiN.shape[1]
        # plateau eigenvalue error + subspace angle
        plat_dl=np.max(np.abs(lamN[pla]-lref[pla])) if len(pla) else 0.0
        ang=subspace_angle(psiref[:,pla],psiN[:,pla],Wc) if len(pla) else 0.0
        # knee eigenvalue + L2
        kdl=abs(lamN[knee]-lref[knee])
        sgn=1.0 if np.sum(psiN[:,knee]*psiref[:,knee]*Wc)>=0 else -1.0
        kL2=np.sqrt(max(0.0,np.sum((psiN[:,knee]-sgn*psiref[:,knee])**2*Wc)))
        # informative separated modes: regime 1 AND lam_ref >= 1e-2 (the plunge/transition band)
        info=[m for m in sep if lref[m]>=1e-2]
        sepL2=0.0
        for m in info:
            sg=1.0 if np.sum(psiN[:,m]*psiref[:,m]*Wc)>=0 else -1.0
            sepL2=max(sepL2,np.sqrt(max(0.0,np.sum((psiN[:,m]-sg*psiref[:,m])**2*Wc))))
        print(f"{P:>4} {plat_dl:>12.1e} {ang:>13.1e} {kdl:>10.1e} {kL2:>10.1e} {sepL2:>13.1e}")
        Ps.append(P); knee_dl.append(kdl)
    # power-law fit knee dlam ~ C P^-p  (log-log)
    Ps=np.array(Ps,float); kd=np.array(knee_dl,float)
    g=kd>0
    p,logC=np.polyfit(np.log(Ps[g]),np.log(kd[g]),1)
    print(f"   plunge-knee eigenvalue error fit:  |dlam| ~ {np.exp(logC):.2f} * P^({p:.2f})")

# per-mode snapshot at the largest P for c=20 and c=40
print("\n"+"="*100); print("PER-MODE SNAPSHOT at largest P"); print("="*100)
for c,P in [(20.0,44),(40.0,56)]:
    ben=np.load(os.path.join(BEN,f"cpswf_n0_c{int(c)}.npz")); xg=ben['x_gl'];wg=ben['w_gl'];Wc=xg*wg
    psiref=ben['psi_gl']; lref=ben['lam']; reg=ben['regime']
    s=np.load(os.path.join(PIL,f"sol_n0_c{int(c)}_P{P}.npz")); psiN=s['psi_gl']; lamN=s['lam']; nm=psiN.shape[1]
    print(f"\n c={c} P={P}:  m  lam_ref   |dlam|   regime    L2err(sign-aligned)")
    for m in range(nm):
        sg=1.0 if np.sum(psiN[:,m]*psiref[:,m]*Wc)>=0 else -1.0
        L2=np.sqrt(max(0.0,np.sum((psiN[:,m]-sg*psiref[:,m])**2*Wc)))
        rg={0:'plateau',1:'sep',2:'tail'}[int(reg[m])]
        print(f"        {m:>2}  {lref[m]:8.5f}  {abs(lamN[m]-lref[m]):7.1e}  {rg:>7}   {L2:.2e}")
