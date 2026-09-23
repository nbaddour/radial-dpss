import numpy as np, os
from scipy.special import jv, jn_zeros
BEN="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
PIL="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/pilot"
def signfix(p):
    for m in range(p.shape[1]):
        j=np.argmax(np.abs(p[:,m]))
        if p[j,m]<0: p[:,m]*=-1
    return p
c=20.0; P=44
ben=np.load(os.path.join(BEN,f"cpswf_n0_c{int(c)}.npz")); xg=ben['x_gl'];wg=ben['w_gl']
psiref=signfix(ben['psi_gl'].copy()); lref=ben['lam']; reg=ben['regime']; nb=int(ben['nmodes'])
sol=np.load(os.path.join(PIL,f"sol_n0_c{int(c)}_P{P}.npz")); psiN=sol['psi_gl']; lamN=sol['lam']
print(f"c={c} P={P}: per-mode individual overlap and eigenvalue error")
print(f"{'m':>2} {'lam_N':>9} {'lam_ref':>9} {'|dlam|':>9} {'regime':>9} {'|<psiN,ref>|':>12}")
for m in range(nb):
    o=abs(np.sum(psiN[:,m]*psiref[:,m]*xg*wg))
    rg={0:'plateau',1:'separated',2:'tail'}[int(reg[m])]
    print(f"{m:>2} {lamN[m]:9.6f} {lref[m]:9.6f} {abs(lamN[m]-lref[m]):9.1e} {rg:>9} {o:12.6f}")

# subspace overlap for the plateau cluster (regime 0) — should be ~1 even though individuals rotate
pla=np.where(reg==0)[0]
if len(pla):
    A=psiref[:,pla]*np.sqrt(xg*wg)[:,None]; Bm=psiN[:,pla]*np.sqrt(xg*wg)[:,None]
    # orthonormalize columns (they are ~orthonormal already) and SVD cross-gram
    Qa,_=np.linalg.qr(A); Qb,_=np.linalg.qr(Bm)
    sv=np.linalg.svd(Qa.T@Qb,compute_uv=False)
    print(f"\nplateau subspace (dim {len(pla)}): principal-angle sines = "
          f"{np.sqrt(np.clip(1-sv**2,0,1))}")
    print(f"  max sin(theta) = {np.sqrt(np.clip(1-sv.min()**2,0,1)):.2e}  (small => subspace captured)")
