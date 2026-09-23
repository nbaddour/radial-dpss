"""
Task 11.3 — Solve the finite radial prolate eigenproblem and reconstruct the continuous approximants.

For each saved pilot matrix B (Task 11.2, data/pilot/Bmat_n0_c{c}_P{P}.npz):
  1. SOLVE  B c^{(m)} = lam_m c^{(m)}  (eigh; lam descending)  -> finite radial prolate sequences c^{(m)}
     (orthonormal eigenvectors, Task 6.9/7.3) and eigenvalues lam_m.
  2. RECONSTRUCT (Task 8.7 nondim formula):
        psi_N^{(m)}(x) = sqrt2 * sum_{k=1}^{P} [ c_k^{(m)} / J_1(j_{0,k}) ] J_0(j_{0,k} x),  x in [0,1],
     evaluated on the benchmark grids (x_gl for L^2 quadrature; x_unif for pointwise/plots).
  3. SIGN: deterministic, matching the Task 10.4 benchmark convention (psi > 0 at its max-|.| node).
Saved to data/pilot/sol_n0_c{c}_P{P}.npz. The formal benchmark error comparison is Task 11.4; here we
verify the reconstruction machinery (unit norm, orthonormality, boundary) and give an informal overlap
preview confirming correctness.
"""
import numpy as np, os, glob
from scipy.special import jv, jn_zeros

BEN="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
PIL="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/pilot"

def reconstruct(jz, cvec, x):
    # Phi[i,k] = sqrt2 J0(jz_k x_i)/J1(jz_k); psi[:,m] = Phi @ cvec[:,m]
    Phi = np.sqrt(2.0)*jv(0, np.outer(x, jz))/jv(1, jz)[None,:]
    return Phi @ cvec

def signfix(psi):
    for m in range(psi.shape[1]):
        j=np.argmax(np.abs(psi[:,m]))
        if psi[j,m]<0: psi[:,m]*=-1.0
    return psi

print("="*96)
print("TASK 11.3 — solve + reconstruct (n=0, F1 fixed-c).  Verify reconstruction machinery.")
print("="*96)
worst={'norm':0,'ortho':0,'bdry':0}
for c in [10.0,20.0,40.0]:
    ben=np.load(os.path.join(BEN,f"cpswf_n0_c{int(c)}.npz"))
    xg=ben['x_gl']; wg=ben['w_gl']; xu=ben['x_unif']
    nben=int(ben['nmodes']); psiref=signfix(ben['psi_gl'].copy()); regime=ben['regime']; lref=ben['lam']
    print(f"\n c={c}  (benchmark modes m=0..{nben-1}; regimes plateau/sep/tail="
          f"{int((regime==0).sum())}/{int((regime==1).sum())}/{int((regime==2).sum())})")
    for fp in sorted(glob.glob(os.path.join(PIL,f"Bmat_n0_c{int(c)}_P*.npz")), key=lambda s:int(s.split('_P')[1].split('.')[0])):
        d=np.load(fp); B=d['B']; P=int(d['P'])
        lam,V=np.linalg.eigh(B); idx=np.argsort(lam)[::-1]; lam=lam[idx]; V=V[:,idx]
        nm=min(nben,P)
        cvec=V[:,:nm]; jz=jn_zeros(0,P)
        psi_gl=signfix(reconstruct(jz,cvec,xg)); psi_unif=signfix(reconstruct(jz,cvec,xu))
        # (1) unit norm via benchmark GL quadrature
        norms=np.sum(psi_gl**2*(xg*wg)[:,None],axis=0)
        e_norm=np.max(np.abs(norms-1))
        # (2) orthonormality
        G=(psi_gl*(xg*wg)[:,None]).T@psi_gl
        e_ortho=np.max(np.abs(G-np.eye(nm)))
        # (3) boundary psi(1)=0  (x_unif[-1]=1)
        e_bdry=np.max(np.abs(psi_unif[-1,:]))
        # (informal) separated-mode overlap with benchmark (sign-aligned)
        sep=np.where(regime[:nm]==1)[0]
        ov=[]
        for m in sep:
            o=abs(np.sum(psi_gl[:,m]*psiref[:,m]*xg*wg)); ov.append(o)
        ovmin=min(ov) if ov else float('nan')
        np.savez_compressed(os.path.join(PIL,f"sol_n0_c{int(c)}_P{P}.npz"),
            n=0,c=c,P=P,lam=lam[:nm],cvec=cvec,jz=jz,x_gl=xg,w_gl=wg,x_unif=xu,
            psi_gl=psi_gl,psi_unif=psi_unif,regime=regime[:nm])
        print(f"   P={P:3d}: ||psi||-1={e_norm:.1e}  ortho={e_ortho:.1e}  psi(1)={e_bdry:.1e}  "
              f"sep-mode min|<psiN,psiref>|={ovmin:.5f}")
        for kk,vv in [('norm',e_norm),('ortho',e_ortho),('bdry',e_bdry)]: worst[kk]=max(worst[kk],vv)

print(f"\nWORST over all (c,P): ||psi||-1={worst['norm']:.1e}  orthonormality={worst['ortho']:.1e}  "
      f"boundary psi(1)={worst['bdry']:.1e}")
print("(Reconstruction unit-norm & orthonormal to quadrature accuracy; psi_N(1)=0 by FB basis. "
      "Separated-mode overlap -> 1 as P grows = correct reconstruction; formal errors in Task 11.4.)")
