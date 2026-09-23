import numpy as np, os
D="/sessions/pensive-keen-heisenberg/mnt/Radial DPSS/data/benchmark"
d=np.load(os.path.join(D,"cpswf_n0_c80.npz")); lam=d['lam']
print("n=0,c=80: per-mode (1-lam), neighbor gap, and the rotation error seen earlier")
rot=[48,9,5.2,9.7,6.4,8.9,4.9,6.9,5.8,7.0,4.5,4.1,6.2,5.4,10,23,47,6.5,4.4e-2,4.7e-4,6.5e-6,1.0e-7,1.9e-9,4.4e-11,1.4e-12,2.2e-12,9.6e-12,4.6e-10,9.9e-9,2.7e-7]
for m in range(len(lam)):
    gap=min([abs(lam[m]-lam[j]) for j in (m-1,m+1) if 0<=j<len(lam)])
    print(f"  m={m:2d}: 1-lam={1-lam[m]:.3e}  gap={gap:.2e}  rot_err~{rot[m]:.1e}")
