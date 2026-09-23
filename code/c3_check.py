import numpy as np
from scipy.special import jn_zeros
print("=== Item 2: is 1-sqrt(1-d^2) <= d^2/2 ? ===")
for d in [0.1,0.3,0.5]:
    lhs=1-np.sqrt(1-d*d); print(f"  d={d}: 1-sqrt(1-d^2)={lhs:.6f}   d^2/2={d*d/2:.6f}   d^2={d*d:.6f}   my bound holds? {lhs<=d*d/2}   critic's? {lhs<=d*d}")
print("\n=== Item 3: eps*_N vs zeta, all orders ===")
print(f"{'n':>2}{'N':>4} | {'eps*_N':>12} | {'zeta':>12} | {'excess %':>9} | violated?")
for n,N in [(0,26),(0,32),(0,45),(1,32),(2,31)]:
    jz=jn_zeros(n,max(N+2,80))
    eps=(N+n/2+0.25)*np.pi-jz[N-1]
    zeta=0.5*np.diff(jz[:N]).min()        # min over i<N as printed
    print(f"{n:>2}{N:>4} | {eps:>12.9f} | {zeta:>12.9f} | {100*(eps-zeta)/zeta:>8.3f}% | {'YES' if eps>zeta else 'no'}")
print("\n=== manuscript's quoted numbers ===")
jz0=jn_zeros(0,80); jz2=jn_zeros(2,80)
print(f"  n=0,P=32: quoted 1.5693, actual {(32+0.25)*np.pi-jz0[31]:.6f}")
print(f"  n=2,P=26: quoted 1.5931, actual {(26+1+0.25)*np.pi-jz2[25]:.6f}   <-- but Table 8 n=2 row has P=31")
print(f"  n=2,P=31 (the actual table row): {(31+1+0.25)*np.pi-jz2[30]:.6f}")
