import numpy as np
from scipy.special import jv, jn_zeros
# Verify the CORRECTED statement at several R and n: Phi^T W_phys Phi = D2 Y^2 D2, D1 W_phys D1 = I
print(f"{'n':>2}{'P':>4}{'R':>6} | {'|D1 Wphys D1 - I|':>18} | {'|Phi^T Wphys Phi - I|':>21} | {'|Y^2-I|':>10} | ratio")
for n in [0,1,2]:
    for P in [5,12,20]:
        for R in [0.5,1.0,2.0,7.3]:
            jz=jn_zeros(n,P+1); jk=jz[:P]; jN=jz[P]; K=jN/R
            r=jk/K
            Phi=(np.sqrt(2)/R)*jv(n,np.outer(r,jk)/R)/jv(n+1,jk)[None,:]
            Wp=2.0*R**2/(jN**2*jv(n+1,jk)**2)
            D1=(jN/(np.sqrt(2)*R))*np.abs(jv(n+1,jk))
            e1=np.abs(D1**2*Wp-1).max()
            G=Phi.T@(Wp[:,None]*Phi); eG=np.abs(G-np.eye(P)).max()
            Y=2*jv(n,np.outer(jk,jk)/jN)/(jN*np.abs(jv(n+1,jk))[:,None]*np.abs(jv(n+1,jk))[None,:])
            eY=np.abs(Y@Y-np.eye(P)).max()
            if R in (1.0,7.3):
                print(f"{n:>2}{P:>4}{R:>6.1f} | {e1:>18.2e} | {eG:>21.3e} | {eY:>10.3e} | {eG/eY:.4f}")
