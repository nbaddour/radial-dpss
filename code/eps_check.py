import numpy as np
from scipy.special import jv, jn_zeros
# At closure c=j_{n,N}, R=1: nodes r_k=j_{n,k}/j_{n,N}; Phi_{k,l}=phi_{n,l}(r_k)
# Standard FB quadrature weights W_k = 2/(j_{n,N}^2 J_{n+1}(j_{n,k})^2)
for n in [0,1]:
    for N in [6,9,13,21,31]:
        jz=jn_zeros(n,N); jk=jz[:N-1]; jN=jz[N-1]; P=N-1
        Phi=np.sqrt(2)*jv(n,np.outer(jk,jk)/jN)/jv(n+1,jk)[None,:]
        W=2.0/(jN**2*jv(n+1,jk)**2)
        G=Phi.T@(W[:,None]*Phi)                    # quadrature Gram of the FB basis
        epsN=np.abs(G-np.eye(P)).max()
        Y=2*jv(n,np.outer(jk,jk)/jN)/(jN*np.abs(jv(n+1,jk))[:,None]*np.abs(jv(n+1,jk))[None,:])
        dht=np.abs(Y@Y-np.eye(P)).max()
        print(f"n={n} N={N:3d} P={P:3d}:  eps_N=|Phi^T W Phi - I| = {epsN:.3e}   DHT defect |Y^2-I| = {dht:.3e}   ratio {epsN/dht:.4f}")
