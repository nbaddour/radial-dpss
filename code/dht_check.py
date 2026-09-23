import numpy as np
from scipy.special import jv, jn_zeros
# Nodal value matrix at disc closure c = j_{n,N}:  Phi_{k,l} = phi_{n,l}(r_k), r_k = j_{n,k}/K
# With R=1, K=c=j_{n,N}:  r_k = j_{n,k}/j_{n,N}
for n in [0,1,2]:
    for N in [6,9,13,21]:
        jz = jn_zeros(n, N)
        P = N-1
        jk = jz[:P]                      # j_{n,1..P}
        jN = jz[N-1]                     # j_{n,N}
        Phi = np.sqrt(2)*jv(n, np.outer(jk, jk)/jN)/jv(n+1, jk)[None,:]   # rows k (nodes), cols l (basis)
        # Baddour DHT Eq.(37):  Y_{i,k} = 2 J_n(j_i j_k / j_N) / [ j_N |J_{n+1}(j_k)| |J_{n+1}(j_i)| ] is orthogonal
        Y = 2*jv(n, np.outer(jk,jk)/jN)/(jN*np.abs(jv(n+1,jk))[:,None]*np.abs(jv(n+1,jk))[None,:])
        orth = np.abs(Y@Y - np.eye(P)).max()
        # Phi = D1 * Y * D2 for diagonal D1,D2 ?
        D1 = np.sqrt(2)*jN*np.abs(jv(n+1,jk))/2.0
        D2 = np.abs(jv(n+1,jk))/jv(n+1,jk)
        rec = (D1[:,None]*Y)*D2[None,:]
        print(f"n={n} N={N} P={P}: |Y@Y - I|={orth:.2e}   |Phi - D1*Y*D2|={np.abs(Phi-rec).max():.2e}   cond(Phi)={np.linalg.cond(Phi):.4f}   |det Phi|={abs(np.linalg.det(Phi)):.3e}")
