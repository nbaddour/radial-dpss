"""
Task 7.6 — the 'commuting excess' that isolates the structural contrast.

For a symmetric matrix M with simple spectrum, the symmetric matrices commuting with M are
exactly poly(M) (dimension P). A GENERIC such matrix (no special / bispectral structure) has
    dim( poly(M) ∩ {symmetric, bandwidth <= p} ) = max(1, P - C(P-p, 2)),   C(m,2)=m(m-1)/2,
because each entry outside the band imposes one independent linear constraint, and I always
survives. A commuting operator of UNUSUALLY low bandwidth (a Slepian-type "lucky" operator)
shows up as an EXCESS:  e_p = dim_measured(p) - dim_generic(p) > 0.

We compute e_p for the 1D DPSS prolate matrix (excess expected at p=1: Slepian's tridiagonal)
and contrast with the finite radial prolate matrix (excess expected = 0 everywhere).
1D is done in double precision (its commuting directions are exact -> machine zero, cleanly
separated from the rest); the radial measured dims are taken from the high-precision run
(task_7_6_hp_confirm.py) and only the generic baseline is recomputed here.
"""
import numpy as np
from math import comb

np.set_printoptions(linewidth=140)


def dpss_prolate(M, Wb):
    P = np.empty((M, M))
    for a in range(M):
        for b in range(M):
            P[a, b] = 2*Wb if a == b else np.sin(2*np.pi*Wb*(a-b))/(np.pi*(a-b))
    return P


def banded_sym_basis(P, p):
    B = []
    for i in range(P):
        for j in range(i, min(i+p+1, P)):
            E = np.zeros((P, P))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0
            B.append(E)
    return B


def measured_dim(M, p, tol=1e-10):
    P = M.shape[0]
    L = np.array([(M@E - E@M).ravel() for E in banded_sym_basis(P, p)]).T
    sv = np.linalg.svd(L, compute_uv=False)/np.linalg.norm(M, 2)
    return int(np.sum(sv < tol))


def generic_dim(P, p):
    if p >= P-1:
        return P
    return max(1, P - comb(P-p, 2))


print("GENERIC baseline  dim_p = max(1, P - C(P-p,2)) :")
for P in (4, 5, 6, 9):
    print(f"  P={P}: " + ", ".join(f"p={p}:{generic_dim(P,p)}" for p in range(P)))

print("\n1D DPSS prolate matrix — measured vs generic (EXCESS = Slepian operator):")
for (M, Wb) in [(6, 0.25), (9, 0.25), (12, 0.2)]:
    Pm = dpss_prolate(M, Wb)
    meas = {p: measured_dim(Pm, p) for p in range(M)}
    gen = {p: generic_dim(M, p) for p in range(M)}
    exc = {p: meas[p]-gen[p] for p in range(M)}
    print(f"  M={M},W={Wb}: measured={meas}")
    print(f"            generic ={gen}")
    print(f"            EXCESS  ={exc}   <-- e_1={exc[1]} (Slepian tridiagonal => +1)")

print("\nRadial finite prolate matrix — measured (from HP run) vs generic:")
# measured dims copied from task_7_6_hp_confirm.py high-precision output (Candidate B):
radial_meas = {
    ("n0", 5): {0: 1, 1: 1, 2: 3, 3: 4},
    ("n0", 6): {0: 1, 1: 1, 2: 2, 3: 4, 4: 5},
    ("n1", 6): {0: 1, 1: 1, 2: 2, 3: 4, 4: 5},
    ("n2", 6): {0: 1, 1: 1, 2: 2, 3: 4, 4: 5},
}
for (tag, N), meas in radial_meas.items():
    P = N-1
    gen = {p: generic_dim(P, p) for p in range(P)}
    exc = {p: meas[p]-gen[p] for p in range(P)}
    print(f"  {tag} N={N} (P={P}): measured={meas}")
    print(f"                 generic ={gen}")
    print(f"                 EXCESS  ={exc}   <-- e_1={exc[1]} (no commuting tridiagonal)")
