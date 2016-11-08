#!/usr/local/bin/python

import numpy as np
"""
Modules for calculating finite temperature properties of the system.
Adeline C. Sun Mar. 28 2016 <chongs0419@gmail.com>
"""

def FT_lanczos_E(hop, v0, T, m=60, ncycle=50,kB=1, Min_b=10e-5, norm = np.linalg.norm):
    r"""Calculating the energy of the system at finite temperature.
    args:
        hop     - function to calculate $H|v\rangle$
        v0      - random initial vector
        T       - temperature
        kB      - Boltzmann const
        n       - size of the Krylov subspace
        ncycle  - number of samples
        Min_b   - min tolerance of b[i]
        Min_m   - min tolerance of m
    return:
        Energy
    """
    def Tri_diag(a1, b1):
        mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
        e, w = np.linalg.eigh(mat)
        return e, w

    N = len(v0)
    beta = 1./(T * kB)
    E, Z = 0., 0.
    cnt = ncycle
    while(cnt > 0):
        a, b = [], []
        v0 = v0/norm(v0)
        Hv = hop(v0)
        a.append(v0.dot(Hv))
        v1 = Hv - a[0] * v0
        b.append(norm(v1))
        if b[0] < Min_b:
            E += np.exp(a[0]) * a[0]
            Z += np.exp(a[0])
            continue

        v1 = v1/b[0]
        Hv = hop(v1)
        a.append(v1.dot(Hv))

        for i in range(1, m - 1):
            v2 = Hv - b[i - 1] * v0 - a[i] * v1
            b.append(norm(v2))
            v2 = v2/b[i]
            if abs(b[i]) < Min_b:
                b.pop()
                break

            Hv = hop(v2)
            a.append(v2.dot(Hv))
            v0 = v1.copy()
            v1 = v2.copy()
    
        a = np.asarray(a)
        b = np.asarray(b)

        eps, phi = Tri_diag(a, b)
        exp_eps = np.exp(-beta * eps)
        for i in range(len(eps)):
            E += exp_eps[i] * eps[i] * phi[0, i]**2
            Z += exp_eps[i] * phi[0, i]**2
        
        cnt -= 1

    E = E/Z
    return E

# not finish!                
def LT_lanczos_E(hop, v0, T, kB=1, m=60, ncycle=100, Min_b=10e-5, Min_m=40, norm=np.linalg.norm):
    r"""
    Calculate the energy of the system at low temperature
    """
    def Tri_diag(a1, b1):
        mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
        e, w = np.linalg.eigh(mat)
        return e, w

    
