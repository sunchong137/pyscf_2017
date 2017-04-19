#!/usr/local/bin/python
# author: Chong Sun <sunchong137@gmail.com>
import numpy as np
from pyscf.ftsolver.utils import logger as log


def Tri_diag(a1, b1):
    mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
    e, w = np.linalg.eigh(mat)
    return e, w

def ftlan_mu1c_time(H_prod_v, mu_prod_v, v0, T, time_step, m=40, Min_b=1e-15, Min_m=3, kB=1.0, norm = np.linalg.norm,E0=0.,**kwargs):
    # H_prod_v - H*c  function
    # mu_prod_v - mu*c  function
    beta = 1./(T * kB)
    a_v, b_v, a_w, b_w = [], [], [], []
    krylov_v, krylov_w = [], []
    v0 = v0/norm(v0)
    w0 = mu_prod_v(v0)
    w0 = w0/norm(w0)
    krylov_v.append(v0)
    krylov_w.append(w0)
    Hv = H_prod_v(v0)
    Hw = H_prod_v(w0)
    a_v.append(v0.dot(Hv))
    a_w.append(w0.dot(Hw))
    v1 = Hv - a_v[0] * v0
    w1 = Hw - a_w[0] * w0
    b_v.append(norm(v1))
    b_w.append(norm(w1))
    if b_v[0]<Min_b or b_w<Min_b:
        log.warning("Insufficient size of the Krylov space!")
        return 0., 1e-10

    v1 = v1/b_v[0]
    w1 = w1/b_w[0]
    krylov_v.append(v1)
    krylov_w.append(w1)
    Hv = H_prod_v(v1)
    Hw = H_prod_v(w1)
    a_v.append(v1.dot(Hv))
    a_w.append(w1.dot(Hw))

    for i in range(1, m-1):
        v2 = Hv - b_v[i-1] * v0 - a_v[i] * v1
        b_v.append(norm(v2))
        if abs(b_v[i])<Min_b:
            if i < Min_m:
                log.warning("Insufficient size of the Krylov space!")
                return 0., 1e-10
            b_v.pop()
            break
        v2 = v2/b_v[i]
        Hv = H_prod_v(v2)
        krylov_v.append(v2)
        a_v.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()

    for i in range(1, m-1):
        w2 = Hw - b_w[i-1] * w0 - a_w[i] * w1
        b_w.append(norm(w2))
        if abs(b_w[i]<Min_b):
            if i < Min_m:
                log.warning("Insufficient size of the Krylov space!")
                return 0., 1e-10
            b_w.pop()
            break
        w2 = w2/b_w[i]
        krylov_w.append(w2)
        Hw = H_prod_v(w2)
        a_w.append(w2.dot(Hw))
        w0 = w1.copy()
        w1 = w2.copy()
    
    a_v,b_v,a_w,b_w = np.asarray(a_v),np.asarray(b_v),np.asarray(a_w),np.asarray(b_w)
    krylov_v, krylov_w = np.asarray(krylov_v), np.asarray(krylov_w)
    eps_v, phi_v = Tri_diag(a_v, b_v)
    eps_w, phi_w = Tri_diag(a_w, b_w)
    estate_v = krylov_v.T.dot(phi_v)
    estate_w = krylov_w.T.dot(phi_w)
    coef_v = np.exp((-beta-1.j*time_step)*eps_v)*(phi_v[0,:].conj())
    coef_w = np.exp(-1.j*time_step*eps_w)*(phi_w[0,:].conj())
    Z = np.sum(np.exp(-beta*(eps_v))*(phi_v[0,:]*phi_v[0,:].conj()))
#    Eo = eps[0]
#    eps = eps-Eo
    bra = estate_v.dot(coef_v.T)
    ket = estate_w.dot(coef_w.T)
    C_t = bra.T.conj().dot(mu_prod_v(ket))

    return C_t, Z

def ftlan_mu1c_freq(H_prod_v, mu_prod_v, v0, T, freq_list, m=40, Min_b=1e-15, Min_m=3, kB=1.0, norm = np.linalg.norm,E0=0.,**kwargs):
    # H_prod_v - H*c  function
    # mu_prod_v - mu*c  function
    # a list of w grid i.e. (5.0, 20) means the range of w is (-5.0, 5.0), and 20 grids are made
    beta = 1./(T * kB)
    a_v, b_v, a_w, b_w = [], [], [], []
    krylov_v, krylov_w = [], []
    v0 = v0/norm(v0)
    w0 = mu_prod_v(v0)
    w0 = w0/norm(w0)
    krylov_v.append(v0)
    krylov_w.append(w0)
    Hv = H_prod_v(v0)
    Hw = H_prod_v(w0)
    a_v.append(v0.dot(Hv))
    a_w.append(w0.dot(Hw))
    v1 = Hv - a_v[0] * v0
    w1 = Hw - a_w[0] * w0
    b_v.append(norm(v1))
    b_w.append(norm(w1))
    if b_v[0]<Min_b or b_w<Min_b:
        log.warning("Insufficient size of the Krylov space!")
        return 0., 1e-10

    v1 = v1/b_v[0]
    w1 = w1/b_w[0]
    krylov_v.append(v1)
    krylov_w.append(w1)
    Hv = H_prod_v(v1)
    Hw = H_prod_v(w1)
    a_v.append(v1.dot(Hv))
    a_w.append(w1.dot(Hw))

    for i in range(1, m-1):
        v2 = Hv - b_v[i-1] * v0 - a_v[i] * v1
        b_v.append(norm(v2))
        if abs(b_v[i])<Min_b:
            if i < Min_m:
                log.warning("Insufficient size of the Krylov space!")
                return 0., 1e-10
            b_v.pop()
            break
        v2 = v2/b_v[i]
        Hv = H_prod_v(v2)
        krylov_v.append(v2)
        a_v.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()

    for i in range(1, m-1):
        w2 = Hw - b_w[i-1] * w0 - a_w[i] * w1
        b_w.append(norm(w2))
        if abs(b_w[i]<Min_b):
            if i < Min_m:
                log.warning("Insufficient size of the Krylov space!")
                return 0., 1e-10
            b_w.pop()
            break
        w2 = w2/b_w[i]
        krylov_w.append(w2)
        Hw = H_prod_v(w2)
        a_w.append(w2.dot(Hw))
        w0 = w1.copy()
        w1 = w2.copy()
    
    a_v,b_v,a_w,b_w = np.asarray(a_v),np.asarray(b_v),np.asarray(a_w),np.asarray(b_w)
    krylov_v, krylov_w = np.asarray(krylov_v), np.asarray(krylov_w)
    eps_v, phi_v = Tri_diag(a_v, b_v)
    eps_w, phi_w = Tri_diag(a_w, b_w)
    estate_v = krylov_v.T.dot(phi_v)
    estate_w = krylov_w.T.dot(phi_w)
    coef_v = np.exp(-beta*eps_v)*(phi_v[0,:].conj())
    coef_w = phi_w[0,:].conj()
    bra = np.einsum('ij,j -> ij', estate_v, coef_v)
    ket = np.einsum('ij,j -> ij', estate_w, coef_w)
    half_Nstep = freq_list[1]
    w_max = freq_list[0]
    lstep = float(w_max)/half_Nstep
    C_omega = np.zeros(2*half_Nstep,dtype=np.complex64)
    for i in range(len(krylov_v)):
        for j in range(len(krylov_w)):
            delta_e = eps_w[j]-eps_v[i]
            if abs(delta_e) > w_max:
                continue
            if delta_e < 0:
                idx = int(delta_e/lstep)+half_Nstep-1
            else:
                idx = int(delta_e/lstep)+half_Nstep
            C_omega[idx] = bra[:,i].T.conj().dot(mu_prod_v(ket[:,j]))
            print C_omega
    Z = np.sum(np.exp(-beta*(eps_v))*(phi_v[0,:]*phi_v[0,:].conj()))

    return C_omega, Z


#if __name__ == '__main__':

