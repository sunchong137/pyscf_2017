#!/usr/local/bin/python
cimport numpy as np
import numpy as np
import logger as log
"""
Modules for calculating finite temperature properties of the system.
Adeline C. Sun Mar. 28 2016 <chongs0419@gmail.com>
Adeline C. Sun Apr. 11 Made some corrections. Added the RDM function
"""

def Tri_diag(a1, b1):
    cdef int l = len(a1)
    cdef np.ndarray[double, ndim=2] mat = np.zeros((l,l))
    cdef np.ndarray[double, ndim=1] e = np.empty(l)
    cdef np.ndarray[double, ndim=2] w = np.zeros((l,l))
    mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
    e, w = np.linalg.eigh(mat)
    # w[:, i] is the ith eigenvector
    return e, w


def ftlan_E1c(hop, v0, double T, int m=50, double Min_b=10e-10, int Min_m=5, double kB=1., norm = np.linalg.norm):
    r"""1 cycle Lanczos... you need to generate a random number first and then
        pass it into this function, iteration is built outside.

    Calculating the energy of the system at finite temperature.
    args:
        hop     - function to calculate $H|v\rangle$
        v0      - random initial vector
        T       - temperature
        kB      - Boltzmann const
        m       - size of the Krylov subspace
        Min_b   - min tolerance of b[i]
        Min_m   - min tolerance of m
    return:
        if succeed: Energy
        if b[0]=0 : 0
    """

    cdef double beta = 1./(T * kB)
    cdef double E = 0.
    cdef double Z = 0.
    cdef int lci = len(v0)
    cdef int i
    cdef np.ndarray[double, ndim=1] Hv=np.empty(lci)
    cdef np.ndarray[double, ndim=1] v1=np.empty(lci)
    cdef np.ndarray[double, ndim=1] v2=np.empty(lci)
    a, b = [], []
    v0 = v0/norm(v0).real
    Hv = hop(v0)
    a.append(v0.dot(Hv))
    v1 = Hv - a[0] * v0
    b.append(norm(v1))
    if b[0] < Min_b:
        return 0

    v1 = v1/b[0]
    Hv = hop(v1)
    a.append(v1.dot(Hv))

    for i in range(1, m - 1):
        v2 = Hv - b[i - 1] * v0 - a[i] * v1
        b.append(norm(v2))
        if abs(b[i]) < Min_b:
            b.pop()
            break

        v2 = v2/b[i]
        Hv = hop(v2)
        a.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()
    
    a = np.asarray(a)
    b = np.asarray(b)

    cdef int l = len(a)
    cdef np.ndarray[double, ndim=1] eps = np.empty(l)
    cdef np.ndarray[double, ndim=1] exp_eps = np.empty(l)
    cdef np.ndarray[double, ndim=2] phi = np.empty((l,l))
    eps, phi = Tri_diag(a, b)
#    l = len(eps)
#    Eo = eps[0]
#    eps = eps-Eo
    exp_eps = np.exp(-beta * eps)
    E = np.sum(exp_eps * eps * phi[0, :]**2.).real
    Z = np.sum(exp_eps * phi[0, :]**2.).real

    return E, Z


def ftlan_rdm1s1c(qud, hop, v0, double T, int norb, int m=60, double Min_b=1e-9, int Min_m=5, double kB=1., norm=np.linalg.norm):
    r'''1 step lanczos
    return the 1-particle reduced density matrix
    at finite temperature $T$.
    args:
        qud    - function for getting the matrix repr
                 of the RDM of given two vectors
        hop    - function to get $H|v\rangle$
        v0     - initial vector (normalized)
        T      - temperature
        kB     - Boltzmann const
        m      - size of the Krylov subspace
        Min_b  - min tolerance of b[i] 
        Min_m  - min tolerance of m
    return:
        RDMs of spin a and b
    '''
#    rdma, rdmb = qud(v0, v0)*0. #so we don't need norb
#    log.info("RDM1s -- 1 cycle\n")
    cdef double beta = 1./(kB*T)
    cdef double Z = 0.
    cdef int lci = len(v0)
    cdef int i
    cdef np.ndarray[double, ndim=1] v1 = np.empty(lci)
    cdef np.ndarray[double, ndim=1] v2 = np.empty(lci)
    cdef np.ndarray[double, ndim=1] Hv = np.empty(lci)
    cdef np.ndarray[double, ndim=2] rdm1 = np.empty((norb, norb))
    
    a, b = [], []
    krylov = []
    v0 = v0/norm(v0).real
    krylov.append(v0)
    Hv = hop(v0)
    a.append(v0.dot(Hv))
    v1=Hv-a[0]*v0
    b.append(norm(v1))
    if b[0] < Min_b:
        return None, None
    v1 = v1/b[0]
    Hv = hop(v1)
    a.append(v1.dot(Hv))
    krylov.append(v1)
    for i in range(1, int(m-1)):
        v2 = Hv-b[i-1]*v0-a[i]*v1
        b.append(norm(v2))
        if abs(b[i])<Min_b:
            if i < Min_m:
                return None, None
            b.pop()
            break
        v2 = v2/b[i]
        krylov.append(v2)
        Hv = hop(v2)
        a.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()
    
    a, b = np.asarray(a), np.asarray(b)
    krylov = np.asarray(krylov)
    cdef int l = len(a)
    cdef np.ndarray[double,ndim=1] eps = np.empty(l)
    cdef np.ndarray[double,ndim=1] phi = np.empty((l,l))
    cdef np.ndarray[double,ndim=1] coef=np.empty(l)
    cdef np.ndarray[double,ndim=2] estate=np.empty((lci, l))

    eps, phi = Tri_diag(a, b)
    estate = krylov.T.dot(phi)
    # estate[:, i] is the ith eigenstate
    coef = np.exp(-beta*eps/2.)*phi[0, :]
    eps = np.exp(-beta*eps)
    Z = np.sum(eps*phi[0, :]**2.)
    psi = estate.dot(coef.T)
    rdm1 = qud(psi)
    return rdm1, Z

def ftlan_rdm12s1c(qud, hop, v0, double T, int norb, int m=60, double Min_b=1e-9, int Min_m=5, double kB=1.0, norm=np.linalg.norm):
    r'''1 step lanczos
    returns the 1-particle and 2-particle reduced
    density matrix at temperature $T$
    args:
        qud    - function for getting the matrix repr
                 of the RDM of given two vectors
        hop    - function to get $H|v\rangle$
        v0     - initial vector (normalized)
        T      - temperature
        kB     - Boltzmann const
        m      - size of the Krylov subspace
        Min_b  - min tolerance of b[i] 
        Min_m  - min tolerance of m
    return:
        1- and 2-RDMs of spin a and b
        E
    '''
    cdef double beta = 1./(kB*T)
    cdef double E = 0.
    cdef double Z = 0.
    cdef int i
    cdef int lci = len(v0)

    cdef np.ndarray[double,ndim=2] rdm1 = np.zeros((norb, norb))
    cdef np.ndarray[double,ndim=4] rdm2 = np.zeros((norb, norb, norb, norb))
    cdef np.ndarray[double,ndim=1] v1 = np.empty(lci)
    cdef np.ndarray[double,ndim=1] v2 = np.empty(lci)
    cdef np.ndarray[double,ndim=1] Hv = np.empty(lci)
    
    # generate the Krylov space
    a, b = [], []
    krylov = []
    v0 = v0/norm(v0).real
    krylov.append(v0)
    Hv = hop(v0)
    a.append(v0.dot(Hv))
    v1=Hv-a[0]*v0
    b.append(norm(v1))
    if b[0] < Min_b:
        log.info("The Lanczos procedure exited because b[0] is zero!\n")
        return None, None, None
    v1 = v1/b[0]
    Hv = hop(v1)
    a.append(v1.dot(Hv))
    krylov.append(v1)
    for i in range(1, int(m-1)):
        v2 = Hv-b[i-1]*v0-a[i]*v1
        b.append(norm(v2))
        if abs(b[i])<Min_b:
            if i < Min_m:
                return None, None
            b.pop()
            break
        v2 = v2/b[i]
        krylov.append(v2)
        Hv = hop(v2)
        a.append(v2.dot(Hv))
        v0 = v1.copy()
        v1 = v2.copy()
    
    a, b = np.asarray(a), np.asarray(b)

    cdef int l = len(a)
    cdef np.ndarray[double,ndim=1] eps=np.empty(l)
    cdef np.ndarray[double,ndim=1] coef=np.empty(l)
    cdef np.ndarray[double,ndim=2] phi=np.empty((l, l))
    cdef np.ndarray[double,ndim=2] estate=np.empty((lci, l))

    krylov = np.asarray(krylov).real
    eps, phi = Tri_diag(a, b)
    estate = krylov.T.dot(phi).real
    coef = np.exp(-beta*eps/2.)*phi[0, :].real
    exp_eps = np.exp(-beta*eps)
    E = np.sum(exp_eps * eps * phi[0, :]**2.).real
    Z = np.sum(exp_eps*phi[0, :]**2.).real
    psi = estate.dot(coef.T)
#    log.section("partition function: %10.10f"%Z)
    rdm1, rdm2 = qud(psi)
    return rdm1, rdm2, E, Z.real

if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.fci import cistring
    from pyscf.fci import direct_spin1

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    ne = mol.nelectron - 2
    nelec = (nelec//2, nelec-nelec//2)
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
     
    T = 1.
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = np.random.randn(na*nb)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud1(v1, v2):
        dma, dmb = direct_spin1.trans_rdm1s(v1, v2, norb, nelec)
        return dma, dmb

    def qud2(v1, v2):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = direct_spin1.trans_rdm12s(v1, v2, norb, nelec)
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

    dma, dmb, Z1 = ftlan_rdm1s1c(qud1, hop, ci0, T, norb, m = 20)
    (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb), Z2 = ftlan_rdm12s1c(qud2, hop, ci0, T, norb, m=20)
    print "partition function difference: ", Z2-Z1
    print "Z1 = %6.6f, Z2 = %6.6f\n"%(Z1, Z2)
    print "rdm1 from v12:\n%s\n"%(rdm1a/Z2)
    print "rdm1 from v1:\n%s\n"%(dma/Z1)
    print "dm1a:", np.linalg.norm(dma-rdm1a)
    print "dm1b:", np.linalg.norm(dmb-rdm1b)
    print "dm1a/Z:", np.linalg.norm(dma/Z1-rdm1a/Z1)
    print "dm1b/Z:", np.linalg.norm(dmb/Z1-rdm1b/Z2)
