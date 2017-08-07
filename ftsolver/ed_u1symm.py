
'''
test the results of FT using exact diagonalization. (for small systems lol)
Adeline C. Sun Apr. 22 2016
'''


import numpy as np
from numpy import linalg as nl
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring
import sys
import os


def kernel_fted(h1e,g2e,norb,nelec,T,symm='RHF',Tmin=1.e-3,\
                dcompl=False,**kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    
    ew, ev = diagH(h1e,g2e,norb,nelec,fcisolver)
    if T < Tmin:
        return ew[0]
     
    Z = np.sum(np.exp(-ew/T))
    E = np.sum(np.exp(-ew/T)*ew)/Z 
    if not dcompl:
        E = E.real
    return E

def rdm12s_fted(h1e,g2e,norb,nelec,T,symm='RHF',Tmin=1.e-3,\
                dcompl=False,**kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    ew, ev = diagH(h1e,g2e,norb,nelec,fcisolver)
    ndim = len(ew) 
    rdm1, rdm2 = [], []

    RDM1, RDM2 = fcisolver.make_rdm12s(ev[:,0].copy(),norb,nelec)
    RDM1=np.asarray(RDM1,dtype=np.complex128)
    RDM2=np.asarray(RDM2,dtype=np.complex128)

    if T < Tmin:
        if symm is not 'UHF' and len(RDM1.shape)==3:
            RDM1 = np.sum(RDM1, axis=0)
            RDM2 = np.sum(RDM2, axis=0)
        if not dcompl:
            return RDM1.real, RDM2.real, ew[0].real
        return RDM1, RDM2, ew[0]

    Z = np.sum(np.exp(-ew/T))
    E = np.sum(np.exp(-ew/T)*ew)/Z 

    for i in range(ndim):
        dm1, dm2 = fcisolver.make_rdm12s(ev[:,i].copy(),norb,nelec)
        rdm1.append(np.asarray(dm1,dtype=np.complex128))
        rdm2.append(np.asarray(dm2,dtype=np.complex128))

#   rdm1 = np.asarray(rdm1)
#   rdm2 = np.asarray(rdm2)

#   RDM1 = np.sum(rdm1*np.exp(-1.j*ew/T))/Z
#   RDM2 = np.sum(rdm2*np.exp(-1.j*ew/T))/Z
    RDM1*=0.
    RDM2*=0.
    for i in range(ndim):
        RDM1 += rdm1[i]*np.exp(-ew[i]/T)
        RDM2 += rdm2[i]*np.exp(-ew[i]/T)

    RDM1 /= Z
    RDM2 /= Z

    if symm is not 'UHF' and len(RDM1.shape)==3:
        RDM1 = np.sum(RDM1, axis=0)
        RDM2 = np.sum(RDM2, axis=0)

    if not dcompl:
        E = E.real
        RDM1 = RDM1.real
        RDM2 = RDM2.real

    return RDM1, RDM2, E

def diagH(h1e,g2e,norb,nelec,fcisolver):
    '''
        exactly diagonalize the hamiltonian.
    '''
    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ndim = na*nb
    eyebas = np.eye(ndim)
    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    Hmat = []
    for i in range(ndim):
        hc = hop(eyebas[i])
        Hmat.append(hc)

    Hmat = np.asarray(Hmat).T
    ew, ev = nl.eigh(Hmat)
    return ew, ev

if __name__ == "__main__":
    # system to be tested
    from pyscf import fci
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
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    g2e = g2e.reshape(norb,norb,norb,norb)
   
    rdm1, rdm2, e = rdm12s_fted(h1e,g2e,norb,nelec,T=0)
    print e
    print rdm1

#   diagH(h1e,eri,norb,nelec,fci.direct_spin1)
#   gen_rdm12s(h1e, eri, norb, nelec, fci.direct_spin1, readfile=True )
#   (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb) = fted_rdm12s(h1e, eri, norb, nelec, 0.1, fci.direct_spin1, readfile=True)
#    print rdm2aa
