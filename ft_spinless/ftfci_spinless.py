#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Adeline C. Sun <chongs@princeton.edu>
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
from pyscf.fci import fci_slow_spinless as spinless
import ftlanczos_spinless as flan
import smpl_spinless as ftsmpl


def kernel_ft_smpl(h1e, g2e, norb, nelec, T, vecgen=0, m=5,\
 nsmpl=20, Tmin=1e-3, nrotation=200, eprofile=None, verbose=0, **kwargs):
    '''
        E at temperature T.
        using importance sampling.
    '''
    if T < Tmin:
        return spinless.kernel(h1e, g2e, norb, nelec)[0]
    disp = numpy.exp(T) * 0.5 # displacement
    #NOTE We want ar to remain ~0.5 for any T, so displacement needs
    #NOTE to depend on T

#    h2e = spinless.absorb_h1e(h1e, g2e, norb, nelec, .5)
    h2e = spinless.absorb_h1e(h1e, g2e, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec)
    ci0 = numpy.random.randn(na)
    ci0 = ci0/numpy.linalg.norm(ci0)
    hdiag = spinless.make_hdiag(h1e, g2e, norb, nelec)
    hdiag = hdiag-hdiag[0]

    def hop(c):
        hc = spinless.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    E = ftsmpl.ft_ismpl_E(hop, ci0, T,\
            nsamp=nsmpl, dr=disp, genci=vecgen, nrot=nrotation,\
            feprof=eprofile, Hdiag=hdiag, verbose=verbose)
    # ar is the acceptance ratio
    return E

def rdm1s_ft_smpl(h1e, g2e, norb, nelec, T, uhf=False, vecgen=0,\
 m=50, nsmpl=2000, Tmin=1e-3, nrotation=200, verbose=0, **kwargs):
    '''
        RDM1a, RDM1b at temperature T.
        using importance sampling.
    '''
    if T < Tmin:
        log.info("\nGROUND STATE solver is used~\n")
        e, c = spinless.kernel(h1e, g2e, norb, nelec)
        rdm1 = spinless.make_rdm1(c, norb, nelec)
        return rdm1

#    h2e = spinless.absorb_h1e(h1e, g2e, norb, nelec, .5)
    h2e = spinless.absorb_h1e(h1e, g2e, norb, nelec, .5)
    na = cistring.num_strings(norb, nelec)
   
#    hdiag = spinless.make_hdiag(h1e, g2e, norb, nelec)
    hdiag = spinless.make_hdiag(h1e, g2e, norb, nelec)

    hdiag = hdiag-hdiag[0]
    disp=numpy.exp(T)*0.5
    ci0 = numpy.random.randn(na)
    ci0 = ci0/numpy.linalg.norm(ci0)

    def hop(c):
        hc = spinless.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud(v1):
        dm1 = spinless.trans_rdm1(v1, norb, nelec)
        return dm1

    rdm1a, rdm1b = ftsmpl.ft_ismpl_rdm1s(qud, hop, ci0, T, norb,\
        genci=vecgen, nrot=nrotation, nsamp=nsmpl, M=m, \
        dr=disp, Hdiag=hdiag, verbose=verbose)
    return rdm1a, rdm1b

def rdm12s_ft_smpl(h1e, g2e, norb, nelec, T, vecgen=0, \
        m=5, nsmpl=20, Tmin=1e-3, nrotation=200, verbose=0, **kwargs):
    '''rdm of spin a and b at temperature T
    '''
    if T < Tmin:
       e, c = spinless.kernel(h1e, g2e, norb, nelec)
       rdm1, rdm2 = spinless.make_rdm12(c, norb, nelec)
       return rdm1, rdm2 #FIXME need to check with Qiming

    h2e = spinless.absorb_h1e(h1e, g2e, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec)
    
#    hdiag = spinless.make_hdiag(h1e, g2e, norb, nelec)
    hdiag = spinless.make_hdiag(h1e, g2e, norb, nelec)

    hdiag = hdiag-hdiag[0]
    disp=numpy.exp(T)*0.5
    ci0 = numpy.random.randn(na)
    ci0 = ci0/numpy.linalg.norm(ci0)

    def hop(c):
        hc = spinless.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud(v1):
        dm1, dm2 = spinless.make_rdm12(v1, norb, nelec)
        return dm1, dm2 #FIXME need to check with Qiming

#    rdma, rdmb = flan.ht_rdm1s(qud, hop, vecgen, T, norb, m, nsamp)
    dm1, dm2, e = ftsmpl.ft_ismpl_rdm12s(qud,\
     hop, ci0, T, norb, genci=vecgen, nrot=nrotation, \
    nsamp=nsmpl, M=m, dr=disp, Hdiag=hdiag, verbose=verbose)

    return dm1, dm2, e

if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

#   mol = gto.Mole()
#   mol.verbose = 0
#   mol.output = None
#   mol.atom = [
#       ['H', ( 1.,-1.    , 0.   )],
#       ['H', ( 0.,-1.    ,-1.   )],
#       ['H', ( 1.,-0.5   ,-1.   )],
#       ['H', ( 0.,-0.    ,-1.   )],
#       ['H', ( 1.,-0.5   , 0.   )],
#       ['H', ( 0., 1.    , 1.   )],
#   ]
#   mol.basis = 'sto-3g'
#   mol.build()

#   m = scf.RHF(mol)
#   m.kernel()
#   norb = m.mo_coeff.shape[1]
#   nelec = mol.nelectron - 2
#   ne = mol.nelectron - 2
#   nelec = (nelec//2, nelec-nelec//2)
#   h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
#   eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
#   eri = eri.reshape(norb,norb,norb,norb)
#    e1, ci0 = kernel(h1e, eri, norb, ne) #FCI kernel
    
#   E = []
#   data = numpy.loadtxt("ft_6_U4.dat") 
#   i = 0
#   for T in numpy.linspace(0.01, 4., 40):
#       E.append([T, data[i]])
#       i += 1
#   E = numpy.asarray(E)
#   numpy.save("1d_6_U4_fci.npy", E)
#   print E

#   exit()
    

    u = 4.
    norb = 8
    nelec = 4
    T = 0.1
    h1e = numpy.zeros((norb, norb))
    for i in range(norb):
        h1e[i, (i+1)%norb] = -1.
        h1e[i, (i-1)%norb] = -1.
    h1e[0, norb-1] = -1.
    h1e[norb-1, 0] = -1.
    eri = numpy.zeros((norb, norb, norb, norb)) 
    for i in range(norb):
        eri[i,i,i,i] = u
    _e =kernel_ft_smpl(h1e, eri, norb, nelec, T, m=40, nsmpl=10) 
#    print (_e/norb).real

#    dm1a, dm1b = rdm1s_ft_smpl(h1e, eri, norb, nelec, T, vecgen=0, m=20, nsmpl=10)
    dm1, dm2, e = rdm12s_ft_smpl(h1e, eri, norb, nelec, T, vecgen=0, m=20, nsmpl=10)
    print (_e - e)
