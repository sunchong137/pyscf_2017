#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Adeline C. Sun <chongs@princeton.edu>
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import direct_uhf
import ftlanczos as flan
import smpl as ftsmpl

def kernel_ft(h1e, g2e, norb, nelec, T, uhf=False, m=50, nsamp=1000, Tmin=1e-3):
    '''E at temperature T
       using random sampling.
    '''
    if T < Tmin:
        e, c = direct_spin1.kernel(h1e, g2e, norb, nelec)
        return e

    if uhf:
        h2e = direct_uhf.absorb_h1e(h1e, g2e, norb, nelec, .5)
    else:
        h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.randn(n1, n2)
        return ci0.reshape(-1)
    def hop(c):
        if uhf:
            hc = direct_uhf.contract_2e(h2e, c, norb, nelec)
        else:
            hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    E = flan.ftlan_E(hop, vecgen, T, m, nsamp)
    return E

def kernel_ft_smpl(h1e, g2e, norb, nelec, T, uhf=False, vecgen=0, m=50,\
 nsmpl=2000, Tmin=1e-3, nrotation=200, eprofile=None, verbose=0, **kwargs):
    '''
        E at temperature T.
        using importance sampling.
    '''
    if T < Tmin:
        e, c = direct_spin1.kernel(h1e, g2e, norb, nelec)
        return e

    disp = numpy.exp(T) * 0.5 # displacement
    #NOTE We want ar to remain ~0.5 for any T, so displacement needs
    #NOTE to depend on T

#    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if uhf:
        h2e = direct_uhf.absorb_h1e(h1e, g2e, norb, nelec, .5)
    else:
        h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)

    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = numpy.random.randn(na, nb)
    if uhf:
        hdiag = direct_uhf.make_hdiag(h1e, g2e, norb, nelec)
    else:
        hdiag = direct_spin1.make_hdiag(h1e, g2e, norb, nelec)
    hdiag = hdiag-hdiag[0]

    def hop(c):
        if uhf:
            hc = direct_uhf.contract_2e(h2e, c, norb, nelec)
        else:
            hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    E = ftsmpl.ft_ismpl_E(hop, ci0, T,\
            nsamp=nsmpl, dr=disp, genci=vecgen, nrot=nrotation,\
            feprof=eprofile, Hdiag=hdiag, verbose=verbose)
    # ar is the acceptance ratio
    return E

  
def ft_rdm1s(h1e, g2e, norb, nelec, T, m=50, nsamp=40, Tmin=1e-3):
    '''rdm of spin a and b at temperature T
    '''
    if T < Tmin:
       e, c = kernel(h1e, g2e, norb, nelec)
       rdma, rdmb = direct_spin1.make_rdm1s(c, norb, nelec)
       return rdma, rdmb

    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.randn(n1, n2)
#        ci0[0, 0] = 1.
        return ci0.reshape(-1)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    def qud(v1, v2):
        dma, dmb = direct_spin1.trans_rdm1s(v1, v2, norb, nelec)
        return dma, dmb

#    rdma, rdmb = flan.ht_rdm1s(qud, hop, vecgen, T, norb, m, nsamp)
    rdma, rdmb = flan.ftlan_rdm1s(qud, hop, vecgen, T, norb, m, nsamp)
    return rdma, rdmb

def ft_rdm1(h1e, g2e, norb, nelec, T, m=50, nsamp=40):
    rdma, rdmb = ft_rdm1s(h1e, g2e, norb, nelec, T, m, nsamp)
    return rdma+rdmb
 
def rdm1s_ft_smpl(h1e, g2e, norb, nelec, T, uhf=False, vecgen=0,\
 m=50, nsmpl=2000, Tmin=1e-3, nrotation=200, verbose=0, **kwargs):
    '''
        RDM1a, RDM1b at temperature T.
        using importance sampling.
    '''
    if T < Tmin:
        log.info("\nGROUND STATE solver is used~\n")
        e, c = kernel(h1e, g2e, norb, nelec)
        rdm1a, rdm1b = direct_spin1.make_rdm1s(c, norb, nelec)
        return rdm1a, rdm1b

#    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if uhf:
        h2e = direct_uhf.absorb_h1e(h1e, g2e, norb, nelec, .5)
    else:
        h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)

    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
   
#    hdiag = direct_spin1.make_hdiag(h1e, g2e, norb, nelec)
    if uhf:
        hdiag = direct_uhf.make_hdiag(h1e, g2e, norb, nelec)
    else:
        hdiag = direct_spin1.make_hdiag(h1e, g2e, norb, nelec)

    hdiag = hdiag-hdiag[0]
    disp=numpy.exp(T)*0.5
    ci0 = numpy.random.randn(na*nb)
    ci0 = ci0/numpy.linalg.norm(ci0)

    def hop(c):
        if uhf:
            hc = direct_uhf.contract_2e(h2e, c, norb, nelec)
        else:
            hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud(v1, v2):
        dma, dmb = direct_spin1.trans_rdm1s(v1, v2, norb, nelec)
        return dma, dmb

    rdm1a, rdm1b = ftsmpl.ft_ismpl_rdm1s(qud, hop, ci0, T, norb,\
        genci=vecgen, nrot=nrotation, nsamp=nsmpl, M=m, \
        dr=disp, Hdiag=hdiag, verbose=verbose)
    return rdm1a, rdm1b

def rdm1_ft_smpl(h1e, g2e, norb, nelec, T, vecgen=0, m=50, nsmpl=2000, \
                nblk=10, Tmin=1e-3, nrotation=200):
    (rdm1a, rdm1b), ar = rdm1s_ft_smpl(h1e, g2e, norb, nelec, T, vecgen, m, nsmpl, Tmin, nrotation)
    return rdm1a+rdm1b, ar
   
def ft_rdm12s(h1e, g2e, norb, nelec, T, m=50, nsamp=40, Tmin=1e-3):
    '''rdm of spin a and b at temperature T
    '''
    if T < Tmin:
       e, c = kernel(h1e, g2e, norb, nelec)
       (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb) = direct_spin1.make_rdm1s(c, norb, nelec)
       return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb)

    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.randn(n1, n2)
#        ci0[0, 0] = 1.
        return ci0.reshape(-1)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    def qud(v1, v2):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = direct_spin1.trans_rdm12s(v1, v2, norb, nelec)
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

#    rdma, rdmb = flan.ht_rdm1s(qud, hop, vecgen, T, norb, m, nsamp)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = flan.ftlan_rdm12s(qud, hop, vecgen, T, norb, m, nsamp)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

def ft_rdm12(h1e, g2e, norb, nelec, T, m=50, nsamp=40, Tmin=1e-3):
    if T < Tmin:
        e, c = kernel(h1e, g2e, norb, nelec)
        (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb) = direct_spin1.make_rdm1s(c, norb, nelec)
        return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb)

    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    def vecgen(n1=na, n2=nb):
        ci0 = numpy.random.randn(n1, n2)
#        ci0[0, 0] = 1.
        return ci0.reshape(-1)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    def qud(v1, v2):
        dm1, dm2 = direct_spin1.trans_rdm12s(v1, v2, norb, nelec)
        return dm1, dm2

    dm1, dm2 = flan.ftlan_rdm12(qud, hop, vecgen, T, norb, m, nsamp)
    return dm1, dm2

def rdm12s_ft_smpl(h1e, g2e, norb, nelec, T, uhf=False, vecgen=0, \
        m=50, nsamp=200, Tmin=1e-3, nrotation=200, verbose=0, **kwargs):
    '''rdm of spin a and b at temperature T
    '''
    if T < Tmin:
       e, c = kernel(h1e, g2e, norb, nelec)
       (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb) = direct_spin1.trans_rdm12s(c, c, norb, nelec)
       return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb)

#    h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if uhf:
        h2e = direct_uhf.absorb_h1e(h1e, g2e, norb, nelec, .5)
    else:
        h2e = direct_spin1.absorb_h1e(h1e, g2e, norb, nelec, .5)

    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    
#    hdiag = direct_spin1.make_hdiag(h1e, g2e, norb, nelec)
    if uhf:
        hdiag = direct_uhf.make_hdiag(h1e, g2e, norb, nelec)
    else:
        hdiag = direct_spin1.make_hdiag(h1e, g2e, norb, nelec)

    hdiag = hdiag-hdiag[0]
    disp=numpy.exp(T)*0.5
    ci0 = numpy.random.randn(na*nb)
    ci0 = ci0/numpy.linalg.norm(ci0)

    def hop(c):
        if uhf:
            hc = direct_uhf.contract_2e(h2e, c, norb, nelec)
        else:
            hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud(v1, v2):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = direct_spin1.trans_rdm12s(v1, v2, norb, nelec)
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

#    rdma, rdmb = flan.ht_rdm1s(qud, hop, vecgen, T, norb, m, nsamp)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = ftsmpl.ft_ismpl_rdm12s(qud,\
     hop, ci0, T, norb, genci=vecgen, nrot=nrotation, \
    nsamp=nsamp, M=m, dr=disp, Hdiag=hdiag, verbose=verbose)

    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

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
    nelec = 8
    h1e = numpy.zeros((norb, norb))
    for i in range(norb):
        h1e[i, (i+1)%norb] = -1.
        h1e[i, (i-1)%norb] = -1.
    h1e[0, norb-1] = 1.
    h1e[norb-1, 0] = 1.
    eri = numpy.zeros((norb, norb, norb, norb)) 
    for i in range(norb):
        eri[i,i,i,i] = u
    E = [] 
    for T in numpy.linspace(0.01, 4., 40):
        _e =kernel_ft_smpl(h1e, eri, norb, nelec, T, m=50, nsmpl=10000) 
        print _e/norb
        E.append(_e/norb)
    E = np.asarray(E)
    numpy.save("1d_%d_U4.npy"%norb, E)
    exit()

    dm1a, dm1b = rdm1s_ft_smpl(h1e, eri, norb, nelec, T, vecgen=0, m=20, nsmpl=10)
    (dm1a_2, dm1b_2), _ = rdm12s_ft_smpl(h1e, eri, norb, nelec, T, vecgen=0, m=20, nsamp=10)
    print numpy.linalg.norm(dm1a-dm1a_2)
    print numpy.linalg.norm(dm1b-dm1b_2)

