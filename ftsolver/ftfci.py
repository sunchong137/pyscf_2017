#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Adeline C. Sun <chongs@princeton.edu>
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
#from pyscf import fci
from pyscf.fci import direct_spin1 as fcisolver
from pyscf.ftsolver.utils import rsmpl as ftsmpl
from pyscf.ftsolver.utils import logger as log


def kernel_ft_smpl(h1e, g2e, norb, nelec, T, m=50,\
 nsmpl=20000, Tmin=1e-3, symm='SOC', **kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver


    if T < Tmin:
        return fcisolver.kernel(h1e, g2e, norb, nelec)[0]
    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)

    if symm is 'SOC':
        na = cistring.num_strings(norb, nelec)
        ci0 = numpy.random.randn(na)
    else:
        na = cistring.num_strings(norb, nelec//2)
        ci0 = numpy.random.randn(na*na)
 
    ci0 = ci0/numpy.linalg.norm(ci0)
    hdiag = fcisolver.make_hdiag(h1e, g2e, norb, nelec)
    hdiag = hdiag-hdiag[0]

    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    E = ftsmpl.ft_smpl_E(hop, ci0, T, nsamp=nsmpl)
    return E

def rdm1s_ft_smpl(h1e, g2e, norb, nelec, T, \
        m=50, nsmpl=20000, Tmin=1e-3, symm='RHF', **kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver

    if T < Tmin:
        e, c = fcisolver.kernel(h1e, g2e, norb, nelec)
        rdm1 = fcisolver.make_rdm1s(c, norb, nelec)
        return numpy.asarray(rdm1), e

    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if symm is 'SOC':
        na = cistring.num_strings(norb, nelec)
        ci0 = numpy.random.randn(na)
    else:
        na = cistring.num_strings(norb, nelec//2)
        ci0 = numpy.random.randn(na*na)
    
    ci0 = ci0/numpy.linalg.norm(ci0)

    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud(v1):
        dm1 = fcisolver.make_rdm1s(v1, norb, nelec)
        return dm1 

    dm1, e = ftsmpl.ft_smpl_rdm1s(qud,\
        hop, ci0, T, norb, nsamp=nsmpl,M=m)

    return dm1, e

def rdm12s_ft_smpl(h1e, g2e, norb, nelec, T, \
        m=50, nsmpl=20000, Tmin=1e-3, symm='RHF', **kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver

    if T < Tmin:
        e, c = fcisolver.kernel(h1e, g2e, norb, nelec)
        dm1, dm2 = fcisolver.make_rdm12s(c, norb, nelec)
        dm1 = numpy.asarray(dm1)
        dm2 = numpy.asarray(dm2)
    else:
        h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
        if symm is 'SOC':
            na = cistring.num_strings(norb, nelec)
            ci0 = numpy.random.randn(na)
        else:
            na = cistring.num_strings(norb, nelec//2)
            ci0 = numpy.random.randn(na*na)
        
        hdiag = fcisolver.make_hdiag(h1e, g2e, norb, nelec)
 
        hdiag = hdiag-hdiag[0]
        disp=numpy.exp(T)*0.5
        ci0 = ci0/numpy.linalg.norm(ci0)
 
        def hop(c):
            hc = fcisolver.contract_2e(h2e, c, norb, nelec)
            return hc.reshape(-1)
 
        def qud(v1):
            dm1, dm2 = fcisolver.make_rdm12s(v1, norb, nelec)
            return dm1, dm2
 
        
        dm1, dm2, e = ftsmpl.ft_smpl_rdm12s(qud,hop, ci0, T, norb, nsamp=nsmpl,M=m)

    if symm is 'UHF':
        return dm1, dm2, e
    elif len(dm1.shape) == 3:
        return numpy.sum(dm1, axis=0), numpy.sum(dm2, axis=0), e
    else:
        return dm1, dm2, e

def test_hubbard(norb, nelec, u, Tlist=None, fname=None, M=50, nsmpl=10000):
    h1e = numpy.zeros((norb, norb))
    for i in range(norb):
        h1e[i, (i+1)%norb] = -1.
        h1e[i, (i-1)%norb] = -1.
    h1e[0, norb-1] = -1.
    h1e[norb-1, 0] = -1.
    eri = numpy.ones((norb, norb, norb, norb))*1e-11
    for i in range(norb):
        eri[i,i,i,i] = u
    if Tlist is None:
        Tlist = []
        for i in range(20):
            Tlist.append(i*0.02)
    log.blank("# T      E")

    E = []
    for T in Tlist:
        rdm1,rdm2,e = rdm12s_ft_smpl(h1e, eri, norb, nelec, T=T, m=M, nsmpl=nsmpl, symm='RHF') 
        E.append([T, e/norb])
        log.blank("%2.2f        %10.10f"%(T, e/norb))
        #log.blank("%s"%rdm1.real)
#        log.blank("%s"%rdm2[0][0][0].real)

    E = numpy.asarray(E)
    if fname is None:
        fname = "%d_U%1.0f_fci.npy"%(norb,u)
    numpy.save(fname, E)

     
if __name__ == '__main__':

    norb = 12
    nelec = 12
    u = 4.0
    h1e = numpy.zeros((norb, norb))
    for i in range(norb):
        h1e[i,(i+1)%norb] = -1.0
        h1e[i,(i-1)%norb] = -1.0
    g2e = numpy.zeros((norb,)*4)
    for i in range(norb):
        g2e[i,i,i,i] = u

    e = kernel_ft_smpl(h1e, g2e, norb, nelec, T=0.0, symm='RHF')

