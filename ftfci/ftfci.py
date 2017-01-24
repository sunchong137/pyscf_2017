#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Adeline C. Sun <chongs@princeton.edu>
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
from pyscf import fci
from pyscf.fci import fci_slow_spinless as spinless
from pyscf.ft_spinless.utils import smpl_spinless as ftsmpl
from pyscf.ft_spinless.utils import logger as log


def kernel_ft_smpl(h1e, g2e, norb, nelec, T, m=50,\
 nsmpl=20000, Tmin=1e-3, symm="rhf", **kwargs):
    if symm is "rhf":
        fcisolver=fci.direct_spin0
    elif symm is "soc":
        fcisolver=fci.fci_slow_spinless
    elif symm is "uhf":
        fcisolver=fci.direct_uhf

    if T < Tmin:
        return fcisolver.kernel(h1e, g2e, norb, nelec)[0]
    disp = numpy.exp(T) * 0.5 # displacement

    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec)
    ci0 = numpy.random.randn(na)
    ci0 = ci0/numpy.linalg.norm(ci0)
    hdiag = fcisolver.make_hdiag(h1e, g2e, norb, nelec)
    hdiag = hdiag-hdiag[0]

    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    E = ftsmpl.ft_ismpl_E(hop, ci0, T, nsamp=nsmpl, dr=disp, \
            Hdiag=hdiag)
    return E

def rdm1s_ft_smpl(h1e, g2e, norb, nelec, T, m=50, nsmpl=2000,\
                  Tmin=1e-3, symm="rhf", **kwargs):

    if symm is "rhf":
        fcisolver=fci.direct_spin0
    elif symm is "soc":
        fcisolver=fci.fci_slow_spinless
    elif symm is "uhf":
        fcisolver=fci.direct_uhf


    if T < Tmin:
        e, c = fcisolver.kernel(h1e, g2e, norb, nelec)
        rdm1 = fcisolver.make_rdm1(c, norb, nelec)
        return numpy.asarray(rdm1)

    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
    na = cistring.num_strings(norb, nelec)
   
    hdiag = fcisolver.make_hdiag(h1e, g2e, norb, nelec)

    hdiag = hdiag-hdiag[0]
    disp=numpy.exp(T)*0.5
    ci0 = numpy.random.randn(na)
    ci0 = ci0/numpy.linalg.norm(ci0)

    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud(v1):
        dm1 = fcisolver.trans_rdm1(v1, norb, nelec)
        return dm1

    rdm1 = ftsmpl.ft_ismpl_rdm1s(qud, hop, ci0, T, norb,\
        nsamp=nsmpl, M=m, dr=disp, Hdiag=hdiag, verbose=verbose)
    return rdm1

def rdm12s_ft_smpl(h1e, g2e, norb, nelec, T, \
        m=5, nsmpl=20, Tmin=1e-3, symm='rhf', **kwargs):

    if symm is "rhf":
        fcisolver=fci.direct_spin0
    elif symm is "soc":
        fcisolver=fci.fci_slow_spinless
    elif symm is "uhf":
        fcisolver=fci.direct_uhf

    if T < Tmin:
       e, c = fcisolver.kernel(h1e, g2e, norb, nelec)
       rdm1, rdm2 = fcisolver.make_rdm12(c, norb, nelec)
       return numpy.asarray(rdm1), numpy.asarray(rdm2)

    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5).real
    na = cistring.num_strings(norb, nelec)
    
    hdiag = fcisolver.make_hdiag(h1e, g2e, norb, nelec)

    hdiag = hdiag-hdiag[0]
    disp=numpy.exp(T)*0.5.real
    ci0 = numpy.random.randn(na)
    ci0 = ci0/numpy.linalg.norm(ci0)

    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1).real

    def qud(v1):
        dm1, dm2 = fcisolver.make_rdm12(v1, norb, nelec)
        return dm1.real, dm2.real 

    dm1, dm2, e = ftsmpl.ft_ismpl_rdm12s(qud,\
        hop, ci0, T, norb, nsamp=nsmpl, M=m, dr=disp, \
        Hdiag=hdiag, verbose=verbose)

    return dm1, dm2, e

def test_hubbard(norb, nelec, Tlist=None, fname=None, M=4, nsmpl=10):
    h1e = numpy.zeros((norb, norb))
    for i in range(norb):
        h1e[i, (i+1)%norb] = -1.
        h1e[i, (i-1)%norb] = -1.
    h1e[0, norb-1] = -1.
    h1e[norb-1, 0] = -1.
    eri = numpy.ones((norb, norb, norb, norb))*1e-11
    if Tlist is None:
        Tlist = []
        for i in range(2):
            Tlist.append(i*0.02)
    log.blank("# T      E")

    E = []
    for T in Tlist:
        e = kernel_ft_smpl(h1e, eri, norb, nelec, T=T, m=M, nsmpl=nsmpl, symm='soc') 
        E.append([T, e/norb])
        log.blank("%2.2f        %10.10f"%(T, e*2./norb))

    E = numpy.asarray(E)
    if fname is None:
        fname = "%d_U0f_fci.npy"%norb
    numpy.save(fname, E)

     
if __name__ == '__main__':

    norb = 10
    nelec = 5
    test_hubbard(norb, nelec)
