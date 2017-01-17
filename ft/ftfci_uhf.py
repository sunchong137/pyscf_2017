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
import sys
sys.path.append("./utils")
import smpl_uhf as ftsmpl
import logger as log

def kernel_ft_smpl(h1e, g2e, norb, nelec, T, uhf=False, m=50,\
 nsmpl=2000, Tmin=1e-3, nrotation=200, eprofile=None, verbose=0, **kwargs):
    '''
        E at temperature T.
        using importance sampling.
    '''
    if T < Tmin:
        if uhf:
            e, c = direct_uhf.kernel(h1e, g2e, norb, nelec)
        else:
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
    E = ftsmpl.ft_ismpl_E(hop, ci0, T, nsamp=nsmpl, dr=disp, \
            feprof=eprofile, Hdiag=hdiag, verbose=verbose)
    # ar is the acceptance ratio
    return E

################################################################
def rdm1s_ft_smpl(h1e, g2e, norb, nelec, T, uhf=False, \
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
        nsamp=nsmpl, M=m, dr=disp, Hdiag=hdiag, verbose=verbose)
    return rdm1a, rdm1b

def rdm1_ft_smpl(h1e, g2e, norb, nelec, T, m=50, nsmpl=2000, \
                nblk=10, Tmin=1e-3):
    rdm1a, rdm1b = rdm1s_ft_smpl(h1e, g2e, norb, nelec, T, m, nsmpl, Tmin)
    return rdm1a+rdm1b
   
################################################################
def rdm12s_ft_smpl(h1e, g2e, norb, nelec, T, uhf=False, \
        m=50, nsamp=200, Tmin=1e-3, verbose=0, **kwargs):
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

    (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = ftsmpl.ft_ismpl_rdm12s(qud,\
     hop, ci0, T, norb, \
    nsamp=nsamp, M=m, dr=disp, Hdiag=hdiag, verbose=verbose)

    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

if __name__ == '__main__':

    u = 0.
    norb = 10
    nelec = 10
    h1ea = numpy.zeros((norb, norb))
    for i in range(norb):
        h1ea[i, (i+1)%norb] = -1.
        h1ea[i, (i-1)%norb] = -1.

    h1e = (h1ea, h1ea)
    eri_ab = numpy.ones((norb, norb, norb, norb))*1e-11
    eri_aa = eri_ab.copy()
    for i in range(norb):
        eri_ab[i,i,i,i] = u
    eri = (eri_aa, eri_ab, eri_aa)
    Tlist = []
    for i in range(2):
        Tlist.append(i*0.02)
    E = []
    for T in Tlist:
        _e =kernel_ft_smpl(h1e, eri, norb, nelec, T, m=5, \
                    nsmpl=10, uhf=True)/norb
        log.blank("%1.2f        %10.10f"%(T, _e))
        E.append([T, _e])

    E = numpy.asarray(E)
    numpy.save("10_%1.1f.npy"%u, E)
