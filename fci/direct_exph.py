#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com> Chong Sun <sunchong137@gmail.com>
#

'''
Electron phonon coupling

Ref:
    arXiv:1602.04195 
    Chem. Mater. 2017, 29, 2513-2520
'''

import numpy
from pyscf import lib
from pyscf.fci import cistring
import pyscf.fci

# Hamiltonian (a: excitons, b: phonons)
# H = \sum_{m,n} t_{mn}a_m^+ a_n                         -> hopping term
#    +\sum_{n,k} hpp_{nk} b_{nk}^+ b_{nk}               -> ph-ph int  
#    +\sum_{n,k} hpp_{nk}g_{nk} n_{n}(b_{nk}^+ + b_{nk}) -> e-p int   
# 
#                                  site-1            , ...,   site-N
#                                     v                         v
# ep_wfn, shape = (nstr,) + (nphonon+1,)*nmode + ...+(nphonon+1,)*nmode
#               = (nstr,) + ([nphonon+1]*nsite*nmode)
#
# For each site, there are nmode modes, and {0,1,...,nphonon} gives 
# nphonon+1 possible confs

# t for hopping, shape = (nsite,nsite)
# g for the exciton-phonon coupling: (nsite,nmode)
# hpp for phonon-phonon interaction: (nsite,nmode)

def contract_all(t, g, hpp, ci0, nsite, nexciton, nmode, nphonon):
    # no e-e interaction
    ci1  = contract_1e        (t  , ci0, nsite, nexciton, nmode, nphonon)
    ci1 += contract_ep        (g  , ci0, nsite, nexciton, nmode, nphonon)
    ci1 += contract_pp        (hpp, ci0, nsite, nexciton, nmode, nphonon)
    return ci1

def make_shape(nsite, nexciton, nmode, nphonon):
    # no spin freedom
    nstr = cistring.num_strings(nsite, nexciton)
    return (nstr,)+(nphonon+1,)*nmode*nsite

def contract_1e(h1e, fcivec, nsite, nexciton, nmode, nphonon):
    # excitons are treated as hard core phonons
    # [a_n, a_m] = 0 (n =\= m)
    # no sign needed

    link_index = cistring.gen_linkstr_index(range(nsite), nexciton)
    cishape = make_shape(nsite, nexciton, nmode, nphonon)

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in tab:
            fcinew[str1] += ci0[str0] * h1e[a,i]
    return fcinew.reshape(fcivec.shape)

def slices_for(psite_id, mode_id, nsite, nmode, nphonon):
#   slices = [slice(None,None,None)] * (2+nsite*nmode)  # +2 for electron indices
#   slices[2+psite_id*nmode+mode_id] = nphonon
    slices = [slice(None,None,None)] * (1+nsite*nmode)  # +2 for electron indices
    slices[1+psite_id*nmode+mode_id] = nphonon
    return tuple(slices)
def slices_for_cre(psite_id, mode_id, nsite, nmode, nphonon):
    return slices_for(psite_id, mode_id, nsite, nmode, nphonon+1)
def slices_for_des(psite_id, mode_id, nsite, nmode, nphonon):
    return slices_for(psite_id, mode_id, nsite, nmode, nphonon-1)

def contract_ep(g, fcivec, nsite, nexciton, nmode, nphonon):
    # N_alpha N_beta * \sum_{p} (p^+ + p)
    # N_alpha, N_beta are particle number operator, p^+ and p are phonon creation annihilation operator
    strs = numpy.asarray(cistring.gen_strings4orblist(range(nsite), nexciton))
    cishape = make_shape(nsite, nexciton, nmode, nphonon)
    nstr = cishape[0]
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)
    nbar = 0.0

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for i in range(nsite):
        mask = (strs & (1<<i)) > 0
        e_part = numpy.zeros((nstr,))
        e_part[mask] += 1
       #e_part[:] -= float(nexcitona+nexcitonb) / nsite
        for j in range(nmode):
            for ip in range(nphonon):
                slices1 = slices_for_cre(i, j, nsite, nmode, ip)
                slices0 = slices_for    (i, j, nsite, nmode, ip)
#               fcinew[slices1] += numpy.einsum('ij...,ij...->ij...', g[i,j]*phonon_cre[ip]*e_part, ci0[slices0])
#               fcinew[slices0] += numpy.einsum('ij...,ij...->ij...', g[i, j]*phonon_cre[ip]*e_part, ci0[slices1])
                fcinew[slices1] += numpy.einsum('i...,i...->i...', g[i,j]*phonon_cre[ip]*e_part, ci0[slices0])
                fcinew[slices0] += numpy.einsum('i...,i...->i...', g[i, j]*phonon_cre[ip]*e_part, ci0[slices1])
    return fcinew.reshape(fcivec.shape)

# Phonon-phonon coupling
def contract_pp(hpp, fcivec, nsite, nexciton, nmode, nphonon):
    cishape = make_shape(nsite, nexciton, nmode, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    t1 = numpy.zeros((nsite,nmode)+cishape)
    for psite_id in range(nsite):
        for mode_id in range(nmode):
            for i in range(nphonon):
                slices1 = slices_for_cre(psite_id, mode_id, nsite, nmode, i)
                slices0 = slices_for    (psite_id, mode_id, nsite, nmode, i)
                t1[(psite_id,mode_id)+slices0] += ci0[slices1] * phonon_cre[i]     # annihilation
    t1 = numpy.einsum('ij,ij...->ij...', hpp, t1)
    #t1 = lib.dot(hpp, t1.reshape(nsite,nmode,-1)).reshape(t1.shape)

    for psite_id in range(nsite):
        for mode_id in range(nmode):
            for i in range(nphonon):
                slices1 = slices_for_cre(psite_id, mode_id, nsite, nmode, i)
                slices0 = slices_for    (psite_id, mode_id, nsite, nmode, i)
                fcinew[slices1] += t1[(psite_id,mode_id)+slices0] * phonon_cre[i]  # creation
    return fcinew.reshape(fcivec.shape)

if __name__ == '__main__':
    nsite = 2
    nexciton = 1
    nmode = 2
    nphonon = 1

    t = numpy.zeros((nsite,nsite))
    idx = numpy.arange(nsite-1)
    t[idx+1,idx] = t[idx,idx+1] = -1.1
    g = numpy.ones((nsite,nmode))*0.5
    hpp = numpy.ones((nsite,nmode))*1.1
    cishape=make_shape(nsite, nexciton, nmode, nphonon)
    ld = numpy.prod(cishape)
    H = numpy.zeros((ld,ld))
    eyemat = numpy.eye(ld)
    for i in range(ld):
        H[:, i] = contract_all(t, g, hpp, eyemat[:,i], nsite, nexciton, nmode, nphonon)
    print "ham", H


