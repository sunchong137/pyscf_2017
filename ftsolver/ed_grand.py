
'''
exact diagonalization solver with grand canonical statistics.
Chong Sun 08/07/17
'''

import numpy as np
from numpy import linalg as nl
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring
from scipy.optimize import minimize
import scipy
import sys
import os


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

    hdm1 = FD(h1e.copy(),nelec,T)
    ew, ev = diagH(h1e,g2e,norb,nelec,fcisolver)
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

    ews, evs = solve_spectrum(h1e,g2e,norb,fcisolver)
    mu = solve_mu(h1e,g2e,norb,nelec,fcisolver,T,mu0=0.,ews=ews,evs=evs)

    Z = 0.
    E = 0.
    RDM1*=0.
    RDM2*=0.

    for ne in range(0,2*norb+1):
        ndim = len(ews[ne]) 
        rdm1, rdm2 = [], []
        Z += np.sum(np.exp((-ews[ne]+mu*ne)/T))
        E += np.sum(np.exp((-ews[ne]+mu*ne)/T)*ews[ne])

        for i in range(ndim):
            dm1, dm2 = fcisolver.make_rdm12s(evs[ne][:,i].copy(),norb,ne)
            dm1 = np.asarray(dm1,dtype=np.complex128)
            dm2 = np.asarray(dm2,dtype=np.complex128)
            RDM1 += dm1*np.exp((ne*mu-ews[ne][i])/T)
            RDM2 += dm2*np.exp((ne*mu-ews[ne][i])/T)

    E    /= Z
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
#    return hdm1*2, RDM2, E

def solve_spectrum(h1e,h2e,norb,fcisolver):
    EW, EV = [], []
    for ne in range(0, 2*norb+1):
        ew, ev = diagH(h1e,h2e,norb,ne,fcisolver)
        EW.append(ew)
        EV.append(ev)
    return EW, EV

def solve_mu(h1e,h2e,norb,nelec,fcisolver,T,mu0=0., ews=None, evs=None):
    if ews is None:
        ews, evs = solve_spectrum(h1e,h2e,norb,fcisolver)

    def Ne_average(mu):
        N = 0
        Z = 0.
        for ne in range(0, 2*norb+1):
            N += ne * np.sum(np.exp((-ews[ne]+ne*mu)/T))
            Z += np.sum(np.exp((-ews[ne]+ne*mu)/T))
        return N/Z
    
    mu_n = minimize(lambda mu:(Ne_average(mu)-nelec)**2, mu0,tol=1e-9).x
    return mu_n

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

    eyemat = np.eye(ndim)

    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    Hmat = []
    for i in range(ndim):
        hc = hop(eyemat[i])
        Hmat.append(hc)

    Hmat = np.asarray(Hmat).T
    ew, ev = nl.eigh(Hmat)
    return ew, ev

def FD(h1e,nelec,T):
    #ew, ev = np.linalg.eigh(h1e)
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec


    ew, ev = np.linalg.eigh(h1e)
    if T < 1.e-3:
        beta = np.inf
    else:
        beta = 1./T
    def fermi(mu):
        return 1./(1.+np.exp((ew-mu)*beta))
    mu0 = 0.
    mu = scipy.optimize.minimize(lambda x:(np.sum(fermi(x))-neleca)**2, mu0, tol=1e-9).x

    eocc = fermi(mu)
    dm1 = np.dot(ev, np.dot(np.diag(eocc), ev.T.conj()))
#   dm1_n = np.zeros_like(dm1)
#   for i in range(dm1.shape[-1]):
#       dm1_n += np.einsum('i,j -> ij', ev[:,i],ev[:,i].conj())*eocc[i]

    return dm1

