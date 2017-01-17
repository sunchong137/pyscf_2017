#!/usr/bin/env python
'''importance sampling of the Hilbert space
   Chong Sun 2016 Jun. 10
'''
cimport numpy as np
import numpy as np
import numpy.linalg as nl
from pyscf.ft_spinless import ftlanczos_spinless as _ftlan
from pyscf.fci import fci_slow_spinless as spinless
import logger as log# this is my logger
import os
import random

def ft_ismpl_E(hop, ci0, double T,\
         int nw=25, int nsamp=2000, int M=50, double dr=0.5,\
         save_prof=False, feprof=None,\
         Hdiag=None, int verbose=0):
    '''
       with 1 initial vector
       note that the probability and the energy is given by the same function (ftlan)
       ci0    ---- array, initial vector
       T      ---- float, temperature
       genci  ---- int, the way to generate new ci
       nrot   ---- int, steps of rotations on ci0, only used when genci = 1
       nw     ---- number of warm-up steps
       nsamp  ---- number of samples (initial vector generated)
       M      ---- Size of the Krylov space
       dr     ---- the size of the sampling box (or sphere)
       dtheta ---- step size of rotational sampling
       feprof ---- the file to store the energy profile
       Hdiag  ---- 1d array, the diagonal terms of H
    '''
    log.info("Importance sampling is used for impurity solver!")
    ci0 = ci0.reshape(-1).copy().real
    ci0 = ci0/np.linalg.norm(ci0)
 
    cdef double beta = 1./T  # kB = 1.
    cdef int lc = len(ci0)
    cdef int Nar #acceptance number
    cdef double e 
    cdef double E
    cdef double tp 
    cdef double tp_n 
    cdef double acc
    cdef double ar
    cdef double tmp
    cdef int i

    def ftE(v0, int m=20):
        if Hdiag is None:
            return _ftlan.ftlan_E1c(hop, v0, T, m=m)[1]
        else:
            _E = np.sum(v0**2*Hdiag)
            return np.exp(-beta*_E)
    cdef np.ndarray[double,ndim=1] ci = np.empty(lc)

    ftlan = _ftlan.ftlan_E1c  # function giving the probability and energy evaluated from the initial vector

    # Warm-up
    if verbose > 2:
        log.info("Warming up ......")
    tp = ftE(ci0)
#    tp = ftlan(hop, ci0, T, m=20)[1]
    for i in range(nw):
        ci = gen_nci(ci0, dr=dr)
        tp_n = ftE(ci)
#        tp_n = ftlan(hop, ci, T, m=20)[1]
        acc = tp_n/tp 
        if acc >= 1.:
            ci0 = ci.copy()
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp = tp_n
                Nar += 1
            else:
                continue
    if verbose > 2:
        log.info("Acceptance ration for Warming-up is %6.6f"%(1.*Nar/(1.*nw)))
    Nar = 0
    Eb = [] # the array of energies per block
    Eprofile = []
    e, tp = ftlan(hop, ci0, T, m=M)
    for i in range(nsamp):
        E = e/tp
        Eprofile.append(E)
        #print "E", e/tp
        ci = gen_nci(ci0, dr=dr)
        e_n, tp_n = ftlan(hop, ci, T, m=M)
        acc = tp_n/tp
        if acc >= 1:
            ci0 = ci.copy()
            e = e_n
            tp = tp_n
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp = tp_n
                e = e_n
                Nar += 1
    Eprofile = np.asarray(Eprofile)

    if save_prof:
        if feprof==None:
            feprof = "Eprof.npy"
        np.save(feprof, Eprofile)
    E = np.mean(Eprofile)
    ar =  (1.* Nar)/(1.* nsamp)

    if verbose > 0:
        log.debug("The acceptance ratio at T=%2.6f is %6.6f"%(T, ar))
    return E  

def ft_ismpl_rdm1s(qud, hop, ci0, double T, int norb, int nw=25,\
         int nsamp=100, int M=50, double dr=0.5, save_prof=False,\
         prof_file=None, Hdiag=None, int verbose=0, **kwargs):

    if verbose > 2:
        log.info("calculating 1 particle RDMs with importance sampling!")

    '''
        qud         - function that returns <i|a^+ a|j>
        norb        - number of orbitals
        prof_file   - string, the path to the file storing the profile of 
                      |rdm|
        Hdiag  ---- 1d array, the diagonal terms of H
    '''
    ci0 = ci0.reshape(-1).copy().real
    ci0 = ci0/np.linalg.norm(ci0)

    cdef double beta = 1./T  # kB = 1.
    cdef int lc = len(ci0)
    cdef int Nar, i
    cdef double tp_e, tp_e_n, tp_rdm, acc, tmp
    cdef np.ndarray[double,ndim=1] ci=np.empty(lc)
    cdef np.ndarray[double,ndim=2] rdm1=np.empty((norb,norb))
    cdef np.ndarray[double,ndim=2] RDM1=np.zeros((norb,norb))

    def ftE(v0, m=M):
        v=v0.copy()
        if Hdiag is None:
            return _ftlan.ftlan_E1c(hop, v, T, m=m)[1]
        else:
            _E = np.sum((v**2)*Hdiag)
            return np.exp(-beta*_E)
        #function that returns E at finite T (ftlan_E1c)

    def ftrdm1s(v):
        dm1, z = _ftlan.ftlan_rdm1s1c(qud, hop, v, T, norb, m=M)
        return dm1, z
        #function that returns rdm1 at finite T (ftlan_rdm1s1c)

    rdm_arr = []

    if save_prof:
        if prof_file is None:
            if not os.path.exists("./data"):
                os.makedirs("./data")
            prof_file = "./data/RDM1prof_T%2.2f.npy"%T

    # Warm-up
    Nar = 0 # acceptance number
    tp_e = ftE(ci0)
    for i in range(nw):
        ci = gen_nci(ci0, dr=dr)
        tp_e_n = ftE(ci)
        acc = tp_e_n/tp_e 
        if acc >= 1:
            ci0 = ci.copy()
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp_e = tp_e_n
                Nar += 1
            else:
                continue
           
    Nar = 0
    tp_e = ftE(ci0)
    rdm1, tp_rdm = ftrdm1s(ci0)
    for i in range(nsamp):
        RDM1 += rdm1/tp_rdm
#        rdm_arr.append(np.linalg.norm((rdm1a+rdm1b)/tp_rdm)/(norb**2.))
        rdm_arr.append(np.linalg.norm(rdm1/tp_rdm))
#        print "E", e/tp
        ci = gen_nci(ci0, dr=dr)
        tp_e_n = ftE(ci)
        acc = tp_e_n/tp_e
        if acc >= 1:
            ci0 = ci.copy()
            tp_e = tp_e_n
            rdm1, tp_rdm = ftrdm1s(ci0)
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp_e = tp_e_n
                rdm1, tp_rdm = ftrdm1s(ci0)
                Nar += 1
        
#    E = E/(1.*nsamp)
    RDM1 = RDM1/(1.*nsamp)
    if verbose > 1:
        log.section("The 1 particle RDM from importance sampling is:\n %s"%RDM1)

    # save the rdm_profile
    if save_prof:
        rdm_arr = np.asarray(rdm_arr)
        np.save(prof_file, rdm_arr)
    ar =  (1.* Nar)/(1.* nsamp)
    if verbose > 1:
        log.debug("The acceptance ratio at T=%2.6f is %6.6f"%(T, ar))
    return RDM1

def ft_ismpl_rdm12s(qud, hop, ci0, double T, int norb,\
         int nw=25, int nsamp=100, int M=50, double dr=0.5, \
         gen_prof=False, prof_file=None,\
         Hdiag=None, int verbose=0, **kwargs):

    if verbose > 2:
        log.info("calculating 2 particle RDMs with importance sampling!")

    '''
        qud       - function that returns <v|a_i^+ a_k^+ a_l a_j|u>
        gen_prof  - if True, the profile will be generated
        prof_file - string, the path to the file storing the profile
                    of double occupancy.
        Hdiag     - 1d array, the diagonal terms of H
    '''

    def ftE(v0, m=M):
        if Hdiag is None:
            return _ftlan.ftlan_E1c(hop, v0, T, m=m)[1]
        else:
            _E=np.sum((v0**2)*Hdiag)
            return np.exp(-beta*_E)
        #function that returns E at finite T (ftlan_E1c)

    def ftrdm12s(v0):
        dm1, dm2, E, Z = _ftlan.ftlan_rdm12s1c(qud, hop, v0, T, norb, m=M)
        return dm1, dm2, E, Z
        #function that returns rdm1 at finite T (ftlan_rdm1s1c)

    ci0 = ci0.reshape(-1).copy().real
    ci0 = ci0/np.linalg.norm(ci0)
    cdef int lc = len(ci0)
    cdef int Nar, i
    cdef double tp, tp_n, tp_e, tp_e_n, tp_rdm, acc, tmp, e_n, E_n
    cdef double beta = 1./T  # kB = 1.
    cdef np.ndarray[double, ndim=1] ci=np.empty(lc)
    cdef np.ndarray[double, ndim=2] rdm1=np.empty((norb,norb))
    cdef np.ndarray[double, ndim=2] RDM1=np.zeros((norb,norb))
    cdef np.ndarray[double, ndim=4] rdm2=np.empty((norb,norb,norb,norb))
    cdef np.ndarray[double, ndim=4] RDM2=np.zeros((norb,norb,norb,norb))

    # Warm-up
    Nar = 0 # acceptance number
    tp = ftE(ci0, m=20)
    for i in range(nw):
        ci = gen_nci(ci0, dr=dr)
        tp_n = ftE(ci,m=20)
        acc = tp_n/tp 
        if acc >= 1:
            ci0 = ci.copy()
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp = tp_n
                Nar += 1
            else:
                continue
           
    if verbose > 2:
        log.info("Acceptance ration for Warming-up is %6.6f"%(1.*Nar/(1.*nw)))
    # Sampling with block average
    Nar = 0
    tp_e = ftE(ci0)
    rdm1, rdm2, e_n, tp_rdm = ftrdm12s(ci0)

#   RDM1 = np.zeros((norb, norb))
#   RDM2 = np.zeros((norb, norb, norb, norb))
    E_n = 0.
    for i in range(nsamp):
        RDM1 += rdm1/tp_rdm
        RDM2 += rdm2/tp_rdm
        E_n += e_n/tp_rdm
        ci = gen_nci(ci0, dr=dr)
        tp_e_n = ftE(ci)
        acc = tp_e_n/tp_e
        if acc >= 1:
            ci0 = ci.copy()
            tp_e = tp_e_n
            rdm1, rdm2, e_n, tp_rdm = ftrdm12s(ci0)
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp_e = tp_e_n
                rdm1, rdm2, e_n, tp_rdm = ftrdm12s(ci0)
                Nar += 1
        
#    E = E/(1.*nsamp)
    RDM1 = RDM1/(1.*nsamp)
    RDM2 = RDM2/(1.*nsamp)
    E_n = E_n/(1.*nsamp)

    if verbose > 1:
        log.section("The 1 particle RDM from importance \
                    sampling- v12 is:\n %s\n"%RDM1)

    # save the rdm_profile
    ar =  (1.* Nar)/(1.* nsamp)

    if verbose > 0:
        log.debug("The acceptance ratio at T=%2.6f is %6.6f"%(T, ar))

    return RDM1, RDM2, E_n

def gen_nci(v0, double dr=0.5):
    # generate the new ci-vector
    cdef int l = len(v0)
    cdef np.ndarray[double, ndim=1] v1 = np.empty(l)
    cdef np.ndarray[double, ndim=1] disp = np.empty(l)
    disp = np.random.randn(l) * dr
    v1 = v0 + disp
    return v1/nl.norm(v1)

if __name__ == "__main__":
    from pyscf.fci import cistring
     
    u = 4.
    norb = 8
    nelec = 4
    T = 0.1
    h1e = np.zeros((norb, norb))
    for i in range(norb):
        h1e[i, (i+1)%norb] = -1.
        h1e[i, (i-1)%norb] = -1.
    h1e[0, norb-1] = -1.
    h1e[norb-1, 0] = -1.
    eri = np.zeros((norb, norb, norb, norb)) 
    for i in range(norb):
        eri[i,i,i,i] = u
    nstr = cistring.num_strings(norb, nelec)
    ci0 = np.random.randn(nstr)
    ci0 = ci0/nl.norm(ci0)
    def hop(c):
        hc = spinless.contract_2e(eri, c, norb, nelec)
        return hc.reshape(-1)

    def qud2(v1): 
        dm1, dm2 = spinless.make_rdm12(v1, norb, nelec)
        return dm1, dm2 
#    e=ft_ismpl_E(hop, ci0, T, nsamp=20)
#    print e
#    dm1, dm2, e = ft_ismpl_rdm12s(qud2, hop, ci0, T, norb, nsamp=20)
#    print e


