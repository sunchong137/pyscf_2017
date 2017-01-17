#!/usr/bin/env python
'''importance sampling of the Hilbert space
   Chong Sun 2016 Jun. 10
'''
import numpy as np
import numpy.linalg as nl
from pyscf.ft_spinless.utils import ftlanczos_spinless as _ftlan
from pyscf.fci import fci_slow_spinless as spinless
from pyscf.ft_spinless.utils import logger as log# this is my logger
import os
import random
from mpi4py import MPI

comm=MPI.COMM_WORLD
MPI_rank=comm.Get_rank()
MPI_size=comm.Get_size()


def ft_ismpl_E(hop, ci0, T, genci=0, nrot=200,\
         nw=25, nsamp=2000, M=50, dr=0.5, dtheta=20.0, \
         save_prof=False, feprof=None, Hdiag=None, verbose=0, **kwargs):
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
    N_per_proc = nsamp//MPI_size

    def ftE(v0, m=M):
        if Hdiag is None:
            return _ftlan.ftlan_E1c(hop, v0, T, m=m)[1]
        else:
            _E = np.sum(v0**2*Hdiag)
            return _E, np.exp(-beta*_E)

    beta = 1./T  # kB = 1.
    ftlan = _ftlan.ftlan_E1c  # function giving the probability and energy evaluated from the initial vector

    # generate the starting vector ----NOTE can also be several starting vectors
    ci0 = ci0.reshape(-1).copy()
    lc = len(ci0)
   
 
    # Warm-up
    if verbose > 2:
        log.info("Warming up ......")
    Nar = 0 # acceptance number
    tp = ftlan(hop, ci0, T, m=20)[1]
    for i in range(nw):
        ci = gen_nci(ci0, genci, dr=dr, dtheta=dtheta)
        tp_n = ftlan(hop, ci, T, m=20)[1]
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
        log.info("Sampling......")
    # Sampling with block average

    Nar = 0
    Eb = [] # the array of energies per block
    Eprofile = []
    if genci == 0:
        move = "disp"
        if verbose > 2:
            log.info("Using displacement to move to the next wavefunction!")
    else:
        move = "rot"
        if verbose > 2:
            log.info("Using rotation to move to the next wavefunction!")

    e, tp = ftlan(hop, ci0, T)
    for i in range(N_per_proc):
        E = e/tp
        Eprofile.append(E)
        #print "E", e/tp
        ci = gen_nci(ci0, genci)
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
            feprof = "Eprof_%s.npy"%(move)
        np.save(feprof, Eprofile)
    E = np.mean(Eprofile)
    ar =  (1.* Nar)/(1.* N_per_proc)

    comm.Barrier()
    E = comm.gather(E, root=0)
    if MPI_rank==0:
        E = np.asarray(E)
        E = np.mean(E)
    else:
        E=None
    E = comm.bcast(E, root=0)
    if verbose > 0:
        log.debug("The acceptance ratio at T=%2.6f is %6.6f"%(T, ar))
    return E  

def ft_ismpl_rdm1s(qud, hop, ci0, T, norb,\
         genci=0, nrot=200, nw=25, nsamp=100, M=50, dr=0.5, \
         dtheta=20.0, save_prof=False, prof_file=None, \
         Hdiag=None, verbose=0,**kwargs):

    if verbose > 2:
        log.info("calculating 1 particle RDMs with importance sampling!")

    '''
        qud         - function that returns <i|a^+ a|j>
        norb        - number of orbitals
        prof_file   - string, the path to the file storing the profile of 
                      |rdm|
        Hdiag  ---- 1d array, the diagonal terms of H
    '''
    N_per_proc=nsamp//MPI_size

    def ftE(v0, m=M):
        v=v0.copy()
        if Hdiag is None:
            return _ftlan.ftlan_E1c(hop, v, T, m=m)[1]
        else:
            _E = np.sum((v**2)*Hdiag)
            return np.exp(-beta*_E)
        #function that returns E at finite T (ftlan_E1c)

    def ftrdm1s(v):
        dm1a, dm1b, z = _ftlan.ftlan_rdm1s1c(qud, hop, v, T, norb, m=M)
        return (np.asarray(dm1a), np.asarray(dm1b)), z
        #function that returns rdm1 at finite T (ftlan_rdm1s1c)

    beta = 1./T  # kB = 1.
    # generate the starting vector ----NOTE can also be several starting vectors
    ci0 = ci0.reshape(-1).copy()
    lc = len(ci0)
    rdm_arr = []

    if genci == 0:
        move = "disp"
        if verbose > 2:
            log.info("Using displacement to move to the next wavefunction!")
    else:
        move = "rot"
        if verbose > 2:
            log.info("Using rotation to move to the next wavefunction!")

    if save_prof:
        if prof_file is None:
            if not os.path.exists("./data"):
                os.makedirs("./data")
            prof_file = "./data/RDM1prof_%sT%2.2f.npy"%(move, T)
    # Warm-up

    Nar = 0 # acceptance number
    tp_e = ftE(ci0)
    for i in range(nw):
        ci = gen_nci(ci0, genci)
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
           
    # Sampling with block average
    Nar = 0
    tp_e = ftE(ci0)
    (rdm1a, rdm1b), tp_rdm = ftrdm1s(ci0)

    RDM1a = np.zeros((norb, norb))
    RDM1b = np.zeros((norb, norb))
    for i in range(N_per_proc):
        RDM1a += rdm1a/tp_rdm
        RDM1b += rdm1b/tp_rdm
#        rdm_arr.append(np.linalg.norm((rdm1a+rdm1b)/tp_rdm)/(norb**2.))
        rdm_arr.append(np.linalg.norm((rdm1a+rdm1b)/tp_rdm))
#        print "E", e/tp
        ci = gen_nci(ci0, genci)
        tp_e_n = ftE(ci)
        acc = tp_e_n/tp_e
        if acc >= 1:
            ci0 = ci.copy()
            tp_e = tp_e_n
            (rdm1a, rdm1b), tp_rdm = ftrdm1s(ci0)
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp_e = tp_e_n
                (rdm1a, rdm1b), tp_rdm = ftrdm1s(ci0)
                Nar += 1
        
#    E = E/(1.*nsamp)
    RDM1a = RDM1a/(1.*N_per_proc)
    RDM1b = RDM1b/(1.*N_per_proc)
    if verbose > 1:
        log.section("The 1 particle RDM from importance sampling is:\n %s\n %s"%(RDM1a, RDM1b))

    comm.Barrier()
    RDM1a=comm.gather(RDM1a, root=0)
    RDM1b=comm.gather(RDM1b, root=0)
    if MPI_rank==0:
        RDM1a=np.sum(np.asarray(RDM1a), axis=0).copy()/MPI_size
        RDM1b=np.sum(np.asarray(RDM1b), axis=0).copy()/MPI_size
    
    RDM1a=comm.bcast(RDM1a, root=0)
    RDM1b=comm.bcast(RDM1b, root=0)
    # save the rdm_profile
    if save_prof:
        rdm_arr = np.asarray(rdm_arr)
        np.save(prof_file, rdm_arr)
    ar =  (1.* Nar)/(1.* nsamp)
    if verbose > 1:
        log.debug("The acceptance ratio at T=%2.6f is %6.6f"%(T, ar))
    return RDM1a, RDM1b  

def ft_ismpl_rdm12s(qud, hop, ci0, T, norb,\
         genci=0, nrot=200, nw=25, nsamp=100, M=50, dr=0.5, \
         dtheta=20.0, gen_prof=False, prof_file=None,\
         Hdiag=None, verbose=0, **kwargs):

    if verbose > 2:
        log.info("calculating 2 particle RDMs with importance sampling!")

    '''
        qud       - function that returns <v|a_i^+ a_k^+ a_l a_j|u>
        gen_prof  - if True, the profile will be generated
        prof_file - string, the path to the file storing the profile
                    of double occupancy.
        Hdiag     - 1d array, the diagonal terms of H
    '''

    N_per_proc=nsamp//MPI_size

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

#   def get_docc(_rdm2ab):
#       res = 0.
#       for i in range(norb):
#           res += _rdm2ab[i,i,i,i]
#       return res/(1.*norb)

    beta = 1./T  # kB = 1.
    # generate the starting vector ----NOTE can also be several starting vectors
    ci0 = ci0.reshape(-1).copy()
    lc = len(ci0)

    if genci == 0:
        move = "disp"
        if verbose > 2:
            log.info("Using displacement to move to the next wavefunction!")
    else:
        move = "rot"
        if verbose > 2:
            log.info("Using rotation to move to the next wavefunction!")

    # Warm-up
    Nar = 0 # acceptance number
    tp = ftE(ci0, m=20)
    for i in range(nw):
        ci = gen_nci(ci0, genci)
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
    RDM1 = np.zeros_like(rdm1)
    RDM2 = np.zeros_like(rdm2)
    for i in range(N_per_proc):
        RDM1 += rdm1/tp_rdm
        RDM2 += rdm2/tp_rdm
        E_n += e_n/tp_rdm
        ci = gen_nci(ci0, genci)
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
    RDM1 = RDM1/(1.*N_per_proc)
    RDM2 = RDM2/(1.*N_per_proc)
    E_n = E_n/(1.*N_per_proc)

    if verbose > 1:
        log.section("The 1 particle RDM from importance \
                    sampling- v12 is:\n %s\n"%RDM1a)

    # save the rdm_profile
    ar =  (1.* Nar)/(1.* nsamp)

    comm.Barrier()
    RDM1=comm.gather(RDM1, root=0)
    RDM2 = comm.gather(RDM2, root=0)
    E_n = comm.gather(E_n, root=0)
   
    if MPI_rank==0:
        RDM1=np.sum(np.asarray(RDM1), axis=0).copy()/MPI_size
        RDM2=np.sum(np.asarray(RDM2), axis=0).copy()/MPI_size
        E_n = np.sum(np.asarray(E_n), axis=0).copy()/MPI_size

    RDM1=comm.bcast(RDM1, root=0)
    RDM2=comm.bcast(RDM2, root=0)
    E_n = comm.bcast(E_n, root=0)

    if verbose > 0:
        log.debug("The acceptance ratio at T=%2.6f is %6.6f"%(T, ar))

    return RDM1, RDM2, E_n

def gen_nci(v0, cases, dr=0.5, dtheta=20.):
    # generate the new ci-vector
    if cases == 0: # generate new vector by displacement
        disp = np.random.randn(len(v0)) * dr
        v1 = v0 + disp
        return v1/nl.norm(v1)
    if cases == 1: # generate new vector by rotational
#        print "generating new vectors by rotation"
        v1 = v0.copy()
        for i in range(nrot):
            addr = np.random.randint(lc, size = 2)         
            theta = random.random() * dtheta # rotational angle
            vec = np.zeros(2)
            vec[0] = v1[addr[0]]
            vec[1] = v1[addr[1]]
            rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            vec = (rotmat.dot(vec)).copy()
            v1[addr[0]] = vec[0]
            v1[addr[1]] = vec[1]
        return v1/nl.norm(v1) 

if __name__ == "__main__":
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.fci import cistring
    import numpy
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
    dm1, dm2, e = ft_ismpl_rdm12s(qud2, hop, ci0, T, norb, nsamp=20)
    print e


