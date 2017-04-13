#!/usr/bin/env python
'''importance sampling of the Hilbert space
   Chong Sun 2016 Jun. 10
'''
import numpy as np
import numpy.linalg as nl
import ftlanczos as _ftlan
import logger as log
#from pyscf.ftfci.utils import ftlanczos as _ftlan
#from pyscf.ftfci.utils import logger as log# this is my logger
import os
import random
from mpi4py import MPI

comm=MPI.COMM_WORLD
MPI_rank=comm.Get_rank()
MPI_size=comm.Get_size()


def ft_smpl_E(hop,ci0,T,nsamp=20000,M=50,**kwargs):

    N_per_proc = nsamp//MPI_size
    ci0 = ci0.reshape(-1).copy()
    ftlan = _ftlan.ftlan_E1c  
    beta = 1./T 
    ld = len(ci0)
    E0 = _ftlan.ftlan_E(hop,ci0)

    def lanczosE(v0):
        return _ftlan.ftlan_E1c(hop,v0,T,E0=E0)
    E,Z = 0., 0.
    for i in range(N_per_proc):
        ci0 = np.random.randn(ld)
        e,z = lanczosE(ci0)
        E += e
        Z += z
    E /= (1.*N_per_proc)
    Z /= (1.*N_per_proc)

    comm.Barrier()
    E = comm.gather(E,root=0)
    Z = comm.gather(Z,root=0)
    if MPI_rank==0:
        E = np.asarray(E)
        Z = np.asarray(Z)
        E = np.mean(E)
        Z = np.mean(Z)
        E = E/Z
    else:
        E=None
        Z=None
    E = comm.bcast(E,root=0)
    Z = comm.bcast(Z,root=0)
    return E

def ft_smpl_rdm1s(qud,hop,ci0,T,norb,nsamp=10000,M=50,**kwargs):

    N_per_proc=nsamp//MPI_size
    beta = 1./T
    E0 = _ftlan.ftlan_E(hop,ci0)
    ld = len(ci0)

    def ftrdm1s(v):
        dm1,E,z = _ftlan.ftlan_rdm1s1c(qud,hop,v,T,norb,m=M,E0=E0)
        return np.asarray(dm1),E,z

    RDM1,_,_ =  _ftlan.ftlan_rdm1s1c(qud,hop,ci0,T,norb,m=20,E0=E0)   
    RDM1 = np.asarray(RDM1)*0.
    E, Z = 0., 0.

    for i in range(N_per_proc):
        ci0 = np.random.randn(ld)    
        rdm1,e,z = ftrdm1s(ci0)
        RDM1 += rdm1
        E    += e
        Z    += z

    RDM1 = RDM1/(1.*N_per_proc)
    E    = E/(1.*N_per_proc)
    Z    = Z/(1.*N_per_proc)

    comm.Barrier()
    RDM1=comm.gather(RDM1, root=0)
    E    = comm.gather(E,root=0)
    Z    = comm.gather(Z,root=0)

    if MPI_rank==0:
        Z   =np.sum(np.asarray(Z), axis=0).copy()/MPI_size
        E   =np.sum(np.asarray(E), axis=0).copy()/(MPI_size*Z)
        RDM1=np.sum(np.asarray(RDM1), axis=0).copy()/(MPI_size*Z)
    
    RDM1=comm.bcast(RDM1, root=0)
    E = comm.bcast(E, root=0)

    return RDM1, E  

def ft_smpl_rdm12s(qud,hop,ci0,T,norb,nsamp=100,M=50,**kwargs):

    N_per_proc=nsamp//MPI_size
    beta = 1./T
    E0 = _ftlan.ftlan_E(hop,ci0)
    ld = len(ci0)
    def ftrdm12s(v0):
        dm1,dm2,E,Z = _ftlan.ftlan_rdm12s1c(qud,hop,v0,T,norb,m=M,E0=E0)
        return np.asarray(dm1), np.asarray(dm2), E, Z
    RDM1,RDM2,_,_, = _ftlan.ftlan_rdm12s1c(qud,hop,ci0,T,norb,m=20,E0=E0)
    RDM1 *= 0.
    RDM2 *= 0.
    E,Z = 0.,0.
       
    for i in range(N_per_proc):
        ci0 = np.random.randn(ld)    
        rdm1,rdm2,e,z = ftrdm12s(ci0)
        RDM1 += rdm1
        RDM2 += rdm2
        E    += e
        Z    += z
        
    RDM1 = RDM1/(1.*N_per_proc)
    RDM2 = RDM2/(1.*N_per_proc)
    E    = E/(1.*N_per_proc)
    Z    = Z/(1.*N_per_proc)

    comm.Barrier()
    RDM1 = comm.gather(RDM1,root=0)
    RDM2 = comm.gather(RDM2,root=0)
    E    = comm.gather(E,root=0)
    Z    = comm.gather(Z,root=0)
   
    if MPI_rank==0:
        Z   =np.sum(np.asarray(Z), axis=0).copy()/MPI_size
        E   =np.sum(np.asarray(E), axis=0).copy()/(MPI_size*Z)
        RDM1=np.sum(np.asarray(RDM1), axis=0).copy()/(MPI_size*Z)
        RDM2=np.sum(np.asarray(RDM2), axis=0).copy()/(MPI_size*Z)

    RDM1=comm.bcast(RDM1, root=0)
    RDM2=comm.bcast(RDM2, root=0)
    E = comm.bcast(E, root=0)
    return RDM1, RDM2, E

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


