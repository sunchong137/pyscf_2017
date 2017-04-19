#!/usr/bin/env python
'''importance sampling of the Hilbert space
   Chong Sun 2016 Jun. 10
'''
import numpy as np
import numpy.linalg as nl
import ftlm_ep as _ftlan
import logger as log
import os
import random
from mpi4py import MPI

comm=MPI.COMM_WORLD
MPI_rank=comm.Get_rank()
MPI_size=comm.Get_size()


def smpl_time(H_prod_v,mu_prod_v,T,time_step,nsamp=10000,M=50,**kwargs):

    N_per_proc=nsamp//MPI_size
    beta = 1./T

    def corr_t(v):
        ct,z = _ftlan.ftlan_mu1c_time(H_prod_v,mu_prod_v,v,T,time_step,m=M)
        return ct,z
    C_t, Z = 0.j, 0.j
    for i in range(N_per_proc):
        ci0  = np.random.randn(ld)    
        cy,z = corr_t(ci0)
        C_t += ct
        Z   += z
    C_t = C_t/(1.*N_per_proc)
    Z   = Z/(1.*N_per_proc)

    comm.Barrier()
    C_t  = comm.gather(C_t,root=0)
    Z    = comm.gather(Z,root=0)
    if MPI_rank==0:
        Z   =np.sum(np.asarray(Z), axis=0).copy()/MPI_size
        C_t =np.sum(np.asarray(C_t), axis=0).copy()/(MPI_size*Z)
    C_t=comm.bcast(C_t, root=0)
    return C_t

def smpl_freq(H_prod_v,mu_prod_v,T,freq_list,nsamp=10000,M=50,**kwargs):

    #XXX ci0 can be replaced by the shape of it
    N_per_proc=nsamp//MPI_size
    beta = 1./T

    def corr_w(v):
        cw,z = _ftlan.ftlan_mu1c_freq(H_prod_v,mu_prod_v,v,T,freq_list,m=M)
        return cw,z
    C_w = np.zeros(freq_list[1], dtype=complex64)
    Z = 0.j
    for i in range(N_per_proc):
        ci0  = np.random.randn(ld)    
        cw,z = corr_w(ci0)
        C_w += cw
        Z   += z
    C_w = C_w/(1.*N_per_proc)
    Z   = Z/(1.*N_per_proc)

    comm.Barrier()
    C_w  = comm.gather(C_w,root=0)
    Z    = comm.gather(Z,root=0)
    if MPI_rank==0:
        Z   =np.sum(np.asarray(Z), axis=0).copy()/MPI_size
        C_w =np.sum(np.asarray(C_w), axis=0).copy()/(MPI_size*Z)
    C_w=comm.bcast(C_w, root=0)
    return C_w


