import numpy as np
import numpy.linalg as nl
from pyscf.ft import ftlanczos as _ftlan
import logger as log# this is my logger
import random

def gen_nci(v0, cases, dr=0.5, dtheta=20.):
    # generate the new ci-vector
    if cases == 0: # generate new vector by displacement
        disp = np.random.randn(len(v0)) * dr
        v1 = v0 + disp
        return v1/nl.norm(v1)
    if cases == 1: # generate new vector by rotational FIXME gives very bad result!!!!! -_-!!!
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

def sample_debug(qud1, qud2, hop, ci0, T, norb,\
         genci=0, nrot=200, nw=25, nsamp=50, M=20, dr=0.5, dtheta=20.0):

    log.info("calculating 1 particle RDMs with importance sampling!")

    '''
        norb   - number of orbitals
        ftE    - function that returns E at finite T (ftlan_E1c)
        ftrdm  - function that returns rdm1 at finite T (ftlan_rdm1s1c)
    '''
    def ftE(v0, m=M):
        return _ftlan.ftlan_E1c(hop, v0, T, m=m)

    def ftrdm1s(v):
        dm1a, dm1b, z = _ftlan.ftlan_rdm1s1c(qud1, hop, v, T, norb, m=M)
        return (np.asarray(dm1a), np.asarray(dm1b)), z
    
    def ftrdm12s(v0):
        (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb), Z = _ftlan.ftlan_rdm12s1c(qud2, hop, v0, T, norb, m=M)
        return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb), Z
    

    beta = 1./T  # kB = 1.
    # generate the starting vector ----NOTE can also be several starting vectors
    ci0 = ci0.reshape(-1).copy()
    lc = len(ci0)

    # Warm-up
    log.section("Warming up ......")
    Nar = 0 # acceptance number
    tp = ftE(ci0, m=20)[1]
    for i in range(nw):
        ci = gen_nci(ci0, genci)
        tp_n = ftE(ci,m=20)[1]
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
           
    log.section("Sampling......")
    # Sampling with block average
    Nar = 0
    Eb = [] # the array of energies per block
    Eprofile = []
    if genci == 0:
        move = "disp"
    else:
        move = "rot"
    e, tp_e = ftE(ci0)
    (dm1a, dm1b), tp_rdm1 = ftrdm1s(ci0)
    (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb), tp_rdm2 = ftrdm12s(ci0)

    log.section("partition function difference: %12.12f\n"%(tp_rdm1 -tp_rdm2))
    log.section("RDM differences: \n")
    log.section("initial :|dm1a - rdm1a| = %12.12f    ; |dm1b - rdm1b| = %12.12f\n"%(np.linalg.norm(dm1a-rdm1a), np.linalg.norm(dm1b/tp_rdm1-rdm1b/tp_rdm2)))
    log.section("DM 1-v1:\n%s"%(dm1a/tp_rdm1))
    log.section("RDM 1-v12:\n%s"%(rdm1a/tp_rdm2))

    E = 0.
    DM1a = np.zeros((norb, norb))
    DM1b = np.zeros((norb, norb))
    RDM1a = np.zeros((norb, norb))
    RDM1b = np.zeros((norb, norb))
    RDM2aa, RDM2ab, RDM2ba, RDM2bb = np.zeros((norb, norb, norb, norb)),\
                                     np.zeros((norb, norb, norb, norb)),\
                                     np.zeros((norb, norb, norb, norb)),\
                                     np.zeros((norb, norb, norb, norb))

    for i in range(nsamp):
        E += e/tp_e
        DM1a += dm1a/tp_rdm1
        DM1b += dm1b/tp_rdm1
        RDM1a += rdm1a/tp_rdm2
        RDM1b += rdm1b/tp_rdm2
        RDM2aa += rdm2aa/tp_rdm2
        RDM2ab += rdm2ab/tp_rdm2
        RDM2ba += rdm2ba/tp_rdm2
        RDM2bb += rdm2bb/tp_rdm2

#        Eprofile.append([i, e/tp])
#        print "E", e/tp
        ci = gen_nci(ci0, genci)
        e_n, tp_e_n = ftE(ci)
        acc = tp_e_n/tp_e
        if acc >= 1:
            ci0 = ci.copy()
            e = e_n, 
            tp_e = tp_e_n
            (dm1a, dm1b), tp_rdm1 = ftrdm1s(ci0)
            (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb), tp_rdm2 = ftrdm12s(ci0)
            log.section("partition function difference: %6.6f\n"%(tp_rdm1 -tp_rdm2))
            log.section("Nsamp = %d :|dm1a - rdm1a| = %12.12f    ; |dm1b - rdm1b| = %12.12f\n"%(i, np.linalg.norm(dm1a/tp_rdm1-rdm1a/tp_rdm2), np.linalg.norm(dm1b/tp_rdm1-rdm1b/tp_rdm2)))

            log.section("DM 1-v1:\n%s"%(dm1a/tp_rdm1))
            log.section("RDM 1-v12:\n%s"%(rdm1a/tp_rdm2))
            Nar += 1
        else:
            tmp = random.random()
            if tmp <= acc:
                ci0 = ci.copy()
                tp_e = tp_e_n
                e = e_n
                (dm1a, dm1b), tp_rdm1 = ftrdm1s(ci0)
                (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2ba, rdm2bb), tp_rdm2 = ftrdm12s(ci0)
                log.section("partition function difference: %12.12f\n"%(tp_rdm1 -tp_rdm2))
                log.section("Nsamp = %d :|dm1a - rdm1a| = %12.12f    ; |dm1b - rdm1b| = %12.12f\n"%(i, np.linalg.norm(dm1a/tp_rdm1-rdm1a/tp_rdm2), np.linalg.norm(dm1b/tp_rdm1-rdm1b/tp_rdm2)))

                log.section("DM 1:\n%s"%(dm1a/tp_rdm1))
                log.section("RDM 1:\n%s"%(rdm1a/tp_rdm2))

                Nar += 1
        
    E = E/(1.*nsamp)
    DM1a = DM1a/(1.*nsamp)
    DM1b = DM1b/(1.*nsamp)
    RDM1a = RDM1a/(1.*nsamp)
    RDM1b = RDM1b/(1.*nsamp)
    RDM2aa = RDM2aa/(1.*nsamp)
    RDM2ab = RDM2ab/(1.*nsamp)
    RDM2ba = RDM2ba/(1.*nsamp)
    RDM2bb = RDM2bb/(1.*nsamp)

    ar =  (1.* Nar)/(1.* nsamp)
    log.debug("The acceptance ratio at T=%2.6f is %6.6f"%(T, ar))
    log.section("The final difference is |dm1a - rdm1a| = %6.6f, |dm1b - rdm1b|= %6.6f"%(np.linalg.norm(DM1a-RDM1a), np.linalg.norm(DM1b-RDM1b)))
    
if __name__ == "__main__":
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.fci import cistring
    from pyscf.fci import direct_spin1

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    ne = mol.nelectron - 2
    nelec = (nelec//2, nelec-nelec//2)
    h1e = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
     
    T = 0.1
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = np.random.randn(na*nb)
    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    def qud1(v1, v2):
        dma, dmb = direct_spin1.trans_rdm1s(v1, v2, norb, nelec)
        return dma, dmb

    def qud2(v1, v2):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = direct_spin1.trans_rdm12s(v1, v2, norb, nelec)
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

    sample_debug(qud1, qud2, hop, ci0, T, norb, M=20)

