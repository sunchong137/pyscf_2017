#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Full CI solver for spin-free Hamiltonian.  This solver can be used to compute
doublet, triplet,...

The CI wfn are stored as a 2D array [alpha,beta], where each row corresponds
to an alpha string.  For each row (alpha string), there are
total-num-beta-strings of columns.  Each column corresponds to a beta string.

Different FCI solvers are implemented to support different type of symmetry.
                    Symmetry
File                Point group   Spin singlet   Real hermitian*    Alpha/beta degeneracy
direct_spin0_symm   Yes           Yes            Yes                Yes
direct_spin1_symm   Yes           No             Yes                Yes
direct_spin0        No            Yes            Yes                Yes
direct_spin1        No            No             Yes                Yes
direct_uhf          No            No             Yes                No
direct_nosym        No            No             No**               Yes

*  Real hermitian Hamiltonian implies (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
** Hamiltonian is real but not hermitian, (ij|kl) != (ji|kl) ...
'''

import sys
import ctypes
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.gto
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci import spin_op

libfci = pyscf.lib.load_library('libfci')

def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    '''Contract the 1-electron Hamiltonian with a FCI vector to get a new FCI
    vector.
    '''
    fcivec = numpy.asarray(fcivec, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert(fcivec.size == na*nb)
    f1e_tril = pyscf.lib.pack_tril(f1e)
    ci1 = numpy.zeros_like(fcivec)
    libfci.FCIcontract_a_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    libfci.FCIcontract_b_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    r'''Contract the 2-electron Hamiltonian with a FCI vector to get a new FCI
    vector.

    Note the input arg eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is

    .. math::

        h2e &= eri_{pq,rs} p^+ q r^+ s \\
            &= (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s

    So eri is defined as

    .. math::

        eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)

    to restore the symmetry between pq and rs,

    .. math::

        eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]

    See also :func:`direct_spin1.absorb_h1e`
    '''
    fcivec = numpy.asarray(fcivec, order='C')
    eri = pyscf.ao2mo.restore(4, eri, norb)
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert(fcivec.size == na*nb)
    ci1 = numpy.empty_like(fcivec)

    libfci.FCIcontract_2e_spin1(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1

def make_hdiag(h1e, eri, norb, nelec):
    '''Diagonal Hamiltonian for Davidson preconditioner
    '''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    h1e = numpy.ascontiguousarray(h1e)
    eri = pyscf.ao2mo.restore(1, eri, norb)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]

    occslista = numpy.asarray(link_indexa[:,:neleca,0], order='C')
    occslistb = numpy.asarray(link_indexb[:,:nelecb,0], order='C')
    hdiag = numpy.empty(na*nb)
    jdiag = numpy.asarray(numpy.einsum('iijj->ij',eri), order='C')
    kdiag = numpy.asarray(numpy.einsum('ijji->ij',eri), order='C')
    libfci.FCImake_hdiag_uhf(hdiag.ctypes.data_as(ctypes.c_void_p),
                             h1e.ctypes.data_as(ctypes.c_void_p),
                             h1e.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             kdiag.ctypes.data_as(ctypes.c_void_p),
                             kdiag.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(neleca), ctypes.c_int(nelecb),
                             occslista.ctypes.data_as(ctypes.c_void_p),
                             occslistb.ctypes.data_as(ctypes.c_void_p))
    return numpy.asarray(hdiag)

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, numpy.number)):
        nelec = sum(nelec)
    eri = eri.copy()
    h2e = pyscf.ao2mo.restore(1, eri, norb)
    f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return pyscf.ao2mo.restore(4, h2e, norb) * fac

def pspace(h1e, eri, norb, nelec, hdiag, np=400):
    '''pspace Hamiltonian to improve Davidson preconditioner. See, CPL, 169, 463
    '''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    h1e = numpy.ascontiguousarray(h1e)
    eri = pyscf.ao2mo.restore(1, eri, norb)
    nb = cistring.num_strings(norb, nelecb)
    if hdiag.size < np:
        addr = numpy.arange(hdiag.size)
    else:
        try:
            addr = numpy.argpartition(hdiag, np-1)[:np]
        except AttributeError:
            addr = numpy.argsort(hdiag)[:np]
    addra, addrb = divmod(addr, nb)
    stra = numpy.array([cistring.addr2str(norb,neleca,ia) for ia in addra],
                       dtype=numpy.uint64)
    strb = numpy.array([cistring.addr2str(norb,nelecb,ib) for ib in addrb],
                       dtype=numpy.uint64)
    np = len(addr)
    h0 = numpy.zeros((np,np))
    libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p),
                            h1e.ctypes.data_as(ctypes.c_void_p),
                            eri.ctypes.data_as(ctypes.c_void_p),
                            stra.ctypes.data_as(ctypes.c_void_p),
                            strb.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(np))

    for i in range(np):
        h0[i,i] = hdiag[addr[i]]
    h0 = pyscf.lib.hermi_triu(h0)
    return addr, h0

# be careful with single determinant initial guess. It may diverge the
# preconditioner when the eigvalue of first davidson iter equals to hdiag
def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           **kwargs):
    return _kfactory(FCISolver, h1e, eri, norb, nelec, ci0, level_shift,
                     tol, lindep, max_cycle, max_space, nroots,
                     davidson_only, pspace_size, **kwargs)
def _kfactory(Solver, h1e, eri, norb, nelec, ci0=None, level_shift=1e-3,
              tol=1e-10, lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
              davidson_only=False, pspace_size=400, **kwargs):
    cis = Solver(None)
    cis.level_shift = level_shift
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.max_space = max_space
    cis.nroots = nroots
    cis.davidson_only = davidson_only
    cis.pspace_size = pspace_size

    unknown = {}
    for k, v in kwargs.items():
        setattr(cis, k, v)
        if not hasattr(cis, k):
            unknown[k] = v
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown.keys()), __name__))
    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, **unknown)
    return e, c

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    '''Compute the FCI electronic energy for given Hamiltonian and FCI vector.
    '''
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))


def make_rdm1s(fcivec, norb, nelec, link_index=None):
    '''Spin searated 1-particle density matrices, (alpha,beta)
    '''
    if link_index is None:
        if isinstance(nelec, (int, numpy.number)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        link_index = (link_indexa, link_indexb)
    rdm1a = rdm.make_rdm1_spin1('FCImake_rdm1a', fcivec, fcivec,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCImake_rdm1b', fcivec, fcivec,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

def make_rdm1(fcivec, norb, nelec, link_index=None):
    '''spin-traced 1-particle density matrix
    '''
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, link_index)
    return rdm1a + rdm1b

def make_rdm12s(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin searated 1- and 2-particle density matrices,
    (alpha,beta) for 1-particle density matrices.
    (alpha,alpha,alpha,alpha), (alpha,alpha,beta,beta),
    (beta,beta,beta,beta) for 2-particle density matrices.

    NOTE the 2pdm is :math:`\langle p^\dagger q^\dagger s r\rangle` but is
    stored as [p,r,q,s]
    '''
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCIrdm12kern_a', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCIrdm12kern_b', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', fcivec, fcivec,
                                    norb, nelec, link_index, 0)
    if reorder:
        dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin traced 1- and 2-particle density matrices,

    NOTE the 2pdm is :math:`\langle p^\dagger q^\dagger s r\rangle` but is
    stored as [p,r,q,s]
    '''
    #(dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
    #        make_rdm12s(fcivec, norb, nelec, link_index, reorder)
    #return dm1a+dm1b, dm2aa+dm2ab+dm2ab.transpose(2,3,0,1)+dm2bb
    dm1, dm2 = rdm.make_rdm12_spin1('FCIrdm12kern_sf', fcivec, fcivec,
                                    norb, nelec, link_index, 1)
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2

def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    '''Spin separated transition 1-particle density matrices
    '''
    rdm1a = rdm.make_rdm1_spin1('FCItrans_rdm1a', cibra, ciket,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCItrans_rdm1b', cibra, ciket,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

# spacial part of DM
def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    '''Spin traced transition 1-particle density matrices
    '''
    rdm1a, rdm1b = trans_rdm1s(cibra, ciket, norb, nelec, link_index)
    return rdm1a + rdm1b

def trans_rdm12s(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    '''Spin separated transition 1- and 2-particle density matrices.
    '''
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket,
                                       norb, nelec, link_index, 2)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket,
                                       norb, nelec, link_index, 2)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket,
                                    norb, nelec, link_index, 0)
    _, dm2ba = rdm.make_rdm12_spin1('FCItdm12kern_ab', ciket, cibra,
                                    norb, nelec, link_index, 0)
    dm2ba = dm2ba.transpose(3,2,1,0)
    if reorder:
        dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

def trans_rdm12(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    '''Spin traced transition 1- and 2-particle density matrices.
    '''
    #(dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = \
    #        trans_rdm12s(cibra, ciket, norb, nelec, link_index, reorder)
    #return dm1a+dm1b, dm2aa+dm2ab+dm2ba+dm2bb
    dm1, dm2 = rdm.make_rdm12_spin1('FCItdm12kern_sf', cibra, ciket,
                                    norb, nelec, link_index, 2)
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2

def get_init_guess(norb, nelec, nroots, hdiag):
    '''Initial guess is the single Slater determinant
    '''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    # The "nroots" lowest determinats based on energy expectation value.
    ci0 = []
    try:
        addrs = numpy.argpartition(hdiag, nroots-1)[:nroots]
    except AttributeError:
        addrs = numpy.argsort(hdiag)[:nroots]
    for addr in addrs:
        x = numpy.zeros((na*nb))
        x[addr] = 1
        ci0.append(x.ravel())

    # Add noise
    ci0[0][0 ] += 1e-5
    ci0[0][-1] -= 1e-5
    return ci0


###############################################################
# direct-CI driver
###############################################################

def kernel_ms1(fci, h1e, eri, norb, nelec, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               max_memory=None, verbose=None, **kwargs):
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size

    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)

    if pspace_size > 0:
        addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag, pspace_size)
        pw, pv = scipy.linalg.eigh(h0)
    else:
        pw = pv = addr = None

    if pspace_size >= na*nb and ci0 is None and not davidson_only:
# The degenerated wfn can break symmetry.  The davidson iteration with proper
# initial guess doesn't have this issue
        if na*nb == 1:
            return pw[0], pv[:,0].reshape(1,1)
        elif nroots > 1:
            civec = numpy.empty((nroots,na*nb))
            civec[:,addr] = pv[:,:nroots].T
            return pw[:nroots], civec.reshape(nroots,na,nb)
        elif abs(pw[0]-pw[1]) > 1e-12:
            civec = numpy.empty((na*nb))
            civec[addr] = pv[:,0]
            return pw[0], civec.reshape(na,nb)

    precond = fci.make_precond(hdiag, pw, pv, addr)

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    def hop(c):
        hc = fci.contract_2e(h2e, c, norb, nelec, (link_indexa,link_indexb))
        return hc.ravel()

    if ci0 is None:
        if hasattr(fci, 'get_init_guess'):
            ci0 = fci.get_init_guess(norb, nelec, nroots, hdiag)
        else:
            ci0 = []
            for i in range(nroots):
                x = numpy.zeros(na*nb)
                x[addr[i]] = 1
                ci0.append(x)
    else:
        if isinstance(ci0, numpy.ndarray) and ci0.size == na*nb:
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    if max_memory is None: max_memory = fci.max_memory
    if verbose is None: verbose = logger.Logger(fci.stdout, fci.verbose)
    #e, c = pyscf.lib.davidson(hop, ci0, precond, tol=fci.conv_tol, lindep=fci.lindep)
    e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                   max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                   max_memory=max_memory, verbose=verbose, **kwargs)
    if nroots > 1:
        return e, [ci.reshape(na,nb) for ci in c]
    else:
        return e, c.reshape(na,nb)

def make_pspace_precond(hdiag, pspaceig, pspaceci, addr, level_shift=0):
    # precondition with pspace Hamiltonian, CPL, 169, 463
    def precond(r, e0, x0, *args):
        #h0e0 = h0 - numpy.eye(len(addr))*(e0-level_shift)
        h0e0inv = numpy.dot(pspaceci/(pspaceig-(e0-level_shift)), pspaceci.T)
        hdiaginv = 1/(hdiag - (e0-level_shift))
        hdiaginv[abs(hdiaginv)>1e8] = 1e8
        h0x0 = x0 * hdiaginv
        #h0x0[addr] = numpy.linalg.solve(h0e0, x0[addr])
        h0x0[addr] = numpy.dot(h0e0inv, x0[addr])
        h0r = r * hdiaginv
        #h0r[addr] = numpy.linalg.solve(h0e0, r[addr])
        h0r[addr] = numpy.dot(h0e0inv, r[addr])
        e1 = numpy.dot(x0, h0r) / numpy.dot(x0, h0x0)
        x1 = r - e1*x0
        #pspace_x1 = x1[addr].copy()
        x1 *= hdiaginv
# pspace (h0-e0)^{-1} cause diverging?
        #x1[addr] = numpy.linalg.solve(h0e0, pspace_x1)
        return x1
    return precond

def make_diag_precond(hdiag, pspaceig, pspaceci, addr, level_shift=0):
    def precond(x, e, *args):
        hdiagd = hdiag-(e-level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return x/hdiagd
    return precond


class FCISolver(pyscf.lib.StreamObject):
    def __init__(self, mol=None):
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.mol = mol
        self.max_cycle = 50
        self.max_space = 12
        self.conv_tol = 1e-10
        self.lindep = 1e-14
        self.max_memory = pyscf.lib.parameters.MEMORY_MAX
# level shift in precond
        self.level_shift = 1e-3
        # force the diagonlization use davidson iteration.  When the CI space
        # is small, the solver exactly diagonlizes the Hamiltonian.  But this
        # solution will ignore the initial guess.  Setting davidson_only can
        # enforce the solution on the initial guess state
        self.davidson_only = False
        self.nroots = 1
        self.pspace_size = 400
# Initialize symmetry attributes for the compatibility with direct_spin1_symm
# solver.  They are not used by direct_spin1 solver.
        self.orbsym = None
        self.wfnsym = None

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        if verbose is None: verbose = self.verbose
        log = logger.Logger(self.stdout, verbose)
        log.info('******** %s flags ********', self.__class__)
        log.info('max. cycles = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('linear dependence = %g', self.lindep)
        log.info('level shift = %d', self.level_shift)
        log.info('max iter space = %d', self.max_space)
        log.info('max_memory %d MB', self.max_memory)
        log.info('davidson only = %s', self.davidson_only)
        log.info('nroots = %d', self.nroots)
        log.info('pspace_size = %d', self.pspace_size)
        return self


    @pyscf.lib.with_doc(absorb_h1e.__doc__)
    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return absorb_h1e(h1e, eri, norb, nelec, fac)

    @pyscf.lib.with_doc(make_hdiag.__doc__)
    def make_hdiag(self, h1e, eri, norb, nelec):
        return make_hdiag(h1e, eri, norb, nelec)

    @pyscf.lib.with_doc(pspace.__doc__)
    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return pspace(h1e, eri, norb, nelec, hdiag, np)

    @pyscf.lib.with_doc(contract_1e.__doc__)
    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    @pyscf.lib.with_doc(contract_2e.__doc__)
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)

    def eig(self, op, x0, precond, **kwargs):
        if kwargs['nroots'] == 1 and x0[0].size > 6.5e7: # 500MB
            lessio = True
        else:
            lessio = False
        return pyscf.lib.davidson(op, x0, precond, lessio=lessio, **kwargs)

    def make_precond(self, hdiag, pspaceig, pspaceci, addr):
        if pspaceig is None:
            return make_diag_precond(hdiag, pspaceig, pspaceci, addr,
                                     self.level_shift)
        else:
            return make_pspace_precond(hdiag, pspaceig, pspaceci, addr,
                                       self.level_shift)

    @pyscf.lib.with_doc(get_init_guess.__doc__)
    def get_init_guess(self, norb, nelec, nroots, hdiag):
        return get_init_guess(norb, nelec, nroots, hdiag)

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, **kwargs):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        return kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
                          tol, lindep, max_cycle, max_space, nroots,
                          davidson_only, pspace_size, **kwargs)

    @pyscf.lib.with_doc(energy.__doc__)
    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = self.contract_2e(h2e, fcivec, norb, nelec, link_index)
        return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

    def spin_square(self, fcivec, norb, nelec):
        if self.nroots == 1:
            return spin_op.spin_square0(fcivec, norb, nelec)
        else:
            ss = [spin_op.spin_square0(c, norb, nelec) for c in fcivec]
            return [x[0] for x in ss], [x[1] for x in ss]
    spin_square.__doc__ = spin_op.spin_square0.__doc__

    @pyscf.lib.with_doc(make_rdm1s.__doc__)
    def make_rdm1s(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1s(fcivec, norb, nelec, link_index)

    @pyscf.lib.with_doc(make_rdm1.__doc__)
    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1(fcivec, norb, nelec, link_index)

    @pyscf.lib.with_doc(make_rdm12s.__doc__)
    def make_rdm12s(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12s(fcivec, norb, nelec, link_index, reorder)

    @pyscf.lib.with_doc(make_rdm12.__doc__)
    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12(fcivec, norb, nelec, link_index, reorder)

    def make_rdm2(self, fcivec, norb, nelec, link_index=None, reorder=True):
        r'''Spin traced 2-particle density matrice

        NOTE the 2pdm is :math:`\langle p^\dagger q^\dagger s r\rangle` but
        stored as [p,r,q,s]
        '''
        return self.make_rdm12(fcivec, norb, nelec, link_index, reorder)[1]

    @pyscf.lib.with_doc(make_rdm1s.__doc__)
    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None):
        return trans_rdm1s(cibra, ciket, norb, nelec, link_index)

    @pyscf.lib.with_doc(trans_rdm1.__doc__)
    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None):
        return trans_rdm1(cibra, ciket, norb, nelec, link_index)

    @pyscf.lib.with_doc(trans_rdm12s.__doc__)
    def trans_rdm12s(self, cibra, ciket, norb, nelec, link_index=None,
                     reorder=True):
        return trans_rdm12s(cibra, ciket, norb, nelec, link_index, reorder)

    @pyscf.lib.with_doc(trans_rdm12.__doc__)
    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
        return trans_rdm12(cibra, ciket, norb, nelec, link_index, reorder)


def _unpack(norb, nelec, link_index):
    if link_index is None:
        if isinstance(nelec, (int, numpy.number)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
        return link_indexa, link_indexb
    else:
        return link_index


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        #['H', ( 0.,-0.5   ,-1.   )],
        #['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    cis = FCISolver(mol)
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    nea = nelec//2 + 1
    neb = nelec//2 - 1
    nelec = (nea, neb)

    e1 = cis.kernel(h1e, eri, norb, nelec)[0]
    print(e1, e1 - -7.7466756526056004)

