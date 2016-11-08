import numpy as np
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

e1 = direct_spin1.kernel(h1e, eri, norb, nelec)[0]
print e1

