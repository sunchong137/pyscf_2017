#!/usr/bin/env python
# SOC version for handling Spin-Orbit coupling
# Edited using Qiming Sun's fci_slow
# Author: Ushnish Ray <ushnish.qmc@gmail.com>
#

import numpy
import pyscf.lib
from pyscf.fci import cistring_soc

#Note f1e needs to be in the correct format
# f1e[0] corresponds to the \sum_{ij} f1e[0][i][j] c+_ia c_jb
# f1e[1] corresponds to the \sun_{ij} f1e[1][i][j] c+_ib c_ja

def contract_1esoc_soc(f1e, fcivec, norb, nelec):

    fcinew = numpy.zeros_like(fcivec,dtype=numpy.complex64)   
    goffset = 0
 
    for neleca in range(nelec+1):
        nelecb = nelec - neleca

        na = cistring_soc.num_strings(norb, neleca)
        nb = cistring_soc.num_strings(norb, nelecb)
        ci0 = fcivec[goffset:goffset+na*nb].reshape(na,nb)

	if(nelecb>0):	
  	    #c+_a c_b
            link_index, vv = cistring_soc.gen_linkstr_index_o0_soc(range(norb), neleca, nelecb, 0)
            nna = cistring_soc.num_strings(norb, neleca+1)
            nnb = cistring_soc.num_strings(norb, nelecb-1)
	    t1 = numpy.zeros((norb,norb,nna,nnb),dtype=numpy.complex64)
            for str0, tab in enumerate(link_index):
                for loc, (a, i, target, sign) in enumerate(tab):
		    t1[a,i,target[0],target[1]] += sign * ci0[vv[str0][loc],str0]
            ci0 = numpy.dot(f1e[0].reshape(-1), t1.reshape(-1,nna*nnb))
            fcinew[goffset+na*nb:goffset+na*nb+nna*nnb] += ci0.reshape(-1)

	#c+_b c_a
	if(neleca>0):
            ci0 = fcivec[goffset:goffset+na*nb].reshape(na,nb)

            link_index, vv = cistring_soc.gen_linkstr_index_o0_soc(range(norb), neleca, nelecb, 1)
            nna = cistring_soc.num_strings(norb, neleca-1)
            nnb = cistring_soc.num_strings(norb, nelecb+1)
	    t1 = numpy.zeros((norb,norb,nna,nnb),dtype=numpy.complex64)
            for str0, tab in enumerate(link_index):
                for loc, (a, i, target, sign) in enumerate(tab):
                    t1[a,i,target[0],target[1]] += sign * ci0[str0,vv[str0][loc]]

            ci0 = numpy.dot(f1e[1].reshape(-1), t1.reshape(-1,nna*nnb))
            fcinew[goffset-nna*nnb:goffset] += ci0.reshape(-1)
	
        goffset += na*nb   
 
    return fcinew.reshape(fcivec.shape)

def contract_1e_soc(f1e, fcivec, norb, nelec):

    fcinew = numpy.zeros_like(fcivec,dtype=numpy.complex64)   
    goffset = 0
 
    for neleca in range(nelec+1):
        nelecb = nelec - neleca

        link_indexa = cistring_soc.gen_linkstr_index_o0(range(norb), neleca)
        link_indexb = cistring_soc.gen_linkstr_index_o0(range(norb), nelecb)
        
        na = cistring_soc.num_strings(norb, neleca)
        nb = cistring_soc.num_strings(norb, nelecb)
        ci0 = fcivec[goffset:goffset+na*nb].reshape(na,nb)

        t1 = numpy.zeros((norb,norb,na,nb),dtype=numpy.complex64)
        for str0, tab in enumerate(link_indexa):
            for a, i, str1, sign in tab:
                t1[a,i,str1] += sign * ci0[str0]
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[a,i,:,str1] += sign * ci0[:,str0]

        ci0 = numpy.dot(f1e.reshape(-1), t1.reshape(-1,na*nb))
        fcinew[goffset:goffset+na*nb] = ci0.reshape(-1)
	goffset += na*nb   

    return fcinew.reshape(fcivec.shape)


def contract_2e_soc(eri, fcivec, norb, nelec, opt=None):
   
    fcinew = numpy.zeros_like(fcivec,dtype=numpy.complex64)
    goffset = 0

    for neleca in range(nelec+1):
	nelecb = nelec - neleca

	na = cistring_soc.num_strings(norb, neleca)
	nb = cistring_soc.num_strings(norb, nelecb)
	ci0 = fcivec[goffset:goffset+na*nb].reshape(na,nb)	   	

    	link_indexa = cistring_soc.gen_linkstr_index_o0(range(norb), neleca)
	link_indexb = cistring_soc.gen_linkstr_index_o0(range(norb), nelecb)

    	t1 = numpy.zeros((norb,norb,na,nb),dtype=numpy.complex64)
    	for str0, tab in enumerate(link_indexa):
	    for a, i, str1, sign in tab:
	        t1[a,i,str1] += sign * ci0[str0]

    	for str0, tab in enumerate(link_indexb):
	    for a, i, str1, sign in tab:
	        t1[a,i,:,str1] += sign * ci0[:,str0]

    	t1 = numpy.dot(eri.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    	t1 = t1.reshape(norb,norb,na,nb)
    	ci0 = numpy.zeros_like(ci0,dtype=numpy.complex64)
    	for str0, tab in enumerate(link_indexa):
	    for a, i, str1, sign in tab:
	        ci0[str1] += sign * t1[a,i,str0]
        for str0, tab in enumerate(link_indexb):
	    for a, i, str1, sign in tab:
	        ci0[:,str1] += sign * t1[a,i,:,str0]

        fcinew[goffset:goffset+na*nb] = ci0.reshape(-1)
	goffset += na*nb
	
    return fcinew

def contract_2e_hubbard_soc(u, fcivec, norb, nelec, opt=None):
    u_aa, u_ab, u_bb = u

    fcinew = numpy.zeros_like(fcivec,dtype=numpy.complex64)
    goffset = 0

    for neleca in range(nelec+1):
        nelecb = nelec - neleca   	 

        strsa = cistring.gen_strings4orblist(range(norb), neleca)
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)

        na = cistring.num_strings(norb, neleca)
        nb = cistring.num_strings(norb, nelecb)
   
        ci0 = fcivec[goffset:goffset+na*nb].reshape(na,nb)
        t1a = numpy.zeros((norb,na,nb),dtype=numpy.complex64)
        t1b = numpy.zeros((norb,na,nb),dtype=numpy.complex64)

        for addr, s in enumerate(strsa):
            for i in range(norb):
                if s & (1<<i):
                    t1a[i,addr] += ci0[addr]
        for addr, s in enumerate(strsb):
            for i in range(norb):
                if s & (1<<i):
                    t1b[i,:,addr] += ci0[:,addr]

	ci0 = numpy.zeros_like(ci0,dtype=numpy.complex64)
        if u_aa != 0:
        # u * n_alpha^+ n_alpha
            for addr, s in enumerate(strsa):
                for i in range(norb):
                    if s & (1<<i):
                        ci0[addr] += t1a[i,addr] * u_aa
        if u_ab != 0:
        # u * n_alpha^+ n_beta
            for addr, s in enumerate(strsa):
                for i in range(norb):
                    if s & (1<<i):
                        ci0[addr] += t1b[i,addr] * u_ab

        # u * n_beta^+ n_alpha
        for addr, s in enumerate(strsb):
            for i in range(norb):
                if s & (1<<i):
                        ci0[:,addr] += t1a[i,:,addr] * u_ab
        if u_bb != 0:
        # u * n_beta^+ n_beta
            for addr, s in enumerate(strsb):
                for i in range(norb):
                    if s & (1<<i):
                        ci0[:,addr] += t1b[i,:,addr] * u_bb

	fcinew[goffset:goffset+na*nb] = ci0
        goffset += na*nb

    return fcinew

def absorb_h1e_soc(h1e, eri, norb, nelec, fac=1):
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
    return h2e * fac


def make_hdiag_soc(h1e, g2e, norb, nelec, opt=None):
    '''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    '''
    hdiag = []

    g2e = pyscf.ao2mo.restore(1, g2e, norb)
    diagj = numpy.einsum('iijj->ij',g2e)
    diagk = numpy.einsum('ijji->ij',g2e)

    for neleca in range(nelec+1):
	nelecb = nelec - neleca

	offseta = cistring_soc.num_strings_soc(norb,neleca)
	offsetb = cistring_soc.num_strings_soc(norb,nelecb)

	if neleca == 0:

	    link_indexb = cistring_soc.gen_linkstr_index_o0(range(norb), nelecb)
	    occslistb = [tab[:nelecb,0] for tab in link_indexb]
	    aocc = offseta
	    for boccb in occslistb:
	        e1 = h1e[boccb,boccb].sum()
	        e2 = diagj[boccb][:,boccb].sum() - diagk[boccb][:,boccb].sum() 
	        hdiag.append(e1 + e2*.5)

	elif nelecb == 0:

      	    link_indexa = cistring_soc.gen_linkstr_index_o0(range(norb), neleca)
   	    occslista = [tab[:neleca,0] for tab in link_indexa]

	    bocc = offsetb
            for aoccb in occslista:
	        e1 = h1e[aoccb,aoccb].sum() 
	        e2 = diagj[aoccb][:,aoccb].sum() - diagk[aoccb][:,aoccb].sum()
	        hdiag.append(e1 + e2*.5)

	else:

      	    link_indexa = cistring_soc.gen_linkstr_index_o0(range(norb), neleca)
	    link_indexb = cistring_soc.gen_linkstr_index_o0(range(norb), nelecb)

   	    occslista = [tab[:neleca,0] for tab in link_indexa]
	    occslistb = [tab[:nelecb,0] for tab in link_indexb]
	
            for aoccb in occslista:
	        for boccb in occslistb:
	            e1 = h1e[aoccb,aoccb].sum() + h1e[boccb,boccb].sum()
	            e2 = diagj[aoccb][:,aoccb].sum() + diagj[aoccb][:,boccb].sum() \
	                 + diagj[boccb][:,aoccb].sum() + diagj[boccb][:,boccb].sum() \
	                 - diagk[aoccb][:,aoccb].sum() - diagk[boccb][:,boccb].sum()
	            hdiag.append(e1 + e2*.5)

    return numpy.array(hdiag)

def kernel(h1e, hsoc, g2e, norb, nelec):

    h2e = absorb_h1e_soc(h1e, g2e, norb, nelec, .5)

    def hop(c):
        hc = contract_2e_soc(h2e, c, norb, nelec) + contract_1esoc_soc(hsoc,c,norb,nelec)
	return hc.reshape(-1)

    e = 0.0
    hdiag = make_hdiag_soc(h1e, g2e, norb, nelec)

    na = hdiag.shape[0]
    ci0 = numpy.zeros(na,numpy.complex64)

    #Need to allow mixing otherwise no overlap
    #This will not be an issue with SOC switched on	
    
    goffset = 0
    for neleca in range(nelec+1):
	nelecb = nelec - neleca
	na = cistring_soc.num_strings(norb, neleca)
	nb = cistring_soc.num_strings(norb, nelecb)
       	goffset += na*nb 
    ci0 = numpy.random.random(goffset)
    ci0 /= numpy.linalg.norm(ci0)

    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = pyscf.lib.davidson(hop, ci0.reshape(-1), precond, max_cycle=100, max_space=100, tol=1.e-15)
    return e, c


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):

    rdm1 = numpy.zeros((2,2,norb,norb),dtype=numpy.complex64)    
    goffset = 0

    for neleca in range(nelec+1):
        nelecb = nelec - neleca

        link_indexa = cistring_soc.gen_linkstr_index_o0(range(norb), neleca)
        link_indexb = cistring_soc.gen_linkstr_index_o0(range(norb), nelecb)
        
        na = cistring_soc.num_strings(norb, neleca)
        nb = cistring_soc.num_strings(norb, nelecb)
        ci0 = fcivec[goffset:goffset+na*nb].reshape(na,nb)

	#sector a,a
	#spin a
        for str0, tab in enumerate(link_indexa):
            for a, i, str1, sign in tab:
                rdm1[0,0,a,i] += sign * numpy.dot(ci0[str1].conjugate(),ci0[str0])
        
	#sector b,b
	#spin b
	for str0, tab in enumerate(link_indexb):
            for k in range(na):
                for a, i, str1, sign in tab:
                    rdm1[1,1,a,i] += sign * ci0[k,str1].conjugate()*ci0[k,str0]

	#sector a,b
	if(nelecb>0):	
  	    #c+_a c_b
            link_indexa, vv = cistring_soc.gen_linkstr_index_o0_soc(range(norb), neleca, nelecb, 0)
            nna = cistring_soc.num_strings(norb, neleca+1)
            nnb = cistring_soc.num_strings(norb, nelecb-1)
            ciT = fcivec[goffset+na*nb:goffset+na*nb+nna*nnb].reshape(nna,nnb)
            for str0, tab in enumerate(link_indexa):
                for loc, (a, i, target, sign) in enumerate(tab):
		    #print a,i,target,sign, str0, vv[str0], ci0[tstr,str0], ciT[target[0],target[1]]
                    tstr = vv[str0][loc]
                    rdm1[0,1,a,i] += sign * ci0[tstr,str0] * ciT[target[0],target[1]].conjugate()

	if(neleca>0):
	    #c+_b c_a
            link_indexb,vv = cistring_soc.gen_linkstr_index_o0_soc(range(norb), neleca, nelecb, 1)
            nna = cistring_soc.num_strings(norb, neleca-1)
            nnb = cistring_soc.num_strings(norb, nelecb+1)
            ciT = fcivec[goffset-nna*nnb:goffset].reshape(nna,nnb)
            for str0, tab in enumerate(link_indexb):
                for loc, (a, i, target, sign) in enumerate(tab):
                    #for tstr in vv[str0]:
                    tstr = vv[str0][loc]
                    rdm1[1,0,a,i] += sign * ci0[str0,tstr] * ciT[target[0],target[1]].conjugate()
	
	goffset += na*nb   
 
    return rdm1 

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):

    goffset = 0
    t1 = numpy.zeros((2,2,len(fcivec),norb,norb),dtype=numpy.complex64)
    for neleca in range(nelec+1):
        nelecb = nelec - neleca

        link_indexa = cistring_soc.gen_linkstr_index_o0(range(norb), neleca)
        link_indexb = cistring_soc.gen_linkstr_index_o0(range(norb), nelecb)
        
        na = cistring_soc.num_strings(norb, neleca)
        nb = cistring_soc.num_strings(norb, nelecb)
        ci0 = fcivec[goffset:goffset+na*nb].reshape(na,nb)
        
        #sector a,a
        #spin a
        for str0, tab in enumerate(link_indexa):
            for a, i, str1, sign in tab:
                for k in range(nb):
                    t1[0,0,goffset+str1*nb+k,a,i] += sign * ci0[str0,k]
        
        #sector b,b
        #spin b
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                for k in range(na):
                    t1[1,1,goffset+k*nb+str1,a,i] += sign * ci0[k,str0]

        #sector a,b
        if(nelecb>0):    
            #c+_a c_b
            link_indexa, vv = cistring_soc.gen_linkstr_index_o0_soc(range(norb), neleca, nelecb, 0)
            nna = cistring_soc.num_strings(norb, neleca+1)
            nnb = cistring_soc.num_strings(norb, nelecb-1)
            
            for str0b, tab in enumerate(link_indexa):
                for loc, (a, i, target, sign) in enumerate(tab):
                        str0a = vv[str0][loc]
                        #print a,i,target,sign, str0, vv[str0], ci0[tstr,str0], ciT[target[0],target[1]]
                        targetloc = goffset + na*nb + nnb*target[0] + target[1]
                        t1[0,1,targetloc,a,i] += sign*ci0[str0a,str0b]
                          
        if(neleca>0):
            #c+_b c_a
            link_indexb,vv = cistring_soc.gen_linkstr_index_o0_soc(range(norb), neleca, nelecb, 1)
            nna = cistring_soc.num_strings(norb, neleca-1)
            nnb = cistring_soc.num_strings(norb, nelecb+1)
            
            for str0a, tab in enumerate(link_indexb):
                for loc, (a, i, target, sign) in enumerate(tab):
                        str0b = vv[str0][loc]
                        targetloc = goffset-nna*nnb+nnb*target[0]+target[1]
                        t1[1,0,targetloc,a,i] += sign * ci0[str0a,str0b]
    
        goffset += na*nb   
    
    
    #Now construct the different sectors for rdm2
    rdm1 = numpy.zeros((2,2,norb,norb),dtype=numpy.complex64)    
    rdm2 = numpy.zeros((2,2,2,2,norb,norb,norb,norb),dtype=numpy.complex64)    
    
    for a in range(2):
        for b in range(2):
            rdm1[a,b] = numpy.einsum('m,mij->ij', fcivec.conjugate(), t1[a,b]) 
            for c in range(2):
                for d in range(2):
                    rdm2[a,b,c,d] = numpy.einsum('mij,mkl->jikl',t1[a,b].conjugate(),t1[c,d])
    
    #Remove additional contribution
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for k in range(norb):
                    rdm2[a,b,b,c,:,k,k,:] -= rdm1[a,c]
    
    return rdm1, rdm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
#    mol.atom = [
#        ['H', ( 1.,-1.    , 0.   )],
#        ['H', ( 0.,-1.    ,-1.   )],
#        ['H', ( 1.,-0.5   ,-1.   )],
#        ['H', ( 0.,-0.    ,-1.   )],
#        ['H', ( 1.,-0.5   , 0.   )],
#        ['H', ( 0., 1.    , 1.   )],
#    ]

    mol.atom = [['H', (1.*i,0,0)] for i in range(0,6)]     
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron 
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1, psi0 = kernel(h1e, [], eri, norb, nelec)
    print e1
	
    rdm1 = make_rdm1(psi0,norb,nelec)
    print rdm1


#    print(e1, e1 - -7.9766331504361414)
