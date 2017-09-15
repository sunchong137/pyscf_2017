from pyscf.ftsolver import ed_grand
import numpy as np

def Hubbard1d(Nsite, Nelec, U, T,symm='RHF'):
    h1e = np.zeros([Nsite,Nsite])
    h2e = np.zeros((Nsite,)*4)
    for i in range(Nsite):
        h1e[i,(i+1)%Nsite] = -1.
        h1e[i,(i-1)%Nsite] = -1.
        h2e[i,i,i,i] = U
    return ed_grand.kernel_fted(h1e,h2e,Nsite,Nelec,T,symm)
    

if __name__ == '__main__':
    
    Nsite = 10
    Nelec = 10
    U = 1
    T = np.linspace(0, 0.7,36)
    E = Hubbard1d(Nsite,Nelec,U,T)
    for i in range(len(T)):
        print('%1.2f           %2.10f'%(T[i],E[i]))
