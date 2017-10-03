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
    
    Nsite = 6
    Nelec = Nsite
    U = 4.0
    h1e = np.zeros([Nsite,Nsite])
    h2e = np.zeros((Nsite,)*4)
    for i in range(Nsite):
        h1e[i,(i+1)%Nsite] = -1.
        h1e[i,(i-1)%Nsite] = -1.
        h2e[i,i,i,i] = U

    T = np.linspace(0, 1.10,56)
    T = np.linspace(2.00, 3.00, 21)
#    T = np.array([0.0, 1.0, 50.0])
    E = Hubbard1d(Nsite,Nelec,U,T)/Nsite
    for i in range(len(T)):
        print('%1.2f           %1.9f'%(T[i],E[i]))
    
