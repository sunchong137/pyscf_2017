#!/usr/local/bin/python
""" This script is used to diagonalze a large sparce matrix with Lanczos method
    History: Adeline C. Sun Mar. 07 2016 first sketch
             Adeline C. Sun Mar. 21 2016 Restart Ohahaha
             Adeline C. Sun Mar. 25 2016 Tested all functions before really use
             them and cried to die in the restroom

    <chongs0419@gmail.com>
"""

import numpy as np
import math
import numpy.linalg as nl
import random

N = 100
M = 60
dens = 0.01
Minb = 10e-5
MinM = 10
Cut_M = 30
Nloop = 100

def gen_mat(L = N, d = dens):
    """Generate a sparse symmetry matrix. This will not really be used... Just for the test of the program
    """
    n = int(L ** 2 * d)
    mat = np.zeros((n, 3)) 
    for i in range(n//2):
        r = random.randint(0, L - 1)
        c = random.randint(r, L - 1) # c > r
        mat[i, 0], mat[i, 1] = r, c
        mat[i, 2] = random.uniform(-1., 1.)
    return mat

def vec_mat_vec(mat, vec): # <vector|matrix|vector> No problem
    res = 0.0
    for i in range(len(mat)):
        res += vec[int(mat[i, 0])] * mat[i, 2] * vec[int(mat[i, 1])]
        if int(mat[i, 0]) != int(mat[i, 1]): # diagonal terms count once
            res += vec[int(mat[i, 1])] * mat[i, 2] * vec[int(mat[i, 0])]
    return res

def nosym_vmv(mat, vec1, vec2):
    res = 0.0
    for i in range(len(mat)):
        res += vec1[int(mat[i, 0])] * mat[i, 2] * vec2[int(mat[i, 1])]
        if int(mat[i, 0]) != int(mat[i, 1]): 
            res += vec1[int(mat[i, 1])] * mat[i, 2] *vec2[int(mat[i, 0])]

    return res

def mat_vec(mat, vec): #matrix|vector>
    res = np.zeros(len(vec))
    for i in range(len(mat)):
        res[int(mat[i, 0])] += mat[i, 2] * vec[int(mat[i, 1])]
        if mat[i, 0] != mat[i, 1]: #diagonal terms count once
          res[int(mat[i, 1])] += mat[i, 2] * vec[int(mat[i, 0])]
    res = np.asarray(res)
    return res

def Lanczos(mat, n = N, m = M, minm = MinM, minb = Minb):
    """ Get the tridiagonal entries
    """
    goon = True
    brk = False
    cnt = 0
    while goon:
        if cnt == 100:
            print "Lanczos can't converge at current M and MinM!!"
            break
        
        brk = False
        cnt += 1
        
        a, b = [], [] # tri diagonal vectors of H
        v0 = np.random.randn(n)
        v0 = v0/np.linalg.norm(v0)
        a.append(vec_mat_vec(mat, v0))
        v1 = mat_vec(mat, v0) - a[0] * v0
        b.append(np.linalg.norm(v1))
        if abs(b[0]) < minb:
            print "****DOWN****"
            continue

        v1 = v1 / b[0]
        a.append(vec_mat_vec(mat, v1))
        
        for i in range(1, m - 1):
            v2 = mat_vec(mat, v1) - b[i - 1] * v0 - a[i] * v1
            b.append(np.linalg.norm(v2))
            v2 = v2/b[i]
            if abs(b[i]) < minb:
                if i < minm:
                    brk = True

                b.pop()
                break
            if brk == True:
                continue
            a.append(vec_mat_vec(mat, v2))
            v0 = v1.copy()
            v1 = v2.copy()
         
        break
            
    b = np.asarray(b)
    a = np.asarray(a)
    return a, b # vn is the trial vector

def Tri_diag(a, b):
    mat = np.diag(b, -1) + np.diag(a, 0) + np.diag(b, 1) # generate the tridiag matrix
    e, w = nl.eigh(mat)
    return e, w

#@profile
def FT_Lanczos(H, A, Beta, n = N, m = M):
    r"""Calculate the canonical ensemble average of $A$ with finite temperature Lanczos algorithm
    Suppose $A$ is also a sparse matrix, usually $A$ is just $H$.
    Here c is an array of $\langle r|\phi_j\rangle$, d is an array of $\langle\phi_j|A|r\rangle$
    """

    AA = 0.
    av_A, av_Z = 0., 0. # <A> * Z and Z
    for cnt in range(Nloop):
        do_again=False
        a, b = [], []
        d = []
#        v0 = np.zeros((7))
#        v0[cnt] = 1
        v0 = np.random.randn(n)
        v0 = v0/np.linalg.norm(v0)
        vn = v0.copy()
        Av = mat_vec(A, v0) # I didn't mean it...I'm pure and inocent

        a.append(vec_mat_vec(H, v0))
        d.append(v0.dot(Av))
        v1 = mat_vec(H, v0) - a[0] * v0
        b.append(np.linalg.norm(v1))
        v1 = v1/b[0]
        a.append(vec_mat_vec(H, v1))
        d.append(v1.dot(Av))

        for i in range(1, m - 1):
            v2 = mat_vec(H, v1) - b[i - 1] * v0 - a[i] * v1
            b.append(np.linalg.norm(v2))
            v2 = v2/b[i]
            if abs(b[i]) < 10e-10:
                b.pop()
                print "(FT:)********** M is truncated!!!***********"
                do_again=True
                break

            a.append(vec_mat_vec(H, v2))
            d.append(v2.dot(Av))
            v0 = v1.copy()
            v1 = v2.copy()
        if do_again==True:
            continue

        a = np.asarray(a)
        b = np.asarray(b)
        d = np.asarray(d)
        eps, phi = Tri_diag(a, b)
        eps = np.exp(-Beta * eps)

        for i in range(len(eps)):
            av_A += eps[i] * phi[0, i] * d.dot(phi[:, i])
            av_Z += eps[i] * phi[0, i] ** 2
        
        AA += av_A/av_Z

    AA = AA/Nloop
    return AA

def LT_Lanczos(H, A, Beta, n = N, m = M, norm = np.linalg.norm):
    r"""Calculate the canonical ensemble average of $A$ with low temperature Lanczos algorithm.
    Suppose $A$ is also a sparse matrix.
    """

    av_A = 0.
    av_Z = 0.
    for cnt in range(Nloop):
        a, b = [], []
        V, D = [], [] #V stores the m vectors, D stores vAv'
        v0 = np.random.randn(n)
        v0 = v0/norm(v0)
        V.append(v0)
        a.append(vec_mat_vec(H, v0))
        v1 = mat_vec(H, v0) - a[0] * v0
        b.append(norm(v1))
        if b[0] < Minb:
            continue
        v1 = v1/b[0]
        V.append(v1)
        a.append(vec_mat_vec(H, v1))

        for i in range(1, m - 1):
            v2 = mat_vec(H, v1) - b[i - 1] * v0 - a[i] * v1
            b.append(norm(v2))
            v2 = v2/b[i]
            if abs(b[i]) < 10e-10:
                b.pop()
                print "(LT)************* M is truncated*******************"
                break
            a.append(vec_mat_vec(H, v2))
            V.append(v2)
            v0 = v1.copy()
            v1 = v2.copy()

        a, b, V = np.asarray(a), np.asarray(b), np.asarray(V)
        eps, phi = Tri_diag(a, b)
        eps = np.exp(-Beta * eps/2)
        for i in range(len(eps)):
            D.append([i, i, vec_mat_vec(A, V[i, :])])
            for j in range(i + 1, len(eps)):
                D.append([i, j, nosym_vmv(A, V[i, :], V[j, :])])

        D = np.asarray(D)
        for i in range(len(eps)):
            for j in range(len(eps)):
                av_A += eps[i] * eps[j] * phi[0, i] *phi[0, j] * nosym_vmv(D, phi[:, i], phi[:, j])
            av_Z += eps[i] * eps[i] * phi[0, i] ** 2

    av_A = av_A/av_Z
    return av_A


def LT_Lanczos_new(H, A, Beta, n=N, m=M, minb=Minb, cutm=Cut_M, norm=np.linalg.norm):
    av_A = 0.
    av_Z = 0.
    cnt = Nloop
    while(cnt > 0):
        do_again=False
        a, b = [], []
        new_A = np.zeros((m, m))
        v0 = np.random.randn(n)
        v0 = v0/norm(v0)
        AV = mat_vec(A, v0)
        Hv = mat_vec(H, v0)
        a.append(v0.dot(Hv))
        new_A[0, 0] = v0.dot(AV)
        v1 = Hv - a[0] * v0
        b.append(norm(v1))
        if b[0] < Minb:
            av_A += v0.dot(mat_vec(A, v0))*math.exp(-Beta * a[0])
            av_Z += math.exp(-Beta * a[0])
            continue

        v1 = v1/b[0]
        Hv = mat_vec(H, v1)
        AV = np.vstack([AV, mat_vec(A, v1)])
        a.append(v1.dot(Hv))
        for i in range(2):
            new_A[i, 1] = v1.dot(AV[i, :])
        
        for i in range(1, m-1):
            v2 = Hv - b[i-1]*v0 - a[i]*v1
            b.append(norm(v2))
            if abs(b[i]) < 10e-10:
                if i < MinM:
                    do_again = True
                b.pop()
                print "LT_new:************M is truncated****************"
                break
                   
            v2 = v2/b[i]
            Hv = mat_vec(H, v2)
            a.append(v2.dot(Hv))
            AV = np.vstack([AV, mat_vec(A, v2)])
            for j in range(i+2):
                new_A[j, i+1]=v2.dot(AV[j, :])
            
            v0 = v1.copy()
            v1 = v2.copy()
        
        if do_again == True:
            continue
        
        a, b = np.asarray(a), np.asarray(b)
        for kr in range(1, len(a)):
            for kc in range(kr):
                new_A[kr, kc] = new_A[kc, kr]
         
        for i in range(1, m - len(a)+1):
            new_A = np.delete(new_A, m-i, 1)
            new_A = np.delete(new_A, m-i, 0)


        eps, phi = Tri_diag(a, b)
        eps = np.exp(-Beta*eps/2)
        for i in range(len(eps)):
            for j in range(len(eps)):
                av_A += eps[i]*eps[j] * phi[0, i] * phi[0, j] * phi[:, i].dot(new_A.dot(phi[:,j]))
            av_Z += (eps[i] * phi[0, i]) ** 2
        cnt -= 1
        
    av_A = av_A/av_Z
    return av_A


#TODO evaluate density matrix

def FT_Lanczos_E(H, Beta, n = N, m = M):
    r"""Calculate the energy with low temperature Lanczos. Simpler than
    LT_Lanczos because $H$ is tridiagonal in $|\phi\rangle$.
    """
    av_E, av_Z = 0., 0.
    for cnt in range(Nloop):
        a, b = [], []
        v0 = np.random.randn(n)
        v0 = v0/np.linalg.norm(v0)
        a.append(vec_mat_vec(H, v0))
        v1 = mat_vec(H, v0) - a[0] * v0
        b.append(np.linalg.norm(v1))
        v1 = v1/b[0]
        a.append(vec_mat_vec(H, v1))

        for i in range(1, m-1):
            v2 = mat_vec(H, v1) - b[i - 1] * v0 - a[i] * v1
            b.append(np.linalg.norm(v2))
            v2 = v2/b[i]
            if abs(b[i]) < 10e-10:
                b.pop()
                break

            a.append(vec_mat_vec(H, v2))
            v0 = v1.copy()
            v1 = v2.copy()

        a = np.asarray(a)
        b = np.asarray(b)
        eps, phi = Tri_diag(a, b)
        exp_eps = np.exp(-Beta * eps)
        for i in range(len(eps)):
            av_E += exp_eps[i] * eps[i] * phi[0, i] ** 2
            av_Z += exp_eps[i] * phi[0, i] ** 2

    av_E = av_E/av_Z
    return av_E


if __name__ == "__main__":
#Test of the functions w.r.t. exact diagonalzation
    F = np.array([[0, 0, 1.],
                  [0, 2, 0.3],
                  [2, 2, 0.5],
                  [1, 3, -0.22],
                  [2, 5, -0.1],
                  [3, 3, 0.07],
                  [4, 4, -0.7],
                  [5, 5, 0.4],
                  [6, 6, 0.1]])
    G = np.array([[1.0, 0.0, 0.3, 0, 0, 0, 0],
                  [0.0, 0.0, 0.0, -0.22, 0, 0, 0],
                  [0.3, 0.0, 0.5, 0, 0, -0.1, 0],
                  [0.0, -0.22, 0.0, 0.07, 0, 0, 0],
                  [0.0, 0.0, 0.0, 0, -0.7, 0, 0],
                  [0.0, 0.0, -0.1, 0, 0, 0.4, 0],
                  [0.0, 0.0, 0.0, 0, 0, 0, 0.1]])

    F1 = np.array([[0, 0, 1.],
                   [0, 1, 2.],
                   [1, 0, 3.],
                   [1, 1, 4.]])

    e, w = np.linalg.eig(G)
    e.sort()
    a, b = Lanczos(F, 7, 7, 7)
    x, y = Tri_diag(a, b)
    x.sort()
    print "e:", e
    print "x:", x

#Test of finite T functions
    def exact_E(vec, Beta): #vec is the vector of eigenvalues
        E = 0.
        Z = 0.
        for i in range(len(vec)):
            E += vec[i] * math.exp(-Beta * vec[i])
            Z += math.exp(-Beta * vec[i])
        return E/Z

    E0 = exact_E(e, 1)
    E1 = FT_Lanczos(F, F, 1, 7, 7)
    E2 = LT_Lanczos(F, F, 1, 7, 7)
    E3 = FT_Lanczos_E(F, 1, 7, 7)
    E4 = LT_Lanczos_new(F, F, 1, 7, 7)
    print "E0 = %0.12f"%E0   
    print "E1:(%0.12f, %0.12f)"%(E1, E1-E0)
    print "E2:(%0.12f, %0.12f)"%(E2, E2-E0)
    print "E3:(%0.12f, %0.12f)"%(E3, E3-E0)
    print "E4:(%0.12f, %0.12f)"%(E4, E4-E0)

#    print w

'''
# Test of FT_Lanczos, LT_Lanczos, LT_Lanczos_E, LT_Lanczos_new
    H = gen_mat() 
    av_A1 = FT_Lanczos(H, H, 20)
    print "A1 = ", av_A1
    av_A2 = LT_Lanczos(H, H, 20)
    print "A2 = ", av_A2
    av_A3 = FT_Lanczos_E(H, 20)
    av_A4 = LT_Lanczos_new(H, H, 20)
    print "A3 = ", av_A3
    print "A4 = ", av_A4
'''


