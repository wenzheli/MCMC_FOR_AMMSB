import numpy as np
cimport numpy as np
cimport cython
from scipy.special import psi
import math
import copy



def reorder(int i, int j):
    if i < j:
        return (i,j)
    else:
        return (j,i)

def update_phi(int i, int itr, double epsilon, int K, int N, double eps_t, double alpha,
                                            np.ndarray[double, ndim=2] pi,
                                            np.ndarray[double, ndim=2] phi, 
                                            np.ndarray[double, ndim=1] beta,
                                            np.ndarray[double, ndim=1] noise,
                                            int n, linked_edges, neighbor_nodes):
    cdef int j 
    cdef int k 
    cdef int y 
    cdef double sum_phi = 0.0
    cdef np.ndarray[double, ndim=1] grads = np.zeros(K)
    cdef np.ndarray[double, ndim=1] phi_star = np.zeros(K)
    cdef np.ndarray[double, ndim=1] probs = np.zeros(K)
    cdef double sum_prob
    
    sum_phi = np.sum(phi[i])
    
    for j in neighbor_nodes:
        if i == j:
            continue
        
        y = 0
        if reorder(i,j) in linked_edges:
            y = 1
        
        for k in range(0, K):
            probs[k] = beta[k]**y * (1-beta[k])**(1-y)*pi[i][k]*pi[j][k]
            probs[k] += epsilon**y *(1-epsilon)**(1-y)*pi[i][k]*(1-pi[j][k])
        
        sum_prob = np.sum(probs)     
        for k in range(0, K):
            grads[k] += (probs[k]/sum_prob)/phi[i][k] - 1.0/sum_phi
           

    # update phi
    for k in range(0, K):
        if n == 0:          
            phi_star[k] = abs(phi[i,k] + eps_t/2 * (alpha - phi[i,k]) + eps_t**.5*phi[i,k]**.5 * noise[k])
        else:
            phi_star[k] = abs(phi[i,k] + eps_t/2 * (alpha - phi[i,k] + \
                           (N/n) * grads[k]) + eps_t**.5*phi[i,k]**.5 * noise[k])
        
    return phi_star


def sample_from_distribution(np.ndarray[double, ndim=2] p, int K):
    cdef int n = K * K
    cdef np.ndarray[double, ndim=1] temp = np.zeros(n)
    cdef int cnt = 0
    cdef int i, idx 
    cdef double u
    
    for i in range(0, K):
        for j in range(0, K):
            temp[cnt] = p[i][j]
            cnt += 1
    
    for i in range(1, n):
        temp[i] += temp[i-1];
    
    
    u = random.random() * temp[n-1]
    idx = 0
    for i in range(0, n):
        if u <= temp[i]:
            idx = i
            break
                
    return (int(idx/K), int(idx%K))

def gibbs_sampler(int K, double epsilon, int y, double alpha,
                  np.ndarray[int, ndim=2] num_n_k,
                  np.ndarray[int, ndim=2] num_kk,
                  np.ndarray[double, ndim=1] eta):
    cdef int k1, k2
    cdef double term
    cdef np.ndarray[double, ndim=2] p = np.zeros((K,K))
    
    for k1 in range(0, self._K):
        for k2 in range(0, self._K):
            if k1 != k2:
                if y == 1:
                    term = epsilon
                else:
                    term = 1 - epsilon
                p[k1][k2] = (alpha + num_n_k[i][k1]) * (alpha + self.num_n_k[j][k2])\
                                        * term
                
            else:
                if y == 1:
                    term = (num_kk[k1][0] +eta[0])/(num_kk[k1][0]+num_kk[k1][1]\
                                                                            +eta[0]+eta[1])
                else:
                    term = (self.num_kk[k1][1] + self._eta[1])/(self.num_kk[k1][0]+self.num_kk[k1][1]\
                                                                            +self._eta[0]+self._eta[1])
                p[k1][k2] = (alpha + num_n_k[i][k1]) * (alpha + num_n_k[j][k2])* term 
                                                   
    return sample_from_distribution(p, K)
       