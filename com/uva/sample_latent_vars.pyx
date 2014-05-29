from __future__ import division
cimport numpy as np
import numpy as np
from libc.stdlib cimport rand

def sample_z_ab_from_edge(int y, np.ndarray[double, ndim=1] pi_a, 
                               np.ndarray[double, ndim=1] pi_b,
                               np.ndarray[double, ndim=1] beta,
                               double epsilon, int K):
    '''
    we need to calculate z_ab. We can use deterministic way to calculate this
    for each k,  p[k] = p(z_ab=k|*) = \sum_{i}^{} p(z_ab=k, z_ba=i|*)
    then we simply sample z_ab based on the distribution p.
    this runs in O(K) 
    '''
    cdef np.ndarray[double, ndim=1] p = np.zeros(K)
    cdef np.ndarray[double, ndim=1] bounds = np.zeros(K)
    cdef double location  = 0.0
    cdef double tmp = 0.0
    
    for i in range(0, K):
        tmp = beta[i]**y*(1-beta[i])**(1-y)*pi_a[i]*pi_b[i]
        tmp += epsilon**y*(1-epsilon)**(1-y)*pi_a[i]*(1-pi_b[i])
        p[i] = tmp
    
    
    bounds[0] = p[0]
    for k in range(1,K):
        bounds[k] = bounds[k-1] + p[k]
    
    location = random.random() * bounds[K-1]
    # get the index of bounds that containing location. 
    for i in range(0, K):
            if location <= bounds[i]:
                return i
    
    # failed, should not happen!
    return -1
        
        