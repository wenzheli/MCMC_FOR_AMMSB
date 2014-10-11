from __future__ import division
cimport cython
import random
import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
ctypedef np.uint32_t uitype_t
from random import randint
from libc.stdlib cimport rand
#cdef extern from "stdlib.h":
#    rand()

    
def sample_latent_vars_for_each_pair(int y_ab, 
                                     np.ndarray[double, ndim=1] pi_a,
                                     np.ndarray[double, ndim=1] pi_b,
                                     np.ndarray[double, ndim=1] beta, 
                                     double epsilon, int K):
        
        cdef int z_ab = 1
        cdef int z_ba = 1

        cdef int burn_in = 500           # stops sampling process after burn-in period. 
        cdef np.ndarray[double, ndim=1] topic_probs = np.zeros(K)
        cdef np.ndarray[double, ndim=1] bounds = np.zeros(K)
        cdef double location = 0.0
        cdef int k,i
        
    
        # run until burn-in...
        while burn_in > 0:
            # sample p(z_ab=k|*)
            z_ab = rand()%K
            #z_ab = random.randint(0, K-1)
            z_ba = rand()%K
            
            for k in range(K):
                topic_probs[k] = epsilon**y_ab*(1-epsilon)**(1-y_ab)*pi_b[z_ba]*pi_a[k]
            topic_probs[z_ba]=beta[z_ba]**y_ab*(1-beta[z_ba])**(1-y_ab)*pi_a[z_ba]*pi_b[z_ba]
            
            bounds[0] = topic_probs[0]
            for k in range(1,K):
                bounds[k] = bounds[k-1] + topic_probs[k]
                
            location = random.random() * bounds[K-1]
            for i in range(0, K-1):
                if location <= bounds[i]:
                    z_ab = i
                    break
         
            # sample p(z_ba=k|*)
            for k in range(K):
                topic_probs[k] = epsilon**y_ab*(1-epsilon)**(1-y_ab)*pi_b[z_ab]*pi_b[k]
            topic_probs[z_ab]=beta[z_ab]**y_ab*(1-beta[z_ab])**(1-y_ab)*pi_a[z_ab]*pi_b[z_ab]
            bounds[0] = topic_probs[0]
            for k in range(1,K):
                bounds[k] = bounds[k-1] + topic_probs[k]
            
           
            location = random.random() * bounds[K-1]
            for i in range(0, K-1):
                if location <= bounds[i]:
                    z_ba = i
                    break
                
            burn_in -= 1
        
        return (z_ab, z_ba)