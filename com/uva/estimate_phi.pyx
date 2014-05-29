import numpy as np
cimport numpy as np
cimport cython
from scipy.special import psi
import math
import copy




def sample_latent_vars_for_each_pair(int a, int b, 
                                     np.ndarray[double, ndim=1] gamma_a,
                                     np.ndarray[double, ndim=1] gamma_b,
                                     np.ndarray[double, ndim=2] lamda, 
                                     int K, double update_threshold, double epsilon,
				     int online_iterations, linked_edges):
    
       
    '''
    calculate (phi_ab, phi_ba) for given edge : (a,b) 
    (a) calculate phi_ab given phi_ba
        if y =0:
    phi_ab[k]=exp(psi(gamma[a][k])+phi_ba[k]*(psi(lamda[k][1])-psi(lambda[k0]+lambda[k][1]))-phi_ba[k]*log(1-epsilon))
        if y=1:
    phi_ab[k]=exp(psi(gamma[a][k])+phi_ba[k]*(psi(lamda[k][0])-psi(lambda[k0]+lambda[k][1]))-phi_ba[k]*log(epsilon))
        
    (b) calculate phi_ba given phi_ab
        if y =0:
    phi_ba[k]=exp(psi(gamma[b][k])+phi_ab[k]*(psi(lamda[k][1])-psi(lambda[k0]+lambda[k][1]))-phi_ab[k]*log(1-epsilon))
        if y=1:
    phi_ba[k]=exp(psi(gamma[b][k])+phi_ab[k]*(psi(lamda[k][0])-psi(lambda[k0]+lambda[k][1]))-phi_ab[k]*log(epsilon))
    '''
        
      
    # initialize 
    cdef np.ndarray[double, ndim=1] phi_ab = np.empty(K)
    cdef np.ndarray[double, ndim=1] phi_ba = np.empty(K)
    phi_ab.fill(1.0/K)
    phi_ba.fill(1.0/K)
    
    cdef double u = 0.0 
    cdef int y = 0
    if (a,b) in linked_edges:
        y = 1
        
    # alternatively update phi_ab and phi_ba, until it converges
    # or reach the maximum iterations. 
    for i in range(online_iterations):
        phi_ab_old = copy.copy(phi_ab)
        phi_ba_old = copy.copy(phi_ba)
            
        # first, update phi_ab
        for k in range(K):
            if y == 1:
                u = -phi_ba[k]* math.log(epsilon)
                phi_ab[k] = math.exp(psi(gamma_a[k])+phi_ba[k]*\
                                        (psi(lamda[k][0])-psi(lamda[k][0]+lamda[k][1]))+u)
            else:
                u = -phi_ba[k]* math.log(1-epsilon)
                phi_ab[k] = math.exp(psi(gamma_a[k])+phi_ba[k]*\
                                         (psi(lamda[k][1])-psi(lamda[k][0]+lamda[k][1]))+u)    
        sum_phi_ab = np.sum(phi_ab)
        phi_ab = phi_ab/sum_phi_ab
                
        # then update phi_ba
        for k in range(K):
            if y == 1:
                u = -phi_ab[k]* math.log(epsilon)
                phi_ba[k] = math.exp(psi(gamma_b[k])+phi_ab[k]*\
                                        (psi(lamda[k][0])-psi(lamda[k][0]+lamda[k][1]))+u)
            else:
                u = -phi_ab[k]* math.log(1-epsilon)
                phi_ba[k] = math.exp(psi(gamma_b[k])+phi_ab[k]*\
                                        (psi(lamda[k][1])-psi(lamda[k][0]+lamda[k][1]))+u)   
               
        sum_phi_ba = np.sum(phi_ba)
        phi_ba = phi_ba/sum_phi_ba
            
        # calculate the absolute difference between new value and old value
        diff1 = np.sum(np.abs(phi_ab - phi_ab_old))
        diff2 = np.sum(np.abs(phi_ba - phi_ba_old))
        if diff1 < update_threshold and diff2 < update_threshold:
            break
        
    return (phi_ab, phi_ba) 
