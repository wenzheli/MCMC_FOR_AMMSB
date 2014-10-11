from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
ctypedef np.uint32_t uitype_t
cdef extern from "stdlib.h"

def add(np.ndarray[double, ndim=1] score, int count):
    cdef int i =0
    cdef int sum = 0
    for i in range(0,count):
        sum +=rand()
        
    return sum

        



def primes(int kmax):
    cdef int n, k, i
    cdef int p[1000]
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n    
            k = k + 1
            result.append(n)
        n = n + 1
    return result