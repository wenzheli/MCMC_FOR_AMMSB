from com.uva.learning.learner import Learner
from sets import Set
import math
import random
import numpy as np
import copy
from com.uva.sample_latent_vars import sample_z_ab_from_edge
import cProfile, pstats, StringIO
import time
from com.uva.core_utils import gibbs_sampler
from array import *

class GibbsSampler(Learner):
    def __init__(self, args, graph):
        # call base class initialization
        Learner.__init__(self, args, graph)
        self._avg_log = []
        self._timing = []
        self._max_iteration = args.max_iteration
        
        # latent variables
        self.z = [ [ 0 for i in range(self._N) ] for j in range(self._N) ]
        
        # counting variables
        self.num_kk = np.zeros((self._K, 2))
        self.num_n_k = np.zeros((self._N, self._K))

        
        self._random_initialize()
    
    def _random_initialize(self):
        
        for i in range(0, self._N):
            for j in range(i+1, self._N):
                if (i,j) in self._network.get_held_out_set():
                    continue
                y = 0
                if (i,j) in self._network.get_linked_edges():
                    y = 1
                self.z[i][j] = random.randint(0, self._K-1)
                self.z[j][i] = random.randint(0, self._K-1)
                
                self.num_n_k[i][self.z[i][j]] += 1
                self.num_n_k[j][self.z[j][i]] += 1
                
                if self.z[i][j] == self.z[j][i]:
                    if y == 1:
                        self.num_kk[self.z[i][j]][0] += 1
                    else:
                        self.num_kk[self.z[i][j]][1] += 1
                            
    def run(self):
        itr = 0
        self._max_iteration = 500
        start = time.time()
        while itr < self._max_iteration:
            """
            pr = cProfile.Profile()
            pr.enable()
            """
            print "iteration: " + str(itr)
            self._process()
            self._update()
            ppx = self._cal_perplexity_held_out()
            
            if itr > 300:
                size = len(self._avg_log)
                ppx = (1-1.0/(itr-300)) * self._avg_log[size-1] + 1.0/(itr-300) * ppx
                self._avg_log.append(ppx)
            else:
                self._avg_log.append(ppx)
            
            self._timing.append(time.time()-start)
            
            if itr % 50 == 0:
                self._save()
            
            itr += 1
            """
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
            """
    def _save(self):
        f = open('ppx_gibbs_sampler.txt', 'wb')
        for i in range(0, len(self._avg_log)):
            f.write(str(math.exp(self._avg_log[i])) + "\t" + str(self._timing[i]) +"\n")
        f.close()
        
            
    def _update(self):
        # update pi
        for i in range(0, self._N):
            for j in range(0, self._K):
                self._pi[i][j] = self.num_n_k[i][j]/(1.0 *np.sum(self.num_n_k[i]))

        # update beta
        for k in range(0, self._K):
            self._beta[k] = (1+self.num_kk[k][0])*1.0/(self.num_kk[k][0]+self.num_kk[k][1]+1)
        
        
    def _sample_from_distribution(self, p, K):
        n = K * K;
        temp = np.zeros(n)
        cnt = 0
        
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
            
        k1_new = int(idx/K)
        k2_new = int(idx%K)
        
        return (k1_new, k2_new)
    
    def _process(self):
        """
        run one iteration of gibbs sampling
        """
        for i in range(0, self._N):
            for j in range(i+1, self._N):
                if (i,j) in self._network.get_held_out_set():
                    continue
                y = 0
                if (i,j) in self._network.get_linked_edges():
                    y = 1
                
                # remove current assignment
                z_ij_old = self.z[i][j]
                z_ji_old = self.z[j][i]
                
                self.num_n_k[i][z_ij_old] -= 1
                self.num_n_k[j][z_ji_old] -= 1
                
                if z_ij_old == z_ji_old:
                    if y == 1:
                        self.num_kk[z_ij_old][0] -= 1
                    else:
                        self.num_kk[z_ij_old][1] -= 1
                
                
                (k1_new, k2_new) = gibbs_sampler(i,j,self._K,self._epsilon, y,self._alpha,self.num_n_k,self.num_kk, self._eta )
                
                """
                # assign new values. 
                p = np.zeros((self._K, self._K))
                for k1 in range(0, self._K):
                    for k2 in range(0, self._K):
                        if k1 != k2:
                            if y == 1:
                                term = self._epsilon
                            else:
                                term = 1 - self._epsilon
                            p[k1][k2] = (self._alpha + self.num_n_k[i][k1]) * (self._alpha + self.num_n_k[j][k2])\
                                        * term
                        else:
                            if y == 1:
                                term = (self.num_kk[k1][0] +self._eta[0])/(self.num_kk[k1][0]+self.num_kk[k1][1]\
                                                                            +self._eta[0]+self._eta[1])
                            else:
                                term = (self.num_kk[k1][1] + self._eta[1])/(self.num_kk[k1][0]+self.num_kk[k1][1]\
                                                                            +self._eta[0]+self._eta[1])
                            p[k1][k2] = (self._alpha + self.num_n_k[i][k1]) * (self._alpha + self.num_n_k[j][k2])\
                                        * term
                
                (k1_new, k2_new) = self._sample_from_distribution(p, self._K)
        
                """
                self.z[i][j] = k1_new
                self.z[j][i] = k2_new
                
                self.num_n_k[i][k1_new] += 1
                self.num_n_k[j][k2_new] += 1
                
                if k1_new == k2_new:
                    if y == 1:
                        self.num_kk[k1_new][0] += 1
                    else:
                        self.num_kk[k1_new][1] += 1
                
                
                
                
                