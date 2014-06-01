
import random
from com.uva.edge import Edge
import math
import numpy as np
import copy
from sets import Set
import cProfile, pstats, StringIO
from scipy.special import psi
from com.uva.learning.learner import Learner
from com.uva.estimate_phi import sample_latent_vars_for_each_pair


class SV(Learner):
    def __init__(self, args, graph):
        '''
        Initialize the sampler using the network object and arguments (i.e prior)
        Arguments:
            network:    representation of the graph. 
            args:       containing priors, control parameters for the model. 
        '''
        Learner.__init__(self, args, graph)
        
        # variational parameters. 
        self.__lamda = np.random.gamma(self._eta[0],self._eta[1],(self._K, 2))      # variational parameters for beta  
        self.__gamma = np.random.gamma(1,1,(self._N, self._K)) # variational parameters for pi
        self.__update_pi_beta()
        # step size parameters. 
        self.__kappa = args.b
        self.__tao = args.c
        
        # control parameters for learning 
        self.__online_iterations = 50
        self.__phi_update_threshold = 0.0001
        
        
    def run(self):
        pr = cProfile.Profile()
        pr.enable()
        
        # running until convergence.
        self._step_count += 1      
            
        while self._step_count < self._max_iteration and not self._is_converged(): 
                
            ppx_score = self._cal_perplexity_held_out()
            print "perplexity for hold out set is: "  + str(ppx_score)
                  
            self.__update()
            self.__update_pi_beta()
            
            self._step_count += 1
        
        
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
         
    
    def __update(self):
        gamma_grad = np.zeros((self._N, self._K))
        lamda_grad = np.zeros((self._K, 2))
        
        for a in range(0, self._N):
            for b in range(a+1, self._N):
                if (a,b) in self._network.get_held_out_set() or (a,b) in self._network.get_test_set():
                    continue
                
        
                (phi_ab, phi_ba) = sample_latent_vars_for_each_pair(a, b, self.__gamma[a], self.__gamma[b],
                                                                self.__lamda, self._K, self.__phi_update_threshold,
                                                                self._epsilon, self.__online_iterations, 
                                                                self._network.get_linked_edges())
                
                print str(phi_ab)
                print str(phi_ba)
                
                # update gamma_grad and lamda_grad
                gamma_grad[a] += phi_ab
                gamma_grad[b] += phi_ba
                
                y = 0
                if (a,b) in self._network.get_linked_edges():
                    y = 1
                
                for k in range(self._K):
                    lamda_grad[k][0] += phi_ab[k] * phi_ba[k] * y 
                    lamda_grad[k][1] += phi_ab[k] * phi_ba[k] * (1-y) 
        
        # update gamma, only update node in the grad
        if self.stepsize_switch == False:
            p_t = (1024 + self._step_count)**(-0.5)
        else:
            p_t = 0.01*(1+self._step_count/1024.0)**(-0.55)
            
        for node in range(0, self._K):
            gamma_star = np.zeros(self._K)
            
            if self._step_count > 400:
                gamma_star = (1-p_t)*self.__gamma[node] + p_t * (self._alpha + gamma_grad[node])
                self.__gamma[node] = (1-1.0/(self._step_count))*self.__gamma[node] + gamma_star
            else:
                self.__gamma[node]=(1-p_t)*self.__gamma[node] + p_t * (self._alpha + gamma_grad[node])
        
        # update lamda
        for k in range(self._K):
            
            if self._step_count > 400:
                lamda_star_0 = (1-p_t)*self.__lamda[k][0] + p_t *(self._eta[0] + lamda_grad[k][0])
                lamda_star_1 = (1-p_t)*self.__lamda[k][1] + p_t *(self._eta[1] + lamda_grad[k][1])
                self.__lamda[k][0] = (1-1/(self._step_count)) * self.__lamda[k][0] +1/(self._step_count)*lamda_star_0
                self.__lamda[k][1] = (1-1.0/(self._step_count)) * self.__lamda[k][1] +1.0/(self._step_count)*lamda_star_1
            else:
                self.__lamda[k][0] = (1-p_t)*self.__lamda[k][0] + p_t *(self._eta[0] + lamda_grad[k][0])
                self.__lamda[k][1] = (1-p_t)*self.__lamda[k][1] + p_t *(self._eta[1] + lamda_grad[k][1])
            
        
                           
    def __update_pi_beta(self):
        
        self._pi = self.__gamma/np.sum(self.__gamma,1)[:,np.newaxis]
        temp = self.__lamda/np.sum(self.__lamda,1)[:,np.newaxis]
        self._beta = temp[:,1]
        
        
   
    def __estimate_phi_for_edge(self, edge, phi):
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
        
        a = edge[0]
        b = edge[1]
        # initialize 
        phi_ab = np.empty(self._K)
        phi_ba = np.empty(self._K)
        phi_ab.fill(1.0/self._K)
        phi_ba.fill(1.0/self._K)
        
        y = 0
        if (a,b) in self._network.get_linked_edges():
            y = 1
        
        # alternatively update phi_ab and phi_ba, until it converges
        # or reach the maximum iterations. 
        for i in range(self.__online_iterations):
            phi_ab_old = copy.copy(phi_ab)
            phi_ba_old = copy.copy(phi_ba)
            
            # first, update phi_ab
            for k in range(self._K):
                if y == 1:
                    u = -phi_ba[k]* math.log(self._epsilon)
                    phi_ab[k] = math.exp(psi(self.__gamma[a][k])+phi_ba[k]*\
                                         (psi(self.__lamda[k][0])-psi(self.__lamda[k][0]+self.__lamda[k][1]))+u)
                else:
                    u = -phi_ba[k]* math.log(1-self._epsilon)
                    phi_ab[k] = math.exp(psi(self.__gamma[a][k])+phi_ba[k]*\
                                         (psi(self.__lamda[k][1])-psi(self.__lamda[k][0]+self.__lamda[k][1]))+u)    
            sum_phi_ab = np.sum(phi_ab)
            phi_ab = phi_ab/sum_phi_ab
                
            # then update phi_ba
            for k in range(self._K):
                if y == 1:
                    u = -phi_ab[k]* math.log(self._epsilon)
                    phi_ba[k] = math.exp(psi(self.__gamma[b][k])+phi_ab[k]*\
                                         (psi(self.__lamda[k][0])-psi(self.__lamda[k][0]+self.__lamda[k][1]))+u)
                else:
                    u = -phi_ab[k]* math.log(1-self._epsilon)
                    phi_ba[k] = math.exp(psi(self.__gamma[b][k])+phi_ab[k]*\
                                         (psi(self.__lamda[k][1])-psi(self.__lamda[k][0]+self.__lamda[k][1]))+u)   
               
            sum_phi_ba = np.sum(phi_ba)
            phi_ba = phi_ba/sum_phi_ba
            
            
            # calculate the absolute difference between new value and old value
            diff1 = np.sum(np.abs(phi_ab - phi_ab_old))
            diff2 = np.sum(np.abs(phi_ba - phi_ba_old))
            if diff1 < self.__phi_update_threshold and diff2 < self.__phi_update_threshold:
                break
        
        phi[(a,b)] = phi_ab
        phi[(b,a)] = phi_ba 
        