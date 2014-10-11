from com.uva.learning.learner import Learner
from sets import Set
import math
import random
import numpy as np
import copy
from com.uva.sample_latent_vars import sample_z_ab_from_edge
import cProfile, pstats, StringIO
import time


class MCMCSamplerBatch(Learner):
    """
    MCMC Sampler for batch learning. Every update go through the whole data sets. 
    """
    def __init__(self, args, graph):
        # call base class initialization
        Learner.__init__(self, args, graph)
        
        # step size parameters. 
        self.__a = args.a
        self.__b = args.b
        self.__c = args.c
        
        # control parameters for learning
        self.__num_node_sample = int(math.sqrt(self._network.get_num_nodes())) 
        
        # model parameters and re-parameterization
        # since the model parameter - \pi and \beta should stay in the simplex, 
        # we need to restrict the sum of probability equals to 1.  The way we
        # restrict this is using re-reparameterization techniques, where we 
        # introduce another set of variables, and update them first followed by 
        # updating \pi and \beta.  
        self._theta = np.random.gamma(1,100,(self._K, 2))      # parameterization for \beta
        self._phi = np.random.gamma(1,1,(self._N, self._K))   # parameterization for \pi
        
        temp = self._theta/np.sum(self._theta,1)[:,np.newaxis]
        self._beta = temp[:,1]
        self._pi = self._phi/np.sum(self._phi,1)[:,np.newaxis]
        
        self._avg_log = []
        self._timing = []
    
    def __sample_neighbor_nodes_batch(self, node):
        neighbor_nodes = Set()
        for i in range(0, self._N):
            if ((min(node, i),max(node,i)) not in self._network.get_held_out_set()) and \
                 ((min(node, i),max(node,i)) not in self._network.get_test_set()):
                neighbor_nodes.add(i)
        
        return neighbor_nodes
    
    def __update_pi_for_node(self, i, z, phi_star, n):
        '''
        update pi for current node i. 
        ''' 
        # update gamma, only update node in the grad
        if self.stepsize_switch == False:
            eps_t = (1024+self._step_count)**(-0.5)
        else:
            eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)                                                                                                                                                                                                                                                                                                                          
    
        phi_i_sum = np.sum(self._phi[i])                                   
        noise = np.random.randn(self._K)                                 # random noise. 
        
        # get the gradients    
        grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
        for k in range(0, self._K):
            grads[k] += 1/self._phi[i,k] * z[k]
        
        # update the phi 
        for k in range(0, self._K):
            phi_star[i][k] = abs(self._phi[i,k] + eps_t/2 * (self._alpha - self._phi[i,k] + \
                                 grads[k]) + eps_t**.5*self._phi[i,k]**.5 * noise[k])
    
    def __update_beta(self):
        grads = np.zeros((self._K, 2))
        sum_theta = np.sum(self._theta,1)  
        
        # update gamma, only update node in the grad
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)
            
        for i in range(0, self._N):
            for j in range(i+1, self._N):
                y = 0
                if (i,j) in self._network.get_linked_edges():
                    y = 1
                
                probs = np.zeros(self._K)
                sum_pi = 0
                for k in range(0, self._K):
                    sum_pi += self._pi[i][k]*self._pi[j][k]
                    probs[k] = self._beta[k]**y*(1-self._beta[k])**(1-y)*self._pi[i][k]*self._pi[j][k]
                prob_0 = self._epsilon**y*(1-self._epsilon)**(1-y)*(1-sum_pi)
                sum_prob = np.sum(probs) + prob_0
                for k in range(0, self._K):
                    grads[k,0] += (probs[k]/sum_prob) * (abs(1-y)/self._theta[k,0]-1/sum_theta[k])
                    grads[k,1] += (probs[k]/sum_prob) * (abs(-y)/self._theta[k,1]-1/sum_theta[k])
            
        # update theta 
        noise = np.random.randn(self._K, 2)   
        theta_star = copy.copy(self._theta)  
        for k in range(0,self._K):
            for i in range(0,2):
                theta_star[k,i] = abs(self._theta[k,i] + eps_t/2 * (self._eta[i] - self._theta[k,i] + \
                                      grads[k,i]) + eps_t**.5*self._theta[k,i] ** .5 * noise[k,i])  
        
        self._theta = theta_star
        #self._theta = theta_star        
        # update beta from theta
        temp = self._theta/np.sum(self._theta,1)[:,np.newaxis]
        self._beta = temp[:,1]
      
    def __sample_z_for_each_edge(self, y, pi_a, pi_b, beta, K):
        '''
        sample latent variables z_ab and z_ba 
        but we don't need to consider all of the cases. i.e  (z_ab = j, z_ba = q) for all j and p. 
        because of the gradient depends on indicator function  I(z_ab=z_ba=k), we only need to consider
        K+1 different cases:  p(z_ab=0, z_ba=0|*), p(z_ab=1,z_ba=1|*),...p(z_ab=K, z_ba=K|*),.. p(z_ab!=z_ba|*)
         
        Arguments:
            y:        observation [0,1]
            pi_a:     community membership for node a
            pi_b:     community membership for node b
            beta:     community strengh. 
            epsilon:  model parameter. 
            K:        number of communities. 
        
        Returns the community index. If it falls into the case that z_ab!=z_ba, then return -1
        '''
        p = np.zeros(K+1)
        for k in range(0,K):
            p[k] = beta[k]**y*(1-beta[k])**(1-y)*pi_a[k]*pi_b[k]
        p[K] = 1 - np.sum(p[0:K])
         
        # sample community based on probability distribution p. 
        bounds = np.cumsum(p)
        location = random.random() * bounds[K]
        
        # get the index of bounds that containing location. 
        for i in range(0, K):
                if location <= bounds[i]:
                    return i
        return -1
        
          
    def __sample_latent_vars(self, node, neighbor_nodes):
        '''
        given a node and its neighbors (either linked or non-linked), return the latent value
        z_ab for each pair (node, neighbor_nodes[i]. 
        '''
        z = np.zeros(self._K)  
        for neighbor in neighbor_nodes:
            y_ab = 0      # observation
            if (min(node, neighbor), max(node, neighbor)) in self._network.get_linked_edges():
                y_ab = 1
            
            z_ab = sample_z_ab_from_edge(y_ab, self._pi[node], self._pi[neighbor], self._beta, self._epsilon, self._K)           
            z[z_ab] += 1
            
        return z
    
    def update_phi(self,i):
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)   
        """ update phi for node i"""
        sum_phi = np.sum(self._phi[i])
        grads = np.zeros(self._K)
        phi_star = np.zeros(self._K)
        noise = np.random.randn(self._K) 
        
        for j in range(0, self._N):
            """ for each node j """
            if i == j:
                continue
            y = 0
            if (min(i,j), max(i,j)) in self._network.get_linked_edges():
                y = 1
            
            probs = np.zeros(self._K)
            for k in range(0, self._K):
                # p(z_ij = k)
                probs[k] = self._beta[k]**y * (1-self._beta[k])**(1-y)*self._pi[i][k]*self._pi[j][k]
                probs[k] += self._epsilon**y *(1-self._epsilon)**(1-y)*self._pi[i][k]*(1-self._pi[j][k])
            
            sum_prob = np.sum(probs)
            for k in range(0, self._K):
                grads[k] += (probs[k]/sum_prob)/self._phi[i][k] - 1.0/sum_phi
        
        # update phi
        for k in range(0, self._K):
            phi_star[k] = abs(self._phi[i,k] + eps_t/2 * (self._alpha - self._phi[i,k] + \
                                grads[k]) + eps_t**.5*self._phi[i,k]**.5 * noise[k])
        
        self._phi[i] = phi_star  
        
    def _save(self):
        f = open('ppx_mcmc_batch.txt', 'wb')
        for i in range(0, len(self._avg_log)):
            f.write(str(math.exp(self._avg_log[i])) + "\t" + str(self._timing[i]) +"\n")
        f.close()
        
        
    def run(self):
        pr = cProfile.Profile()
        pr.enable()
     
        self._max_iteration = 300
        start = time.time()
        while self._step_count < self._max_iteration and not self._is_converged():
            #print "step: " + str(self._step_count)
            ppx_score = self._cal_perplexity_held_out()
            print str(ppx_score)
            self._ppxs_held_out.append(ppx_score)
            
            size = len(self._avg_log)
            self._avg_log.append(ppx_score)
            self._timing.append(time.time()-start)
            
            if self._step_count % 50 == 0:
                self._save()
            
            for i in range(0, self._N):
                # update parameter for pi_i
                #print "updating: " + str(i)
                #neighbor_nodes = self.__sample_neighbor_nodes_batch(i)
                self.update_phi(i)
                
            self._pi = self._phi/np.sum(self._phi,1)[:,np.newaxis]
            # update beta
            self.__update_beta()
                                  
            self._step_count += 1
            
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
                