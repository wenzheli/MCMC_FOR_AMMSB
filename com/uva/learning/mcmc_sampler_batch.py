from com.uva.learning.learner import Learner
from sets import Set
import math
import random
import numpy as np
import copy
from com.uva.sample_latent_vars import sample_z_ab_from_edge
import cProfile, pstats, StringIO



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
        self.__theta = np.random.gamma(self._eta[0],self._eta[1],(self._K, 2))      # parameterization for \beta
        self.__phi = np.random.gamma(1,1,(self._N, self._K))   # parameterization for \pi
        
        temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        self._beta = temp[:,1]
        self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
    
    
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
    
        phi_i_sum = np.sum(self.__phi[i])                                   
        noise = np.random.randn(self._K)                                 # random noise. 
        
        # get the gradients    
        grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
        for k in range(0, self._K):
            grads[k] += 1/self.__phi[i,k] * z[k]
        
        # update the phi 
        for k in range(0, self._K):
            phi_star[i][k] = abs(self.__phi[i,k] + eps_t/2 * (self._alpha - self.__phi[i,k] + \
                                 grads[k]) + eps_t**.5*self.__phi[i,k]**.5 * noise[k])
    
    def __update_beta(self):
        grads = np.zeros((self._K, 2))
        sums = np.sum(self.__theta,1)  
        
        # update gamma, only update node in the grad
        if self.stepsize_switch == False:
            eps_t = (1024+self._step_count)**(-0.5)
        else:
            eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)
            
        for i in range(0, self._N):
            for j in range(i+1, self._N):
                if (i,j) in self._network.get_held_out_set() or (i,j) in self._network.get_test_set():
                    continue
            
                y_ab = 0
                if (i,j) in self._network.get_linked_edges():
                    y_ab = 1
            
                z = self.__sample_z_for_each_edge(y_ab, self._pi[i], self._pi[j], \
                                          self._beta, self._K)   
                if z == -1:
                    continue
            
                grads[z,0] += abs(1-y_ab)/self.__theta[z,0] - 1/ sums[z]
                grads[z,1] += abs(-y_ab)/self.__theta[z,1] - 1/sums[z] 
        
        
        # update theta 
        noise = np.random.randn(self._K, 2)   
        theta_star = copy.copy(self.__theta)  
        for k in range(0,self._K):
            for i in range(0,2):
                theta_star[k,i] = abs(self.__theta[k,i] + eps_t/2 * (self._eta[i] - self.__theta[k,i] + \
                                      grads[k,i]) + eps_t**.5*self.__theta[k,i] ** .5 * noise[k,i])  
        
        if  self._step_count < 50000:
            self.__theta = theta_star 
        else:
            self.__theta = theta_star * 1.0/(self._step_count) + (1-1.0/(self._step_count))*self.__theta
        #self.__theta = theta_star        
        # update beta from theta
        temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
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
        
    
    def run(self):
        pr = cProfile.Profile()
        pr.enable()
        """ run mini-batch based MCMC sampler """
        while self._step_count < self._max_iteration and not self._is_converged():
            #print "step: " + str(self._step_count)
            ppx_score = self._cal_perplexity_held_out()
            print str(ppx_score)
            self._ppxs_held_out.append(ppx_score)
            
            phi_star = copy.copy(self._pi)
            # iterate through each node, and update parameters pi_a
            for i in range(0, self._N):
                # update parameter for pi_i
                #print "updating: " + str(i)
                neighbor_nodes = self.__sample_neighbor_nodes_batch(i)
                z = self.__sample_latent_vars(i, neighbor_nodes)
                self.__update_pi_for_node(i, z, phi_star, len(neighbor_nodes))
            
            self.__phi = phi_star
            self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
            
            # update beta
            z = self.__update_beta()
                                  
            self._step_count += 1
            
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
                