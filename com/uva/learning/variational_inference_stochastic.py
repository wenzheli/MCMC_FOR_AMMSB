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

class SVI(Learner):
    '''
    Stochastic variational inference for assortive mixture membership stochastic model. 
    The implementation is based on the paper: 
        http://www.cs.princeton.edu/~blei/papers/GopalanMimnoGerrishFreedmanBlei2012.pdf
    
    Formally, each node can be belong to multiple communities which we can represent it by 
    distribution of communities. For instance, if we assume there are total K communities
    in the graph, then each node a, is attached to community distribution \pi_{a}, where
    \pi{a} is K dimensional vector, and \pi_{ai} represents the probability that node a 
    belongs to community i, and \pi_{a0} + \pi_{a1} +... +\pi_{aK} = 1
    
    Also, there is another parameters called \beta representing the community strength, where 
    \beta_{k} is scalar. 
    
    In summary, the model has the parameters:
    Prior: \alpha, \eta
    Parameters: \pi, \beta
    Latent variables: z_ab, z_ba
    Observations: y_ab for every link. 
    
    And our goal is to estimate the posterior given observations and priors:
    p(\pi,\beta | \alpha,\eta, y). 
    
    Due to the intractability of this posterior, we adopt approximate inference - variational inference
    More specifically, using the mean-field variational inference. 
    
    In this implementation, we introduce sets of variational parameters. 
    q(z_ab) ~ Mul(phi_ab)     phi_ab is K dimensional vector, where sum equals to 1
    q(z_ba) ~ Mul(phi_ba)     phi_ba is K dimensional vector, where sum equals to 1
    q(pi_a) ~ Dir(gamma_a)    gamma_a is K dimensional vector,  for each node a. 
    q(beta_k) ~ Beta(lamda_k) lamda_k is 2 dimensional vector, each denotes the beta shape parameters. 
    
    TODO:  need to create base class for sampler, and MCMC sampler and variational inference should inherit
           from that base class.
    '''
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
        '''
        stochastic variational optimization. 
        while not converge
            sample mini-batch node pairs E_t  from network (trianing)
            for each node pair (a,b) in E_t
                optimize (phi_ab, phi_ba) 
            for n = [0,..,N-1], k=[0,..K-1]
                calculate the gradient for gamma_nk
            for k = [0,,,K-1], i=[0,1]
                calculate the gradient for lammda_ki
            update (gamma, lamda) using gradient:
                new_value = (1-p_t)*old_value + p_t * new value. 
        '''
        
        pr = cProfile.Profile()
        pr.enable()
        
        # running until convergence.
        self._step_count += 1      
            
        while self._step_count < self._max_iteration and not self._is_converged(): 
            (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
            
             # evaluate model after processing every 10 mini-batches. 
            if self._step_count % 2 ==  1:
                ppx_score = self._cal_perplexity_held_out()
                print "perplexity for hold out set is: "  + str(ppx_score)
                self._ppxs_held_out.append(ppx_score)
                if ppx_score < 13.0:
                    # we will use different step size schema
                    self.stepsize_switch = True
            
            # update (phi_ab, phi_ba) for each edge
            phi = {}               # mapping (a,b) => (phi_ab, phi_ba)
            self.__sample_latent_vars_for_edges(phi, mini_batch)
            self.__update_gamma_and_lamda(phi, mini_batch, scale)
            self.__update_pi_beta()
            
            self._step_count += 1
        
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()    
    def __sample_latent_vars_for_edges(self, phi, mini_batch):
        
        for edge in mini_batch:
            a = edge[0]
            b = edge[1]
            #self.__estimate_phi_for_edge(edge, phi)  # this can be done in parallel. 
            
            (phi_ab, phi_ba) = sample_latent_vars_for_each_pair(a, b, self.__gamma[a], self.__gamma[b],
                                                                self.__lamda, self._K, self.__phi_update_threshold,
                                                                self._epsilon, self.__online_iterations, 
                                                                self._network.get_linked_edges())
            phi[(a,b)]=phi_ab
            phi[(b,a)]=phi_ba
            
            #self.__estimate_phi_for_edge(edge, phi)
            
    
    def __update_pi_beta(self):
        
        self._pi = self.__gamma/np.sum(self.__gamma,1)[:,np.newaxis]
        temp = self.__lamda/np.sum(self.__lamda,1)[:,np.newaxis]
        self._beta = temp[:,1]

            
    def __update_gamma_and_lamda(self, phi, mini_batch, scale):
        
        flag = 0
        # calculate the gradient for gamma
        grad_lamda = np.zeros((self._K, 2))
        grad_gamma = {}   # ie. grad[a] = array[] which is K dimensional vector
        counter = {}   # used for scaling 
        for edge in mini_batch:
            '''
            calculate the gradient for gamma
            '''
            a = edge[0]
            b = edge[1]
            phi_ab = phi[(a,b)]
            phi_ba = phi[(b,a)]
            if a in grad_gamma.keys():
                grad_gamma[a] += phi_ab
                counter[a] += 1
            else:
                grad_gamma[a] = phi_ab
                counter[a] = 1
                
            if b in grad_gamma.keys():
                grad_gamma[b] += phi_ba
                counter[b] += 1
            else:
                grad_gamma[b] = phi_ba
                counter[b] = 1
        
        for edge in mini_batch:    
            """
            calculate the gradient for lambda
            """
            y = 0
            if edge in self._network.get_linked_edges():
                y = 1
                flag = 1
            for k in range(self._K):
                grad_lamda[k][0] += phi_ab[k] * phi_ba[k] * y 
                grad_lamda[k][1] += phi_ab[k] * phi_ba[k] * (1-y) 
                
                
        # update gamma, only update node in the grad
        if self.stepsize_switch == False:
            p_t = (1024 + self._step_count)**(-0.5)
        else:
            p_t = 0.01*(1+self._step_count/1024.0)**(-0.55)
            
        for node in grad_gamma.keys():
            gamma_star = np.zeros(self._K)
            scale1 = 1.0
            if flag == 0:
                scale1 = self._N/counter[node]*1.0
            
            if self._step_count > 400:
                gamma_star = (1-p_t)*self.__gamma[node] + p_t * (self._alpha + scale1 * grad_gamma[node])
                self.__gamma[node] = (1-1.0/(self._step_count))*self.__gamma[node] + 1.0/(self._step_count)*gamma_star
            else:
                self.__gamma[node]=(1-p_t)*self.__gamma[node] + p_t * (self._alpha + scale1 * grad_gamma[node])
        
        # update lamda
        for k in range(self._K):
            
            if self._step_count > 400:
                lamda_star_0 = (1-p_t)*self.__lamda[k][0] + p_t *(self._eta[0] + scale * grad_lamda[k][0])
                lamda_star_1 = (1-p_t)*self.__lamda[k][1] + p_t *(self._eta[1] + scale * grad_lamda[k][1])
                self.__lamda[k][0] = (1-1/(self._step_count)) * self.__lamda[k][0] +1/(self._step_count)*lamda_star_0
                self.__lamda[k][1] = (1-1.0/(self._step_count)) * self.__lamda[k][1] +1.0/(self._step_count)*lamda_star_1
            else:
                self.__lamda[k][0] = (1-p_t)*self.__lamda[k][0] + p_t *(self._eta[0] + scale * grad_lamda[k][0])
                self.__lamda[k][1] = (1-p_t)*self.__lamda[k][1] + p_t *(self._eta[1] + scale * grad_lamda[k][1])
            
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
        