from com.uva.learning.learner import Learner
from sets import Set
import math
import random
import numpy as np
import copy
from com.uva.sample_latent_vars import sample_z_ab_from_edge
import cProfile, pstats, StringIO
import time
from com.uva.core_utils import update_phi

class MCMCSamplerStochastic(Learner):
    '''
    Mini-batch based MCMC sampler for community overlapping problems. Basically, given a 
    connected graph where each node connects to other nodes, we try to find out the 
    community information for each node. 
    
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
    
    Because of the intractability, we use MCMC(unbiased) to do approximate inference. But 
    different from classical MCMC approach, where we use ALL the examples to update the 
    parameters for each iteration, here we only use mini-batch (subset) of the examples.
    This method is great marriage between MCMC and stochastic methods.  
    '''
    def __init__(self, args, graph):
        # call base class initialization
        Learner.__init__(self, args, graph)
        
        # step size parameters. 
        self.__a = args.a
        self.__b = args.b
        self.__c = args.c
        
        # control parameters for learning
        #self.__num_node_sample = int(math.sqrt(self._network.get_num_nodes())) 
        
        self.__num_node_sample = 10
        #self.__num_node_sample = int(self._N/5)
        # model parameters and re-parameterization
        # since the model parameter - \pi and \beta should stay in the simplex, 
        # we need to restrict the sum of probability equals to 1.  The way we
        # restrict this is using re-reparameterization techniques, where we 
        # introduce another set of variables, and update them first followed by 
        # updating \pi and \beta.  
        self._theta = np.random.gamma(1,1,(self._K, 2))      # parameterization for \beta
        self._phi = np.random.gamma(1,1,(self._N, self._K))       # parameterization for \pi
        
        temp = self._theta/np.sum(self._theta,1)[:,np.newaxis]
        self._beta = temp[:,1]
        self._pi = self._phi/np.sum(self._phi,1)[:,np.newaxis]
        
        self._avg_log = []
        self._timing = []
        
    def run1(self):
        while self._step_count < self._max_iteration and not self._is_converged():
            """
            pr = cProfile.Profile()
            pr.enable()
            """
            (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
            
            if self._step_count % 10 == 0:
               
                ppx_score = self._cal_perplexity_held_out()
                #print "perplexity for hold out set is: "  + str(ppx_score)
                self._ppxs_held_out.append(ppx_score)
            
            self.__update_pi1(mini_batch, scale)
            
            # sample (z_ab, z_ba) for each edge in the mini_batch. 
            # z is map structure. i.e  z = {(1,10):3, (2,4):-1}
            #z = self.__sample_latent_vars2(mini_batch)
            self.__update_beta(mini_batch, scale)    
            
            """
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
            """
            self._step_count += 1
        
        print "terminated"
            
            
    def run(self):
        start = time.time()
        self._max_iteration = 10000  
        """ run mini-batch based MCMC sampler, based on the sungjin's note """
        while self._step_count < self._max_iteration and not self._is_converged():
            if self._step_count % 10 == 0:
                ppx_score = self._cal_perplexity_held_out()
                #print str(ppx_score)
                self._ppxs_held_out.append(ppx_score)
                
                if self._step_count > 200000:
                    size = len(self._avg_log)
                    ppx_score = (1-1.0/(self._step_count-19000)) * self._avg_log[size-1] + 1.0/(self._step_count-19000) * ppx_score
                    self._avg_log.append(ppx_score)
                else:
                    self._avg_log.append(ppx_score)
            
                self._timing.append(time.time()-start)
                
            self._step_count += 1
            eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)   
            (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
            size = {}
            
            """
            pr = cProfile.Profile()
            pr.enable()
            """
            # iterate through each node in the mini batch. 
            for node in self.__nodes_in_batch(mini_batch):
                #noise = np.random.randn(self._K) 
                noise = np.zeros(self._K)
                # sample a mini-batch of neighbors
                neighbor_nodes = self.__sample_neighbor_nodes(self.__num_node_sample, node)                
                self._update_phi(node, neighbor_nodes)
                
                #phi_star = update_phi(node, self._step_count, self._epsilon, self._K, self._N,eps_t, self._alpha,
                #           self._pi, self._phi, self._beta, noise, len(neighbor_nodes),
                #           self._network.get_linked_edges(), neighbor_nodes)
                #self._phi[node] = phi_star
                
            self._pi = self._phi/np.sum(self._phi,1)[:,np.newaxis]
            
          
            # sample (z_ab, z_ba) for each edge in the mini_batch. 
            # z is map structure. i.e  z = {(1,10):3, (2,4):-1}
            #z = self.__sample_latent_vars2(mini_batch)
            self.__update_beta(mini_batch, scale)    
            #print self._beta;
            
           
            
            if self._step_count % 1000 == 0:
                self._save()
            
            """
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue() 
            """
            
    def __update_pi1(self, mini_batch, scale):
        
        grads = np.zeros((self._N, self._K))
        counter = np.zeros(self._N)
        phi_star = np.zeros((self._N, self._K))
        
        for edge in mini_batch:                                  
            a = edge[0]
            b = edge[1]
            
            y = 0      # observation
            if (min(a, b), max(a, b)) in self._network.get_linked_edges():
                y = 1
            # z_ab
            prob_a = np.zeros(self._K)
            prob_b = np.zeros(self._K)
            
            for k in range(0, self._K):
                prob_a[k] = self._beta[k]**y*(1-self._beta[k])**(1-y)*self._pi[a][k]*self._pi[b][k]
                prob_a[k] += self._epsilon**y*(1-self._epsilon)**(1-y)*self._pi[a][k]*(1-self._pi[b][k])
                prob_b[k] = self._beta[k]**y*(1-self._beta[k])**(1-y)*self._pi[b][k]*self._pi[a][k]
                prob_b[k] += self._epsilon**y*(1-self._epsilon)**(1-y)*self._pi[b][k]*(1-self._pi[a][k])
            
            sum_prob_a = np.sum(prob_a)
            sum_prob_b = np.sum(prob_b)
            
            for k in range(0, self._K):
                grads[a][k] += (prob_a[k]/sum_prob_a)/self._phi[a][k]-1.0/np.sum(self._phi[a]) 
                grads[b][k] += (prob_b[k]/sum_prob_b)/self._phi[b][k]-1.0/np.sum(self._phi[b])
            #z_ab = self.sample_z_ab_from_edge(y_ab, self._pi[a], self._pi[b], self._beta, self._epsilon, self._K)           
            #z_ba = self.sample_z_ab_from_edge(y_ab, self._pi[b], self._pi[a], self._beta, self._epsilon, self._K)
            #print str(grads[a])
            counter[a] += 1
            counter[b] += 1
            
            #grads[a][z_ab] += 1/self._phi[a][z_ab]
            #grads[b][z_ba] += 1/self._phi[b][z_ba]
        
        
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)

        for i in range(0, self._N):
            #noise = np.random.randn(self._K)  
            noise = np.zeros(self._K);
            for k in range(0, self._K):
                if counter[i] < 1:
                    phi_star[i][k] = abs((self._phi[i,k]) + eps_t*(self._alpha - self._phi[i,k])+(2*eps_t)**.5*self._phi[i,k]**.5 * noise[k])
                else:
                    phi_star[i][k] = abs(self._phi[i,k] + eps_t * (self._alpha - self._phi[i,k] + \
                               (self._N/counter[i]) * grads[i][k]) \
                                + (2*eps_t)**.5*self._phi[i,k]**.5 * noise[k])
                
        self._phi = phi_star        
        self._pi = self._phi/np.sum(self._phi,1)[:,np.newaxis]                        
        
            
    def _update_phi(self,i, neighbor_nodes):
                                                                                                                                                                                                                                                                                                                                 
        n = len(neighbor_nodes)
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)   
        """ update phi for node i"""
        sum_phi = np.sum(self._phi[i])
        grads = np.zeros(self._K)
        phi_star = np.zeros(self._K)
        noise = np.random.randn(self._K) 
        #noise = np.zeros(self._K);
         
        for j in neighbor_nodes:
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
                                (self._N/n)*grads[k]) + eps_t**.5*self._phi[i,k]**.5 * noise[k])
        
        self._phi[i] = phi_star                                  
    
    def __update_beta(self, mini_batch, scale):
        '''
        update beta for mini_batch. 
        '''
        grads = np.zeros((self._K, 2))
        sum_theta = np.sum(self._theta,1)  
        
        # update gamma, only update node in the grad
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)                                  # gradients K*2 dimension                                                         # random noise. 
        
        
        for edge in mini_batch:
            i = edge[0]
            j = edge[1]
            y = 0
            if edge in self._network.get_linked_edges():
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
        
        noise = np.random.randn(self._K, 2)   
        #noise = np.zeros((self._K,2))
        theta_star = copy.copy(self._theta)  
        for k in range(0,self._K):
            for i in range(0,2):
                theta_star[k,i] = abs(self._theta[k,i] + eps_t/2 * (self._eta[i] - self._theta[k,i] + \
                                    scale* grads[k,i]) + eps_t**.5*self._theta[k,i] ** .5 * noise[k,i])  
        
        self._theta = theta_star
        #self._theta = theta_star        
        # update beta from theta
        temp = self._theta/np.sum(self._theta,1)[:,np.newaxis]
        self._beta = temp[:,1]
    
    def __update_pi_for_node(self, i, z, n, scale):
        '''
        update pi for current node i. 
        ''' 
        # update gamma, only update node in the grad
        #if self.stepsize_switch == False:
        #    eps_t = (1024+self._step_count)**(-0.5)
        #else:
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)                                                                                                                                                                                                                                                                                                                          
    
        phi_star = copy.copy(self._phi[i])                              # updated \phi
        phi_i_sum = np.sum(self._phi[i])                                   
        noise = np.random.randn(self._K)                                 # random noise. 
        #noise = np.zeros(self._K)
        # get the gradients    
        grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
        for k in range(0, self._K):
            grads[k] += 1/self._phi[i,k] * z[k]
        
        # update the phi 
        for k in range(0, self._K):
            phi_star[k] = abs(self._phi[i,k] + eps_t/2 * (self._alpha - self._phi[i,k] + \
                                (self._N/n) * grads[k]) + eps_t**.5*self._phi[i,k]**.5 * noise[k])
        
        self._phi[i] = phi_star
       
        # update pi
        sum_phi = np.sum(self._phi[i])
        self._pi[i] = [self._phi[i,k]/sum_phi for k in range(0, self._K)]
            

    def __sample_latent_vars2(self, mini_batch):
        '''
        sample latent variable (z_ab, z_ba) for each pair of nodes. But we only consider 11 different cases,
        since we only need indicator function in the gradient update. More details, please see the comments 
        within the sample_z_for_each_edge function. 
        '''
        z = {}  
        for edge in mini_batch:
            y_ab = 0
            if edge in self._network.get_linked_edges():
                y_ab = 1
            
            z[edge] = self.__sample_z_for_each_edge(y_ab, self._pi[edge[0]], self._pi[edge[1]], \
                                          self._beta, self._K)            

        return z
    
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
        for k in range(1,K+1):
            p[k] += p[k-1] 
        #bounds = np.cumsum(p)
        location = random.random() * p[K]
        
        # get the index of bounds that containing location. 
        for i in range(0, K):
                if location <= p[i]:
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
            
            z_ab = self.sample_z_ab_from_edge(y_ab, self._pi[node], self._pi[neighbor], self._beta, self._epsilon, self._K)           
            z[z_ab] += 1
            
        return z
    """
    def __sample_z_ab_from_edge(self, y, pi_a, pi_b, beta, epsilon, K):
        '''
        we need to calculate z_ab. We can use deterministic way to calculate this
        for each k,  p[k] = p(z_ab=k|*) = \sum_{i}^{} p(z_ab=k, z_ba=i|*)
        then we simply sample z_ab based on the distribution p.
        this runs in O(K) 
        '''
        p = np.zeros(K)
        for i in range(0, K):
            tmp = beta[i]**y*(1-beta[i])**(1-y)*pi_a[i]*pi_b[i]
            tmp += epsilon**y*(1-epsilon)**(1-y)*pi_a[i]*(1-pi_b[i])
            p[i] = tmp
        # sample community based on probability distribution p. 
        bounds = np.cumsum(p)
        location = random.random() * bounds[K-1]
        
        # get the index of bounds that containing location. 
        for i in range(0, K):
                if location <= bounds[i]:
                    return i
        # failed, should not happen!
        return -1
    """    
        
    def __sample_neighbor_nodes(self, sample_size, nodeId):
        '''
        Sample subset of neighborhood nodes. 
        '''    
        p = sample_size
        neighbor_nodes = Set()
        held_out_set = self._network.get_held_out_set()
        test_set = self._network.get_test_set()
        
        while p > 0:
            nodeList = random.sample(list(xrange(self._N)), sample_size * 2)
            for neighborId in nodeList:
                    if p < 0:
                        break
                    if neighborId == nodeId:
                        continue
                    # check condition, and insert into mini_batch_set if it is valid. 
                    edge = (min(nodeId, neighborId), max(nodeId, neighborId))
                    if edge in held_out_set or edge in test_set or neighborId in neighbor_nodes:
                        continue
                    else:
                        # add it into mini_batch_set
                        neighbor_nodes.add(neighborId)
                        p -= 1
                        
        return neighbor_nodes

    def __nodes_in_batch(self, mini_batch):
        """
        Get all the unique nodes in the mini_batch. 
        """
        node_set = Set()
        for edge in mini_batch:
            node_set.add(edge[0])
            node_set.add(edge[1])
        return node_set
    
    def _save(self):
        f = open('ppx_mcmc_stochastic.txt', 'wb')
        for i in range(0, len(self._avg_log)):
            f.write(str(math.exp(self._avg_log[i])) + "\t" + str(self._timing[i]) +"\n")
        f.close()
    
     
    
    def sample_z_ab_from_edge(self, y, pi_a, pi_b, beta, epsilon, K):
        p = np.zeros(K)
   
        tmp = 0.0
    
        for i in range(0, K):
            tmp = beta[i]**y*(1-beta[i])**(1-y)*pi_a[i]*pi_b[i]
            tmp += epsilon**y*(1-epsilon)**(1-y)*pi_a[i]*(1-pi_b[i])
            p[i] = tmp
    
    
        for k in range(1,K):
            p[k] += p[k-1]
    
        location = random.random() * p[K-1]
        # get the index of bounds that containing location. 
        for i in range(0, K):
            if location <= p[i]:
                return i
    
        # failed, should not happen!
        return -1    