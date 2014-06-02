from com.uva.learning.learner import Learner
from sets import Set
import math
import random
import numpy as np
import copy
from com.uva.sample_latent_vars import sample_z_ab_from_edge
import cProfile, pstats, StringIO


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
        
        self.__num_node_sample = int(self._N/5)
        # model parameters and re-parameterization
        # since the model parameter - \pi and \beta should stay in the simplex, 
        # we need to restrict the sum of probability equals to 1.  The way we
        # restrict this is using re-reparameterization techniques, where we 
        # introduce another set of variables, and update them first followed by 
        # updating \pi and \beta.  
        self.__theta = np.random.gamma(100,0.01,(self._K, 2))      # parameterization for \beta
        self.__phi = np.random.gamma(1,1,(self._N, self._K))       # parameterization for \pi
        
        temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
        self._beta = temp[:,1]
        self._pi = self.__phi/np.sum(self.__phi,1)[:,np.newaxis]
        
    def run1(self):
        while self._step_count < self._max_iteration and not self._is_converged():
            """
            pr = cProfile.Profile()
            pr.enable()
            """
            (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
            #print "iteration: " + str(self._step_count)
            
            if self._step_count % 1 == 0:
                #print str(self._beta)
                ppx_score = self._cal_perplexity_held_out()
                #print "perplexity for hold out set is: "  + str(ppx_score)
                self._ppxs_held_out.append(ppx_score)
            
            self.__update_pi1(mini_batch, scale)
            
            # sample (z_ab, z_ba) for each edge in the mini_batch. 
            # z is map structure. i.e  z = {(1,10):3, (2,4):-1}
            z = self.__sample_latent_vars2(mini_batch)
            self.__update_beta(mini_batch, scale,z)    
            
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
        """ run mini-batch based MCMC sampler, based on the sungjin's note """
        while self._step_count < self._max_iteration and not self._is_converged():
           
            (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")
            latent_vars = {}
            size = {}
            
            # iterate through each node in the mini batch. 
            for node in self.__nodes_in_batch(mini_batch):
                # sample a mini-batch of neighbors
                neighbor_nodes = self.__sample_neighbor_nodes(self.__num_node_sample, node)                
                size[node] = len(neighbor_nodes)
                # sample latent variables z_ab for each pair of nodes
                z = self.__sample_latent_vars(node, neighbor_nodes)
                # save for a while, in order to update together. 
                latent_vars[node] = z
                
            # update pi for each node
            for node in self.__nodes_in_batch(mini_batch):
                self.__update_pi_for_node(node, latent_vars[node], size[node], scale)
            
            
            # sample (z_ab, z_ba) for each edge in the mini_batch. 
            # z is map structure. i.e  z = {(1,10):3, (2,4):-1}
            z = self.__sample_latent_vars2(mini_batch)
            self.__update_beta(mini_batch, scale,z)    
            
            
            if self._step_count % 1 == 0:
                ppx_score = self._cal_perplexity_held_out()
                print str(ppx_score)
                self._ppxs_held_out.append(ppx_score)
                            
            self._step_count += 1
            
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
            
            y_ab = 0      # observation
            if (min(a, b), max(a, b)) in self._network.get_linked_edges():
                y_ab = 1
            
            z_ab = self.sample_z_ab_from_edge(y_ab, self._pi[a], self._pi[b], self._beta, self._epsilon, self._K)           
            z_ba = self.sample_z_ab_from_edge(y_ab, self._pi[b], self._pi[a], self._beta, self._epsilon, self._K)
            
            counter[a] += 1
            counter[b] += 1
            
            grads[a][z_ab] += 1/self.__phi[a][z_ab]
            grads[b][z_ba] += 1/self.__phi[b][z_ba]
        
         # update gamma, only update node in the grad
        if self.stepsize_switch == False:
            eps_t = (1024+self._step_count)**(-0.5)
        else:
            eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)

        for i in range(0, self._N):

            noise = np.random.randn(self._K)  
            sum_phi_i = np.sum(self.__phi[i])
            for k in range(0, self._K):
                if counter[i] < 1:
                    phi_star[i][k] = abs((self.__phi[i,k]) + eps_t*(self._alpha - self.__phi[i,k])+(2*eps_t)**.5*self.__phi[i,k]**.5 * noise[k])
                else:
                    phi_star[i][k] = abs(self.__phi[i,k] + eps_t * (self._alpha - self.__phi[i,k] + \
                                scale * (grads[i][k]-(1.0/sum_phi_i)*counter[i])) \
                                + (2*eps_t)**.5*self.__phi[i,k]**.5 * noise[k])
                
                if self._step_count < 50000:
                    self.__phi[i][k] = phi_star[i][k]
                else:
                    self.__phi[i][k] = phi_star[i][k] * (1.0/(self._step_count)) + \
                                                (1-(1.0/(self._step_count)))*self.__phi[i][k]
                                                
            sum_phi = np.sum(self.__phi[i])
            self._pi[i] = [self.__phi[i,k]/sum_phi for k in range(0, self._K)]
            
        
    def __update_beta(self, mini_batch, scale,z):
        '''
        update beta for mini_batch. 
        '''
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)
            
        grads = np.zeros((self._K, 2))                               # gradients K*2 dimension
        sums = np.sum(self.__theta,1)                                 
        noise = np.random.randn(self._K, 2)                          # random noise. 
        
        for  edge in z.keys():
            y_ab = 0
            if edge in self._network.get_linked_edges():
                y_ab = 1
            k = z[edge]
            # if k==-1 means z_ab != z_ba => gradient is 0. 
            if k == -1:
                continue
            
            grads[k,0] += abs(1-y_ab)/self.__theta[k,0] - 1/ sums[k]
            grads[k,1] += abs(-y_ab)/self.__theta[k,1] - 1/sums[k]
        
        #if len(mini_batch) < 1:
        #    scale = 1;
        #else:
        #    scale = (self._N * (self._N-1)/2)/(len(mini_batch))
        # update theta 
        theta_star = copy.copy(self.__theta)  
        for k in range(0,self._K):
            for i in range(0,2):
                theta_star[k,i] = abs(self.__theta[k,i] + eps_t * (self._eta[i] - self.__theta[k,i] + \
                                    scale * grads[k,i]) + (2*eps_t)**.5*self.__theta[k,i] ** .5 * noise[k,i])  
        self.__theta = theta_star 
        temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
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
    
        phi_star = copy.copy(self.__phi[i])                              # updated \phi
        phi_i_sum = np.sum(self.__phi[i])                                   
        noise = np.random.randn(self._K)                                 # random noise. 
        
        # get the gradients    
        grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
        for k in range(0, self._K):
            grads[k] += 1/self.__phi[i,k] * z[k]
        
        # update the phi 
        for k in range(0, self._K):
            phi_star[k] = abs(self.__phi[i,k] + eps_t/2 * (self._alpha - self.__phi[i,k] + \
                                self._N/n * grads[k]) + eps_t**.5*self.__phi[i,k]**.5 * noise[k])
        
        self.__phi[i] = phi_star
       
        # update pi
        sum_phi = np.sum(self.__phi[i])
        self._pi[i] = [self.__phi[i,k]/sum_phi for k in range(0, self._K)]
            

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
        f = open('ppx_mcmc.txt', 'wb')
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