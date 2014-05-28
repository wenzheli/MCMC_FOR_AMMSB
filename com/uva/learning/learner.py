import numpy as np
import abc
import math

class Learner(object):
    """
    This is base class for all concrete learners, including MCMC sampler, variational
    inference,etc. 
    """
    def __init__(self, args, network):
        """
        initialize base learner parameters.
        """    
        self._network = network
        # model priors
        self._alpha = args.alpha
        self._eta = np.zeros(2)
        self._eta[0] = args.eta0
        self._eta[1] = args.eta1
        
        # parameters related to control model 
        self._K = args.K
        self._epsilon = args.epsilon
        
        # parameters related to network 
        self._N = network.get_num_nodes()
        
        # model parameters to learn
        self._beta = np.zeros(self._K)
        self._pi = np.zeros((self._N, self._K))
                
        # parameters related to sampling
        self._mini_batch_size = args.mini_batch_size
        if self._mini_batch_size < 1:
            self._mini_batch_size = self._N/2   # default option. 
        
        # ration between link edges and non-link edges
        self._link_ratio = network.get_num_linked_edges()/(self._N*(self._N-1)/2.0)
        # check the number of iterations. 
        self._step_count = 1
        # store perplexity for all the iterations
        self._ppxs_held_out = []
        self._ppxs_test = []
        
        self._max_iteration = args.max_iteration
        self.CONVERGENCE_THRESHOLD = 0.000000000001
        
        self.stepsize_switch = False
        
        
    @abc.abstractmethod
    def run(self):
        """
        Each concrete learner should implement this. It basically
        iterate the data sets, then update the model parameters, until
        convergence. The convergence can be measured by perplexity score. 
         
        We currently support four different learners:
            1. MCMC for batch learning
            2. MCMC for mini-batch training
            3. Variational inference for batch learning
            4. Stochastic variational inference
        """
    
    def get_ppxs_held_out(self):
        return self._ppxs_held_out
    
    def get_ppxs_test(self):
        return self._ppxs_test
    
    def set_max_iteration(self, max_iteration):
        self._max_iteration = max_iteration
    
    def _cal_perplexity_held_out(self):
        return self.__cal_perplexity(self._network.get_held_out_set())
   
    def _cal_perplexity_test(self):
        return self.__cal_perplexity(self._network.get_test_set())
        
    def _is_converged(self):
        n = len(self._ppxs_held_out)
        if n < 2:
            return False
        if abs((self._ppxs_held_out[n-1] - self._ppxs_held_out[n-2])/self._ppxs_held_out[n-2]) \
                                                    > self.CONVERGENCE_THRESHOLD:
            return False
        
        return True
            
    def __cal_perplexity(self, data):
        """
        calculate the perplexity for data.
        perplexity defines as exponential of negative average log likelihood. 
        formally:
            ppx = exp(-1/N * \sum){i}^{N}log p(y))
        
        we calculate average log likelihood for link and non-link separately, with the 
        purpose of weighting each part proportionally. (the reason is that we sample 
        the equal number of link edges and non-link edges for held out data and test data,
        which is not true representation of actual data set, which is extremely sparse. 
        """
        
        link_likelihood = 0
        non_link_likelihood = 0
        edge_likelihood = 0 
        link_count = 0
        non_link_count = 0
        
        for edge in data.keys():
            edge_likelihood = self.__cal_edge_likelihood(self._pi[edge[0]], self._pi[edge[1]], \
                                                       data[edge], self._beta)
            if edge in self._network.get_linked_edges():
                link_count += 1
                link_likelihood += edge_likelihood
            else:
                non_link_count += 1
                non_link_likelihood += edge_likelihood
        
        # weight each part proportionally. 
        #avg_likelihood = self._link_ratio*(link_likelihood/link_count) + \
        #                   (1-self._link_ratio)*(non_link_likelihood/non_link_count) 
        
        # direct calculation. 
        avg_likelihood = (link_likelihood + non_link_likelihood)/(link_count+non_link_count)
        #print "perplexity score is: " + str(math.exp(-avg_likelihood))    
        
        return math.exp(-avg_likelihood)            
    
    
    def __cal_edge_likelihood(self, pi_a, pi_b, y, beta):
        """
        calculate the log likelihood of edge :  p(y_ab | pi_a, pi_b, \beta)
        in order to calculate this, we need to sum over all the possible (z_ab, z_ba)
        such that:  p(y|*) = \sum_{z_ab,z_ba}^{} p(y, z_ab,z_ba|pi_a, pi_b, beta)
        but this calculation can be done in O(K), by using some trick.  
        """
        prob = 0
        s = 0
        for k in range(0, self._K):
            if y == 0:
                prob += pi_a[k] * pi_b[k] * (1-beta[k])
            else:
                prob += pi_a[k] * pi_b[k] * beta[k]
            s += pi_a[k] * pi_b[k]
        
        if y == 0:
            prob += (1-s) * (1-self._epsilon)
        else:
            prob += (1-s) * self._epsilon
        if prob < 0:
            print "adsfadsfadsf"
        return math.log(prob)
    
    