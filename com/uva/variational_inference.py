import random
from com.uva.edge import Edge
import math
import numpy as np
import copy
from sets import Set
import cProfile, pstats, StringIO
from scipy.special import psi


class SVI(object):
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
    def __init__(self, network, args):
        '''
        Initialize the sampler using the network object and arguments (i.e prior)
        Arguments:
            network:    representation of the graph. 
            args:       containing priors, control parameters for the model. 
        '''
        
        # parameters related to network 
        self.network = network
        self.N = network.num_nodes  # total number of nodes in the graph
        self.K = args.K             # number of communities 
        
        # model priors 
        self.alpha = args.alpha     # prior for the community membership \pi
        self.eta0 = args.eta0       # prior for the community strength 
        self.eta1 = args.eta1       # prior for the community strength. 
          
        # model parameters              
        self.beta = np.zeros(self.K)        # community strength.  size: K
        self.pi = np.zeros((self.N, self.K))  # community membership for each node. size: N * K
        
        # variational parameters. 
        self.lamda = np.random.gamma(1,1,(self.K, 2))      # variational parameters for beta  
        self.gamma = np.random.gamma(1,1,(self.N, self.K)) # variational parameters for pi
        self.phi = None   # local parameters, be created for each mini-batch 
        
        # other parameters. 
        self.epsilon = args.epsilon
        
        # step size parameters. 
        self.a = args.a
        self.b = args.b
        self.c = args.c
        
        # control parameters for learning 
        self.online_iterations = 50
        self.phi_update_threshold = 0.01
        self.step_count = 1
        self.hold_out_prob = args.hold_out_prob  # percentage of samples used for validation, testing
        self.mini_batch_size = args.mini_batch_size
        if args.mini_batch_size < 1:
            # use default option. 
            self.mini_batch_size = self.N/2    
        
        # ratio between link edges and non-link edges
        self.link_ratio = len(self.network.edges_set)/(self.N*(self.N-1)/2.0)
        
        
        # validation and test sets. Divide the data sets into (training, hold-out and testing). 
        # the number of samples used for hold-out and testing are equal, and the number depends
        # on the parameter - self.hold_out_prob
        self.num_hold_out = int(self.network.num_links * args.hold_out_prob)  
        self.hold_out_map = {}
        self.test_map = {}
        self.train_link_map = {}    # For each node, we store all the neighborhood nodes. 
        
        self.init_train_link_map()
        self.init_hold_out_set()
        self.init_test_set()
     
    def run_sampler(self, sample_strategy):
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
        # running until convergence. 
        while True:        
            # evaluate model after processing every 10 mini-batches. 
            if self.step_count % 1 == 0:
                ppx_score = self.cal_perplexity()
                print "perplexity for hold out set is: "  + str(ppx_score)
            
            #pr = cProfile.Profile()
            #pr.enable()
            
            mini_batch = self.sample_mini_batch(self.mini_batch_size, sample_strategy)
            # update (phi_ab, phi_ba) for each edge
            phi = {}               # mapping (a,b) => (phi_ab, phi_ba)
            for edge in mini_batch:
                self.estimate_phi_for_edge(edge, phi)  # this can be done in parallel. 
            self.update_gamma_and_lamda(phi, mini_batch)
            
            
            #pr.disable()
            #s = StringIO.StringIO()
            #sortby = 'cumulative'
            #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            #ps.print_stats()
            #print s.getvalue()
            #self.step_count += 1
    
    def update_gamma_and_lamda(self, phi, mini_batch):
        # calculate the gradient for gamma
        grad_lamda = np.zeros((self.K, 2))
        grad_gamma = {}   # ie. grad[a] = array[] which is K dimensional vector
        counter = {}   # used for scaling 
        for edge in mini_batch:
            '''
            calculate the gradient for gamma
            '''
            a = edge.first
            b = edge.second
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
                
            '''
            calculate the gradient for lamda
            '''
            y = 0
            if edge in self.network.edges_set:
                y = 1
            for k in range(self.K):
                    grad_lamda[k][0] += phi_ab[k] * phi_ba[k] * y 
                    grad_lamda[k][1] += phi_ab[k] * phi_ba[k] * (1-y) 
                
                
        # update gamma, only update node in the grad
        p_t = (1024 + self.step_count)**(-0.5)
        for node in grad_gamma.keys():
            scale = self.N/counter[node]*1.0
            self.gamma[node] = (1-p_t)*self.gamma[node] + p_t * (self.alpha + scale * grad_gamma[node])
            
        # update lamda
        size = len(mini_batch)
        scale = (self.N * self.N/2)/size*1.0
        for k in range(self.K):
            self.lamda[k][0] = (1-p_t)*self.lamda[k][0] + p_t *(self.eta0 + scale * grad_lamda[k][0])
            self.lamda[k][1] = (1-p_t)*self.lamda[k][1] + p_t *(self.eta1 + scale * grad_lamda[k][0])
            
        
    def estimate_phi_for_edge(self, edge, phi):
        '''
        calculate (phi_ab, phi_ba) for given edge : (a,b)
        '''
        a = edge.first
        b = edge.second
        # initialize 
        phi_ab = np.empty(self.K)
        phi_ba = np.empty(self.K)
        phi_ab.fill(1.0/self.K)
        phi_ba.fill(1.0/self.K)
        
        y = 0
        if Edge(a,b) in self.network.edges_set:
            y = 1
        
        # alternatively update phi_ab and phi_ba, until it converges
        # or reach the maximum iterations. 
        for i in range(self.online_iterations):
            phi_ab_old = copy.copy(phi_ab)
            phi_ba_old = copy.copy(phi_ba)
            
            # first, update phi_ab
            for k in range(self.K):
                if y == 1:
                    u = (1-phi_ba[k])* math.log(self.epsilon)
                    phi_ab[k] = math.exp(psi(self.gamma[a][k])+phi_ba[k]*\
                                         (psi(self.lamda[k][0])-psi(self.lamda[k][0]+psi(self.lamda[k][1])))+u)
                else:
                    phi_ab[k] = math.exp(psi(self.gamma[a][k])+phi_ba[k]*\
                                         (psi(self.lamda[k][1])-psi(self.lamda[k][0]+self.lamda[k][1])))    
            sum_phi_ab = np.sum(phi_ab)
            phi_ab = phi_ab/sum_phi_ab
                
            # then update phi_ba
            for k in range(self.K):
                if y == 1:
                    u = (1-phi_ab[k])* math.log(self.epsilon) 
                    phi_ba[k] = math.exp(psi(self.gamma[b][k])+phi_ab[k]*\
                                         (psi(self.lamda[k][0])-psi(self.lamda[k][0]+psi(self.lamda[k][1])))+u)
                else:
                    phi_ba[k] = math.exp(psi(self.gamma[b][k])+phi_ab[k]*\
                                         (psi(self.lamda[k][1])-psi(self.lamda[k][0]+self.lamda[k][1])))   
               
            sum_phi_ba = np.sum(phi_ba)
            phi_ba = phi_ba/sum_phi_ba
            
            
            # calculate the absolute difference between new value and old value
            diff1 = np.sum(np.abs(phi_ab - phi_ab_old))
            diff2 = np.sum(np.abs(phi_ba - phi_ba_old))
            if diff1 < self.phi_update_threshold and diff2 < self.phi_update_threshold:
                break
        
        phi[(a,b)] = phi_ab
        phi[(b,a)] = phi_ba 
        
            
    def sample_mini_batch(self, mini_batch_size, sample_strategy):
        '''
        sample mini-batch of samples from the whole training set
        Arguments: 
            mini_batch_size:    number of samples for each mini-batch. 
            sample_strategy:    implies different sampling strategies. 
                                i.e random-link: uniformally sample the edges (link/non-links)
                                from the while network. For detailed sampling approach, please
                                refer to the paper:
                                http://www.cs.princeton.edu/~blei/papers/GopalanMimnoGerrishFreedmanBlei2012.pdf
        Returns list of samples (mini-batch)
        '''
        
        if sample_strategy == "random-link":
            return self.sample_mini_batch_random_link(mini_batch_size)
        else:
            return self.sample_mini_batch_stratified_random_node(mini_batch_size)
    
    
    def cal_perplexity(self):
        '''
        calculate the perplexity for held_out data set/test data set.
        perplexity defines as exponential of negative average log likelihood. 
        formally:
            ppx = exp(-1/N * \sum){i}^{N}log p(y))
        
        we calculate average log likelihood for link and non-link separately, with the 
        purpose of weighting each part proportionally. (the reason is that we sample 
        the equal number of link edges and non-link edges for held out data and test data,
        which is not true representation of actual data set, which is extremely sparse. 
        '''
        # estimate the pi and beta
        for i in range(self.N):
            for k in range(self.K):
                s = np.sum(self.gamma[i])
                self.pi[i][k] = self.gamma[i][k]/s
        
        for k in range(self.K):
            self.beta[k] = self.lamda[k][0]/(self.lamda[k][0]+self.lamda[k][1])
        
        link_likelihood = 0
        non_link_likelihood = 0
        edge_likelihood = 0 
        link_count = 0
        non_link_count = 0
        
        for edge in self.hold_out_map.keys():
            edge_likelihood = self.cal_edge_likelihood(self.pi[edge.first], self.pi[edge.second], \
                                                       self.hold_out_map[edge], self.beta)
            if self.hold_out_map[edge] == 1:
                link_count += 1
                link_likelihood += edge_likelihood
            else:
                non_link_count += 1
                non_link_likelihood += edge_likelihood
        
    v    # weight each part proportionally. 
        avg_likelihood = self.link_ratio*(link_likelihood/link_count) + \
                            (1-self.link_ratio)*(non_link_likelihood/non_link_count) 
        
        # direct calculation. 
        #avg_likelihood = (link_likelihood + non_link_likelihood)/(link_count+non_link_count)
        #print "perplexity score is: " + str(math.exp(-avg_likelihood))    
        
        return math.exp(-avg_likelihood)            
    
    
    def cal_edge_likelihood(self, pi_a, pi_b, y, beta):
        '''
        calculate the log likelihood of edge :  p(y_ab | pi_a, pi_b, \beta)
        in order to calculate this, we need to sum over all the possible (z_ab, z_ba)
        such that:  p(y|*) = \sum_{z_ab,z_ba}^{} p(y, z_ab,z_ba|pi_a, pi_b, beta)
        but this calculation can be done in O(K), by using some trick.  
        '''
        prob = 0
        s = 0
        for k in range(0, self.K):
            if y == 0:
                prob += pi_a[k] * pi_b[k] * (1-beta[k])
            else:
                prob += pi_a[k] * pi_b[k] * beta[k]
            s += pi_a[k] * pi_b[k]
        
        if y == 0:
            prob += (1-s) * (1-self.epsilon)
        else:
            prob += (1-s) * self.epsilon
            
        return math.log(prob)
   
                
    def init_train_link_map(self):
        '''
        create a set for each node, which contains list of 
        neighborhood nodes. i.e {0: Set[2,3,4], 1: Set[3,5,6]...}
        This is used for sub-sampling 
        in the later. 
        '''  
        for i in range(0, self.N):
            self.train_link_map[i] = Set()
        
        for edge in self.network.edges_set:
            self.train_link_map[edge.first].add(edge.second)
            self.train_link_map[edge.second].add(edge.first)
              
    def init_hold_out_set(self):
        ''' 
        sample hold out set. we draw equal number of 
        links and non-links from the graph. 
        '''
        p = self.num_hold_out/2
        # sample p links from network. 
        sampled_links = random.sample(self.network.edges_set, p)
        for edge in sampled_links:
            self.hold_out_map[edge] = True
            self.train_link_map[edge.first].remove(edge.second)
            self.train_link_map[edge.second].remove(edge.first)
            
        # sample p non-links from network 
        while p > 0:
            edge = self.sample_hold_out_non_link_edge()
            self.hold_out_map[edge] = False
            p -= 1
        
    def init_test_set(self):
        '''
        sample test set. we draw equal number of samples for 
        links and non-links edges
        '''
        p = int(self.num_hold_out/2)
        # sample p links from network 
        while p > 0:
            edges = random.sample(self.network.edges_set, 2*int(self.num_hold_out/2))
            for edge in edges:   # check whether it is already used in hold_out set
                if p < 0:
                    break
                if edge in self.hold_out_map or edge in self.test_map:
                    continue
                else:
                    self.test_map[edge] = True
                    self.train_link_map[edge.first].remove(edge.second)
                    self.train_link_map[edge.second].remove(edge.first)
                    p -= 1
            
        # sample p non-links from network 
        p = int(self.num_hold_out/2)
        while p > 0:
            edge = self.sample_test_non_link_edge()
            self.test_map[edge] = False
            p -= 1
        
    def sample_hold_out_non_link_edge(self):
        '''
        sample one non-link edge from the network. We should make sure the edge is not 
        been used already, so we need to check the condition before we add it into 
        hold out sets
        '''
        while True:
            firstIdx = random.randint(0,self.N-1)
            secondIdx = random.randint(0, self.N-1)
        
            if (firstIdx == secondIdx):
                continue
            # ensure the first index is smaller than the second one.  
            edge = Edge(min(firstIdx, secondIdx), max(firstIdx, secondIdx))
        
            # check conditions. 
            if edge in self.network.edges_set or edge in self.hold_out_map:
                continue
        
            return edge
    
    def sample_test_non_link_edge(self):
        '''
        sample one non-link edge from the network. First, randomly generate one 
        edge. Second, check conditions for that edge. If that edge passes all the 
        conditions, return that edge. 
        ''' 
        while True:
            firstIdx = random.randint(0,self.N-1)
            secondIdx = random.randint(0, self.N-1)
        
            if (firstIdx == secondIdx):
                continue
            # ensure the first index is smaller than the second one.  
            edge = Edge(min(firstIdx, secondIdx), max(firstIdx, secondIdx))
        
            # check conditions: 
            if edge in self.network.edges_set or edge in self.hold_out_map or edge in self.test_map:
                continue
        
            return edge
            
    def sample_mini_batch_random_link(self, mini_batch_size):
        '''
        sample list of edges from the while training network uniformally, regardless
        of links or non-links edges.The sampling approach is pretty simple: randomly generate 
        one edge and then check if that edge passes the pre-defined conditions. The iteration
        stops until we get enough (mini_batch_size) edges. 
        '''
        p = mini_batch_size
        mini_batch_set = Set()     # list of samples in the mini-batch 
        
        # iterate until we get $p$ valid edges. 
        while p > 0:
            firstIdx = random.randint(0,self.N-1)
            secondIdx = random.randint(0, self.N-1)
            if firstIdx == secondIdx:
                continue
            # make sure the first index is smaller than the second one, since
            # we are dealing with undirected graph. 
            edge = Edge(min(firstIdx, secondIdx), max(firstIdx, secondIdx))
            
            # the edge should not be in  1)hold_out set, 2)test_set  3) mini_batch_set (avoid duplicate)
            if edge in self.hold_out_map or edge in self.test_map or edge in mini_batch_set:
                continue
            
            # great, we put it into the mini_batch list. 
            mini_batch_set.add(edge)
            p -= 1
        
        return mini_batch_set
    
    def sample_mini_batch_stratified_random_node(self, num_pieces):
        '''
        stratified sampling approach gives more attention to link edges (the edge is connected by two
        nodes). The sampling process works like this: 
        a) randomly choose one node $i$ from all nodes (1,....N)
        b) decide to choose link edges or non-link edges with (50%, 50%) probability. 
        c) if we decide to sample link edge:
                return all the link edges for the chosen node $i$
           else 
                sample edges from all non-links edges for node $i$. The number of edges
                we sample equals to  number of all non-link edges / num_pieces
        '''
        # randomly select the node ID
        nodeId = random.randint(0, self.N-1)
        # decide to sample links or non-links
        flag = random.randint(0,1)      # flag=0: non-link edges  flag=1: link edges
        
        mini_batch_set = Set()
        
        if flag == 0:
            # this is approximation, since the size of self.train_link_map[nodeId]
            # greatly smaller than N. 
            mini_batch_size = int((self.N - len(self.train_link_map[nodeId])/2)/num_pieces)  
            p = mini_batch_size
            while p > 0:
                # because of the sparsity, when we sample $mini_batch_size*2$ nodes, the list likely
                # contains at least mini_batch_size valid nodes. 
                nodeList = random.sample(list(xrange(self.N)), mini_batch_size * 2)
                for neighborId in nodeList:
                    if p < 0:
                        return mini_batch_set
                    if neighborId == nodeId:
                        continue
                    # check condition, and insert into mini_batch_set if it is valid. 
                    edge = Edge(min(nodeId, neighborId), max(nodeId, neighborId))
                    if edge in self.network.edges_set or edge in self.hold_out_map or \
                            edge in self.test_map or edge in mini_batch_set:
                        continue
                    else:
                        # add it into mini_batch_set
                        mini_batch_set.add(edge)
                        p -= 1
                        
            return mini_batch_set
        
        else:
            # return all link edges
            for neighborId in self.train_link_map[nodeId]:
                mini_batch_set.add(Edge(min(nodeId, neighborId),max(nodeId, neighborId)))
            
        return mini_batch_set   
        
        
        
        
        
        
        
        