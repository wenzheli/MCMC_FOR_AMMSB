import random
from com.uva.edge import Edge
import math
import numpy as np
import copy
from sets import Set
import cProfile, pstats, StringIO
from com.uva.sample_latent_vars import sample_z_ab_from_edge


class Sampler(object):
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
        self.eta = args.eta         # prior for the community strength 
        
        # model parameters and re-parameterization
        # since the model parameter - \pi and \beta should stay in the simplex, 
        # we need to restrict the sum of probability equals to 1.  The way we
        # restrict this is using re-reparameterization techniques, where we 
        # introduce another set of variables, and update them first followed by 
        # updating \pi and \beta.  
        self.theta = np.random.gamma(1,1,(self.K, 2))      # parameterization for \beta
        self.phi = np.random.gamma(1,1,(self.N, self.K))   # parameterization for \pi
        temp = self.theta/np.sum(self.theta,1)[:,np.newaxis]
        self.beta = temp[:,1]
        self.pi = self.phi/np.sum(self.phi,1)[:,np.newaxis]
        
        # other parameters. 
        self.epsilon = args.epsilon
        
        # step size parameters. 
        self.a = args.a
        self.b = args.b
        self.c = args.c
        
        # control parameters for learning
        self.num_node_sample = int(math.sqrt(network.num_nodes)) 
        self.step_count = 1
        self.hold_out_prob = args.hold_out_prob  # percentage of samples used for validation, testing
        if args.mini_batch_size < 1:
            # use default option. 
            self.mini_batch_size = self.N/2
        else:
            self.mini_batch_size = args.mini_batch_size    
        
        
        # ration between link edges and non-link edges
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
        MCMC sampler. This is alternating sampling process for parameter \pi and \beta
        The pseudo code is:
            initialize \theta, \beta, \pi, \phi,
            while sampling do
                sample a mini-batch of node pairs xi, denote nodes in this mini-batch by N(xi)
                for each node a in N(xi)
                    sample a mini-batch of nodes V_t \in V
                    update \phi conditioning on \beta
                    update \pi from \phi
                for k=1....K do
                update \theta_{k} conditioning on \pi
                update \beta_{k} from \theta_{k}
        '''   
        
        # running until convergence. 
        while True:        
            # evaluate model after processing every 10 mini-batches. 
            if self.step_count % 1 == 0:
                ppx_score = self.cal_perplexity()
                print "perplexity for hold out set is: "  + str(ppx_score)
                #print "beta is: " + str(self.beta)
                #print "alpha is" + str(self.pi[0])
            
            mini_batch = self.sample_mini_batch(self.mini_batch_size, sample_strategy)
            # iterate through each node in the mini batch. 
            for node in self.nodes_in_batch(mini_batch):
                # sample a mini-batch of neighbors. 
                neighborhood_nodes = self.sample_neighbor_nodes(self.num_node_sample, node)
                # sample latent variables z_ab for each pair of nodes
                z = self.sample_latent_vars(node, neighborhood_nodes)
                # update \phi and \pi. 
                self.update_pi_for_node(node, z, len(neighborhood_nodes))
            
            # update \theta and \beta 
            self.update_beta(mini_batch)
                   
            self.step_count += 1
            
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
    
    def sample_neighbor_nodes(self, sample_size, nodeId):
        '''
        Sample subset of neighborhood nodes. Since our update equation for each node 
        depends on its neighborhood nodes, we do sub-sampling. 
        '''    
        p = sample_size
        neighborhood_nodes = Set()
        while p > 0:
            nodeList = random.sample(list(xrange(self.N)), sample_size * 2)
            for neighborId in nodeList:
                    if p < 0:
                        return neighborhood_nodes
                    if neighborId == nodeId:
                        continue
                    # check condition, and insert into mini_batch_set if it is valid. 
                    edge = Edge(min(nodeId, neighborId), max(nodeId, neighborId))
                    if edge in self.hold_out_map or edge in self.test_map or neighborId in neighborhood_nodes:
                        continue
                    else:
                        # add it into mini_batch_set
                        neighborhood_nodes.add(neighborId)
                        p -= 1
                        
        return neighborhood_nodes
    
    def nodes_in_batch(self, mini_batch):
        '''
        get all the nodes in the mini_batch. avoid duplicate. 
        '''
        node_set = Set()
        for edge in mini_batch:
            node_set.add(edge.first)
            node_set.add(edge.second)
        return node_set
    
    
    def update_beta(self, mini_batch):
        '''
        update beta for mini_batch. 
        '''
        num_total_pairs = self.N * (self.N-1) / 2
        eps_t  = self.a*((1 + self.step_count/self.b)**-self.c)     # step size 
        
        # sample (z_ab, z_ba) for each edge in the mini_batch. 
        # z is map structure. i.e  z = {(1,10):3, (2,4):-1}
        z = self.sample_latent_vars2(mini_batch)
        
        grads = np.zeros((self.K, 2))                               # gradients K*2 dimension
        sums = np.sum(self.theta,1)                                 
        noise = np.random.randn(self.K, 2)                          # random noise. 
        
        for edge in z.keys():
            y_ab = 0
            if Edge(edge.first, edge.second) in self.network.edges_set:
                y_ab = 1
            k = z[edge]
            # if k==-1 means z_ab != z_ba => gradient is 0. 
            if k == -1:
                continue
            
            grads[k,0] += abs(1-y_ab)/self.theta[k,0] - 1/ sums[k]
            grads[k,1] += abs(-y_ab)/self.theta[k,1] - 1/sums[k]
        
        # update theta 
        theta_star = copy.copy(self.theta)  
        for k in range(0,self.K):
            for i in range(0,2):
                theta_star[k,i] = abs(self.theta[k,i] + eps_t/2 * (self.eta - self.theta[k,i] + \
                                    num_total_pairs/len(mini_batch) * grads[k,i]) + eps_t**.5*self.theta[k,i] ** .5 * noise[k,i])  
        self.theta = theta_star * 1.0/self.step_count + (1-1.0/self.step_count)*self.theta
                
        # update beta from theta
        temp = self.theta/np.sum(self.theta,1)[:,np.newaxis]
        self.beta = temp[:,1]
    
    
    def update_pi_for_node(self, i, z, n):
        '''
        update pi for current node i. 
        '''                                                                                                                                                                                                                                                                                                                           
        eps_t  = self.a*((1 + self.step_count/self.b)**-self.c)        # step size 
        phi_star = copy.copy(self.phi[i])                              # updated \phi
        phi_i_sum = np.sum(self.phi[i])                                   
        noise = np.random.randn(self.K)                                 # random noise. 
        
        # get the gradients    
        grads = [-n * 1/phi_i_sum * j for j in np.ones(self.K)]
        for k in range(0, self.K):
            grads[k] += 1/self.phi[i,k] * z[k]
        
        # update the phi 
        for k in range(0, self.K):
            phi_star[k] = abs(self.phi[i,k] + eps_t/2 * (self.alpha - self.phi[i,k] + \
                                self.N/n * grads[k]) + eps_t**.5*self.phi[i,k]**.5 * noise[k])
        
        self.phi[i] = phi_star * (1.0/self.step_count) + (1-1.0/self.step_count)*self.phi[i]
        
        # update pi
        sum_phi = np.sum(self.phi[i])
        self.pi[i] = [self.phi[i,k]/sum_phi for k in range(0, self.K)]
            

    def sample_latent_vars2(self, mini_batch):
        '''
        sample latent variable (z_ab, z_ba) for each pair of nodes. But we only consider 11 different cases,
        since we only need indicator function in the gradient update. More details, please see the comments 
        within the sample_z_for_each_edge function. 
        '''
        z = {}  
        for edge in mini_batch:
            y_ab = 0
            if Edge(edge.first, edge.second) in self.network.edges_set:
                y_ab = 1
            
            z[edge] = self.sample_z_for_each_edge(y_ab, self.pi[edge.first], self.pi[edge.second], \
                                          self.beta, self.K)            

        return z
    
    def sample_z_for_each_edge(self, y, pi_a, pi_b, beta, K):
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
        
        
    def sample_latent_vars(self, node, neighbor_nodes):
        '''
        given a node and its neighbors (either linked or non-linked), return the latent value
        z_ab for each pair (node, neighbor_nodes[i]. 
        '''
        z = np.zeros(self.K)  
        for neighbor in neighbor_nodes:
            y_ab = 0      # observation
            if Edge(min(node, neighbor), max(node, neighbor)) in self.network.edges_set:
                y_ab = 1
            
            z_ab = sample_z_ab_from_edge(y_ab, self.pi[node], self.pi[neighbor], self.beta, self.epsilon, self.K)           
            z[z_ab] += 1
            
        return z
    
    def sample_z_ab_from_edge(self, y, pi_a, pi_b, beta, epsilon, K):
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
        
        # weight each part proportionally. 
        #avg_likelihood = self.link_ratio*(link_likelihood/link_count) + \
        #                    (1-self.link_ratio)*(non_link_likelihood/non_link_count) 
        
        # direct calculation. 
        avg_likelihood = (link_likelihood + non_link_likelihood)/(link_count+non_link_count)
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
    