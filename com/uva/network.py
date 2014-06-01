import random
from sets import Set

class Network(object):
    """
    Network class represents the whole graph that we read from the 
    data file. Since we store all the edges ONLY, the size of this 
    information is much smaller due to the graph sparsity (in general, 
    around 0.1% of links are connected)
    
    We use the term "linked edges" to denote the edges that two nodes
    are connected, "non linked edges", otherwise. If we just say edge,
    it means either linked or non-link edge. 

    The class also contains lots of sampling methods that sampler can utilize. 
    This is great separation between different learners and data layer. By calling
    the function within this class, each learner can get different types of 
    data.
    """
    
    def __init__(self, data, held_out_ratio):
        """
        In this initialization step, we separate the whole data set
        into training, validation and testing sets. Basically, 
        Training ->  used for tuning the parameters. 
        Held-out/Validation -> used for evaluating the current model, avoid over-fitting
                      , the accuracy for validation set used as stopping criteria
        Testing -> used for calculating final model accuracy. 
        
        Arguments:
            data:   representation of the while graph. 
            vlaidation_ratio:  the percentage of data used for validation and testing. 
        """
        
        self.__N = data.N                          # number of nodes in the graph
        self.__linked_edges = data.E                       # all pair of linked edges. 
        self.__num_total_edges = len(self.__linked_edges)  # number of total edges. 
        self.__held_out_ratio = held_out_ratio             # percentage of held-out data size
        
        # Based on the a-MMSB paper, it samples equal number of 
        # linked edges and non-linked edges. 
        self.__held_out_size = int(held_out_ratio * len(self.__linked_edges))  
        
        # it is used for stratified random node sampling. By default 10 
        self.__num_pieces = 10    
        
        # The map stores all the neighboring nodes for each node, within the training
        # set. The purpose of keeping this object is to make the stratified sampling
        # process easier, in which case we need to sample all the neighboring nodes
        # given the current one. The object looks like this:
        # {
        #     0: [1,3,1000,4000]
        #     1: [0,4,999]
        #   .............
        # 10000: [0,441,9000]
        #                         }
        self.__train_link_map = {}                        
        self.__held_out_map = {}                            # store all held out edges
        self.__test_map = {}                                # store all test edges
        
        
        # initialize train_link_map 
        self.__init_train_link_map()
        # randomly sample hold-out and test sets. 
        self.__init_held_out_set()
        self.__init_test_set()
    
    def sample_mini_batch(self, mini_batch_size, strategy):
        """
        Sample a mini-batch of edges from the training data. 
        There are four different sampling strategies for edge sampling
        1.random-pair sampling
          sample node pairs uniformly at random.This method is an instance of independent 
          pair sampling, with h(x) equal to 1/(N(N-1)/2) * mini_batch_size
          
        2.random-node sampling
          A set consists of all the pairs that involve one of the N nodes: we first sample one of 
          the node from N nodes, and sample all the edges for that node. h(x) = 1/N
        
        3.stratified-random-pair sampling
          We divide the edges into linked and non-linked edges, and each time either sample
          mini-batch from linked-edges or non-linked edges.  g(x) = 1/N_0 for non-link and
          1/N_1 for link, where N_0-> number of non-linked edges, N_1-> # of linked edges.
        
        4.stratified-random-node sampling
          For each node, we define a link set consisting of all its linkes, and m non-link sets 
          that partition its non-links. We first selct a random node, and either select its link
          set or sample one of its m non-link sets. h(x) = 1/N if linked set, 1/Nm otherwise
        
        Returns (sampled_edges, scale)
            scale equals to 1/h(x), insuring the sampling gives the unbiased gradients.   
        
        """
        if strategy == "random-pair":
            return self.__random_pair_sampling(mini_batch_size)
        elif strategy == "random-node":
            return self.__random_node_sampling()
        elif strategy == "stratified-random-pair":
            return self.__stratified_random_pair_sampling(mini_batch_size)
        elif strategy == "stratified-random-node":
            return self.__stratified_random_node_sampling(10)
        else:
            print "Invalid sampling strategy, please make sure you are using the correct one:\
                [random-pair, random-node, stratified-random-pair, stratified-random-node]"
            return None
    
    
    def get_num_linked_edges(self):
        return len(self.__linked_edges)
    
    def get_num_total_edges(self):
        return self.__num_total_edges
    
    def get_num_nodes(self):
        return self.__N
    
    def get_linked_edges(self):
        return self.__linked_edges
    
    def get_held_out_set(self):
        return self.__held_out_map
    
    def get_test_set(self):
        return self.__test_map
    
    def set_num_pieces(self, num_pieces):
        self.__num_pieces = num_pieces
    
    
    def __random_pair_sampling(self, mini_batch_size):
        """
        sample list of edges from the whole training network uniformly, regardless
        of links or non-links edges.The sampling approach is pretty simple: randomly generate 
        one edge and then check if that edge passes the conditions. The iteration
        stops until we get enough (mini_batch_size) edges. 
        """
        p = mini_batch_size
        mini_batch_set = Set()     # list of samples in the mini-batch 
        
        # iterate until we get $p$ valid edges. 
        while p > 0:
            firstIdx = random.randint(0,self.__N-1)
            secondIdx = random.randint(0, self.__N-1)
            if firstIdx == secondIdx:
                continue
            # make sure the first index is smaller than the second one, since
            # we are dealing with undirected graph. 
            edge = (min(firstIdx, secondIdx), max(firstIdx, secondIdx))
            
            # the edge should not be in  1)hold_out set, 2)test_set  3) mini_batch_set (avoid duplicate)
            if edge in self.__held_out_map or edge in self.__test_map or edge in mini_batch_set:
                continue
            
            # great, we put it into the mini_batch list. 
            mini_batch_set.add(edge)
            p -= 1
        
        scale = (self.__N*(self.__N-1)/2)/mini_batch_size
         
        return (mini_batch_set, scale)
    
    
    def __random_node_sampling(self):
        """
        A set consists of all the pairs that involve one of the N nodes: we first sample one of 
        the node from N nodes, and sample all the edges for that node. h(x) = 1/N
        """
        mini_batch_set = Set()
        # randomly select the node ID
        nodeId = random.randint(0, self.__N-1)
        for i in range(0, self.__N):
            
            # make sure the first index is smaller than the second one, since
            # we are dealing with undirected graph.
            edge = (min(nodeId, i), max(nodeId, i))          
            if edge in self.__held_out_map or edge in self.__test_map \
                        or edge in mini_batch_set:
                continue
            mini_batch_set.add(edge)
        
        return (mini_batch_set, self.__N)
    
        
    def __stratified_random_pair_sampling(self, mini_batch_size):
        """
        We divide the edges into linked and non-linked edges, and each time either sample
        mini-batch from linked-edges or non-linked edges.  g(x) = 1/N_0 for non-link and
        1/N_1 for link, where N_0-> number of non-linked edges, N_1-> # of linked edges.
        """
        p = mini_batch_size
        mini_batch_set = Set()
        flag = random.randint(0,1)
        if flag == 0:
            """ sample mini-batch from linked edges """
            while p > 0:
                sampled_linked_edges = random.sample(self.__linked_edges, mini_batch_size * 2)
                for edge in sampled_linked_edges:
                    if p < 0:
                        break
                    
                    if edge in self.__held_out_map or edge in self.__test_map or edge in mini_batch_set:
                        continue
                    mini_batch_set.add(edge)
                    p -= 1
            
            return (mini_batch_set, len(self.__linked_edges)/mini_batch_size)
        
        else:
            """ sample mini-batch from non-linked edges """
            while p > 0:
                firstIdx = random.randint(0,self.__N-1)
                secondIdx = random.randint(0, self.__N-1)
                
                if (firstIdx == secondIdx):
                    continue
                # ensure the first index is smaller than the second one.  
                edge = (min(firstIdx, secondIdx), max(firstIdx, secondIdx))
        
                # check conditions: 
                if edge in self.__linked_edges or edge in self.__held_out_map \
                        or edge in self.__test_map or edge in mini_batch_set:
                    continue
                mini_batch_set.add(edge)
                p -= 1
                
            return (mini_batch_set, ((self.__N*(self.__N-1))/2 - len(self.__linked_edges)/mini_batch_size))
        
    
    def __stratified_random_node_sampling(self, num_pieces):
        """
        stratified sampling approach gives more attention to link edges (the edge is connected by two
        nodes). The sampling process works like this: 
        a) randomly choose one node $i$ from all nodes (1,....N)
        b) decide to choose link edges or non-link edges with (50%, 50%) probability. 
        c) if we decide to sample link edge:
                return all the link edges for the chosen node $i$
           else 
                sample edges from all non-links edges for node $i$. The number of edges
                we sample equals to  number of all non-link edges / num_pieces
        """
        # randomly select the node ID
        nodeId = random.randint(0, self.__N-1)
        # decide to sample links or non-links
        flag = random.randint(0,1)      # flag=0: non-link edges  flag=1: link edges
        
        mini_batch_set = Set()
        
        if flag == 0:
            """ sample non-link edges """
            # this is approximation, since the size of self.train_link_map[nodeId]
            # greatly smaller than N. 
            mini_batch_size = int(self.__N/self.__num_pieces)  
            p = mini_batch_size
            while p > 0:
                # because of the sparsity, when we sample $mini_batch_size*2$ nodes, the list likely
                # contains at least mini_batch_size valid nodes. 
                nodeList = random.sample(list(xrange(self.__N)), mini_batch_size * 2)
                for neighborId in nodeList:
                    if p < 0:
                        break
                    if neighborId == nodeId:
                        continue
                    # check condition, and insert into mini_batch_set if it is valid. 
                    edge = (min(nodeId, neighborId), max(nodeId, neighborId))
                    if edge in self.__linked_edges or edge in self.__held_out_map or \
                            edge in self.__test_map or edge in mini_batch_set:
                        continue
                    
                    # add it into mini_batch_set
                    mini_batch_set.add(edge)
                    p -= 1
                        
            return (mini_batch_set, self.__N * self.__num_pieces)
        
        else:
            """ sample linked edges """
            # return all linked edges
            for neighborId in self.__train_link_map[nodeId]:
                mini_batch_set.add((min(nodeId, neighborId),max(nodeId, neighborId)))
            
            return (mini_batch_set, self.__N)   
    

    def __init_train_link_map(self):
        """
        create a set for each node, which contains list of 
        neighborhood nodes. i.e {0: Set[2,3,4], 1: Set[3,5,6]...}
        This is used for sub-sampling 
        in the later. 
        """  
        for i in range(0, self.__N):
            self.__train_link_map[i] = Set()
        
        for edge in self.__linked_edges:
            self.__train_link_map[edge[0]].add(edge[1])
            self.__train_link_map[edge[1]].add(edge[0])
            
            
    def __init_held_out_set(self):
        """
        Sample held out set. we draw equal number of 
        links and non-links from the whole graph. 
        """
        p = self.__held_out_size/2
        
        # Sample p linked-edges from the network.
        if len(self.__linked_edges) < p:
            print "There are not enough linked edges that can sample from. \
                    please use smaller held out ratio."
            
        sampled_linked_edges = random.sample(self.__linked_edges, p)
        for edge in sampled_linked_edges: 
            self.__held_out_map[edge] = True
            self.__train_link_map[edge[0]].remove(edge[1])
            self.__train_link_map[edge[1]].remove(edge[0])
        
        # sample p non-linked edges from the network 
        while p > 0:
            edge = self.__sample_non_link_edge_for_held_out()
            self.__held_out_map[edge] = False
            p -= 1
    
    
    def __init_test_set(self):
        """
        sample test set. we draw equal number of samples for 
        linked and non-linked edges
        """
        p = int(self.__held_out_size/2)
        # sample p linked edges from the network 
        while p > 0:
            # Because we already used some of the linked edges for held_out sets,
            # here we sample twice as much as links, and select among them, which
            # is likely to contain valid p linked edges.  
            sampled_linked_edges = random.sample(self.__linked_edges, 2*p)
            for edge in sampled_linked_edges:  
                if p < 0:
                    break
                # check whether it is already used in hold_out set
                if edge in self.__held_out_map or edge in self.__test_map:
                    continue
                else:
                    self.__test_map[edge] = True
                    self.__train_link_map[edge[0]].remove(edge[1])
                    self.__train_link_map[edge[1]].remove(edge[0])
                    p -= 1
            
        # sample p non-linked edges from the network 
        p = int(self.__held_out_size/2)
        while p > 0:
            edge = self.__sample_non_link_edge_for_test()
            self.__test_map[edge] = False
            p -= 1
            
    
    
    def __sample_non_link_edge_for_held_out(self):
        '''
        sample one non-link edge for held out set from the network. We should make sure the edge is not 
        been used already, so we need to check the condition before we add it into 
        held out sets
        TODO: add condition for checking the infinit-loop
        '''
        while True:
            firstIdx = random.randint(0,self.__N-1)
            secondIdx = random.randint(0, self.__N-1)
        
            if (firstIdx == secondIdx):
                continue
            
            # ensure the first index is smaller than the second one.  
            edge = (min(firstIdx, secondIdx), max(firstIdx, secondIdx))
        
            # check conditions. 
            if edge in self.__linked_edges or edge in self.__held_out_map:
                continue
        
            return edge
    
    
    def __sample_non_link_edge_for_test(self):
        """
        Sample one non-link edge for test set from the network. We first randomly generate one 
        edge, then check conditions. If that edge passes all the conditions, return that edge. 
        TODO prevent the infinit loop
        """ 
        while True:
            firstIdx = random.randint(0,self.__N-1)
            secondIdx = random.randint(0, self.__N-1)
        
            if (firstIdx == secondIdx):
                continue
            # ensure the first index is smaller than the second one.  
            edge = (min(firstIdx, secondIdx), max(firstIdx, secondIdx))
        
            # check conditions: 
            if edge in self.__linked_edges or edge in self.__held_out_map \
                                                    or edge in self.__test_map:
                continue
        
            return edge
    
        