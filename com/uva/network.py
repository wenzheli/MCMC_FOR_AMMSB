from com.uva.preprocess import netscience
from com.uva.preprocess import relativity
import random
from sets import Set
from random import shuffle
from itertools import islice

'''
Network class represents the whole graph that we read from the 
data file. Since we store all the edges ONLY, the size of this 
information is much smaller due to the graph sparsity (in genernal, 
around 0.3% of links are connected)

The network encodes the basic information about the graph.
'''
class Network(object):
    '''
    create network object. 
    Arguments:
    file_name:  the input file that stores graph.   
    '''
    
    def __init__(self, file_name):
        # N: number of nodes in the graph
        # E: the set of all edges (links ONLY)
        # V: "node id" to "node attribute" map. Optional. 
        if file_name == "netscience":
            (N,E,V) = netscience.process()  
        elif file_name == "relativity":
            (N,E,V) = relativity.process()      
        
        self.num_nodes = N
        self.edges_set = E
        self.id_to_name = V
        self.num_links = len(E)
        print str(self.num_links)
           
    def create_network_links(self, hold_out_percent):
        # for hold-out set and validation set, we choose the equal number of 
        # links and non-links.   
        num_hold_out_links = int(self.num_links * hold_out_percent)
        links = Set()
        non_links = Set()
        
        # separate links and non-links. 
        for i in range(self.num_nodes):                                                                                                                                                                                                 
            for j in range(i+1, self.num_nodes):
                if self.links_sparse_matrix[i,j] == 1:
                    links.add((i,j))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                    non_links.add((i,j))
                    
        hold_out_links = random.sample(links, num_hold_out_links)
        hold_out_non_links = random.sample(non_links, num_hold_out_links)
        
        # remove them from the original lists
        for link in hold_out_links:
            links.remove(link)
        for non_link  in hold_out_non_links:
            non_links.remove(non_link)
            
        # sample again for the test set. 
        validation_links = random.sample(links, num_hold_out_links)
        validation_non_links =random.sample(non_links, num_hold_out_links)
        
        # remove them from the original list
        for link in validation_links:
            links.remove(link)
        for non_link in validation_non_links:
            non_links.remove(non_link)
            
        train_edges = list(links.union(non_links))
        shuffle(train_edges) 
        
        # create dictionary for each node. 
        dict = {}
        for i in range(self.num_nodes):
            dict[i] = Set()
        for link in train_edges:
            dict[link[0]].add(link[1])
            dict[link[1]].add(link[0])
                
        return (train_edges, hold_out_links, hold_out_non_links, validation_links, validation_non_links, dict)
    

