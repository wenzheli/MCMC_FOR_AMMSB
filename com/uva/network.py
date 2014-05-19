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
           
   