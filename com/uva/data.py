
class Data(object):
    """
    Data class is an abstraction for the raw data, including vertices and edges.
    It's possible that the class can contain some pre-processing functions to clean 
    or re-structure the data.   
    
    The data can be absorbed directly by sampler. 
    """
    def __init__(self, V, E, N):
        self.V = V            # mapping between vertices and attributes. 
        self.E = E            # all pair of "linked" edges. 
        self.N = N            # number of vertices