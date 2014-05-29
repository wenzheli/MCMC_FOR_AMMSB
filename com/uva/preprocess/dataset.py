import abc

class DataSet(object):
    """
    Served as the abstract base class for different types of data sets. 
    For each data set, we should inherit from this class. 
    """
    __metaclass__  = abc.ABCMeta
    
    def __init__(self):
        # do nothing. 
        pass
    
    @abc.abstractmethod
    def _process(self):
        """
        Function to process the document. The document can be in any format. (i.e txt, xml,..)
        The subclass will implement this function to handle specific format of
        document. Finally, return the Data object can be consumed by any learner. 
        """
        