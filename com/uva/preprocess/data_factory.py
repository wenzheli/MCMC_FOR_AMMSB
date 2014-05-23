from com.uva.preprocess.netscience import NetScience
from com.uva.preprocess.relativity import Relativity


class DataFactory(object):
    """
    Factory class for creating Data object can be absorbed by sampler. 
    """
    
    @staticmethod
    def get_data(dataset_name):
        """
        Get data function, according to the input data set name. 
        """
        dataObj = None    # point to @DataSet object        
        if dataset_name == "netscience":
            dataObj = NetScience()
        elif dataset_name == "relativity":
            dataObj  = Relativity()
        else:
            pass
        
        return dataObj._process()
        
        
    
