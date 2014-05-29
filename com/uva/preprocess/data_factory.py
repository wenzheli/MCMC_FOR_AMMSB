from com.uva.preprocess.netscience import NetScience
from com.uva.preprocess.relativity import Relativity
from com.uva.preprocess.hep_ph import HepPH
from com.uva.preprocess.astro_ph import AstroPh
from com.uva.preprocess.condmat import CondMat
from com.uva.preprocess.hep_th import HepTh


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
        elif dataset_name == "hep_ph":
            dataObj = HepPH()
        elif dataset_name == "astro_ph":
            dataObj = AstroPh()
        elif dataset_name == "condmat":
            dataObj = CondMat()
        elif dataset_name == "hep_th":
            dataObj = HepTh()
        else:
            pass
        
        return dataObj._process()
        
        
    
