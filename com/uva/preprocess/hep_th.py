from sets import Set
from com.uva.preprocess.dataset import DataSet
from com.uva.data import Data

class HepTh(DataSet):
    """ Process relativity data set """
    
    def __init__(self):
        pass
    
    def _process(self):
        """
        The data is stored in .txt file. The format of data is as follows, the first column
        is line number. Within each line, it is tab separated. 
        
        [1] some texts
        [2] some texts
        [3] some texts
        [4] some texts
        [5] 1    100
        [6] 1    103
        [7] 4    400
        [8] ............
        
        However, the node ID is not increasing by 1 every time. Thus, we re-format
        the node ID first. 
        """
        
        # TODO: try catch block.  
        f= open("/home/liwenzhe/workspace/SGRLDForMMSB/datasets/CA-HepTh.txt", 'r')
        lines = f.readlines()
        nodes = Set()
        n = len(lines)
        
        # start from the 5th line. 
        for i in range(4, n):
            strs = lines[i].split()
            nodes.add(int(strs[0]))
            nodes.add(int(strs[1]))
        nodelist = list(nodes)
        nodelist.sort()
        N = len(nodelist)
        
        # change the node ID to make it start from 0
        node_id_map = {}
        i = 0
        for node_id in nodelist:
            node_id_map[node_id] = i 
            i += 1
            
        E = Set()   # store all pair of edges.     
        for i in range(4, n):
            strs = lines[i].split()
            strs[0] = int(strs[0])
            strs[1] = int(strs[1])
            node1 = node_id_map[strs[0]]
            node2 = node_id_map[strs[1]]
            if node1 == node2:
                continue
            E.add((min(node1,node2), max(node1,node2)))
        
        return Data({}, E, N)
    
