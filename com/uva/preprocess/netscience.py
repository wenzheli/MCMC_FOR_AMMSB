import xml.etree.ElementTree as ET
from sets import Set
from com.uva.preprocess.dataset import DataSet
from com.uva.data import Data

class NetScience(DataSet):
    """ Process netscience data set """
    
    def __init__(self):
        pass
              
    def _process(self):
        
        max_n = 500
        
        """
        The netscience data is stored in xml format. The function just reads all the vertices
        and edges.
        * if vertices are not record as the format of 0,1,2,3....., we need to do some 
          process.  Fortunally, there is no such issue with netscience data set.   
        """
        # V stores the mapping between node ID and attribute. i.e title, name. etc
        # i.e {0: "WU, C", 1 :CHUA, L"}
        V = {}               
        # file path of netscience data set. 
        tree = ET.parse("/home/liwenzhe/workspace/SGRLDForMMSB/datasets/netscience.xml")
        for node in tree.iter("node"):
            attrs = node.attrib
            V[attrs['id']] = attrs['title']
            
        N = min(len(V), max_n)   
        
        # iterate every link in the graph, and store those links into Set<Edge> object. 
        E = Set()
        for link in tree.iter("link"):
            attrs = link.attrib
            if int(attrs['target']) >= N or int(attrs['source'])>=N:
                   continue
            E.add((int(attrs['target']), int(attrs['source'])))
            
        return Data(V, E, N)
    
             
# file path of netscience data set. 
tree = ET.parse("/home/liwenzhe/workspace/SGRLDForMMSB/datasets/netscience.xml")
        
f = open('netscience.txt', 'wb')      
 # iterate every link in the graph, and store those links into Set<Edge> object. 
for link in tree.iter("link"):
    attrs = link.attrib
    f.write(attrs['target']+"\t"+attrs['source']+"\n")
f.close()
    