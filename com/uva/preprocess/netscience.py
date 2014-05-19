import xml.etree.ElementTree as ET
from sets import Set
from com.uva.edge import Edge
 
def process():
    '''
    read the .xml file, and create network object by reading link data. 
    '''
    
    # id_to_title_pair stores the attribute for each node. i.e title, name. etc
    # i.e {0: "WU, C", 1 :CHUA, L"}
    id_to_title_pair = {}               
    tree = ET.parse("/home/liwenzhe/workspace/SGRLDForMMSB/datasets/netscience.xml")
    for node in tree.iter("node"):
        attrs = node.attrib
        id_to_title_pair[attrs['id']] = attrs['title']
    
    # D = total number of nodes. 
    D = len(id_to_title_pair)           
    # iterate every link in the graph, and store those links into Set<Edge> object. 
    edges_set = Set()
    for link in tree.iter("link"):
        attrs = link.attrib
        edges_set.add(Edge(int(attrs['target']), int(attrs['source'])))
      
    return (D,edges_set, id_to_title_pair)