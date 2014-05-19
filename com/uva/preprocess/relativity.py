from scipy.sparse import lil_matrix
from sets import Set


def process():
    f= open("/home/liwenzhe/workspace/SGRLDForMMSB/datasets/CA-GrQc.txt", 'r')
    lines = f.readlines()
    nodes = Set()
    n = len(lines)
    for i in range(4, n):
        strs = lines[i].split()
        nodes.add(int(strs[0]))
        nodes.add(int(strs[1]))
    nodelist = list(nodes)
    nodelist.sort()
    num_nodes = len(nodelist)
    
    # change the node ID to make it start from 0
    node_id_map = {}
    i = 0
    for id in nodelist:
        node_id_map[id] = i 
        i += 1
    # iterate again, and read all the links into sparse matrix. 
    sparse_matrix = lil_matrix((num_nodes,num_nodes))   # store the edges of the graph
    for i in range(4, n):
        strs = lines[i].split()
        strs[0] = int(strs[0])
        strs[1] = int(strs[1])
        node1 = node_id_map[strs[0]]
        node2 = node_id_map[strs[1]]
        sparse_matrix[min(node1,node2), max(node1,node2)] = True
        
    return (num_nodes, sparse_matrix, {})
    
