# the file is for testing only. no other purpose. 

import numpy as np
from scipy.sparse import lil_matrix
import xml.etree.ElementTree as ET
import cProfile, pstats, StringIO
from sets import Set
import random

def primes1(kmax):
    n = 0
    k =0  
    i=0
    p = np.zeros(1000)
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n    
            k = k + 1
            result.append(n)
        n = n + 1
    return result

def getMatrix():
    A = lil_matrix((10,10))
    A[1,1] =True
    A[1,2] =True
    A[1,4]=True
     
    return A

pr = cProfile.Profile()
pr.enable()

tree = ET.parse("/home/liwenzhe/workspace/SGRLDForMMSB/datasets/netscience.xml")

root = tree.getroot() 
print root.tag
print root.attrib
aaa = {};
for node in root.iter("node"):
    d =  node.attrib
    aaa[d['id']]=d['title']



A = lil_matrix((10,10))
A[1,1] =True
A[1,2] =True
A[1,4]=True
if A[1,1] == 1:
    print "yes"



testSet = Set()
testList = []
for i in range(0,100000):
    testSet.add(i)
    testList.append(i)



pr = cProfile.Profile()
pr.enable()

random.sample(testSet,2000)

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()

def add(b):
    b[1] = 1
    b[2] = 1
    

aaa = {}
add(aaa)
print aaa