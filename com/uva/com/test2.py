import random
import abc

class Sampler(object):
    network = 0
    """ adfadfa """
    def __init__(self):
        self._param4 = 0
    def param4(self):
        return self.__param4 
        
class MCMC(Sampler):
    def __init__(self):
        Sampler.__init__(self)
        self.param1 = 1  
        """ this is comments """
        
        self.param2 = 2
        self._param3 = 3
    
    def to_string(self):
        '''
        adfadfadsf
        '''
        return "global_vars: " + str(Sampler.network) + " local: " + str(self.param1) + ", "\
            +str(self.param2) + ", " + str(self._param3) +", " + str(self._param4)



        
class abBase(object):
    #__metaclass__  = abc.ABCMeta
    @abc.abstractmethod
    def what(self):
        """
        """

class car(abBase):
    def what(self):
        print "aaaa"
class bike(abBase):
    def what(self):
        print "bbb"    
      
class C1(object):
    def __init__(self):
        self.test()
    def test(self):
        return (1,1)
    
class C2(object):
    def __init__(self, a, b):    
        self.a = a
        self.b = b 
        print str(self.a)
        print str(self.b)
    



class cat(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
c1 = cat(1,2)
c2 = cat(1,3)
c3 = cat(1,2)

from sets import Set
sss = Set()
sss.add((1,2))
sss.add((1,2))
sss.add((1,3))

for edge in sss:
    print edge[0]
    
    
    
f = open('workfile1.txt', 'wb')
f.write("hello\n")
f.write("world\n")
f.write("world\n")
f.write("world\n")
f.write("world\n")
f.write("world\n")
f.write("world")
f.close()
print f