import argparse
from com.uva.network import Network
from com.uva.sampler import Sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('alpha', type=float, default=0.01)
    parser.add_argument('eta', type=float, default=0.05)
    parser.add_argument('K', type=int, default=100)  
    parser.add_argument('mini_batch_size', type=int, default=500)   # mini-batch size
    parser.add_argument('epsilon', type=float, default=0.05)
    
    # parameters for step size
    parser.add_argument('a', type=float, default=0.01)
    parser.add_argument('b', type=float, default=1000)
    parser.add_argument('c', type=float, default=0.55)
    
    parser.add_argument('num_updates', type=int, default=1000)
    parser.add_argument('hold_out_prob', type=float, default=0.1)
    parser.add_argument('output_dir', type=str,default='.')
    args = parser.parse_args()
    
    # create network with data set name. 
    network = Network("netscience")
    # initialize sampler. 
    sampler = Sampler(network, args)
    # run by specifying sampling strategy. 
    sampler.run_sampler("random-link")
     
if __name__ == '__main__':
    main()