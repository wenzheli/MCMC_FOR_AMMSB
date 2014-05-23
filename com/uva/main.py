import argparse
from com.uva.network import Network
from com.uva.preprocess.data_factory import DataFactory
from com.uva.learning.mcmc_sampler_stochastic import MCMCSamplerStochastic
from com.uva.learning.variational_inference_stochastic import SVI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('alpha', type=float, default=0.01)
    parser.add_argument('eta0', type=float, default=1)
    parser.add_argument('eta1', type=float, default=1)
    parser.add_argument('K', type=int, default=100)  
    parser.add_argument('mini_batch_size', type=int, default=500)   # mini-batch size
    parser.add_argument('epsilon', type=float, default=0.05)
    parser.add_argument('max_iteration', type=int, default=1000)
    
    # parameters for step size
    parser.add_argument('a', type=float, default=0.01)
    parser.add_argument('b', type=float, default=1000)
    parser.add_argument('c', type=float, default=0.55)
    
    parser.add_argument('num_updates', type=int, default=1000)
    parser.add_argument('hold_out_prob', type=float, default=0.1)
    parser.add_argument('output_dir', type=str,default='.')
    args = parser.parse_args()
    
    data = DataFactory.get_data("netscience")
    network = Network(data, 0.1)
    sampler  = SVI(args, network)
    sampler.run()
    

if __name__ == '__main__':
    main()