import argparse
from com.uva.network import Network
from com.uva.preprocess.data_factory import DataFactory
from com.uva.learning.mcmc_sampler_stochastic import MCMCSamplerStochastic
from com.uva.learning.variational_inference_stochastic import SVI
from com.uva.learning.variational_inference_batch import SV
from com.uva.learning.mcmc_sampler_batch import MCMCSamplerBatch
from com.uva.learning.gibbs_sampler import GibbsSampler
import threading

def work_mcmc (sampler, ppxs): 
    threading.Timer(2, work_mcmc, [sampler, ppxs]).start (); 
    ppx = sampler._cal_perplexity_held_out()
    print "MCMC perplexity: " + str(ppx)
    ppxs.append(ppx)
    if len(ppxs) % 100 == 0:
        f = open('result_mcmc.txt', 'wb')
        for i in range(len(ppxs)):
            f.write(str(ppxs[i]) + "\n")
        f.close()

def work_svi (sampler, ppxs): 
    threading.Timer(2, work_svi, [sampler, ppxs]).start (); 
    ppx = sampler._cal_perplexity_held_out()
    print "SVI perplexity: " + str(ppx)
    ppxs.append(ppx)
    if len(ppxs) % 100 == 0:
        f = open('result_svi.txt', 'wb')
        for i in range(len(ppxs)):
            f.write(str(ppxs[i]) + "\n")
        f.close()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('alpha', type=float, default=0.01)
    parser.add_argument('eta0', type=float, default=1)
    parser.add_argument('eta1', type=float, default=1)
    parser.add_argument('K', type=int, default=300)  
    parser.add_argument('mini_batch_size', type=int, default=50)   # mini-batch size
    parser.add_argument('epsilon', type=float, default=0.05)
    parser.add_argument('max_iteration', type=int, default=10000000)
    
    # parameters for step size
    parser.add_argument('a', type=float, default=0.01)
    parser.add_argument('b', type=float, default=1024)
    parser.add_argument('c', type=float, default=0.55)
    
    parser.add_argument('num_updates', type=int, default=1000)
    parser.add_argument('hold_out_prob', type=float, default=0.1)
    parser.add_argument('output_dir', type=str,default='.')
    args = parser.parse_args()
    
    data = DataFactory.get_data("hep_ph")
    
    network = Network(data, 0.01)
    
    
    print len(data.E)
    
    #sampler = SVI(args, network)
    #sampler.run()
    #sampler= GibbsSampler(args, network)
    #sampler.run()
    sampler = MCMCSamplerStochastic(args, network)
    sampler.run()
    
    # Collapsed Gibbs sampler

    # Stochastic MCMC
    
 

    
    
if __name__ == '__main__':
    main()