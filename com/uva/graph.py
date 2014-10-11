import matplotlib.pyplot as plt
import math
from pylab import *
import numpy as np
from numpy import convolve


def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


mcmc= open("/home/liwenzhe/workspace/SGRLDForMMSB/results/Netscience_k100_400/result_mcmc.txt", 'r')
svi = open("/home/liwenzhe/workspace/SGRLDForMMSB/results/Netscience_k100_400/result_svi.txt", 'r')

lines_mcmc = mcmc.readlines()
lines_svi = svi.readlines()

n = len(lines_mcmc)
m = len(lines_svi)
ppx_mcmc = np.zeros(n)
ppx_svi = np.zeros(n)

for i in range(0,n):
    strs_mcmc = lines_mcmc[i].split()
    
    ppx_mcmc[i] = float(strs_mcmc[0])
    if i >= m:
        ppx_svi[i] = ppx_svi[m-1]
    else:
        strs_svi = lines_svi[i].split()
        ppx_svi[i] = float(strs_svi[0])

t =arange(0.0, n, 1)
print n
print(len(t))
"""
p1, =plot(t/12, ppx_svi)
p2, =plot(t/12,ppx_mcmc)
legend([p1,p2], ["Stochastic variational inference", "Mini-batch MCMC Sampling"])
xlabel('time (m)')
ylabel('perplexity')
title('Perplexity for relativity data set(using stratified random node sampling ')
grid(True)
savefig("relativity.png")
show()
"""

##############################################################
######              Figure 1                                 #
##############################################################

gibbs= open("results/testdata_k10/ppx_gibbs_sampler.txt", 'r')
svi = open("results/testdata_k10/ppx_variational_sampler.txt", 'r')
mcmc_online = open("results/testdata_k10/ppx_mcmc_stochastic.txt", 'r')
mcmc_batch=open("results/testdata_k10/ppx_mcmc_batch.txt", 'r')
lines_gibbs = gibbs.readlines()
lines_svi = svi.readlines() 
lines_mcmc_online = mcmc_online.readlines()
lines_mcmc_batch = mcmc_batch.readlines()

n1 = len(lines_gibbs)
n2 = len(lines_svi)
n3 = len(lines_mcmc_batch)
n4 = len(lines_mcmc_online)

# plot the gibbs sampler
ppx_gibbs =[]
times_gibbs = np.zeros(n1)
ppx_svi = []
times_svi = np.zeros(n2)
ppx_mcmc_batch = []
times_mcmc_batch=np.zeros(n3)
ppx_mcmc_online = []
times_mcmc_online=np.zeros(n4)


avg_mcmc = []
avg_svi = []
avg_gibbs = []
avg_batch = []

for i in range(0, n1):
    strs = lines_gibbs[i].split()
    ppx_gibbs.append(float(strs[0]))
    avg_gibbs.append(np.mean(ppx_gibbs))
    times_gibbs[i] = float(strs[1])
for i in range(0, n2):
    strs = lines_svi[i].split()
    ppx_svi.append(float(strs[0]))
    times_svi[i] = float(strs[1])
    avg_svi.append(np.mean(ppx_svi))

for i in range(0, n3):
    strs = lines_mcmc_batch[i].split()
    ppx_mcmc_batch.append(float(strs[0]))
    times_mcmc_batch[i] = float(strs[1])  
    avg_batch.append(np.mean(ppx_mcmc_batch))
    
for i in range(0, n4):
    strs = lines_mcmc_online[i].split()
    ppx_mcmc_online.append(float(strs[0]))
    times_mcmc_online[i] = float(strs[1])   
    avg_mcmc.append(np.mean(ppx_mcmc_online))



figure(1)     
p1, =plot(times_gibbs, avg_gibbs)
p2, =plot(times_svi,avg_svi)
p3, =plot(times_mcmc_batch, avg_batch)
p4, =plot(times_mcmc_online, avg_mcmc)
legend([p1,p2,p3,p4], ["Collapsed Gibbs Sampler", "Stochastic Variational Inference","Batch MCMC", "Mini-batch MCMC"])
xlabel('time (s)')
ylabel('perplexity')
title('Perplexity for testing data set')
xlim([1,1000])
grid(True)
savefig("small_data_4_methods_k10.png")
show()

##################################################### 
###              Figure 2                           #
#####################################################

svi = open("ppx_gibbs_sampler.txt", 'r')
mcmc_online = open("ppx_mcmc_stochastic.txt", 'r')

lines_svi = svi.readlines() 
lines_mcmc_online = mcmc_online.readlines()

n2 = len(lines_svi)
n4 = len(lines_mcmc_online)


ppx_svi = []
times_svi = np.zeros(n2)
ppx_mcmc_online = []
times_mcmc_online=np.zeros(n4)

avg_mcmc = []
avg_svi = []

for i in range(0, n2):
    strs = lines_svi[i].split()
    ppx_svi.append(float(strs[0])+1.3)
    times_svi[i] = i*10;
    avg_svi.append(np.mean(ppx_svi))

for i in range(0, n4):
    strs = lines_mcmc_online[i].split()
    ppx_mcmc_online.append(float(strs[0])+1.3)
    times_mcmc_online[i] = float(strs[1]);
    avg_mcmc.append(np.mean(ppx_mcmc_online))

axis_font = {'size':'18'}
params = {'legend.fontsize': 18,
          'legend.linewidth': 2}
plt.figure()
plt.rcParams.update(params)
p2, =plot(times_svi ,avg_svi,'r',linewidth=3.0)
p4, =plot(times_mcmc_online, avg_mcmc,'b',linewidth=3.0)

plt.legend(loc=2,prop={'fontsize':18})
legend([p4,p2], ["Stochastic mini-batch MCMC","Collapsed Gibbs Sampler"])

xlabel('time (seconds)',**axis_font)
ylabel('perplexity',**axis_font)
plt.title('US-air Data (K=15)',**axis_font)
xlim([0,1600])
ylim([2,10])
grid(True)  
savefig("us_air_mcmc_gibbs.png")
show()
