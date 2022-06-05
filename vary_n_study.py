#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time as timer
import math
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from jax import jit
from jax import partial, random, device_get, device_put
import jax.numpy as jnp
from jax.experimental import loops
from jax import random
import numpyro.distributions as dist
import jax.ops as ops

from models.logPDF_sir import LogPosterior
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC

Transform = True
iterations = 1000000
Y= np.loadtxt('./data/flu_data.txt')

burnin = 500000
end = 1000000
thin = 500


### Run the SDE and save its output
Fourier = False
n_Bases = None
logP = LogPosterior(Y, num_particles=1000, dt=0.1, transform=Transform, fourier=Fourier,e=n_Bases)
start = [0.5,0.45,0.5, 0.5, 0.5, 0.9]
X0 = np.hstack(start)
X0 = logP._transform_from_constraint(X0)
Init_scale = 1*np.abs(X0)
cov = None
sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)
MCMCsampler = MCMC(sampler, logP, X0, iterations)
trace, _, _ = MCMCsampler.run(random.PRNGKey(1))
trace_post_burn = trace[burnin:end,:]
mcs_params = trace_post_burn[::thin,:]
param_filename = './results/paper_n_vary/sde.p'
pickle.dump(mcs_params, open(param_filename, 'wb'))

### Run the SA for different values of n and save outputs
bases = [3,5,10,15,20,25,30]
timing =[]
for k in range(len(bases)):
  Fourier = True
  n_Bases = bases[k]
  logP = LogPosterior(Y, num_particles=1, dt=0.1, transform=True, fourier=Fourier,e=n_Bases)
  start = [0.5,0.45,0.5, 0.5, 0.5, 0.9,*np.random.randn(n_Bases)]
  X0 = np.hstack(start)
  X0 = logP._transform_from_constraint(X0)
  Init_scale = 1*np.abs(X0)
  cov = None
  sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)

  MCMCsampler = MCMC(sampler, logP, X0, iterations)
  t0 = timer.time()
  trace, _, X, Rt, lfx = MCMCsampler.run(random.PRNGKey(1))
  t1 = timer.time()
  total = t1-t0
  print('SA time: '+str(n_Bases), total)
  timing.append(total)
  trace_post_burn = trace[burnin:end,:]
  mco_params = trace_post_burn[::thin,:]
  param_filename = './results/paper_n_vary/ode'+str(n_Bases)+'.p'
  pickle.dump(mco_params, open(param_filename, 'wb'))

### Record the runtimes of SA with different n
runtimes = np.array(timing)
np.savetxt('./results/paper_n_vary/odetimes.txt',runtimes)





