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

from models.logPDF import LogPosterior
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC

Transform = True
iterations = 1000000

P = True
Y= np.array([3,8,26,76,225,298,258,233,189,128,68,29,14,4])
burnin = 500000
end = 1000000
thin = 500

sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(11,7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
          "xtick.labelsize":15, "ytick.labelsize":15},style="white")
param_names = [r"$\beta_1$",r"$\beta_2$",r"$\beta_3$",r"$\gamma$",\
  r"$x_0$",r"$i_0$"]
 

bases = [3,5,10,15,20,25,30]
timing =[]
for k in range(len(bases)):
  """
  Fourier = False
  n_Bases = bases

  logP = LogPosterior(Y, num_particles=1000, dt=0.1, transform=Transform, fourier=Fourier,e=n_Bases)

  start = [0.5,0.45,0.5, 0.5, 0.5, 0.9]
  X0 = np.hstack(start)
  X0 = logP._transform_from_constraint(X0)
  Init_scale = 1*np.abs(X0)
  cov = None
  # Now run the AMGS sampler
  sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)

  MCMCsampler = MCMC(sampler, logP, X0, iterations)
  trace, _, _, _, _ = MCMCsampler.run(random.PRNGKey(1))
  trace_post_burn = trace[burnin:end,:]
  mcs_params = trace_post_burn[::thin,:]
  """


  Fourier = True
  n_Bases = bases[k]
  logP = LogPosterior(Y, num_particles=1, dt=0.1, transform=True, fourier=Fourier,e=n_Bases)
  start = [0.5,0.45,0.5, 0.5, 0.5, 0.9,*np.random.randn(n_Bases)]#[.3,.6,.2,0.15,1,.9,*np.random.randn(n_Bases)]
  X0 = np.hstack(start)
  X0 = logP._transform_from_constraint(X0)
  Init_scale = 1*np.abs(X0)
  cov = None
  # Now run the AMGS sampler
  sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)

  MCMCsampler = MCMC(sampler, logP, X0, iterations)
  #MCMCsampler = SubBlockRegionalAMGS(logP, X0, X0, iterations)
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

runtimes = np.array(timing)
np.savetxt('./results/paper_n_vary/odetimes.txt',runtimes)
 





