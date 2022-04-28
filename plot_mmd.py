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
from util.mmd import GaussianKernel

burnin = 250000
end = 500000
thin = 250
param_filename = './chains_sir_sde.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:]
pm_params =  trace_post_burn[::thin,:]

burnin = 500000
end = 1000000
thin = 500
param_filename = './results/chains_sir_ode3.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:6]
mc_params_3bs =  trace_post_burn[::thin,:]


param_filename = './results/chains_sir_ode5.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:6]
mc_params_5bs =  trace_post_burn[::thin,:] 

param_filename = './results/chains_sir_ode10.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:6]
mc_params_10bs =  trace_post_burn[::thin,:]

param_filename = './results/chains_sir_ode15.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:6]
mc_params_15bs =  trace_post_burn[::thin,:] 

param_filename = './results/chains_sir_ode20.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:6]
mc_params_20bs =  trace_post_burn[::thin,:]

param_filename = './results/chains_sir_ode25.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:6]
mc_params_25bs =  trace_post_burn[::thin,:]

param_filename = './results/chains_sir_ode30.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:6]
mc_params_30bs =  trace_post_burn[::thin,:]

kern=GaussianKernel(1.5)
sig=kern.get_sigma_median_heuristic(pm_params,500)
kernmd=GaussianKernel(sig)
bs_3=kernmd.estimateMMD(mc_params_3bs,pm_params,True)
bs_5=kernmd.estimateMMD(mc_params_5bs,pm_params,True)
bs_10=kernmd.estimateMMD(mc_params_10bs,pm_params,True)
bs_15=kernmd.estimateMMD(mc_params_15bs,pm_params,True)
bs_20=kernmd.estimateMMD(mc_params_20bs,pm_params,True)
bs_25=kernmd.estimateMMD(mc_params_15bs,pm_params,True)
bs_30=kernmd.estimateMMD(mc_params_15bs,pm_params,True)
mmds_data1 = [bs_3,bs_5,bs_10,bs_15,bs_20,bs_25,bs_30]


sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(11,7),"font.size":18,"axes.titlesize":18,"axes.labelsize":18,
           "xtick.labelsize":18, "ytick.labelsize":18},style="white")
x_axis = [3,5,10,15,20,25,30]
plt.plot(x_axis,mmds_data1,'o--', ms=12,color='purple') 
plt.xlabel('Number of basis: n')
plt.ylabel('MMD between SDE and SA posteriors')
plt.xlim([2.8, 30.2])
plt.xticks(x_axis)
plt.legend(fontsize=20)
plt.show()