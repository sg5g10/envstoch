#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import scipy.stats as stats
from util.mmd import GaussianKernel


param_filename = './results/paper_n_vary/sde.p'
pm_params = pickle.load( open( param_filename , "rb" ) )
print(pm_params.shape)

param_filename = './results/paper_n_vary/ode3.p'
mc_params_3bs = pickle.load( open( param_filename , "rb" ) )[:,:6]
print(mc_params_3bs.shape)
param_filename = './results/paper_n_vary/ode5.p'
mc_params_5bs = pickle.load( open( param_filename , "rb" ) )[:,:6]

param_filename = './results/paper_n_vary/ode10.p'
mc_params_10bs = pickle.load( open( param_filename , "rb" ) )[:,:6]

param_filename = './results/paper_n_vary/ode15.p'
mc_params_15bs = pickle.load( open( param_filename , "rb" ) )[:,:6]

param_filename = './results/paper_n_vary/ode20.p'
mc_params_20bs = pickle.load( open( param_filename , "rb" ) )[:,:6]

param_filename = './results/paper_n_vary/ode25.p'
mc_params_25bs = pickle.load( open( param_filename , "rb" ) )[:,:6]

param_filename = './results/paper_n_vary/ode30.p'
mc_params_30bs = pickle.load( open( param_filename , "rb" ) )[:,:6]

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