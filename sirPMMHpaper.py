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
from jax import random
from models.logPDF import LogPosterior
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC

Transform = True
iterations = 100000

P = True
Y= np.array([3,8,26,76,225,298,258,233,189,128,68,29,14,4])


if not P:
    param_filename = 'sim_data2.p'
    sim_dataS = pickle.load( open( param_filename , "rb" ) )  
    Fourier = False
    n_Bases = None
    #sim_dataS = Y  
    """
    logP = LogPosterior(Y, num_particles=1000, transform=True, fourier=Fourier,e=n_Bases)
    start = np.array([.2,2.5,.05, .65, 0, 0.995])
    sim_dataS, sim_S=logP(start,random.PRNGKey(1),True)
    
    plt.plot(sim_dataS)
    plt.show()
    plt.plot(sim_S[:,0])
    plt.show()
    
    param_filename = 'sim_data302.p'
    pickle.dump(sim_dataS, open(param_filename, 'wb'))

    param_filename = 'sim_traces302.p'
    pickle.dump(sim_S, open(param_filename, 'wb'))
    """
    logP = LogPosterior(sim_dataS, num_particles=1000, transform=True, fourier=Fourier,e=n_Bases)
    start = [0.5,0.45,0.5, 0.5, 0.5, 0.9]
    X0 = np.hstack(start)
    X0 = logP._transform_from_constraint(X0)
    Init_scale = 1*np.abs(X0)
    cov = None
    # Now run the AMGS sampler
    sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)
    t0 = timer.time()
    MCMCsampler = MCMC(sampler, logP, X0, iterations)
    trace, _, X, Rt, lfx = MCMCsampler.run(random.PRNGKey(1))
    t1 = timer.time()
    total = t1-t0
    print('SDE time: ', total)
    
    param_filename = 'chains_sir_sdechn1.p'
    pickle.dump(trace, open(param_filename, 'wb'))
    
    param_filename = 'X_sir_sdechn1.p'
    pickle.dump(X, open(param_filename, 'wb'))
    
    param_filename = 'Rt_sir_sdechn1.p'
    pickle.dump(Rt, open(param_filename, 'wb'))
    
    param_filename = 'lfx_sir_sdechn1.p'
    pickle.dump(lfx, open(param_filename, 'wb'))
    
    
    param_filename = 'sim_data2.p'
    sim_dataS = pickle.load( open( param_filename , "rb" ) )
    #sim_dataS = Y
    Fourier = True
    n_Bases = 15
    logP = LogPosterior(sim_dataS, num_particles=1, transform=True, fourier=Fourier,e=n_Bases)
    start = [0.5,0.45,0.5, 0.5, 0.5, 0.9,*np.random.randn(n_Bases)]#[.3,.6,.2,0.15,1,.9,*np.random.randn(n_Bases)]
    X0 = np.hstack(start)
    X0 = logP._transform_from_constraint(X0)
    Init_scale = 1*np.abs(X0)
    cov = None
    # Now run the AMGS sampler
    sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)
    t0 = timer.time()
    MCMCsampler = MCMC(sampler, logP, X0, iterations)
    #MCMCsampler = SubBlockRegionalAMGS(logP, X0, X0, iterations)
    trace, _, X, Rt, lfx = MCMCsampler.run(random.PRNGKey(1))
    t1 = timer.time()
    total = t1-t0
    print('ODE time: ', total)

    param_filename = 'chains_sir_odechn1.p'
    pickle.dump(trace, open(param_filename, 'wb'))
 
    param_filename = 'X_sir_odechn1.p'
    pickle.dump(X, open(param_filename, 'wb'))
    
    param_filename = 'Rt_sir_odechn1.p'
    pickle.dump(Rt, open(param_filename, 'wb'))
    
    param_filename = 'lfx_sir_odechn1.p'
    pickle.dump(lfx, open(param_filename, 'wb'))
    
    
### works till 20

burnin = 500000
end = 1000000
thin = 500
param_filename = 'chains_sir_ode10k3.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:]
mco_params = trace_post_burn[::thin,:]


burnin = 500000
end = 1000000
thin = 500
param_filename = './chains_sir_sde10k3.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:end,:]
#trace_post_burn = trace[N00:1000000,:]
mcs_params = trace_post_burn[::thin,:]

sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(11,7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
param_names = [r"$\beta_1$",r"$\beta_2$",r"$\beta_3$",r"$\gamma$",\
  r"$x_0$",r"$i_0$"]
real_params = np.array([.3,.6,.1, .45, .3/.6, 0.995])
for i, p in enumerate(param_names):
        
        # Add histogram subplot
        plt.subplot(2, 3, i+1)
        plt.axvline(real_params[i], linewidth=2.5, color='black')
        if i==0:
            sns.kdeplot(mcs_params[:, i], color='magenta', linewidth = 2.5, label='SDE')
            sns.kdeplot(mco_params[:, i], color='orange', linewidth = 2.5, label='SA (n=15)')
        elif i==1:
          sns.kdeplot(mcs_params[:, i], color='magenta', linewidth = 2.5)
          sns.kdeplot(mco_params[:, i], color='orange', linewidth = 2.5)
        else:
          sns.kdeplot(mcs_params[:, i], linewidth = 2.5, color='magenta')
          sns.kdeplot(mco_params[:, i], linewidth = 2.5, color='orange')  

#        if i==0 and i==3:
          
        plt.xlabel(param_names[i])    
        plt.ylabel('Frequency')    
        if i<1:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2,fontsize=18)
plt.subplots_adjust(hspace=0.7)
plt.tight_layout()
plt.show()

"""
param_filename = './lfx_sir_sde10k3.p'
lfx_sde = pickle.load( open( param_filename , "rb" ) )
param_filename = './lfx_sir_ode10k3.p'
lfx_ode = pickle.load( open( param_filename , "rb" ) )
plt.hist(lfx_sde[50:],30, alpha=0.5, color='magenta',label='SDE')
plt.hist(lfx_ode[50:],30, alpha=0.5, color='orange',label='ODE')
plt.legend()
plt.show()
"""


N=15000
param_filename = './sim_traces3.p'
sim_S = pickle.load( open( param_filename , "rb" ) )
param_filename = './sim_data3.p'
sim_dataS = pickle.load( open( param_filename , "rb" ) )

param_filename = './X_sir_sde10k3.p'
X_sde = pickle.load( open( param_filename , "rb" ) )
X_sde = np.array(X_sde).squeeze()
X_sde_traj = np.random.poisson(X_sde[-N:,:,2]*763)#:50



param_filename = './X_sir_ode10k3.p'
X_ode = pickle.load( open( param_filename , "rb" ) )
X_ode = np.array(X_ode).squeeze()
X_ode_traj = np.random.poisson(X_ode[-N:,1:,2]*763)

mean_sde_X = np.percentile(X_sde_traj,q=50,axis=0)
CriL_sde_X = np.percentile(X_sde_traj,q=2.5,axis=0)
CriU_sde_X = np.percentile(X_sde_traj,q=97.5,axis=0)

mean_ode_X = np.percentile(X_ode_traj,q=50,axis=0)
CriL_ode_X = np.percentile(X_ode_traj,q=2.5,axis=0)
CriU_ode_X = np.percentile(X_ode_traj,q=97.5,axis=0)

mean_sde_D = np.percentile(X_sde[-N:,:,0],q=50,axis=0)
CriL_sde_D = np.percentile(X_sde[-N:,:,0],q=2.5,axis=0)
CriU_sde_D = np.percentile(X_sde[-N:,:,0],q=97.5,axis=0)

mean_ode_D = np.percentile(X_ode[-N:,1:,0],q=50,axis=0)
CriL_ode_D = np.percentile(X_ode[-N:,1:,0],q=2.5,axis=0)
CriU_ode_D = np.percentile(X_ode[-N:,1:,0],q=97.5,axis=0)

times = np.arange(1,15,1)#
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(17, 7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
plt.figure(figsize=(17, 7))
plt.subplot(2,2,1)
plt.plot(times,sim_dataS,'o', color='k', lw=4, ms=10.5, label='Observations')
times = np.arange(.1,14.1,.1)#times = np.arange(.1,14.1,.1)
plt.plot(times,mean_sde_X, color='magenta', lw=4, label='SDE')
plt.plot(times,CriL_sde_X, '--', color='magenta', lw=1)
plt.plot(times,CriU_sde_X, '--',  color='magenta', lw=1)
plt.plot(times,mean_ode_X, color='orange', lw=4, label='SA (n=15)')
plt.plot(times,CriL_ode_X, '--', color='orange', lw=1)
plt.plot(times,CriU_ode_X, '--',  color='orange', lw=1)
plt.xlim([1,14])#plt.xlim([1,14])
plt.ylabel('Counts', fontsize=25)
plt.xlabel('Days', fontsize=25)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=3,fontsize=18)

#plt.xticks(times,rotation=45, fontsize=25)
#plt.yticks(fontsize=25)


plt.subplot(2,2,2)
#plt.plot(times,sim_S[:,0], color='black', lw=4, label='True OU')
plt.plot(times,mean_sde_D, color='magenta', lw=3, label='SDE')
plt.plot(times,CriL_sde_D, '--', color='magenta', lw=1)
plt.plot(times,CriU_sde_D, '--',  color='magenta', lw=1)
plt.plot(times,mean_ode_D, color='orange', lw=3, label='SA (n=15)')
plt.plot(times,CriL_ode_D, '--', color='orange', lw=1)
plt.plot(times,CriU_ode_D, '--',  color='orange', lw=1)

plt.xlim([1,14])#plt.xlim([1,14])

plt.ylabel(r"$x_t$", fontsize=25)
plt.xlabel('Days', fontsize=25)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=3,fontsize=18)

plt.subplot(2,2,3)
for i in range(1,200):
  plt.plot(times,X_sde[-i,:,0], '--',  alpha=0.5, color='magenta', lw=0.25)
#plt.plot(times,sim_S[:,0], color='black', lw=4, label='True OU')
plt.xlim([1,14])#plt.xlim([1,14])
plt.ylim([-1,2])
plt.ylabel(r"$x_t$", fontsize=25)
plt.xlabel('Days', fontsize=25)


plt.subplot(2,2,4)
for i in range(1,200):
  plt.plot(times,X_ode[-i,1:,0], '--',  alpha=0.5, color='orange', lw=0.25)
#plt.plot(times,sim_S[:,0], color='black', lw=4, label='True OU')
plt.xlim([1,14])#plt.xlim([1,14])
plt.ylim([-1,2])
plt.ylabel(r"$x_t$", fontsize=25)
plt.xlabel('Days', fontsize=25)
plt.subplots_adjust(wspace = 0.2, hspace=0.5)
plt.show()


