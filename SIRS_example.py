#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import time as timer
import seaborn as sns
from jax import random
from models.logPDF_sirs import LogPosterior
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc_sirs import MCMC
from models.logPDF_sirs import SDEsimulate, SDElikelihood
import jax.numpy as jnp


def mcmc_runner(init, logprob, iters):
  X0 = np.hstack(init)

  sampler = AdaptiveMetropolis(logprob, mean_est=X0, cov_est=None)
  MCMCsampler = MCMC(sampler, logprob, X0, iterations=iters)  
  return MCMCsampler.run(random.PRNGKey(1))

def plot_marginals(sa_params):
  # plot parameter marginals 
  sns.set_context("paper", font_scale=1)
  sns.set(rc={"figure.figsize":(11,10),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            "xtick.labelsize":15, "ytick.labelsize":15},style="white")
  param_names = [r"$x_0$",r"$\sigma$",r"$1/\alpha$",r"$1/\gamma$", r"$S_0$",\
    r"$I_0$"]
  for i, p in enumerate(param_names):
          
          # Add histogram subplot
          plt.subplot(3, 3, i+1)
          if i==0:
              sns.kdeplot(sa_params[:, i], color='orange', linewidth = 2.5, label='SA (n=15)')
          elif i==1:
            sns.kdeplot(sa_params[:, i], color='orange', linewidth = 2.5)
          else:
            sns.kdeplot(sa_params[:, i], linewidth = 2.5, color='orange')  

          plt.xlabel(param_names[i])    
          plt.ylabel('Frequency')  
#          if i==6:
#            plt.xlim([0,1.1])  
          if i<1:
              plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2,fontsize=18)
  plt.subplots_adjust(hspace=0.7)
  plt.tight_layout()
  plt.show()  

def plot_gof(states, inc, y, sine, burnin):
  N=burnin

  #X_sde_traj = np.random.poisson(X_sde[-N:,:,2]*763)
  
  inc_traj = np.random.poisson(inc[-N:,:])
  mean_sa_traj = np.percentile(inc_traj,q=50,axis=0)
  CriL_sa_traj = np.percentile(inc_traj,q=2.5,axis=0)
  CriU_sa_traj = np.percentile(inc_traj,q=97.5,axis=0)

  X_sa = states
  mean_sa_states = np.percentile(X_sa[-N:,:,0],q=50,axis=0)
  CriL_sa_states = np.percentile(X_sa[-N:,:,0],q=2.5,axis=0)
  CriU_sa_states = np.percentile(X_sa[-N:,:,0],q=97.5,axis=0)

  times = np.arange(1,157,1)#
  sns.set_context("paper", font_scale=1)
  sns.set(rc={"figure.figsize":(5, 5),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            "xtick.labelsize":15, "ytick.labelsize":15},style="white")
  plt.figure(figsize=(10, 7))

  plt.subplot(2,1,1)
  
  
  plt.plot(times,mean_sa_traj, color='orange', lw=4, label='SA (n=15)')
  plt.plot(times,CriL_sa_traj, '--', color='orange', lw=2)
  plt.plot(times,CriU_sa_traj, '--',  color='orange', lw=2)
  plt.plot(times, y, 'o', color='k', lw=4, ms=5.5, label='Observations')
  plt.xlim([1,157])
  plt.ylabel('Incidence', fontsize=25)
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=3,fontsize=18)  


  plt.subplot(2,1,2)

  plt.plot(times,mean_sa_states[::7], color='orange', lw=3, label='SA (n=15)')
  plt.plot(times,CriL_sa_states[::7], '--', color='orange', lw=1)
  plt.plot(times,CriU_sa_states[::7], '--',  color='orange', lw=1)
  plt.plot(times,sine[::7], color='black', lw=4, label='True sinewave')
  plt.xlim([1,157])
  plt.ylabel(r"$x_t$", fontsize=25)
  plt.xlabel('Weeks', fontsize=25)

  plt.subplots_adjust(hspace=0.7)
  plt.tight_layout()
  plt.show()

def main(args):
  
  transform_pars_to_real = args.transform
  iterations = args.iterations
  burnin = args.burnin
  end = args.iterations
  thin = args.thin
  # data of the 14 days long influenza epidemic 1/(7*365)
  Y= np.zeros(156)
  sim_y, sim_x, t = SDEsimulate(random.PRNGKey(10),Y, np.array([0.65,0.4, 1/(50*365),1/(7*365),1/14,600,30]), 1, 7, 5, 7)
  np.savetxt('./data/sim_y.txt',sim_y)
  np.savetxt('./data/sim_x.txt',sim_x)
  print(sim_x[:14,0])
  
  plt.plot(sim_x[:,0])
  plt.savefig('./figures/sis_sim_x0.png')
  plt.close()
  plt.plot(sim_y)
  plt.savefig('./figures/sis_sim_i.png')
  plt.close()
  
  
    # Fit the SA model
  Fourier = True
  n_Bases = 20
  logP = LogPosterior(sim_y, num_particles=1, \
    transform=transform_pars_to_real, fourier=Fourier,e=n_Bases,dt=1,obs_interval=7)
  pars_init = [0.39,.01,2500,12, 610, 28, *np.random.randn(n_Bases)]
  print(len(pars_init))
  t0 = timer.time()
  trace_sa, X_sa, incidence = mcmc_runner(pars_init,logP,iterations)
  t1 = timer.time()
  total = t1-t0
  print('\n SA time: \n', total)
  trace_post_burn = trace_sa[burnin:end,:]
  sa_params = trace_post_burn[::thin,:]
  plot_marginals(sa_params)
  sim_y = np.loadtxt('./data/sim_y.txt')
  sim_x = np.loadtxt('./data/sim_x.txt')
  x_sa = np.array(X_sa)
  sa_inc = np.array(incidence)
  plot_gof(x_sa, sa_inc, sim_y, sim_x[:,0], 5000)
  
  param_filename = './results/sa_params_sirs.p'
  pickle.dump(sa_params, open(param_filename, 'wb'))

  param_filename = './results/sa_states_sirs.p'
  pickle.dump(X_sa, open(param_filename, 'wb'))  

  param_filename = './results/sa_inc_sirs.p'
  pickle.dump(incidence, open(param_filename, 'wb'))  
 

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='Fit the SDE and SA models to the Influenza epidemic')
  parser.add_argument('--transform', type=bool, default=False, metavar='N',
                      help='Transform to real line')
  parser.add_argument('--iterations', type=int, default=500000, metavar='N',
                      help='Number of iterations of MCMC')
  parser.add_argument('--burnin', type=int, default=250000, metavar='N',
                      help='Number of burnin iterations of MCMC')
  parser.add_argument('--thin', type=int, default=250, metavar='N',
                      help='Thinning factor of MCMC')
  parser.add_argument('--n_bases', type=int, default=15, metavar='N',
                      help='Number of basis functions')
  parser.add_argument('--n_particles', type=int, default=100, metavar='N',
                      help='Number of particles for PMMH') 
                    
  args = parser.parse_args()
  main(args)


  








