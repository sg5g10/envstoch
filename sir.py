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
from models.logPDF import LogPosterior
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC


def mcmc_runner(init, logprob, iters):
  X0 = np.hstack(init)
  X0 = logprob._transform_from_constraint(X0)
  sampler = AdaptiveMetropolis(logprob, mean_est=X0, cov_est=None)
  MCMCsampler = MCMC(sampler, logprob, X0, iterations=iters)  
  return MCMCsampler.run(random.PRNGKey(1))

def plot_marginals(sde_params, sa_params):
  # plot parameter marginals 
  sns.set_context("paper", font_scale=1)
  sns.set(rc={"figure.figsize":(11,7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            "xtick.labelsize":15, "ytick.labelsize":15},style="white")
  param_names = [r"$\beta_1$",r"$\beta_2$",r"$\beta_3$",r"$\gamma$",\
    r"$x_0$",r"$i_0$"]
  for i, p in enumerate(param_names):
          
          # Add histogram subplot
          plt.subplot(2, 3, i+1)
          if i==0:
              sns.kdeplot(sde_params[:, i], color='magenta', linewidth = 2.5, label='SDE')
              sns.kdeplot(sa_params[:, i], color='orange', linewidth = 2.5, label='SA (n=15)')
          elif i==1:
            sns.kdeplot(sde_params[:, i], color='magenta', linewidth = 2.5)
            sns.kdeplot(sa_params[:, i], color='orange', linewidth = 2.5)
          else:
            sns.kdeplot(sde_params[:, i], linewidth = 2.5, color='magenta')
            sns.kdeplot(sa_params[:, i], linewidth = 2.5, color='orange')  

          plt.xlabel(param_names[i])    
          plt.ylabel('Frequency')    
          if i<1:
              plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2,fontsize=18)
  plt.subplots_adjust(hspace=0.7)
  plt.tight_layout()
  plt.show()  

def plot_gof(sample_paths, data, burnin):
  N=burnin
  X_sde = np.array(sample_paths[0]).squeeze()
  X_sde_traj = np.random.poisson(X_sde[-N:,:,2]*763)
  X_sa = np.array(sample_paths[1]).squeeze()
  X_sa_traj = np.random.poisson(X_sa[-N:,1:,2]*763)

  mean_sde_X = np.percentile(X_sde_traj,q=50,axis=0)
  CriL_sde_X = np.percentile(X_sde_traj,q=2.5,axis=0)
  CriU_sde_X = np.percentile(X_sde_traj,q=97.5,axis=0)

  mean_sa_X = np.percentile(X_sa_traj,q=50,axis=0)
  CriL_sa_X = np.percentile(X_sa_traj,q=2.5,axis=0)
  CriU_sa_X = np.percentile(X_sa_traj,q=97.5,axis=0)

  mean_sde_D = np.percentile(X_sde[-N:,:,0],q=50,axis=0)
  CriL_sde_D = np.percentile(X_sde[-N:,:,0],q=2.5,axis=0)
  CriU_sde_D = np.percentile(X_sde[-N:,:,0],q=97.5,axis=0)

  mean_sa_D = np.percentile(X_sa[-N:,1:,0],q=50,axis=0)
  CriL_sa_D = np.percentile(X_sa[-N:,1:,0],q=2.5,axis=0)
  CriU_sa_D = np.percentile(X_sa[-N:,1:,0],q=97.5,axis=0)

  times = np.arange(1,15,1)#
  sns.set_context("paper", font_scale=1)
  sns.set(rc={"figure.figsize":(17, 7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            "xtick.labelsize":15, "ytick.labelsize":15},style="white")
  plt.figure(figsize=(17, 7))

  plt.subplot(2,2,1)
  plt.plot(times,data,'o', color='k', lw=4, ms=10.5, label='Observations')
  times = np.arange(.1,14.1,.1)
  plt.plot(times,mean_sde_X, color='magenta', lw=4, label='SDE')
  plt.plot(times,CriL_sde_X, '--', color='magenta', lw=1)
  plt.plot(times,CriU_sde_X, '--',  color='magenta', lw=1)
  plt.plot(times,mean_sa_X, color='orange', lw=4, label='SA (n=15)')
  plt.plot(times,CriL_sa_X, '--', color='orange', lw=1)
  plt.plot(times,CriU_sa_X, '--',  color='orange', lw=1)
  plt.xlim([1,14])
  plt.ylabel('Counts', fontsize=25)
  plt.xlabel('Days', fontsize=25)
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=3,fontsize=18)


  plt.subplot(2,2,2)
  plt.plot(times,mean_sde_D, color='magenta', lw=3, label='SDE')
  plt.plot(times,CriL_sde_D, '--', color='magenta', lw=1)
  plt.plot(times,CriU_sde_D, '--',  color='magenta', lw=1)
  plt.plot(times,mean_sa_D, color='orange', lw=3, label='SA (n=15)')
  plt.plot(times,CriL_sa_D, '--', color='orange', lw=1)
  plt.plot(times,CriU_sa_D, '--',  color='orange', lw=1)
  plt.xlim([1,14])
  plt.ylabel(r"$x_t$", fontsize=25)
  plt.xlabel('Days', fontsize=25)

  plt.subplot(2,2,3)
  n_samples_to_plot =200
  for i in range(1,n_samples_to_plot):
    plt.plot(times,X_sde[-i,:,0], '--',  alpha=0.5, color='magenta', lw=0.25)
  plt.xlim([1,14])
  plt.ylim([-1,2])
  plt.ylabel(r"$x_t$", fontsize=25)
  plt.xlabel('Days', fontsize=25)


  plt.subplot(2,2,4)
  for i in range(1,200):
    plt.plot(times,X_sa[-i,1:,0], '--',  alpha=0.5, color='orange', lw=0.25)
  plt.xlim([1,14])
  plt.ylim([-1,2])
  plt.ylabel(r"$x_t$", fontsize=25)
  plt.xlabel('Days', fontsize=25)
  plt.subplots_adjust(wspace = 0.2, hspace=0.5)
  plt.show()

def main(args):
  transform_pars_to_real = args.transform
  iterations = args.iterations
  burnin = args.burnin
  end = args.iterations
  thin = args.thin
  # data of the 14 days long influenza epidemic
  Y= np.array([3,8,26,76,225,298,258,233,189,128,68,29,14,4])

  # Fit the SDE model
  Fourier = False
  n_Bases = None
  logP = LogPosterior(Y, num_particles=args.n_particles, \
    transform=transform_pars_to_real, fourier=Fourier,e=n_Bases)
  pars_init = np.array([0.5,0.45,0.5, 0.5, 0.5, 0.9])
  t0 = timer.time()
  trace_sde, X_sde, _ = mcmc_runner(pars_init,logP,iterations)
  t1 = timer.time()
  total = t1-t0
  print('\n SDE time: \n', total)
  trace_post_burn = trace_sde[burnin:end,:]
  sde_params = trace_post_burn[::thin,:]

  # Fit the SA model
  Fourier = True
  n_Bases = args.n_bases
  logP = LogPosterior(Y, num_particles=1, \
    transform=transform_pars_to_real, fourier=Fourier,e=n_Bases)
  pars_init = [0.5,0.45,0.5, 0.5, 0.5, 0.9,*np.random.randn(n_Bases)]
  t0 = timer.time()
  trace_sa, X_sa, _ = mcmc_runner(pars_init,logP,iterations)
  t1 = timer.time()
  total = t1-t0
  print('\n SA time: \n', total)
  trace_post_burn = trace_sa[burnin:end,:]
  sa_params = trace_post_burn[::thin,:]

  plot_marginals(sde_params, sa_params)

  plot_gof([X_sde,X_sa], Y, burnin)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='Fit the SDE and SA models to the Influenza epidemic')
  parser.add_argument('--transform', type=bool, default=True, metavar='N',
                      help='Transform to real line')
  parser.add_argument('--iterations', type=int, default=10000, metavar='N',
                      help='Number of iterations of MCMC')
  parser.add_argument('--burnin', type=int, default=5000, metavar='N',
                      help='Number of burnin iterations of MCMC')
  parser.add_argument('--thin', type=int, default=5, metavar='N',
                      help='Thinning factor of MCMC')
  parser.add_argument('--n_bases', type=int, default=15, metavar='N',
                      help='Number of basis functions')
  parser.add_argument('--n_particles', type=int, default=12, metavar='N',
                      help='Number of particles for PMMH') 
                    
  args = parser.parse_args()
  main(args)


  








