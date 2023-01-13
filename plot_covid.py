#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from models.logPDF_rw import LogPosteriorRegionWise as LogPosteriorRegionWise_rw
from models.forwardModel_rw import ForwardModelRegionWise as ForwardModelRegionWise_rw
from models.logPDF_bma import LogPosteriorRegionWise as LogPosteriorRegionWise_bma
from models.forwardModel_bma import ForwardModelRegionWise as ForwardModelRegionWise_bma
from contact_m.generate_contact_matrices import breakpoint, daily_mixing_matrices
import scipy.stats as stats
import pandas as pd
import time as timer
import random
TEST_ENABLED = False
#######################################################################################
####################      DEFINE CONSTANTS     ########################################
#######################################################################################
dt = 0.5
time = np.arange(0,167,dt)
regions = int(7)
A = int(8)
T = int(len(time))
N = np.array([74103, 318183, 804260, 704025, 1634429, 1697206, 683583, 577399, 
122401, 493480, 1123981, 1028009, 3063113, 2017884, 575433, 483780, 
118454, 505611, 1284743, 1308343, 2631847, 2708355, 1085138, 895188, 
92626, 396683, 1014827, 1056588, 2115517, 2253914, 904953, 731817, 
79977, 340962, 851539, 845215, 1786666, 1820128, 710782, 577678, 
94823, 412569, 1085466, 1022927, 2175624, 2338669, 921243, 801040, 
55450, 241405, 633169, 644122, 1304392, 1496240, 666261, 564958]).reshape((regions,A)).T

#######################################################################################
############## Create Date-time and generate Mixing Matrices    #######################
#######################################################################################
idx_w = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='W-TUE')
idx_d = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='12H')
ts_d = pd.Series(range(len(idx_d)), index=idx_d)
breakpoints = np.array(breakpoint(ts_d, idx_d, idx_w)[1])
list_brk_beta = [breakpoints[5],breakpoints[-3], breakpoints[-1]]
list_brk_mix = [breakpoints[5],*breakpoints[-2:]]
daily_mix_matrix = daily_mixing_matrices(dt, idx_d, list_brk_mix, regions)

#######################################################################################
####################      DATA LOADING         ########################################
#######################################################################################
death_data = np.zeros((int(len(time)/2),A,regions))
days_to_load = int(len(time)/2)
sero_data = np.zeros((152,A,regions))

data = [death_data, sero_data]
regions_names = ["East_of_England",
                 "London",
                 "Midlands",
                 "North_East_and_Yorkshire",
                 "North_West",
                 "South_East",
                 "South_West"]
sero_denom_path_prefix = "20200717_"
death_path_prefix = "deaths20200807_"
sero_denom_path_postfix = "_8ag_samples"
sero_assay_path_postfix = "_8ag_positives"
death_path_postfix = "_8agCONF"
sero_denom = np.zeros((152,A,regions))
for r, region_name in enumerate(regions_names):
        sero_denom_path = sero_denom_path_prefix + \
                region_name + sero_denom_path_postfix + '.txt'
        sero_denom[:,:,r] = np.loadtxt('./data/' + \
                sero_denom_path,usecols=(1,2,3,4,5,6,7,8))
        sero_assay_path = sero_denom_path_prefix + \
                region_name + sero_assay_path_postfix + '.txt'
        sero_data[:,:,r] = np.loadtxt('./data/' + \
                sero_assay_path,usecols=(1,2,3,4,5,6,7,8))
        death_path = death_path_prefix + \
                region_name + death_path_postfix+'.txt'     
        death_data[:,:,r] = np.loadtxt('./data/' + \
                death_path,usecols=(1,2,3,4,5,6,7,8))[:days_to_load,:] 
death_data_int = death_data.astype(int)
sero_data_int = sero_data.astype(int)
data = [death_data_int, sero_data_int]

burnin = -2000000
thin = 2000

### Load rw mcmc params and set up region markers
param_filename = './results/covid_mcmc_rw.p'
trace_rw = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace_rw[burnin:,:]
py_thinned_trace_amgs_rw = trace_post_burn[::thin,:]
regional_size_rw = 21
global_size_rw = 12
region_starts_rw = [global_size_rw + (i*regional_size_rw) for i in range(regions)]

### Load bma mcmc params and set up region markers
param_filename = './results/covid_mcmc_bma.p'
trace_bma = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace_bma[burnin:,:]
py_thinned_trace_amgs_bma = trace_post_burn[::thin,:]
regional_size_bma = 15
global_size_bma = 12
region_starts_bma = [global_size_bma + (i*regional_size_bma) for i in range(regions)]

ppc_death_rw = []
ppc_nni_rw = []
ppc_rt_rw = []
ppc_death_bma = []
ppc_nni_bma = []
ppc_rt_bma = []
for RUN_REGION in range(regions):
        ### load rw model
        forward_model_rw = ForwardModelRegionWise_rw(dt, time, A, N[:,RUN_REGION], daily_mix_matrix, \
                        breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
        forward_model_rw.real_data()
        logP_rw = LogPosteriorRegionWise_rw(forward_model_rw, RUN_REGION, data, sero_denom[:,:,RUN_REGION])

        ### load bma model
        forward_model_bma = ForwardModelRegionWise_bma(dt, time, A, N[:,RUN_REGION], daily_mix_matrix, \
                        breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
        forward_model_bma.real_data()
        logP_bma = LogPosteriorRegionWise_bma(forward_model_bma, RUN_REGION, data, sero_denom[:,:,RUN_REGION])

        ### gather rw parameters
        trace_regional_rw = py_thinned_trace_amgs_rw[:, \
                region_starts_rw[RUN_REGION]:region_starts_rw[RUN_REGION]+regional_size_rw]
        trace_global_rw = py_thinned_trace_amgs_rw[:,:global_size_rw]
        trace_ppc_rw = np.concatenate((trace_global_rw,trace_regional_rw),axis=1)

        ### gather bma parameters
        trace_regional_bma = py_thinned_trace_amgs_bma[:, \
                region_starts_bma[RUN_REGION]:region_starts_bma[RUN_REGION]+regional_size_bma]
        trace_global_bma = py_thinned_trace_amgs_bma[:,:global_size_bma]
        trace_ppc_bma = np.concatenate((trace_global_bma,trace_regional_bma),axis=1)
        
        
        death_rw = []
        nni_rw = []
        rt_rw = []
        death_bma = []
        nni_bma = []
        rt_bma = []
        rand_range = random.sample(range(0, 1000), 500)
        for i in rand_range:
                _death_rw, _nni_rw, _rt_rw = logP_rw.sample(trace_ppc_rw[i,:])
                death_rw.append(_death_rw)
                nni_rw.append(_nni_rw)
                rt_rw.append(_rt_rw)
                _death_bma, _nni_bma, _rt_bma = logP_bma.sample(trace_ppc_bma[i,:])
                death_bma.append(_death_bma)
                nni_bma.append(_nni_bma)
                rt_bma.append(_rt_bma)               

        death_rw = np.array(death_rw)
        nni_rw = np.array(nni_rw)
        rt_rw = np.array(rt_rw)      
        ppc_death_rw.append(death_rw)
        ppc_nni_rw.append(nni_rw)
        ppc_rt_rw.append(rt_rw)

        death_bma = np.array(death_bma)
        nni_bma = np.array(nni_bma)
        rt_bma = np.array(rt_bma)      
        ppc_death_bma.append(death_bma)
        ppc_nni_bma.append(nni_bma)
        ppc_rt_bma.append(rt_bma)
        print('Done', RUN_REGION)

### ppc death data
ppc_death_rw = np.array(ppc_death_rw)
rw_d_mean = np.percentile(ppc_death_rw,50,axis=1).sum(axis=(0,2))
rw_d_uq = np.percentile(ppc_death_rw,97.5,axis=1).sum(axis=(0,2))
rw_d_lq = np.percentile(ppc_death_rw,2.5,axis=1).sum(axis=(0,2))
ppc_death_bma = np.array(ppc_death_bma)
bma_d_mean = np.percentile(ppc_death_bma,50,axis=1).sum(axis=(0,2))
bma_d_uq = np.percentile(ppc_death_bma,97.5,axis=1).sum(axis=(0,2))
bma_d_lq = np.percentile(ppc_death_bma,2.5,axis=1).sum(axis=(0,2))

### posterior latent infections
ppc_nni_rw = np.array(ppc_nni_rw)
rw_nni_mean = np.percentile(ppc_nni_rw,50,axis=1).sum(axis=(0,2))
rw_nni_uq = np.percentile(ppc_nni_rw,97.5,axis=1).sum(axis=(0,2))
rw_nni_lq = np.percentile(ppc_nni_rw,2.5,axis=1).sum(axis=(0,2))
ppc_nni_bma = np.array(ppc_nni_bma)
bma_nni_mean = np.percentile(ppc_nni_bma,50,axis=1).sum(axis=(0,2))
bma_nni_uq = np.percentile(ppc_nni_bma,97.5,axis=1).sum(axis=(0,2))
bma_nni_lq = np.percentile(ppc_nni_bma,2.5,axis=1).sum(axis=(0,2))




########## Plot Deaths ######################################
death_all_age_all_region = np.sum(death_data_int,axis=(1,2))
x_axis=idx_d
x_axis1 = pd.date_range('2020-02-17', '2020-08-01', freq='D')
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(17, 7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
plt.figure(figsize=(10, 7))
plt.subplot(2,1,1)
plt.axvline(x_axis1[36], linewidth=.85, color='black', label = 'Lockdown')
plt.plot(x_axis1,death_all_age_all_region,'o', color='k', lw=2, ms=4.5, label='Observations')
plt.plot(x_axis1, bma_d_lq, '--', color='orange', lw=1)
plt.plot(x_axis1, bma_d_uq, '--', color='orange', lw=1)
plt.plot(x_axis1, rw_d_lq, '--', color='magenta', lw=1)
plt.plot(x_axis1, rw_d_uq, '--', color='magenta', lw=1)
plt.plot(x_axis1, rw_d_mean, color='magenta', lw=3, label='Random-Walk')
plt.plot(x_axis1, bma_d_mean, color='orange', lw=3, label='BMA')

plt.ylabel('Deaths', fontsize=15)
plt.xlabel('Days', fontsize=15)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=4,fontsize=12)
plt.xlim([x_axis1[0],x_axis1[-1]])
########## Plot NNIs ######################################
plt.subplot(2,1,2)

plt.axvline(idx_d[72], linewidth=.85, color='black')
plt.plot(x_axis, bma_nni_lq, '--', color='orange', lw=1)
plt.plot(x_axis, bma_nni_uq, '--', color='orange', lw=1)
plt.plot(x_axis, rw_nni_lq, '--', color='magenta', lw=1)
plt.plot(x_axis, rw_nni_uq, '--', color='magenta', lw=1)
plt.plot(x_axis, rw_nni_mean, color='magenta', lw=3, label='Random-Walk')
plt.plot(x_axis, bma_nni_mean, color='orange', lw=3, label='BMA')

plt.ylabel('Infections', fontsize=15)
plt.xlabel('Days', fontsize=15)
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=3,fontsize=18)
plt.xlim([x_axis[0],x_axis[-1]])
plt.subplots_adjust(wspace = 0.5, hspace=0.5)
plt.show()


########## Plot Rts ######################################
########### All England Brownian Half Rt 
ppc_rt_rw = np.array(ppc_rt_rw)
ppc_rt_bma = np.array(ppc_rt_bma)
rw_all_eng_rt = (ppc_rt_rw.squeeze()*ppc_nni_rw.sum(axis=-1)).sum(axis=0)/ppc_nni_rw.sum(axis=(0,-1))
bma_all_eng_rt = (ppc_rt_bma.squeeze()*ppc_nni_bma.sum(axis=-1)).sum(axis=0)/ppc_nni_bma.sum(axis=(0,-1))

bma_rt_mean = np.percentile(bma_all_eng_rt,50,axis=0)
bma_rt_uq = np.percentile(bma_all_eng_rt,97.5,axis=0)
bma_rt_lq = np.percentile(bma_all_eng_rt,2.5,axis=0)

rw_rt_mean = np.percentile(rw_all_eng_rt,50,axis=0)
rw_rt_uq = np.percentile(rw_all_eng_rt,97.5,axis=0)
rw_rt_lq = np.percentile(rw_all_eng_rt,2.5,axis=0)

sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(17, 7),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")
plt.figure(figsize=(10, 5))
plt.axvline(x_axis[2*36], linewidth=.85, color='black', label = 'Lockdown')
plt.plot(x_axis, bma_rt_lq, '--', color='orange', lw=1)
plt.plot(x_axis, bma_rt_uq, '--', color='orange', lw=1)
plt.plot(x_axis, rw_rt_lq, '--', color='magenta', lw=1)
plt.plot(x_axis, rw_rt_uq, '--', color='magenta', lw=1)
plt.plot(x_axis, rw_rt_mean, color='magenta', lw=3, label='Random-Walk')
plt.plot(x_axis, bma_rt_mean, color='orange', lw=3, label='SA (n=10)')

plt.ylabel('All England ' + r"$R_{t,E}$", fontsize=12)
plt.xlabel('Days', fontsize=12)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=3,fontsize=12)
plt.xlim([x_axis[0],x_axis[-1]])
plt.show()





