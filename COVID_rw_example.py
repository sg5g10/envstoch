#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models.logPDF_rw import LogPosteriorRegionWise
from models.forwardModel_rw import ForwardModelRegionWise
from mcmc.sub_block_adapt_mcmc import SubBlockRegionalAMGS
from contact_m.generate_contact_matrices import breakpoint, daily_mixing_matrices
import scipy.stats as stats
import pandas as pd
import time as timer
TEST_ENABLED = False
#######################################################################################
####################      DEFINE CONSTANTS     ########################################
#######################################################################################
dt = 0.5
time = np.arange(0,167,dt)
regions = int(7)
A = int(8)
T = int(len(time))
N = np.loadtxt('./data/population.txt')
#######################################################################################
############## Create Date-time and generate Mixing Matrices    #######################
#######################################################################################
idx_w =pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='W-TUE')
idx_d =pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='12H')
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

#######################################################################################
########################      DEFINE REGIONWISE MODELS    #############################
#######################################################################################
regional_logPs = []
for reg_model in range(regions):
        forward_model = ForwardModelRegionWise(dt, time, A, N[:,reg_model], daily_mix_matrix, \
                        breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
        forward_model.real_data()
        regional_logPs.append(LogPosteriorRegionWise(\
                forward_model, reg_model, data, sero_denom[:,:,reg_model]))
#######################################################################################
########################    DEFINE starting PARAMETERS    #############################
#######################################################################################
dI = [0.54033387]
ksens = [0.54078755]
kspec = [0.4650125]
eta = [0.35]
psir = np.array([0.281224110810985,0.246300679874443, 0.230259384150778, 
        0.307383663711624, 0.249492140587071, 0.224509782739688, 0.234528728809235])
I0 = np.array([-16.7064769395683, -14.3327333035582,
                -15.0542472007424, -17.785749594464, -15.8038617705659,
                -15.3720269985272, -16.3667281951197])
p_dead = [0.00000457188478860128,0.00000557867182705255, 
                0.000025278018816958, 0.000623870211139221,
                0.00171791669192503, 0.0516645720110774, 0.102480672513791]
beta_rw_sd = [0.4510573]
m_mul = np.array(np.split(np.loadtxt('./model_data/m_mul_starts.txt'),\
        regions,axis=0)).reshape((regions,3)).T 
beta_rw = np.random.randn(17,7)*beta_rw_sd

start = []
start.append(dI)
start.append(ksens)
start.append(kspec)
start.append(eta)
start.append(p_dead)
start.append(beta_rw_sd)
for region in range(regions):
        start.append(psir[region])
        start.append(I0[region])
        start.append(m_mul[:,region])
        start.append(beta_rw[1:,region])

X0 = np.hstack(start)
#######################################################################################
#############    RUN AMGS regionwise independent MCMC SAMPLER    ######################
#######################################################################################
t0 = timer.time()
MCMCsampler = SubBlockRegionalAMGS(regional_logPs, X0, X0, global_size=12, \
                                   regional_size=21, iterations=3000)
trace = MCMCsampler.run()       
t1 = timer.time()
total = t1-t0
print('Time for SubBlock all regions AMGS RW is: ',total)
param_filename = './results/covid_mcmc_rw.p'
pickle.dump(trace, open(param_filename, 'wb'))

