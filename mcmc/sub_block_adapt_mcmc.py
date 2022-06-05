from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import time
from joblib import Parallel, delayed
import multiprocessing
import time as timer

class SubBlockRegionalAMGS(object):
    """
    Implementation of the adaptive MwG
    algorithm, for the covid model
    """     
    def __init__(self, target, x0, mean_est, global_size=12, \
                 regional_size=21, iterations=150000):

        self._target = target
        self._iterations = iterations
        self._x0 = x0
        self._regions = int(7)
        self._sub_blocks = int(2)
        self._global_size = int(global_size)
        self._regional_size = int(regional_size)

        # This iterator churns out start index of each regional block of parameters
        self.loop_iterator = \
            range(self._global_size, ((self._regional_size*self._regions) + \
            self._global_size),self._regional_size)
        self.region_starts = np.array([i for i in self.loop_iterator])
        self._verbose = True

        # One scale for each region
        self.regionalscale = np.zeros(self._regions)
        self.globalscale = 0.0

        # One mean  and cov block with same global 
        # but different regional parameters
        self._glob_mean_est = mean_est[:self._global_size]
        self._reg_mean_est = []
        for l in self.loop_iterator:
            self._reg_mean_est.append(mean_est[l:l+self._regional_size]) 

        self._glob_cov_est = np.abs(self._glob_mean_est)
        self._glob_cov_est[self._glob_cov_est == 0] = 1
        self._glob_cov_est = np.diag(0.01 * self._glob_cov_est)     

        self._reg_cov_est = []
        for num_cov in range(self._regions):
            reg_cov_est = np.abs(self._reg_mean_est[num_cov]) 

            reg_cov_est[reg_cov_est == 0] = 1
            reg_cov_est = np.diag(0.01 * reg_cov_est)
            self._reg_cov_est.append(reg_cov_est)
            
        self._j = 12

    def global_priors(self, global_parameters):
        assert len(global_parameters) == self._global_size
        
        dI = global_parameters[0]
        ksens, kspec, eta = global_parameters[1:4]
        p_dead = global_parameters[4:11]  
        beta_rw_sd =  global_parameters[11]             

        # infectious_period 
        logPrior_dI = stats.gamma.logpdf(dI, a = 1.43, scale = 1/0.549)

        # sero sensitivity
        logPrior_ksens = stats.beta.logpdf(ksens, 23.5, 9.5)# 23.5, 9.5
 
        # sero specificity
        logPrior_kspec = stats.beta.logpdf(kspec, 569.5, 5.5) #569.5, 5.5

        # negbin_overdispersion
        logPrior_eta = stats.gamma.logpdf(eta, a = 1.0, scale = 1/0.2)# rate parameterisation


        # prop_of_death (IFR)       
        p_dead_loc = np.array([1., 1., 1., 1., 1., 1., 9.5])
        p_dead_scale = np.array([62110.801242236, 23363.4859813084, \
            5290.00529100529, 1107.64745011086, 120.951219512195, \
            31.1543408360129, 112.])
        logPrior_p_dead = np.sum(stats.beta.logpdf(p_dead, p_dead_loc, p_dead_scale))

        # random-walk/bma sd
        logPrior_beta_rw_sd = stats.gamma.logpdf(beta_rw_sd, a = 1.0, scale = 1.0/100)
               

        logGlobPrior = logPrior_dI + logPrior_ksens + logPrior_kspec + \
            logPrior_eta + logPrior_p_dead + logPrior_beta_rw_sd    
        return logGlobPrior
        
    def _run_independent(self):
        # Report the current settings
        if self._verbose:
            print('Running subblock (all regions) AMGS')
            print('Total number of iterations: ' + str(self._iterations))
    
        # Problem dimension
        d = (self._regional_size*self._regions) + self._global_size

        # Initial starting parameters
        current = self._x0

        # Chain of stored samples
        chain = np.zeros((self._iterations, d))
        chain_log_target = np.zeros(self._iterations)

        # Initial acceptance rate 
        acceptance_reg = 0
        acceptance_glob = 0

        # Setup Parallel processing
        def ParallelLogP(thr,glob,local):
            local_glob = np.hstack((glob,local[thr,:]))
            return self._target[thr](local_glob,joint_reg=True)
        num_cores = multiprocessing.cpu_count()
        print('Using total threads: ', num_cores)
        with Parallel(n_jobs=num_cores) as parallel:
            for i in range(self._iterations):
                for s in range(self._sub_blocks):                
                    # Copy the current whole parameter vector
                    proposed = current.copy()
                    if s == 0:
                        # Update one region's parameters
                        j = int(np.random.choice(self.region_starts,1))
                        pos_j = list(self.region_starts).index(j)                    
                        oneoldreg = current[j:(j+self._regional_size)]
                        onenewreg = np.random.multivariate_normal(oneoldreg, \
                                    np.exp(self.regionalscale[pos_j]) * \
                                        self._reg_cov_est[pos_j])

                        proposed[j:j+self._regional_size] = onenewreg
                    elif s == 1:
                        # Update global parameters                   
                        oldglob = current[:self._global_size]
                        newglob = np.random.multivariate_normal(oldglob, \
                                    np.exp(self.globalscale) * \
                                        self._glob_cov_est)

                        proposed[:self._global_size] = newglob                         

                    # Evaluate the logP if first iteration of the chain            
                    if i==0:
                        allreg_current = current[self._global_size:].reshape((self._regions,\
                            self._regional_size))
                        
                        current_log_targets = parallel(
                            delayed(ParallelLogP)(thread,current[:self._global_size],
                            allreg_current) for thread in range(self._regions))
                        current_log_target = np.sum(current_log_targets) + \
                            self.global_priors(current[:self._global_size])

                    # Evaluate the logP of the proposed
                    allreg_proposed = proposed[self._global_size:].reshape((self._regions,\
                        self._regional_size))
                
                    proposed_log_targets = parallel(
                        delayed(ParallelLogP)(thread,proposed[:self._global_size],
                        allreg_proposed) for thread in range(self._regions))
                    proposed_log_target = np.sum(proposed_log_targets) + \
                        self.global_priors(proposed[:self._global_size])

                    # Routine MH step
                    log_ratio = proposed_log_target - current_log_target
                    log_ratio = min(np.log(1), log_ratio)

                    if s==1:
                        accepted_glob = 0
                    elif s==0:
                        accepted_reg = 0
                    if np.isfinite(proposed_log_target):
                        if log_ratio > np.log(np.random.rand(1)):
                            if s==1:
                                accepted_glob = 1
                            elif s==0:
                                accepted_reg = 1
                            current = proposed
                            current_log_target = proposed_log_target  

                    # Free the current's copy
                    proposed = None     
                
                    # Update acceptance rates
                    if s==1:
                        acceptance_glob = (i * acceptance_glob + float(accepted_glob)) / (i + 1) 
                    elif s==0:
                        acceptance_reg = (i * acceptance_reg + float(accepted_reg)) / (i + 1)

                # Start the adaptations
                if i > 199:
                    learning_rate = (i - 199 + 1.0) ** -0.6

                    # Adapt proposal scales
                    self.globalscale += learning_rate * (accepted_glob - 0.234)  
                    self.regionalscale[pos_j] += learning_rate * (accepted_reg - 0.234) 

                    # Etch out the current's global and regional components
                    glob_current_block = current[:self._global_size]
                    reg_current_block = current[j:j+self._regional_size] 

                    # Etch out the changes for global and regional components
                    glob_difference = glob_current_block - self._glob_mean_est
                    reg_difference = reg_current_block - self._reg_mean_est[pos_j]

                    # Adapt proposal means
                    self._glob_mean_est = (1-learning_rate) * self._glob_mean_est + \
                        learning_rate*glob_current_block
                    self._reg_mean_est[pos_j] = (1-learning_rate) * self._reg_mean_est[pos_j] + \
                        learning_rate*reg_current_block

                    # Adapt proposal covariances
                    self._glob_cov_est = (1-learning_rate) * self._glob_cov_est + \
                        learning_rate*(np.outer(glob_difference, glob_difference))
                    self._reg_cov_est[pos_j] = (1-learning_rate) * self._reg_cov_est[pos_j] + \
                        learning_rate*(np.outer(reg_difference, reg_difference))
                            
                # Store the current
                chain[i,:] = current 
                chain_log_target[i] = current_log_target
                # Print-outs
                if self._verbose and ((i % 2000 == 0) or (i==self._iterations-1)):
                    print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                    print('Global  Acceptance rate: ' + str(acceptance_glob))
                    print('Regional  Acceptance rate: ' + str(acceptance_reg))
                    if i>100:
                        print('dI value: ', np.mean(chain[i-100:i,0]))
                        print('Beta sd value: ', np.mean(chain[i-100:i,11]))
                        print('Log target: ', np.mean(chain_log_target[i-100:i]))

            return chain
    
    def run(self):
        return self._run_independent()


