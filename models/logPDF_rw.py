import numpy as np
import scipy
import scipy.stats as stats
import scipy.special as special
from .COVID_CPP import death_lik

class LogPosteriorRegionWise(object):
    """
    Implementation of the covid19 model.
    This contains code for evaluating the regionwise
    loglikelihood and logprior densities.

    ## The negbinom likelihood is implemented
    in C++.
        
    ## Priors for the global params
    are to be found in the MwG code:
    `sub_block_adapt_mcmc.py`

    ## This code evaluates the epidemic model
    with the random-walk
    """    
    def __init__(self, forward_model, which_region, data, sero_denom, testing = False):

        self._forward_model = forward_model
        self.death_data = data[0][:,:,which_region]
        self.sero_data = data[1][:,:,which_region]
        self.sero_denom = sero_denom
        self.sero_length = 152
        
        self._testing = testing

        self.n_params = 33
        self.n_params_global = 12
        self.n_params_regional = 21
        self._transform = False

    def death_lik(self, ag_mu_death, eta, sample = False):      
        if sample:
            r = ag_mu_death/(eta)
            prob = (r/(r+ag_mu_death))
            return stats.nbinom(n=r, p=prob).rvs()
        else:
            if np.any(ag_mu_death<0):
                return -np.inf
            else:
                return death_lik.NegBinomial_LogLikelihood(self.death_data, ag_mu_death, eta)

    def sero_lik(self, S, ksens, kspec, sample = False):

        ag_S = S[1::2,:]
        ag_S =ag_S[:self.sero_length,:] ### change this later
        sero_ratio = ag_S/self._forward_model._N
        prob = (ksens*(1 - sero_ratio)) + ((1 - kspec)*sero_ratio)
        if sample:           
            return stats.binom(n=np.array(self.sero_denom, dtype=np.int64), p=prob).rvs()
        else:
            return np.sum(stats.binom(n=np.array(self.sero_denom, dtype=np.int64), 
            p=prob).logpmf(self.sero_data))

    def __call__(self, parameters, joint_reg=False):
        assert len(parameters) == self.n_params
        if not joint_reg:
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            psir, I0, p_dead  = parameters[4], parameters[5], parameters[6:13]
            beta_rw_sd = parameters[13]
               
        else:
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            p_dead = parameters[4:11]
            beta_rw_sd = parameters[11]
            psir, I0  = parameters[12], parameters[13]  
                     

        m_mul = parameters[14:17]          
        beta_rw = parameters[17:33]  
        beta_rw = np.hstack(([0], beta_rw))        
    
        # Call the simulator to get the expected death, sero series
        mu_death, S_ = self._forward_model.simulate(dI, psir, beta_rw, I0, p_dead, m_mul)

        # Aggregate half days for expected death series
        ag_mu_gp = np.add.reduceat(mu_death,np.arange(0,len(mu_death),2))

        # Now calculate the individual data stream log Likelihoods
        log_death_lik = self.death_lik(ag_mu_gp, eta)
        log_sero_lik = self.sero_lik(S_, ksens, kspec)
                
        # Now calculate the random-walk likelihood
        first_beta = beta_rw[:1]
        assert first_beta == 0
        log_rw_lik = np.sum(stats.norm.logpdf(beta_rw[1:],0.0,beta_rw_sd))

        log_likelihood =  log_death_lik + log_sero_lik + log_rw_lik

        ## Now the priors ####
        # exponential_growth_rate
        logPrior_psir = stats.gamma.logpdf(psir, a = 31.36, scale = 1/224)
        # initial infectives
        logPrior_I0 = stats.norm.logpdf(I0, -17.5, 1.25)
        # contact multipliers
        logPrior_m_mul = np.sum(stats.norm.logpdf(m_mul,-0.1115718, 0.4723807)) 
        
        log_prior = logPrior_psir + logPrior_I0 + logPrior_m_mul 
        return log_likelihood + log_prior
    
    def sample(self, parameters):
        assert len(parameters) == self.n_params                 
        dI = parameters[0]
        ksens, kspec, eta = parameters[1:4]
        p_dead = parameters[4:11]
        beta_rw_sd = parameters[11]
        psir, I0  = parameters[12], parameters[13]                     

        m_mul = parameters[14:17]          
        beta_rw = parameters[17:33]  
        beta_rw = np.hstack(([0], beta_rw)) 

        first_beta = beta_rw[:1]
        assert first_beta == 0
        expected_deaths, nnis, Rt = self._forward_model.simulate(dI, psir, \
            beta_rw, I0, p_dead, m_mul, generate=True)
        ag_mu_gp = np.add.reduceat(expected_deaths,np.arange(0,len(expected_deaths),2))
        return  self.death_lik(ag_mu_gp, eta, sample = True), nnis, Rt
       
    def n_parameters(self):
        return self.n_params
            
        


