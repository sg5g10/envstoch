from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from jax import random


class MCMC(object):
    
    _verbose=True
    def __init__(self, sampler, target, x0, iterations=150000):

        self._target = target
        self._iterations = iterations 

        self._sampler = sampler   

        self._x0 = x0 
        self._acceptance_target = None
        

    def run(self, prng_key):
        # Report the current settings
        if self._verbose:
            print('Running adaptive Blocked MCMC')
            print('Total number of iterations: ' + str(self._iterations))

    
        # Problem dimension
        d = self._target.n_parameters()#n_params#

        # Initial starting parameters
        current = self._x0 

        # Chain of stored samples

        chain = np.zeros((self._iterations,d)) 
        if self._target._transform:
            tx_chain = np.zeros((self._iterations,d))

        log_pdfs = np.zeros(self._iterations) 
        if self._target._transform:
            tx_log_pdfs = np.zeros(self._iterations)            

        # Initial acceptance rate (value doesn't matter)

        acceptance = 0 
        X = []
        for i in range(self._iterations):

            proposed = self._sampler.proposal(current)
            prng_key, skey = random.split(prng_key)
            if i==0:
                prng_key, key = random.split(prng_key)
                current_log_target, _x = self._target(current, key)

            proposed_log_target, _x = self._target(proposed, skey)
            
            log_ratio = (proposed_log_target - current_log_target)

            log_ratio = min(np.log(1), log_ratio)
            accepted = 0
            
            if np.isfinite(proposed_log_target):
                if log_ratio > np.log(np.random.rand(1)):
                    accepted = 1
                    current = proposed
                    X.append(_x)
                    current_log_target = proposed_log_target

            proposed = None
            
            # Store the current in the chain
            if self._target._transform:
                chain[i,:] = self._target._transform_to_constraint(current)
                tx_chain[i,:] = current
                tx_log_pdfs[i] = current_log_target
            else:
                chain[i,:] = current
                log_pdfs[i] = current_log_target

            # Update acceptance rate
            acceptance = (i * acceptance + float(accepted)) / (i + 1)
            if self._target._transform:
                self._sampler.adapt(i, current, accepted, log_ratio, tx_chain[:i,:])
            else:
                self._sampler.adapt(i, current, accepted, log_ratio, chain[:i,:])

            # Report
            if self._verbose and i % 1000 == 0:
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))

                print('  Acceptance rate: ' + str(acceptance))
                print('  Current params: ', np.mean(chain[i-100:i,:],axis=0))
                
                
        # Return generated chain
        if self._target._transform:
            return chain, X, tx_chain
        else:
            return chain
