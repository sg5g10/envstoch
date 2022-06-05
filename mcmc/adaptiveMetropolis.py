import scipy.stats as stats
import numpy as np


class AdaptiveMetropolis(object):
    '''
    RWMH proposal adaptation based on algorithm 4
    in the paper: `A tutorial on adaptive MCMC`.
    '''
    def __init__(self, target, mean_est=None, cov_est=None, tune_interval = 1):

        self.method ='AM'
        self._target = target
        self.globalscale = 0

        if mean_est is None:
            self._mean_est = 2*np.ones(self._target.n_params)
        else:
            self._mean_est = mean_est

        if cov_est is None:

            self._cov_est = np.abs(self._mean_est)
            self._cov_est[self._cov_est == 0] = 1
            self._cov_est = np.diag(0.01 * self._cov_est)
    
        else:
            self._cov_est = cov_est

        assert (len(self._mean_est) == self._target.n_params)
        assert (len(self._cov_est) == self._target.n_params)

        self._tune_interval = tune_interval
        self._discard = 199

    def params_adapt(self, current, learning_rate):

        difference = current - self._mean_est
        self._mean_est = (1-learning_rate) * self._mean_est + learning_rate*current
        self._cov_est = (1-learning_rate) * self._cov_est + \
            learning_rate*(np.outer(difference, difference))


    def scale_adapt(self, learning_rate, accepted_or_not):

        self.globalscale += learning_rate * (accepted_or_not - 0.234) 

    def adapt(self, iteration, samples, accepted_or_not, acceptance, chain_history):

        if iteration>self._discard:
            learning_rate = (iteration - self._discard + 1.0) ** -0.6
            self.scale_adapt(learning_rate,accepted_or_not)
            if iteration % self._tune_interval == 0:
                self.params_adapt(samples,learning_rate)

    def proposal(self, y):            
            return np.random.multivariate_normal(y, \
                np.exp(self.globalscale) * self._cov_est)

    