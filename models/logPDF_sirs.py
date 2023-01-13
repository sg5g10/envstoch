import numpy as np
import scipy
import scipy.stats as stats
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample

from jax.scipy.stats import poisson
from jax import jit
from jax import partial, random, device_get, device_put

import jax.numpy as jnp
from jax.experimental import loops
import jax.ops as ops
from jax.config import config
from jax import lax
config.update("jax_enable_x64", True)

class NegativeBinomialProbs(dist.GammaPoisson):
    arg_constraints = {
        "total_count": constraints.positive,
        "probs": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer

    def __init__(self, total_count, probs, *, validate_args=None):
        self.total_count, self.probs = promote_shapes(total_count, probs)
        concentration = total_count
        rate = 1.0 / probs - 1.0
        super().__init__(concentration, rate, validate_args=validate_args)

from models.euler_sir import EulerFourierSirs, EulerMaruyamaSirs, EulerSineSirs
#@partial(jit)
def create_indices(positions, weights):
    n = 1000#len(weights)
    indices = jnp.zeros(n, jnp.uint32)
    cumsum = jnp.cumsum(weights)
    def true_fun(args):
        a, b, c = args
        a= a.at[b].set(c)
        b += 1
        return (a,b,c)
    def false_fun(args):
        a, b, c = args
        c += 1
        return (a,b,c)

    with loops.Scope() as s:

        s.i, s.j = 0, 0
        s.operands = ()
        s.flag = True
        while s.i < 1000:
            s.operands = (indices,s.i,s.j)
            s.flag = positions[s.i] < cumsum[s.j]
            indices,s.i,s.j = lax.cond(s.flag, true_fun, false_fun, s.operands)

    return indices

#@partial(jit)
def systematic_resample(key, weights):
    n = len(weights)
    positions = (jnp.arange(n) + random.uniform(key)) / n
    return create_indices(positions, weights)

#@partial(jit)
def residual_resample(key, weights):
    n = len(weights)
    
    # take int(N*w) copies of each weight
    num_copies = (n * weights)
    num_copies = jnp.array(num_copies, jnp.int32)
    with loops.Scope() as s:

        s.indices = jnp.zeros(n)
        s.k = 0
        for i in s.range(n):
            for _ in s.range(n):  # make n copies
                s.indices = s.indices.at[s.k].set(i)  
              
                s.k += 1
    # use multinormial resample on the residual to fill up the rest.
        s.residual = weights - num_copies  # get fractional part
        s.residual /= jnp.sum(s.residual)
        s.cumsum = jnp.cumsum(s.residual)
        s.cumsum = s.cumsum.at[-1].set(1)  
        s.P = n - s.k
        s.indices = s.indices.at[s.k:n].set(jnp.searchsorted(s.cumsum, random.uniform(key, shape= (s.P,))))  
        
    return s.indices

@partial(jit, static_argnums=(3,4,5,6))
def SDElikelihood(key, y, param, dt, P, D, obs_interval):
    """
    Implements a vectorised Bootstrap SMC to
    extract a smaple path and 
    unbiased likelihood estimate
    """
    N = int(((obs_interval + dt) /dt)) - 1
    
    num_steps = y.shape[0]
    with loops.Scope() as s:

        s.zz = jnp.ones((P,D))*jnp.array([param[0],param[4],param[5],0.,10000-(param[4] + param[5])])
        s.X = jnp.zeros((P,1,D))
        s.X = jnp.zeros((P,num_steps,N,D))
        s.temp_state = jnp.zeros((P,N,D))
        s.s = 0.
        s.m = 0.
        s.w = jnp.ones(P)
        s.lw = jnp.ones(P)
        s.wt = s.w/P
        s.mLik = 0.
        s.ind = random.choice(key, a=P, shape=(P,), p=s.wt)
        #s.zz = s.zz[s.ind,:]
        s.t = jnp.zeros(num_steps)
        s.key2 = key

        s.temp_time = 0.
        for i in s.range(num_steps):      

            s.key2, euler_subkey, resample_subkey = random.split(s.key2,3)
            s.temp_state, s.temp_time = EulerMaruyamaSirs(euler_subkey, param[1:4], s.zz, s.temp_time, dt, P, N, D)   
            s.zz = s.temp_state[:,-1,:]
            s.t = s.t.at[i].set(s.temp_time)
            #s.w = jnp.exp(dist.Poisson(s.zz[...,2]*763).log_prob(y[i]))
            #s.lw = poisson.logpmf(y[i],s.zz[:,3]-s.temp_state[:,0,3])
            s.lw = NegativeBinomial2(s.zz[:,3]-s.temp_state[:,0,3],50).log_prob(y[i])
            s.m = s.lw.max()
            
            s.w = jnp.exp(s.lw - s.m)
            s.s = s.w.sum()
            s.wt = s.w/s.s
            s.mLik += s.m + jnp.log(s.s / P)
            #s.ind = random.categorical(resample_subkey, s.lw, shape=(P,))
            s.ind = random.choice(resample_subkey, a=P, shape=(P,), p=s.wt)
            s.zz = s.zz[s.ind,:]
            s.X = s.X.at[:,i,:,:].set(s.temp_state[s.ind,...])
            
        s.X =  s.X.reshape((P,num_steps*N,D),order='C')  
    return s.mLik , s.X[0,...]


@partial(jit, static_argnums=(3,4,5,6))
def SDEsimulate(key, y, param, dt, P, D, obs_interval):
    N = int(((obs_interval + dt) /dt)) - 1 
    print('size is ',N)
    
    num_steps = 156
    with loops.Scope() as s:
        s.temp_time = 0.
        s.beta0 = param[0]*(1 + param[1]*jnp.sin(((2.*jnp.pi*s.temp_time)/365.) + (2.*jnp.pi*-0.2) ) )
        s.zz = jnp.array([s.beta0,param[5],param[6],0.,10000-(param[5] + param[6])])
        s.X = jnp.zeros((num_steps,N,D))
        s.temp_state = jnp.zeros((P,N,D))
        s.data = jnp.zeros(num_steps)
        s.t = jnp.zeros(num_steps)
        s.key2 = key
        s.eta = 1.2

        
        for i in s.range(num_steps):      

            s.key2, euler_subkey, resample_subkey = random.split(s.key2,3)
            s.temp_state, s.temp_time = EulerSineSirs(euler_subkey, param[:5], s.zz, s.temp_time, dt, P, N, D)# for OU :4
            print(s.temp_state)
            s.zz = s.temp_state[0,-1,:]
            s.t = s.t.at[i].set(s.temp_time)
            s.data = s.data.at[i].set(dist.Poisson(s.zz[3]-s.temp_state[0,0,3]).sample(resample_subkey))
            s.X = s.X.at[i,:,:].set(s.temp_state[0,...])
            
        s.X =  s.X.reshape((num_steps*N,D),order='C')  
    return s.data , s.X, s.t



@partial(jit, static_argnums=(2,3,4,5))
def ODElikelihood(y, param, dt, D, obs_interval, e):
    N = int(((obs_interval + dt) /dt)) - 1
    
    num_steps = y.shape[0]
    with loops.Scope() as s:
        s.temp_time = 0.
        s.incidence = jnp.zeros_like(y)
        
        s.zz = jnp.array([param[0],param[4],param[5],0.,10000-(param[4] + param[5])])
        s.X = jnp.zeros((num_steps,N,D))
        s.temp_state = jnp.zeros((N,D))
        
        s.t = jnp.zeros(num_steps)
        s.lik = 0.
        _param = [*param[1:4],*param[6:]]
        for i in s.range(num_steps):      
            s.temp_state, s.temp_time = EulerFourierSirs(_param, s.zz, s.temp_time, dt, 1, N, D, e)# for OU :4
            
            s.zz = s.temp_state[-1,:]
            s.t = s.t.at[i].set(s.temp_time)
            s.lik += poisson.logpmf(y[i],s.zz[3]-s.temp_state[0,3])
            s.incidence = s.incidence.at[i].set(s.zz[3]-s.temp_state[0,3])
            s.X = s.X.at[i,:,:].set(s.temp_state)
            
        s.X =  s.X.reshape((num_steps*N,D),order='C')  
        #print(s.lik)
    return s.lik , s.X, s.incidence

class LogPosterior(object):
    def __init__(self, data, num_particles=100, obs_interval=1, dt=0.1, \
        transform=False, fourier=False, e=None):

        self._y = device_put(data)
        self._P = num_particles
        self._D = 5
        self._dt = dt
        self._obs_interval = obs_interval
        self._fourier = fourier
        if fourier:
            self.n_params = 6 + e
            self.e = e
        else:
            self.n_params = 6
        self._transform = transform   
        self._type = "SIRS"    
 

    def _transform_to_constraint(self, transformed_parameters):
        Tx_THETA = transformed_parameters
        Utx_beta1  = np.exp(Tx_THETA[0])
        Utx_beta2  = np.exp(Tx_THETA[1]) 
        Utx_a  = Tx_THETA[2]
        Utx_gamma  = Tx_THETA[3]
        Utx_S0  = np.exp(Tx_THETA[4])
        Utx_I0  = np.exp(Tx_THETA[5]) 
        if self._fourier:
            Utx_z  = Tx_THETA[6:]
            return np.array([Utx_beta1, Utx_beta2, Utx_a, \
                Utx_gamma, Utx_S0, Utx_I0, *Utx_z])
        return np.array([Utx_beta1, Utx_beta2, Utx_a, \
                Utx_gamma, Utx_S0, Utx_I0])

    def _transform_from_constraint(self, untransformed_parameters):
        
        Utx_THETA = untransformed_parameters
        tx_beta1  = np.log(Utx_THETA[0] - .1)
        tx_beta2  = np.log(Utx_THETA[1])
        tx_a  = Utx_THETA[2]
        tx_gamma  = Utx_THETA[3]
        tx_S0  = np.log(Utx_THETA[4] - 500)
        tx_I0  = np.log(Utx_THETA[5] - 27)
        if self._fourier:
            tx_z = Utx_THETA[6:]
            return np.array([tx_beta1, tx_beta2, tx_a, \
                tx_gamma, tx_S0, tx_I0, *tx_z])
        else:
            return np.array([tx_beta1, tx_beta2, tx_a, \
                tx_gamma, tx_S0, tx_I0])

    def __call__(self, parameters, key, simulate=False):
        if simulate:
            return SDEsimulate(key, self._y, parameters, self._dt, \
            self._P, self._D, self._obs_interval)
        if self._transform:
            _Tx_THETA = parameters.copy()
            THETA = self._transform_to_constraint(_Tx_THETA)
            _Tx_beta1  = _Tx_THETA[0]
            _Tx_beta2  = _Tx_THETA[1]
            _Tx_a  = _Tx_THETA[2]
            _Tx_gamma  = _Tx_THETA[3]
            _Tx_S0  = _Tx_THETA[4]
            _Tx_I0 = _Tx_THETA[5]
            if self._fourier:
                _Tx_z = _Tx_THETA[6:]
        else:
            THETA = parameters.copy()

        beta1  = THETA[0]
        beta2  = THETA[1]
        a  = THETA[2]
        gamma  = THETA[3]
        S0  = THETA[4]
        I0 = THETA[5]
        if self._fourier:
            z = THETA[6:]
        theta_scaled = THETA
        theta = device_put(theta_scaled)
        N_ = int(((self._obs_interval + self._dt) /self._dt)) - 1
        if (S0<500 or S0>700) or (I0<27 or I0>33) or (beta2<0):#or (I0<27 or I0>33) or (beta1<.09 or beta1>.7) or (beta2<0 or beta2>.07)
            return -np.inf, np.zeros((self._y.shape[0]*N_,self._D)), np.zeros((*self._y.shape))
        else:

            if self._fourier:
                log_likelihood, X, incidence = device_get(ODElikelihood(self._y, theta, self._dt, self._D, \
                    self._obs_interval, self.e))
                #print(log_likelihood)
            else:
                log_likelihood, X  = device_get(SDElikelihood(key, self._y, theta, self._dt, \
                self._P, self._D, self._obs_interval))
                #print(log_likelihood)
                #print(theta)

            if self._transform:
                logPrior_beta1 = stats.uniform.logpdf(beta1,loc=.1,scale=0.6) + _Tx_beta1
                print(stats.uniform.logpdf(beta1,loc=.1,scale=0.6))
                logPrior_beta2 = stats.uniform.logpdf(beta2,loc=0,scale=0.06) + _Tx_beta2     
                logPrior_a = stats.norm.logpdf(a, loc=2555, scale = 120)  
                logPrior_gamma = stats.norm.logpdf(gamma, loc=14, scale = 1.05) 
                logPrior_S0 = stats.uniform.logpdf(S0,loc=500.0,scale=200.)  + _Tx_S0
                logPrior_I0 = stats.uniform.logpdf(I0,loc=27,scale=5.)  + _Tx_I0
                if self._fourier:
                    logPrior_z = stats.norm.logpdf(z,loc=0,scale=1).sum()
                    
            else:
                logPrior_beta1 = stats.uniform.logpdf(beta1,loc=0.1,scale=0.6)
                #logPrior_beta1 = stats.uniform.logpdf(beta1,loc=10,scale=60)
                logPrior_beta2 = stats.uniform.logpdf(beta2,loc=0.0,scale=0.06) 
                #logPrior_beta2 = stats.uniform.logpdf(beta2,loc=.5,scale=55.5)     
                logPrior_a = stats.norm.logpdf(a, loc=2555, scale = 120)  
                logPrior_gamma = stats.norm.logpdf(gamma, loc=14, scale = 1.05) 
                logPrior_S0 = stats.uniform.logpdf(S0,loc=500.0,scale=200.)
                logPrior_I0 = stats.uniform.logpdf(I0,loc=27,scale=33.)      
                if self._fourier:
                    logPrior_z = stats.norm.logpdf(z,loc=0,scale=1).sum()
            log_prior = logPrior_beta1 
            + logPrior_beta2 
            + logPrior_a 
            + logPrior_gamma  
            + logPrior_S0 
            + logPrior_I0
            if self._fourier:
                log_prior += logPrior_z
            #print('prior',log_prior)
            #print('likelihood',log_likelihood)

            return log_likelihood + log_prior, X, incidence
                
    def n_parameters(self):
        return self.n_params

            
        


