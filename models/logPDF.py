import numpy as np
import scipy
import scipy.stats as stats
import numpyro.distributions as dist
from jax import jit
from jax import partial, random, device_get, device_put
import jax.numpy as jnp
from jax.experimental import loops
import jax.ops as ops
from jax.config import config
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=(3,4,5,6,7))
def EulerFourier(param, init_val, t, dt, P, N, D, e):
        
    flam = lambda x,y,p:p*(jnp.sqrt(2/s.Tt)*jnp.cos(((\
             ((2.0*(x+1))-1.0)*jnp.pi)/(2.0*s.Tt))*y))
    with loops.Scope() as s:
        s.beta1 = param[0]
        s.beta2 = param[1]
        s.beta3 = param[2]
        s.gamma = param[3]
        s.p = jnp.array(param[4:])
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.tt = jnp.zeros(N+1)
        s.z = s.z.at[:,0,:].set(init_val)

        s.expn = jnp.zeros(e)
        s.bases = jnp.arange(e)
        s.Tt = 14
        
        
        for i in s.range(N):

            s.expn = flam(s.bases,s.t,s.p)
            s.z = s.z.at[:,i+1,0].set(s.z[:,i,0] + ( (s.beta1 - s.beta2*s.z[:,i,0]) + (s.beta3*s.expn.sum()) )*s.dt )
            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                (-jnp.exp(s.z[:,i,0])*s.z[:,i,1]*s.z[:,i,2])*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ((jnp.exp(s.z[:,i,0])*s.z[:,i,1]*s.z[:,i,2]) - (s.gamma*s.z[:,i,2]))*s.dt)
            s.z = s.z.at[:,i+1,3].set(s.z[:,i,3] + (s.gamma*s.z[:,i,2])*s.dt)                
            s.t = s.t + s.dt
            s.tt = s.tt.at[i+1].set(s.tt[i]+s.dt)
    zar = s.z
    return zar, s.tt

@partial(jit, static_argnums=(4,5,6,7))
def EulerMaruyama(key, param, init_val, t, dt, P, N, D):
    
    with loops.Scope() as s:
        s.beta1 = param[0]
        s.beta2 = param[1]
        s.beta3 = param[2]
        s.gamma = param[3]
    
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.z = s.z.at[:,0,:].set(init_val)
        s.dw = jnp.zeros(P)
        s.subkey = key
        for i in s.range(N):
            key, s.subkey = random.split(s.subkey,2)
            s.dw = random.normal(s.subkey, shape=(P,))
            s.z = s.z.at[:,i+1,0].set(s.z[:,i,0] + ((s.beta1 - s.beta2*s.z[:,i,0])*s.dt) +\
                 (jnp.sqrt(s.dt)*s.beta3*s.dw))

            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                (-jnp.exp(s.z[:,i,0])*s.z[:,i,1]*s.z[:,i,2])*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ((jnp.exp(s.z[:,i,0])*s.z[:,i,1]*s.z[:,i,2]) - (s.gamma*s.z[:,i,2]))*s.dt)
            s.z = s.z.at[:,i+1,3].set(s.z[:,i,3] + (s.gamma*s.z[:,i,2])*s.dt)                
            s.t = s.t + s.dt
    zar = s.z
    return zar[:,1:,:], s.t

@partial(jit, static_argnums=(3,4,5,6))
def SDElikelihood(key, y, param, dt, P, D, obs_interval):
    N = int(((obs_interval + dt) /dt)) - 1
    
    num_steps = y.shape[0]
    with loops.Scope() as s:

        s.zz = jnp.ones((P,D))*jnp.array([param[4],param[5],1-param[5],0.0])
        s.X = jnp.zeros((P,1,D))
        s.X = jnp.zeros((P,num_steps,N,D))
        s.temp_state = jnp.zeros((P,N,D))

        s.w = jnp.ones(P)
        s.wt = s.w/P
        s.mLik = 0.
        s.ind = random.choice(key, a=P, shape=(P,), p=s.wt)
        s.zz = s.zz[s.ind,:]
        s.t = jnp.zeros(num_steps)
        s.key2 = key

        s.temp_time = 0.
        for i in s.range(num_steps):      

            s.key2, euler_subkey, resample_subkey = random.split(s.key2,3)
            s.temp_state, s.temp_time = EulerMaruyama(euler_subkey, param[:4], s.zz, s.temp_time, dt, P, N, D)   
            s.zz = s.temp_state[:,-1,:]
            s.t = s.t.at[i].set(s.temp_time)
            s.w = jnp.exp(dist.Poisson(s.zz[...,2]*763).log_prob(y[i]))
            s.mLik += jnp.log(s.w.mean())
            s.wt = s.w/jnp.sum(s.w)
    
            s.ind = random.choice(resample_subkey, a=P, shape=(P,), p=s.wt)
            s.zz = s.zz[s.ind,:]
            s.X = s.X.at[:,i,:,:].set(s.temp_state[s.ind,...])
            
        s.X =  s.X.reshape((P,num_steps*N,D),order='C')  
    return s.mLik , s.X[0,...], jnp.exp(s.X[0,:,0])/param[3], s.t

@partial(jit, static_argnums=(3,4,5,6))
def SDEsimulate(key, y, param, dt, P, D, obs_interval):
    N = int(((obs_interval + dt) /dt)) - 1
    
    num_steps = y.shape[0]
    with loops.Scope() as s:

        s.zz = jnp.array([param[4],param[5],1-param[5],0.0])
        s.X = jnp.zeros((num_steps,N,D))
        s.temp_state = jnp.zeros((P,N,D))
        s.data = jnp.zeros(num_steps)
        s.t = jnp.zeros(num_steps)
        s.key2 = key

        s.temp_time = 0.
        for i in s.range(num_steps):      

            s.key2, euler_subkey, resample_subkey = random.split(s.key2,3)
            s.temp_state, s.temp_time = EulerMaruyama(euler_subkey, param[:4], s.zz, s.temp_time, dt, P, N, D)
            s.zz = s.temp_state[0,-1,:]
            s.t = s.t.at[i].set(s.temp_time)
            s.data = s.data.at[i].set(dist.Poisson(s.zz[2]*763).sample(resample_subkey))

            s.X = s.X.at[i,:,:].set(s.temp_state[0,...])
            
        s.X =  s.X.reshape((num_steps*N,D),order='C')  
    return s.data , s.X

@partial(jit, static_argnums=(2,3,4,5))
def ODElikelihood(y, param, dt, D, obs_interval, e):
    N = int(obs_interval/dt)
    
    num_steps = y.shape[0]
    init_val = [param[4],param[5],1-param[5],0.0]
    _param = [*param[:4],*param[6:]]
    ind = ind=(1*N,2*N,3*N,4*N,5*N,6*N,7*N,8*N,9*N,10*N,\
        11*N,12*N,13*N,14*N)

    z,t = EulerFourier(_param, init_val, 0.0, dt, 1, N*num_steps, D, e)
    lik = dist.Poisson(763*z.squeeze()[(ind),2]).log_prob(y).sum()      
    return lik, z, jnp.exp(z[...,0])/param[3] #,t#, s.X, s.temp_state

class LogPosterior(object):
    def __init__(self, data, num_particles=100, obs_interval=1, dt=0.1, \
        transform=False, fourier=False, e=None):

        self._y = data
        self._P = num_particles
        self._D = 4
        self._dt = dt
        self._obs_interval = obs_interval
        self._fourier = fourier
        if fourier:
            self.n_params = 6 + e
            self.e = e
        else:
            self.n_params = 6
        self._transform = transform       
 

    def _transform_to_constraint(self, transformed_parameters):
        Tx_THETA = transformed_parameters
        Utx_beta1  = np.exp(Tx_THETA[0])
        Utx_beta2  = np.exp(Tx_THETA[1]) 
        Utx_beta3  = np.exp(Tx_THETA[2]) 
        Utx_gamma  = np.exp(Tx_THETA[3]) 
        Utx_x0  = Tx_THETA[4] 
        Utx_i0  = np.exp(Tx_THETA[5]) 
        if self._fourier:
            Utx_z  = Tx_THETA[6:]
            return np.array([Utx_beta1, Utx_beta2, Utx_beta3,\
            Utx_gamma, Utx_x0, Utx_i0, *Utx_z])
        return np.array([Utx_beta1, Utx_beta2, Utx_beta3,\
            Utx_gamma, Utx_x0, Utx_i0])

    def _transform_from_constraint(self, untransformed_parameters):
        
        Utx_THETA = untransformed_parameters
        tx_beta1  = np.log(Utx_THETA[0])
        tx_beta2  = np.log(Utx_THETA[1])
        tx_beta3  = np.log(Utx_THETA[2])
        tx_gamma  = np.log(Utx_THETA[3])
        tx_x0  = Utx_THETA[4]
        tx_i0  = np.log(Utx_THETA[5])
        if self._fourier:
            tx_z = Utx_THETA[6:]
            return np.array([tx_beta1, tx_beta2, tx_beta3, tx_gamma, \
                tx_x0, tx_i0, *tx_z])
        return np.array([tx_beta1, tx_beta2, tx_beta3, tx_gamma, tx_x0, tx_i0])

    def __call__(self, parameters, key, simulate=False):
        if simulate:
            return SDEsimulate(key, self._y, parameters, self._dt, \
            self._P, self._D, self._obs_interval)
        if self._transform:
            _Tx_THETA = parameters.copy()
            THETA = self._transform_to_constraint(_Tx_THETA)
            _Tx_beta1  = _Tx_THETA[0]
            _Tx_beta2  = _Tx_THETA[1]
            _Tx_beta3  = _Tx_THETA[2]
            _Tx_gamma  = _Tx_THETA[3]
            _Tx_x0  = _Tx_THETA[4]
            _Tx_i0 = _Tx_THETA[5]
            if self._fourier:
                _Tx_z = _Tx_THETA[6:]
        else:
            THETA = parameters.copy()

        beta1  = THETA[0]
        beta2  = THETA[1]
        beta3  = THETA[2]
        gamma  = THETA[3]
        x0  = THETA[4]
        i0 = THETA[5]
        if self._fourier:
            z = THETA[6:]
        theta_scaled = THETA
        theta = device_put(theta_scaled)

        if self._fourier:
            log_likelihood, X = ODElikelihood(self._y, theta, self._dt, self._D, \
                self._obs_interval, self.e)
        else:
            log_likelihood, X  = SDElikelihood(key, self._y, theta, self._dt, \
            self._P, self._D, self._obs_interval)

        if self._transform:
            logPrior_beta1 = stats.gamma.logpdf(beta1,a=2.,scale=1/2) + _Tx_beta1
            logPrior_beta2 = stats.gamma.logpdf(beta2,a=2.,scale=1/2) + _Tx_beta2     
            logPrior_beta3 = stats.gamma.logpdf(beta3,a=2.,scale=1/2) + _Tx_beta3   
            logPrior_gamma = stats.gamma.logpdf(gamma,a=2, scale = 1/2) + _Tx_gamma 
            logPrior_x0 = stats.norm.logpdf(x0,loc=(beta1/beta2),scale=((beta3**2)/(2*beta2)))
            logPrior_i0 = stats.beta.logpdf(i0,a=2.,b=1.) + _Tx_i0 
            if self._fourier:
                logPrior_z = stats.norm.logpdf(z,loc=0,scale=1).sum()
                   
        else:
            logPrior_beta1 = stats.gamma.logpdf(beta1,a=2.,scale=1/2)
            logPrior_beta2 = stats.gamma.logpdf(beta2,a=2.,scale=1/2)     
            logPrior_beta3 = stats.gamma.logpdf(beta3,a=2, scale = 1/2)  
            logPrior_gamma = stats.gamma.logpdf(gamma,a=2, scale = 1/2) 
            logPrior_x0 = stats.normal.logpdf(x0,loc=(beta1/beta2),scale=((beta3**2)/(2*beta2)))
            logPrior_i0 = stats.beta.logpdf(i0,a=2.,b=1.)      

        log_prior = logPrior_beta1 + logPrior_beta2 + logPrior_beta3 +\
             logPrior_gamma + logPrior_x0 + logPrior_i0
        if self._fourier:
            log_prior += logPrior_z
        return log_likelihood + log_prior, X, log_likelihood
                   
    def n_parameters(self):
        return self.n_params

            
        


