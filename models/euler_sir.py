from jax import jit
from jax import partial, random, device_get, device_put
import jax.numpy as jnp
from jax.experimental import loops
import jax.ops as ops
from jax.config import config
config.update("jax_enable_x64", True)

@partial(jit, static_argnums=(3,4,5,6,7))
def EulerFourier(param, init_val, t, dt, P, N, D, e):
    """
    Implementation of the vectorised
    Euler ODE solver, 
    where the ODE comes from series approx. 
    """

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
        s.Tt = 14 # Time interval    
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
    """
    Implementation of the vectorised
    Euler-Maruyama solver, 
    for the SDE models. 
    """    
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