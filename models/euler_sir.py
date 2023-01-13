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

@partial(jit, static_argnums=(4,5,6,7))
def EulerStep(key, param, init_val, t, dt, P, N, D):
    """
    Implementation of the vectorised
    Euler-Maruyama solver, 
    for the SDE models. 
    """  
     
    with loops.Scope() as s:
        s.beta1 = param[0]
        s.beta2 = param[1]
        s.gamma = param[2]
    
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.z = s.z.at[:,0,:].set(init_val)
        s.dw = jnp.zeros(P)
        s.subkey = key
        s.update=0.
        for i in s.range(N):
            key, s.subkey = random.split(s.subkey,2)
            s.dw = random.normal(s.subkey, shape=(P,))
            s.update=jnp.where((s.t) < 8.0, 2.2,0.5)#s.update=jnp.where((s.t) < 5.0, 0.2,1.5)

            s.z = s.z.at[:,i+1,0].set(s.update)  #s.z.at[:,i+1,0].set(s.beta1*(1 - s.beta2*jnp.sin((2.*jnp.pi*s.t)/7. ) )  )#

            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                (-s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2]) - (s.gamma*s.z[:,i,2]))*s.dt)
            s.z = s.z.at[:,i+1,3].set(s.z[:,i,3] + (s.gamma*s.z[:,i,2])*s.dt)                
            s.t = s.t + s.dt
    zar = s.z
    return zar[:,1:,:], s.t

@partial(jit, static_argnums=(4,5,6,7))
def EulerSine(key, param, init_val, t, dt, P, N, D):
    """
    Implementation of the vectorised
    Euler-Maruyama solver, 
    for the SDE models. 
    """  
     
    with loops.Scope() as s:
        s.beta1 = param[0]
        s.beta2 = param[1]
        s.gamma = param[2]
    
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.z = s.z.at[:,0,:].set(init_val)
        #s.z = s.z.at[:,0,0].set(s.beta1*(1 + s.beta2*jnp.sin(((2.*jnp.pi*s.t)/28.) + (2.*jnp.pi*0.2) ) ))
        s.dw = jnp.zeros(P)
        s.subkey = key
        s.update=0.
        for i in s.range(N):
            key, s.subkey = random.split(s.subkey,2)
            s.dw = random.normal(s.subkey, shape=(P,))
            
            s.z = s.z.at[:,i+1,0].set(s.beta1*(1 + s.beta2*jnp.sin(((2.*jnp.pi*s.t)/28.) + (2.*jnp.pi*0.2) ) ))#

            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                ((-s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2]) + s.gamma*s.z[:,i,2])*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2]) - (s.gamma*s.z[:,i,2]))*s.dt)               
            s.t = s.t + s.dt
    zar = s.z
    return zar[:,1:,:], s.t#

@partial(jit, static_argnums=(3,4,5,6,7))
def EulerFourierSis(param, init_val, t, dt, P, N, D, e):
    """
    Implementation of the vectorised
    Euler ODE solver, 
    where the ODE comes from series approx. 
    """

    flam = lambda x,y,p:p*(jnp.sqrt(2/s.Tt)*jnp.cos(((\
             ((2.0*(x+1))-1.0)*jnp.pi)/(2.0*s.Tt))*y))
    with loops.Scope() as s:
        s.beta1 = param[0]
        #s.beta2 = param[1]
        #s.beta3 = param[2]
        s.gamma = param[1]
        s.p = jnp.array(param[2:])
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.tt = jnp.zeros(N+1)
        s.z = s.z.at[:,0,:].set(init_val)

        s.expn = jnp.zeros(e)
        s.bases = jnp.arange(e)
        s.Tt = 70 # Time interval    
        for i in s.range(N):

            s.expn = flam(s.bases,s.t,s.p)
            s.z = s.z.at[:,i+1,0].set(s.z[:,i,0] + ( s.beta1*s.expn.sum() )*s.dt )
            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                ((-s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2]) + s.gamma*s.z[:,i,2])*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2]) - (s.gamma*s.z[:,i,2]))*s.dt)                  
            s.t = s.t + s.dt
            s.tt = s.tt.at[i+1].set(s.tt[i]+s.dt)
    zar = s.z
    return zar, s.tt

    
@partial(jit, static_argnums=(4,5,6,7))
def EulerSineSirs(key, param, init_val, t, dt, P, N, D):
    """
    Implementation of the vectorised
    Euler-Maruyama solver, 
    for the SDE models. 
    """  
     
    with loops.Scope() as s:
        s.beta1 = param[0]
        s.beta2 = param[1]
        s.mu = param[2]
        s.a = param[3]
        s.gamma = param[4]
        s.N = 10000.
    
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.z = s.z.at[:,0,:].set(init_val)
        #s.z = s.z.at[:,0,0].set(s.beta1*(1 + s.beta2*jnp.sin(((2.*jnp.pi*s.t)/28.) + (2.*jnp.pi*0.2) ) ))
        s.dw = jnp.zeros(P)
        s.subkey = key
        s.update=init_val[0]
        for i in s.range(N):
            key, s.subkey = random.split(s.subkey,2)
            s.dw = random.normal(s.subkey, shape=(P,))
            
            
            
            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                ((s.mu*(s.N - s.z[:,i,1])) - ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) + (s.a*s.z[:,i,4]))*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ( ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) - ((s.gamma + s.mu)*s.z[:,i,2])  )*s.dt)  
            s.z = s.z.at[:,i+1,3].set(s.z[:,i,3] + \
                ( ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) )*s.dt)    
            s.z = s.z.at[:,i+1,4].set(s.z[:,i,4] + \
                ( (s.gamma*s.z[:,i,2]) - ((s.a + s.mu)*s.z[:,i,4])  )*s.dt)  

            s.t = s.t + s.dt
            s.update = s.beta1*(1 + s.beta2*jnp.sin(((2.*jnp.pi*s.t)/365.) + (2.*jnp.pi*-0.2) ) )
            s.z = s.z.at[:,i+1,0].set(s.update)             
            
    zar = s.z
    return zar[:,1:,:], s.t#

@partial(jit, static_argnums=(3,4,5,6,7))
def EulerFourierSirs(param, init_val, t, dt, P, N, D, e):
    """
    Implementation of the vectorised
    Euler ODE solver, 
    where the ODE comes from series approx. 
    """

    flam = lambda x,y,p:p*(jnp.sqrt(2/s.Tt)*jnp.cos(((\
             ((2.0*(x+1))-1.0)*jnp.pi)/(2.0*s.Tt))*y))
    with loops.Scope() as s:
        #s.beta1 = param[0]
        s.beta2 = param[0]
        s.mu = 1/(50*365)#param[2]
        s.a = 1/param[1]
        s.gamma = 1/param[2]
        s.N = 10000.
        s.p = jnp.array(param[3:])
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.tt = jnp.zeros(N+1)
        s.z = s.z.at[:,0,:].set(init_val)

        s.expn = jnp.zeros(e)
        s.bases = jnp.arange(e)
        s.Tt = 1100 # Time interval    
        for i in s.range(N):
            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                ((s.mu*(s.N - s.z[:,i,1])) - ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) + (s.a*s.z[:,i,4]))*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ( ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) - ((s.gamma + s.mu)*s.z[:,i,2])  )*s.dt)  
            s.z = s.z.at[:,i+1,3].set(s.z[:,i,3] + \
                ( ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) )*s.dt)    
            s.z = s.z.at[:,i+1,4].set(s.z[:,i,4] + \
                ( (s.gamma*s.z[:,i,2]) - ((s.a + s.mu)*s.z[:,i,4])  )*s.dt)                  
            s.t = s.t + s.dt
            s.expn = flam(s.bases,s.t,s.p)
            s.z = s.z.at[:,i+1,0].set(s.z[:,i,0] + ( s.beta2*s.expn.sum() )*s.dt )
    zar = s.z
    return zar[0,1:,:], s.t#

@partial(jit, static_argnums=(4,5,6,7))
def EulerMaruyamaSirs(key, param, init_val, t, dt, P, N, D):
    """
    Implementation of the vectorised
    Euler-Maruyama solver, 
    for the SDE models. 
    """    
    with loops.Scope() as s:
        s.beta2 = param[0]
        s.mu = 1/(50*365)#param[2]
        s.a = 1/param[1]
        s.gamma = 1/param[2]
        s.N = 10000.
    
        s.dt = dt
        s.z = jnp.zeros((P,N+1,D))
        s.t = t
        s.z = s.z.at[:,0,:].set(init_val)
        s.dw = jnp.zeros(P)
        s.subkey = key
        for i in s.range(N):
            key, s.subkey = random.split(s.subkey,2)

            s.z = s.z.at[:,i+1,1].set(s.z[:,i,1] + \
                ((s.mu*(s.N - s.z[:,i,1])) - ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) + (s.a*s.z[:,i,4]))*s.dt)
            s.z = s.z.at[:,i+1,2].set(s.z[:,i,2] + \
                ( ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) - ((s.gamma + s.mu)*s.z[:,i,2])  )*s.dt)  
            s.z = s.z.at[:,i+1,3].set(s.z[:,i,3] + \
                ( ((s.z[:,i,0]*s.z[:,i,1]*s.z[:,i,2])/s.N) )*s.dt)    
            s.z = s.z.at[:,i+1,4].set(s.z[:,i,4] + \
                ( (s.gamma*s.z[:,i,2]) - ((s.a + s.mu)*s.z[:,i,4])  )*s.dt)                
            s.t = s.t + s.dt
            s.dw = random.normal(s.subkey, shape=(P,))
            s.z = s.z.at[:,i+1,0].set(s.z[:,i,0] + (jnp.sqrt(s.dt)*s.beta2*s.dw))            
    zar = s.z
    return zar[:,1:,:], s.t