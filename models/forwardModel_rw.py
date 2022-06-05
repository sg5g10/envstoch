import numpy as np
import scipy
import scipy.stats as stats
import scipy.signal as signal
from .COVID_CPP import seeiir_ode
class ForwardModelRegionWise(object):
    """
    Implementation of the covid19 model.
    This contains code for evaluating mainly the forward 
    epidemic model. Computationally intensive
    parts are coded in C++ and wrapped uisng
    pybind11. 

    ## This code evaluates the epidemic model
    with the random-walk
    """        
    def __init__(self, dt, time, A, pop_size_N, mixing_matrix, \
                        seventh_breakpoint, lockdn_breakpoint, 
                        breakpoints_list, dayindex, \
                        MaxConv_dead = None, dL = None):
        self._dt = dt
        self._time = time
        self._T = len(time)
        self._N = pop_size_N
        self._A = A
        self._regions = 1
        if dL == None:
            self._dL = 3.0
        else:
            self._dL = dL
        if MaxConv_dead == None:
            self._MaxConv_dead = int(200)
        else:
            self._MaxConv_dead = MaxConv_dead

        self._pi = np.ones(self._A)
        self._seventh_breakpoint = seventh_breakpoint        
        self._lockdn_breakpoint = lockdn_breakpoint
        self._breakpoints_list = breakpoints_list
        self._dayindex = dayindex
    
        def conv_cdfs(dt, MaxConv_dead):
            mean = (4.0 + 15.0)
            sd = np.sqrt(1.41**2 + 12.1**2)
            alpha = mean**2/sd**2
            beta = mean/sd**2 
            delay_rv_dead = stats.gamma(a=alpha, scale=1/beta)
            cdfs_dead = np.zeros(MaxConv_dead)
            for n in range(MaxConv_dead-1):
                cdfs_dead[n] = (delay_rv_dead.cdf((n+1)*dt)-delay_rv_dead.cdf((n)*dt))
            cdfs_dead[MaxConv_dead-1] = 1-delay_rv_dead.cdf((MaxConv_dead-1)*dt)
            return cdfs_dead

        self.cdf_dead = conv_cdfs(self._dt, self._MaxConv_dead)
        self._mixing_matrix = mixing_matrix
        self._real_data = False

    def real_data(self):
        self._real_data = False  
          
    def init_transmission(self, dI, psir, ngen_M0, I0):
        dt = self._dt
        A = self._A
        N = self._N
        dL = self._dL
        region = self._regions
        pi = self._pi
        
        sigma = 2./dL
        gamma = 2./dI
        
        r0_init = psir*dI*((psir*dL/2 + 1)**2)/ (1- (1/(psir*dI/2 + 1)**2))
        alpha = np.exp(psir*dt) - 1
        p = 0.1
        I_init = (dI * np.exp(I0) * N.sum())/(p*r0_init) # later parameterise p
        eig, eigvec =  np.linalg.eig(ngen_M0)
        dom_eigvec = eigvec[:,np.argsort(abs(eig))[-1]] ## Find dominant eigenvector of nextgen matrix
        i_tot_a = I_init*(dom_eigvec/dom_eigvec.sum()) ## Normalise
        

        i1_init = np.zeros(A)
        i2_init = np.zeros(A)
        e1_init = np.zeros(A)
        e2_init = np.zeros(A)
        r_init = np.zeros(A)
        s_init = np.zeros(A)
        i1_init = i_tot_a*(1 + (gamma*dt/(alpha + gamma*dt)))**-1
        i2_init = i_tot_a - i1_init
        e2_init = i1_init*(alpha + gamma*dt)/(sigma*dt)
        e1_init = e2_init*(alpha + sigma*dt)/(sigma*dt)
        r_init = (1 - pi)*N
        s_init = pi*N
        
        return i1_init, i2_init, e1_init, e2_init, r_init, s_init, r0_init   

    def transmission(self, dI, psir, mix_mat, m_mul, I0, beta, tlock, sampleR = False):
        sigma = 2./self._dL
        gamma = 2./dI
        dt = self._dt
        dL = self._dL
        A = self._A
        region = self._regions
        T = self._T
        N = self._N
        pi = self._pi
    

        m_prelock, m_postlock = np.ones((A,A)), np.ones((A,A))
        m_prelock[A-1,:]=np.exp(m_mul[0])
        

        m_postlock[:A-1,:]= np.exp(m_mul[1]) 

        m_postlock[A-1,:]= np.exp(m_mul[2])
        
        init_mix_mat = mix_mat[:,:,0]*m_prelock    
        nxtgen_mix_mat = np.array([init_mix_mat[rw,:]*N[rw]*dI for rw in range(A)])
        
        r0star = np.sort(abs(np.linalg.eigvals(nxtgen_mix_mat)))[-1]   
        
        s = np.zeros((T,A))
        r = np.zeros((T,A))
        lamda = np.zeros((T,A))
        i1 = np.zeros((T,A))
        i2 = np.zeros((T,A))
        e1 = np.zeros((T,A))
        e2 = np.zeros((T,A))
        

        i1[0,:], i2[0,:], e1[0,:], e2[0,:], r[0,:], s[0,:], r0_init \
        = self.init_transmission(dI, psir, nxtgen_mix_mat, I0)
        Tlock = tlock
        

        beta = beta.reshape((-1,))
        seeiir_ode.SEEIIR_ODE(beta, lamda, i1, i2, e1, e2, r, s, \
            mix_mat, m_prelock, m_postlock, r0_init, r0star, \
                Tlock, gamma, sigma, dt, T, A)

        if sampleR:
            Rt = np.zeros((T,1))
            Rt[0,:] = r0_init
            for i in range(1,T):                
                if i < Tlock+1:
                    scaled_Lambda = s[i,:]*(mix_mat[:,:,i]*m_prelock)*dI
                    r_star_t = np.sort(abs(np.linalg.eigvals(scaled_Lambda)))[-1]
                    Rt[i] = r0_init*(r_star_t/r0star)
                else:
                    scaled_Lambda = s[i,:]*(mix_mat[:,:,i]*m_postlock)*dI
                    r_star_t = np.sort(abs(np.linalg.eigvals(scaled_Lambda)))[-1]
                    Rt[i] = np.exp(beta[i])*r0_init*(r_star_t/r0star)                                           
            return Rt

        return s, lamda

    def convolution(self, S, lamda, p_dead):
        T = self._T
        A = self._A
        region = self._regions
        cdf_dead = self.cdf_dead
            
        MaxConv_dead = self._MaxConv_dead
        delta_infec = S*lamda
        shifted = np.roll(delta_infec,1,axis=0) ### Shift in time dimension 
        shifted[0,:] = shifted[1,:]     ### Copy \Delta (t=0) = \Delta(t=1)
        NNI = shifted
        delta = NNI
        con_len = len(cdf_dead)-1
        conv_series = np.zeros((T,A))
        seeiir_ode.convolution_2D(delta, conv_series, cdf_dead, con_len)
        p = np.zeros(8)
        p[0] = p_dead[0]
        p[1:] = p_dead
        for a in range(A):
            conv_series[:,a] = conv_series[:,a]*p[a]     
        return conv_series, NNI

    def daily_betas(self, beta):
        dt = self._dt
        seventh_breakpoint = self._seventh_breakpoint 
        breakpoints_list = self._breakpoints_list 
        region = 1 
        dayindex = self._dayindex 

        grid_space = int(1/dt)            
        betasplit = beta[:,None]
        beta_mat = np.zeros((seventh_breakpoint,region))
        rw_series = []
        for i in range(2, betasplit.shape[0]+1):
            beta_mat = np.vstack((beta_mat,np.tile(np.sum(betasplit[:i,:],axis=0),\
                (grid_space*7,1)))) 
            rw_series.append(np.sum(betasplit[:i,:],axis=0))
        beta_mat = np.vstack((beta_mat,np.tile(np.sum(betasplit,axis=0),\
            (((breakpoints_list[2]+1) - (breakpoints_list[1])),1)))) 
        assert len(dayindex) == beta_mat.shape[0]
        return beta_mat, np.array(rw_series)

    def simulate(self, _dI, _psir, _beta_rw, _I0, _p_dead, _m_mul, generate=False): 
        
        ### Solve the SEEIIR ODE using Euler regionwise
        _dI = _dI + 2.0
        daily_beta_rw, rw_series = self.daily_betas(_beta_rw)
        daily_mix_matrix = self._mixing_matrix

        s, lamda = self.transmission(_dI, _psir, daily_mix_matrix, _m_mul, 
                                _I0, daily_beta_rw, self._lockdn_breakpoint)            
        expected_deaths, nnis = self.convolution(s, lamda, _p_dead)
        expected_deaths = expected_deaths.reshape((s.shape[0],self._A))  
        if generate:
            Rt = self.transmission(_dI, _psir, daily_mix_matrix, _m_mul, 
                                _I0, daily_beta_rw, self._lockdn_breakpoint, True)
            return expected_deaths, nnis, Rt
        else:
            return expected_deaths, s



