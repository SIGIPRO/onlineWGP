import numpy as np

""" FUNCTIONS """
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

def ddtanh(x):
    return 2 * ( np.tanh(x)**3 - np.tanh(x)  )

def array_2_dic(array,keylist):  
    return dict(zip(keylist,array.squeeze()))  

def dic_2_array(dic,keylist):
    return np.array([dic[key] for key in keylist]).reshape(-1,1)

""" CLASSES """
class AdamOptimizer:
    ''' pytorch documentation: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
    '''
    def __init__(self,
                 lr,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 eps = 1e-8):
        # (handshake) placeholder
        self.warping = None
        # hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # state variables
        self.t = 0
        self.m = None
        self.v = None
    
    def step(self):
        params = dic_2_array( self.warping.parameters , self.warping.parameter_keylist )
        grad = dic_2_array( self.warping.gradient , self.warping.parameter_keylist )
        self.t = self.t + 1
        # lazy initialization (of the state moments)
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        # update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1-self.beta1) * grad
        self.v = self.beta2 * self.v + (1-self.beta2) * (grad ** 2)
        # bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        # update parameters
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.warping.parameters = array_2_dic( params, self.warping.parameters.keys() )


class WarpingFunction():
    ''' '''
    def __init__(self,
                 n=2,
                 learning_rate=1e-3,
                 seed=0,
                 jitter=1e-8):
        # (handshake) placeholder
        self.model = None
        # attributes
        self.n = n
        self.jitter = jitter
        self.number_parameters = 1 + 3*n 
        self.parameter_keylist = ['p1'] + [f'{prefix}{i}' for i in range(1,n+1) for prefix in ['p2','p3','p4']]
        ## parameter value - derivative initialization
        self._initialize(seed)
        # optimizer
        self.optimizer = AdamOptimizer(learning_rate)
        self.optimizer.warping = self
        # flag for idle behaviour (accessed from outside)
        self.idle_warping = False

    # PRIVATE
    def _initialize(self,seed):
        np.random.seed(seed)
        n, p, dp = self.n, {}, {}
        # p['p1'] = np.random.rand()
        p['p1'] = 1
        dp['p1'] = 0
        for i in range(1,n+1):
            p[f'p2{i}'] = np.random.rand()
            p[f'p3{i}'] = np.random.rand()
            p[f'p4{i}'] = np.random.randn()
            dp[f'p2{i}'] = dp[f'p3{i}'] = dp[f'p4{i}'] = 0
        self.parameters = p
        self.gradient = dp

    def _dg_dy(self,observation):
        p = self.parameters
        output = p['p1']
        for i in range(1,self.n+1):
           output = output + p[f'p2{i}'] * dtanh( p[f'p3{i}'] * (observation + p[f'p4{i}']) ) * p[f'p3{i}']
        return output

    def _d_dp_log_dg_dy(self,observation):
        p = self.parameters
        d_dp_log_dg_dy = {}
        dg_dy = self._dg_dy(observation) 
        # d_dp_log_dg_dy['p1'] = (1 / dg_dy)
        d_dp_log_dg_dy['p1'] = 0
        for i in range(1,self.n+1):
            d_dp_log_dg_dy[f'p2{i}'] = (1 / dg_dy) * dtanh( p[f'p3{i}']*(observation + p[f'p4{i}']) ) * p[f'p3{i}']
            d_dp_log_dg_dy[f'p3{i}'] = (1 / dg_dy) * ( p[f'p2{i}'] * dtanh( p[f'p3{i}']*(observation + p[f'p4{i}']) ) + p[f'p2{i}'] * p[f'p3{i}'] * ddtanh( p[f'p3{i}']*(observation + p[f'p4{i}']) ) * (observation + p[f'p4{i}']) )
            d_dp_log_dg_dy[f'p4{i}'] = (1 / dg_dy) * p[f'p2{i}'] * p[f'p3{i}'] * ddtanh( p[f'p3{i}']*(observation + p[f'p4{i}']) ) * p[f'p3{i}']
        return d_dp_log_dg_dy

    def _project_2_feasible(self):
        self.parameters['p1'] = np.maximum( self.parameters['p1'] , self.jitter )
        for i in range(1,self.n+1):
            self.parameters[f'p2{i}'] = np.maximum( self.parameters[f'p2{i}'] , self.jitter )
            self.parameters[f'p3{i}'] = np.maximum( self.parameters[f'p3{i}'] , self.jitter )

    # PUBLIC
    def transform(self,observation):
        ''' g(y;p) = p1 * r + sum^n_{i=1} p2_i * tanh( p3_i * (y + p4_i) ) '''
        if self.idle_warping:
            return observation
        p = self.parameters
        output = p['p1']*observation
        for i in range(1,self.n+1):
            output = output + p[f'p2{i}']*tanh( p[f'p3{i}']*(observation + p[f'p4{i}']) )
        return output.squeeze()
    
    def dtransform(self,observation,flag_array=True):
        ''' d_dp g(y;p) '''
        if self.idle_warping:
            return np.zeros_like(observation).reshape(-1,1)
        p = self.parameters
        dg_dp = {}
        # dg_dp['p1'] = observation
        dg_dp['p1'] = 0
        for i in range(1,self.n+1):
            dg_dp[f'p2{i}'] = tanh( p[f'p3{i}']*(observation + p[f'p4{i}']) )
            dg_dp[f'p3{i}'] = p[f'p2{i}'] * dtanh( p[f'p3{i}']*(observation + p[f'p4{i}']) ) * (observation + p[f'p4{i}'])
            dg_dp[f'p4{i}'] = p[f'p2{i}'] * dtanh( p[f'p3{i}']*(observation + p[f'p4{i}']) ) * p[f'p3{i}']
        if flag_array:
            z_dot = np.array([dg_dp[key] for key in self.parameter_keylist]).reshape(-1,1)
            return z_dot
        else:
            return dg_dp

    def update_parameters(self,observation):
        if self.idle_warping:
            pass
        else:
            # gradient
            tilde_dic_b = array_2_dic(self.model.tilde_vector_b,self.parameter_keylist)
            d_dp_log_dg_dy = self._d_dp_log_dg_dy(observation)
            for key in self.gradient.keys():
                self.gradient[key] = self.model.tilde_e * self.model.inverse_tilde_s * tilde_dic_b[key] - d_dp_log_dg_dy[key]
            # parameters
            self.optimizer.step()
            ## projection
            self._project_2_feasible()

    def inverse_transform(self, target_z, max_iter=1000, tol=1e-8):
        # TODO: clean
        ''' numerically computes y = g^-1(z) using Damped Newton-Raphson '''
        if self.idle_warping:
            return target_z
            
        # Ensure we are working with arrays for vectorized math
        is_scalar = np.isscalar(target_z)
        target_z = np.atleast_1d(target_z).astype(float)
        
        # 1. THE ASYMPTOTIC GUESS
        # For large numbers, g(y) is dominated by the linear term p1 * y.
        # If p1 was removed from the dictionary (anchored at 1), p1_val defaults to 1.0
        p1_val = self.parameters.get('p1', 1.0)
        y_hat = np.copy(target_z) / p1_val
        
        # Damping factor prevents infinite 2-cycle bouncing
        damping = 0.5
        
        for _ in range(max_iter):
            # Evaluate current guess
            z_hat = np.atleast_1d(self.transform(y_hat))
            error = z_hat - target_z
            
            # 2. ACTIVE MASKING
            # Only update the elements that haven't converged yet
            active = np.abs(error) > tol
            
            if not np.any(active):
                break # All points have converged!
                
            # Evaluate derivative only for active points
            dg_dy = np.atleast_1d(self._dg_dy(y_hat))[active]
            
            # Calculate raw step
            raw_step = error[active] / dg_dy
            
            # 3. EXPANDED CLIP WITH DAMPING
            # We allow larger steps (e.g., 10.0) so it can travel faster, 
            # but apply damping so overshoots decay into the root.
            # safe_step = damping * np.clip(raw_step, -1.0, 1.0)
            safe_step = damping * np.clip(raw_step, -0.5, 0.5)
            
            # Update only the active points
            y_hat[active] = y_hat[active] - safe_step
            
        else:
            # If the loop hits max_iter without breaking, warn the user
            print(f"Warning: inverse_transform partially failed. Max error: {np.max(np.abs(error)):.4e}")
            
        return float(y_hat[0]) if is_scalar else y_hat.squeeze()

   

