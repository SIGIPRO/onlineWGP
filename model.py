import numpy as np

""" CLASSES """
class GaussianKernelFunction():
    def __init__(self,
                 kernel_width = 1,
                 kernel_amplitude = 1):
        self.kernel_width = kernel_width
        self.kernel_amplitude = kernel_amplitude
    def __call__(self,x1,x2):
        x1 = np.array(x1).reshape(-1)
        x2 = np.array(x2).reshape(-1)
        return self.kernel_amplitude * np.exp( - (x1-x2).T @ (x1-x2) / (2 * self.kernel_width**2) )

# WGP
class WarpedGaussianProcess():
    def __init__(self,
                 warping,
                 kernel_function,
                 noise_std,
                 ald_threshold,
                 x0): # initial input location
        self.tol = 1e-8
        # WARPING FUNCTION
        self.warping = warping
        ## handshake
        self.warping.model = self
        # KERNEL FUNCTION
        self.kernel_function = kernel_function
        # NOISE FUNCTION
        self.s2 = noise_std**2
        # INSTANTIATION
        self.nu = ald_threshold
        self.inverse_tilde_matrix_K = np.array( 1 / kernel_function(x0,x0) ).reshape(-1,1)
        self.dictionary = [x0] 
        self.a = np.ones(1).reshape(-1,1)
        self.tilde_vector_alpha = np.zeros(1).reshape(-1,1)
        self.tilde_matrix_C = np.zeros(1).reshape(-1,1)
        self.tilde_matrix_B = np.zeros((self.warping.number_parameters,1))

    # PUBLIC
    def forward(self,input_location,observation):
        x = input_location
        y = observation
        ## tilde_vector_k's
        tilde_vector_k = self.compute_tilde_vector_k(x)
        ## hat_a (solution to the ALD problem)
        hat_a = self.inverse_tilde_matrix_K @ tilde_vector_k
        ## delta (conditional variance of the ALD reconstruction from a Bayessian perspective)
        delta = self.kernel_function(x,x) - tilde_vector_k.T @ hat_a 
        delta = max(delta.squeeze(),self.tol)
        ## warping the observation
        z = self.warping.transform(y)
        z_dot = self.warping.dtransform(y) 
        ## tilde_e
        tilde_e = z - tilde_vector_k.T @ self.tilde_vector_alpha
        self.tilde_e = tilde_e.squeeze()
        ## tilde_vector_b
        self.tilde_vector_b = z_dot - self.tilde_matrix_B @ tilde_vector_k 
        ## -* new element into the dictionary *-
        if delta > self.nu:
            self.dictionary.append( x )
            self._update_inverse_tilde_matrix_K(delta,hat_a)
            tilde_vector_c = np.vstack(( - self.tilde_matrix_C @ tilde_vector_k , np.array([1]) ))
            tilde_s = self.kernel_function(x,x) + self.s2 - tilde_vector_k.T @ self.tilde_matrix_C @ tilde_vector_k
            ### reshaping
            self.tilde_vector_alpha = np.vstack(( self.tilde_vector_alpha , np.array([0]) ))
            zero_column_B = np.zeros(self.tilde_matrix_B.shape[0]).reshape(-1,1)
            self.tilde_matrix_B = np.hstack(( self.tilde_matrix_B, zero_column_B ))
            zero_column_C = np.zeros(self.tilde_matrix_C.shape[0]).reshape(-1,1)
            self.tilde_matrix_C = np.block( [ [self.tilde_matrix_C , zero_column_C] , [zero_column_C.T , np.array([[0]]) ] ] )
        ## -* the dictionary remains the same *-
        else:
            tilde_vector_c = hat_a - self.tilde_matrix_C @ tilde_vector_k
            tilde_s = tilde_vector_k.T @ hat_a + self.s2 - tilde_vector_k.T @ self.tilde_matrix_C @ tilde_vector_k
        ## tilde_vector_alpha, tilde_matrix_C, tilde_matrix_B
        self.inverse_tilde_s = 1 / tilde_s
        self.tilde_vector_alpha = self.tilde_vector_alpha + self.tilde_e * tilde_vector_c * self.inverse_tilde_s
        self.tilde_matrix_C = self.tilde_matrix_C + tilde_vector_c @ tilde_vector_c.T * self.inverse_tilde_s
        self.tilde_matrix_B = self.tilde_matrix_B + self.tilde_vector_b @ tilde_vector_c.T * self.inverse_tilde_s
        ## warping parameters
        self.warping.update_parameters(y)

    def compute_tilde_vector_k(self,x):
        return np.array([ self.kernel_function(xi,x) for xi in self.dictionary ]).reshape(-1,1)
    
    ## MEAN, VARIANCE
    def compute_mean(self,x):
        tilde_k = self.compute_tilde_vector_k(x)
        return_mean = tilde_k.T @ self.tilde_vector_alpha
        return return_mean.squeeze()
    
    def compute_variance(self,x):
        tilde_k = self.compute_tilde_vector_k(x)
        return_var = self.kernel_function(x,x) - tilde_k.T @ self.tilde_matrix_C @ tilde_k
        return return_var.squeeze()

    def _update_inverse_tilde_matrix_K(self,delta,hat_a):
        self.inverse_tilde_matrix_K = (1/delta) * np.block( [ [delta * self.inverse_tilde_matrix_K + hat_a @ hat_a.T , -hat_a] , [-hat_a.T , np.array([[1]])] ] )
        ## enforcing symmetry to prevent compounding precision errors
        self.inverse_tilde_matrix_K = 0.5 * (self.inverse_tilde_matrix_K + self.inverse_tilde_matrix_K.T)
