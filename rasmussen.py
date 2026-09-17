''' ------------------
Example adapted from the book: Gaussian processes for machine learning by Carl Rasmussen (Ch. 2, Fig. 2.2)
------------------ '''
import numpy as np
import matplotlib.pyplot as plt

# classic LaTeX font (computer modern)
plt.rcParams['mathtext.fontset'] = 'cm'
# squared exponential Kernel
def squared_exponential_kernel(X1, X2, length_scale=1.0, variance=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return variance * np.exp(-0.5 / length_scale**2 * sqdist)
# input domain
n_test = 200
X_test = np.linspace(-5, 5, n_test).reshape(-1, 1)
# random seed for reproducibility 
np.random.seed(42)
# jitter for numerical stability 
jitter = 1e-8

# PRIOR
## covariance matrix of the test points
K_ss = squared_exponential_kernel(X_test, X_test)
## mean and standard deviation of the prior
mu_prior = np.zeros(n_test)
std_prior = np.sqrt(np.diag(K_ss))
## draw 3 sample functions from the GP prior
L_prior = np.linalg.cholesky(K_ss + jitter * np.eye(n_test))
f_prior_samples = np.dot(L_prior, np.random.normal(size=(n_test, 3)))

# POSTERIOR
## noise-free training data (replicating the points from the book)
X_train = np.array([-4, -3, -1, 0, 2]).reshape(-1, 1)
y_train = np.array([-2, 0, 1, 2, -1]).reshape(-1, 1)
## covariance matrices for the posterior
K = squared_exponential_kernel(X_train, X_train)
K_s = squared_exponential_kernel(X_train, X_test)
## Cholesky decomposition of the training covariance matrix
L = np.linalg.cholesky(K + jitter * np.eye(len(X_train)))
## posterior mean: mu_post = K_s.T * K^-1 * y
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
mu_post = np.dot(K_s.T, alpha).flatten()
## posterior covariance: K_ss - K_s.T * K^-1 * K_s
v = np.linalg.solve(L, K_s)
sigma2 = 0.005
Cov_post = K_ss - np.dot(v.T, v) + sigma2
std_post = np.sqrt(np.diag(Cov_post))
## draw 3 sample functions from the GP posterior
L_post = np.linalg.cholesky(Cov_post + jitter * np.eye(n_test))
f_post_samples = mu_post.reshape(-1, 1) + np.dot(L_post, np.random.normal(size=(n_test, 3)))

# PLOTS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
## prior
### 95% confidence interval (1.96 standard deviations)
ax1.fill_between(X_test.flatten(), mu_prior - 1.96*std_prior, mu_prior + 1.96*std_prior, 
                 color='lightgray',alpha=0.6,label='95% confidence interval')
### prior samples
ax1.plot(X_test, f_prior_samples, color='gray', linewidth=1)
ax1.set_xlim([-5, 5])
ax1.set_ylim([-3, 3])

## posterior
### 95% confidence interval
ax2.fill_between(X_test.flatten(), mu_post - 1.96*std_post, mu_post + 1.96*std_post, 
                 color='lightgray',alpha=0.6)
### posterior samples
ax2.plot(X_test, f_post_samples, color='gray', linewidth=1)

## training data
ax2.plot(X_train, y_train, 'k+', markersize=10, markeredgewidth=1.5, label='Observations')
ax2.set_xlim([-5, 5])
ax2.set_ylim([-3, 3])
x_locs = X_train.flatten()
x_labels = [rf'$x_{{{i+1}}}$' for i in range(len(x_locs))]

## custom x-axis ticks and labels to BOTH plots
FONTSIZE = 22
ax1.set_xticks(x_locs)
ax1.set_xticklabels([])
ax2.set_xticks(x_locs)
ax2.set_xticklabels(x_labels)
ax2.tick_params(axis='x',labelsize=FONTSIZE+7)

## custom y-axis ticks and labels to the LEFT plot
y_locs = y_labels = [0]
ax1.set_yticks(y_locs)
ax1.tick_params(axis='y',labelsize=FONTSIZE)

## same y-axis ticks to the RIGHT plot, but REMOVING the labels
ax2.set_yticks(y_locs)
ax2.tick_params(axis='y', labelsize=FONTSIZE)

## text labels on the right side of the plots
ax1.text(1.02, 0.5, r'$f(x)$', transform=ax1.transAxes, 
         fontsize=FONTSIZE+4, verticalalignment='center', horizontalalignment='left')
ax2.text(1.02, 0.5, r'$f(x)|\boldsymbol{y}_5$', transform=ax2.transAxes, 
         fontsize=FONTSIZE+4, verticalalignment='center', horizontalalignment='left')

plt.tight_layout()
plt.show()