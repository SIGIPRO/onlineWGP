import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'
import argparse
import json
import pickle
import os

from model import WarpedGaussianProcess,GaussianKernelFunction
from warping import WarpingFunction

""" AUXILIARY """
def save_experiment(model,settings,expid):
    ''' saves the GP model and its training settings '''
    path = os.path.join('storage',expid)
    os.makedirs(path, exist_ok=True)
    # save settings as human-readable JSON
    settings_file = os.path.join(path, 'settings.json')
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
    # save the trained model object
    model_file = os.path.join(path, 'model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

def load_experiment(expid):
    ''' loads the GP model '''
    path = os.path.join('storage',expid)    
    # model    
    model_file = os.path.join(path, 'model.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    # settings
    settings_file = os.path.join(path, 'settings.json')
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    return model,settings


""" EXPERIMENTS """
# original
def experiment00():
    ''' True distribution + samples '''
    N = 101
    sigma = 1/3
    x = np.linspace(-np.pi,np.pi,N)
    z = np.sin(x) + np.random.normal(0,sigma,N)
    y = np.cbrt(z)
    z_base = np.sin(x)
    y_median = np.cbrt( z_base )
    y_up = np.cbrt( z_base + 1.96*sigma )
    y_down = np.cbrt( z_base - 1.96*sigma )
    plt.plot(x,y_median,':k')
    plt.plot(x,y_up,':k')
    plt.plot(x,y_down,':k')
    plt.plot(x,y,'.k')
    plt.show()

# GP
def experiment10():
    ''' recursive GP '''
    # settings
    conf = {
        "NUM_REALIZATIONS": 1000,
        "NUM_OBSERVATIONS": 101,
        "true_sigma": 1/3,
        "scale_kw": 2,
        "ka": 2,
        "NUM_WARPING_TERMS": 10,
        "lr": 0.005,
        "idle_warping": True,
        "ald_threshold": 0.1}
    x = np.linspace(-np.pi,np.pi,conf['NUM_OBSERVATIONS'])
    conf['x0'] = x[0]
    # GP model
    ## kernel
    kw = conf['scale_kw'] * (2*np.pi / conf['NUM_OBSERVATIONS'])
    conf['kw'] = kw
    ka = conf['ka']
    kernel_function = GaussianKernelFunction(kernel_width=kw,kernel_amplitude=ka)
    ## noise 
    model_sigma = 0.1
    conf['model_sigma'] = model_sigma
    ## warping 
    warping = WarpingFunction(n=conf['NUM_WARPING_TERMS'],learning_rate=conf['lr'])
    warping.idle_warping = conf['idle_warping']
    ## initialization
    model = WarpedGaussianProcess(warping,kernel_function,model_sigma,conf['ald_threshold'],conf['x0'])
    # training
    for _ in range(conf['NUM_REALIZATIONS']):
        for xi in x:
            yi = np.cbrt( np.sin(xi) + np.random.normal(0,conf['true_sigma']) )
            model.forward(xi,yi)
    ## dictionary size
    conf['dictionary_size'] = len(model.dictionary)
    # saving
    save_experiment(model,conf,'experiment10')

def experiment11():
    ''' recursive GP visualization '''
    # loading the model
    model,conf = load_experiment('experiment10')
    # training data
    x = np.linspace(-np.pi,np.pi,conf['NUM_OBSERVATIONS'])
    y = []
    for _ in range(conf['NUM_REALIZATIONS']):
        for xi in x:
            yi = np.cbrt( np.sin(xi) + np.random.normal(0,conf['true_sigma']) )
            y.append(yi)
    y = np.array(y)
    # evaluation
    NUM_EVAL = 401
    x_grid = np.linspace(-np.pi,np.pi,NUM_EVAL)
    f_median = []
    f_var = []
    for xi in x_grid:
        f_median.append( model.compute_mean(xi) )
        f_var.append( model.compute_variance(xi) )
    f_median = np.array(f_median) 
    f_std = np.sqrt( np.array(f_var) + conf['model_sigma']**2 )
    # plots
    plt.plot(np.tile(x, conf['NUM_REALIZATIONS']), y, marker='.', color='gray', linestyle='none', alpha=0.05)
    plt.plot(x_grid,f_median,'--k')
    plt.plot(x_grid,f_median-1.96*f_std,'--k')
    plt.plot(x_grid,f_median+1.96*f_std,'--k')
    plt.show()

# warped GP
def experiment20():
    ''' recursive warped GP '''
    # settings
    conf = {
        "NUM_REALIZATIONS": 1000,
        "NUM_OBSERVATIONS": 101,
        "true_sigma": 1/3,
        "scale_kw": 2,
        "ka": 2,
        "NUM_WARPING_TERMS": 10,
        "lr": 0.005,
        "idle_warping": False,
        "ald_threshold": 0.1}
    x = np.linspace(-np.pi,np.pi,conf['NUM_OBSERVATIONS'])
    conf['x0'] = x[0]
    # GP model
    ## kernel
    kw = conf['scale_kw'] * (2*np.pi / conf['NUM_OBSERVATIONS'])
    conf['kw'] = kw
    ka = conf['ka']
    kernel_function = GaussianKernelFunction(kernel_width=kw,kernel_amplitude=ka)
    ## noise 
    model_sigma = 3
    conf['model_sigma'] = model_sigma
    ## warping 
    warping = WarpingFunction(n=conf['NUM_WARPING_TERMS'],learning_rate=conf['lr'])
    warping.idle_warping = conf['idle_warping']
    ## initialization
    model = WarpedGaussianProcess(warping,kernel_function,model_sigma,conf['ald_threshold'],conf['x0'])
    # training
    for _ in range(conf['NUM_REALIZATIONS']):
        for xi in x:
            yi = np.cbrt( np.sin(xi) + np.random.normal(0,conf['true_sigma']) )
            model.forward(xi,yi)
    ## dictionary size
    conf['dictionary_size'] = len(model.dictionary)
    # saving
    save_experiment(model,conf,'experiment20')

def experiment21():
    ''' recursive warped GP visualization '''
    # loading the model
    model,conf = load_experiment('experiment20')
    # training data
    x = np.linspace(-np.pi,np.pi,conf['NUM_OBSERVATIONS'])
    y = []
    for _ in range(conf['NUM_REALIZATIONS']):
        for xi in x:
            yi = np.cbrt( np.sin(xi) + np.random.normal(0,conf['true_sigma']) )
            y.append(yi)
    y = np.array(y)
    # evaluation
    NUM_EVAL = 401
    x_grid = np.linspace(-np.pi,np.pi,NUM_EVAL)
    ## (in) latent
    f_median_latent = []
    f_var_latent = []
    for xi in x_grid:
        f_median_latent.append( model.compute_mean(xi) )
        f_var_latent.append( model.compute_variance(xi) )
    f_median_latent = np.array(f_median_latent) 
    f_std_latent = np.sqrt( np.array(f_var_latent) + conf['model_sigma']**2 )
    f_up_latent = f_median_latent + 1.96*f_std_latent
    f_down_latent = f_median_latent - 1.96*f_std_latent
    ## (in) observation
    f_median_obs = model.warping.inverse_transform( f_median_latent )
    f_up_obs = model.warping.inverse_transform( f_up_latent ) 
    f_down_obs = model.warping.inverse_transform( f_down_latent ) 
    # plots
    plt.plot(np.tile(x, conf['NUM_REALIZATIONS']), y, marker='.', color='gray', linestyle='none', alpha=0.05)
    plt.plot(x_grid, f_median_obs, '-k')
    plt.plot(x_grid, f_down_obs, '-k')
    plt.plot(x_grid, f_up_obs, '-k')
    plt.show()

# (pre)paper
def experiment30():
    ''' process plot '''
    model, conf = load_experiment('experiment20')
    # data
    x = np.linspace(-np.pi,np.pi,conf['NUM_OBSERVATIONS'])
    y = []
    for _ in range(conf['NUM_REALIZATIONS']):
        for xi in x:
            yi = np.cbrt( np.sin(xi) + np.random.normal(0,conf['true_sigma']) )
            y.append(yi)
    y = np.array(y)
    # evaluation
    plt.figure(figsize=(10, 5))
    NUM_EVAL = 401
    x_grid = np.linspace(-np.pi,np.pi,NUM_EVAL)
    ## data
    plt.plot(np.tile(x, conf['NUM_REALIZATIONS']), y, marker='.', color='lightgray', linestyle='none', alpha=0.1)
    ## original
    zor = np.sin(x_grid)
    yor_median = np.cbrt(zor)
    yor_up = np.cbrt(zor + 1.96*conf['true_sigma'])
    yor_down = np.cbrt(zor - 1.96*conf['true_sigma'])
    plt.plot(x_grid,yor_median,':k',label=r'$p(y)$')
    plt.plot(x_grid,yor_up,':k')
    plt.plot(x_grid,yor_down,':k')
    ## warped GP
    f_med = []
    f_var = []
    for xi in x_grid:
        f_med.append( model.compute_mean(xi) )
        f_var.append( model.compute_variance(xi) )
    f_med = np.array( f_med )
    f_var = np.array( f_var )
    f_std = np.sqrt( np.array(f_var) + conf['model_sigma']**2 )
    ### (in) observation
    f_obs = model.warping.inverse_transform( f_med )
    f_up = model.warping.inverse_transform( f_med + 1.96*f_std )
    f_down = model.warping.inverse_transform( f_med - 1.96*f_std )
    plt.plot(x_grid,f_obs,'k',label=r'$p(y|\boldsymbol{\theta})$')
    plt.plot(x_grid,f_up,'k')
    plt.plot(x_grid,f_down,'k')
    ## axis
    FONTSIZE = 30
    xtick_locs = [-np.pi, -np.pi/2, -np.pi/12, np.pi/2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{12}$', r'$\frac{\pi}{2}$', r'$\pi$']
    plt.xticks(xtick_locs,xtick_labels,fontsize=FONTSIZE)
    ytick_locs = [-1,0,1]
    ytick_labels = ['$-1$','$0$','$1$']
    plt.yticks(ytick_locs,ytick_labels,fontsize=FONTSIZE-4)
    ## slice
    plt.axvline(x=-np.pi/12, color='gray', linestyle='--')

    plt.legend(fontsize=FONTSIZE,loc='lower right')
    plt.tight_layout()
    plt.show()

def experiment31():
    ''' distribution slice plot '''
    model, conf = load_experiment('experiment20')

    y_grid = np.linspace(-1.5, 1.5, 500)
    # true generating distribution (z = y^3)
    x_eval = - np.pi / 12
    true_mu = np.sin(x_eval)
    true_sigma = conf['true_sigma'] 
    z_true = y_grid**3
    dg_dy_true = 3 * y_grid**2
    true_pdf = (1.0 / (true_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((z_true - true_mu) / true_sigma)**2)
    true_pdf = true_pdf * np.abs(dg_dy_true)
    # learned distribution
    learned_mu = model.compute_mean(true_mu)
    learned_sigma = np.sqrt( model.compute_variance(true_mu) + conf['model_sigma']**2 )
    z = model.warping.transform(y_grid)
    dg_dy = model.warping._dg_dy(y_grid)
    learned_pdf = (1.0 / (learned_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((z - learned_mu) / learned_sigma)**2)
    learned_pdf = learned_pdf * np.abs(dg_dy)
    # plot(s)
    plt.figure(figsize=(7,6))
    plt.plot(y_grid,true_pdf,':k',label=r'$p(y|x=-\frac{\pi}{12})$')
    plt.plot(y_grid, learned_pdf, '-k',label=r'$p(y|\boldsymbol{\theta},x=-\frac{\pi}{12})$')

    FONTSIZE = 30
    xtick_locs = [-1,0,1]
    xtick_labels = ['$-1$','$0$','$1$']
    plt.xticks(xtick_locs,xtick_labels,fontsize=FONTSIZE-4)
    plt.yticks([])
    plt.legend(fontsize=FONTSIZE,loc='upper right')
    plt.tight_layout()
    plt.show()

# paper
def experiment40():
    ''' learned warping transformation - forward and inverse '''
    # retrieving a trained warping transformation
    model,_ = load_experiment('experiment20')
    warping = model.warping
    # data
    y = np.linspace(-1.5,1.5,1000)
    # warping: forward - inverse
    z = warping.transform(y)
    y_hat = warping.inverse_transform(z)

    # plot(s)
    FONTSIZE = 27
    fig = plt.figure(figsize=(10,8))
    ## grid specs
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ## top left plot (row 0 - column 0)
    plt.subplot(gs[0, 0])
    plt.plot(y, z, 'k', label=r'$z=g(y;\boldsymbol{\theta})$')
    plt.plot(y, y**3, ':k', label=r'$y^3$')
    plt.legend(fontsize=FONTSIZE,loc='upper left')
    plt.xlabel('$y$', fontsize=FONTSIZE+2)
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xticks([-1,0,1],fontsize=FONTSIZE)
    plt.yticks([0],fontsize=FONTSIZE)
    ## top right plot (row 0 - column 1)
    plt.subplot(gs[0, 1])
    plt.plot(z, y_hat, 'k', label=r'$\hat{y} = g^{-1}(z;\boldsymbol{\theta})$')
    plt.plot(z, np.cbrt(z), ':k', label=r'$z^{\frac{1}{3}}$')
    plt.legend(fontsize=FONTSIZE,loc='upper left')
    plt.xlabel('$z$', fontsize=FONTSIZE+2)
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xticks([-20,0,20],fontsize=FONTSIZE)
    # plt.xticks([])
    plt.yticks([0],fontsize=FONTSIZE)
    ## bottom full-width plot (row 1 - spanning all columns)
    plt.subplot(gs[1, :])
    error = np.abs(y - y_hat)
    plt.plot(y, error, 'k', label=r'$|y-\hat{y}|$')
    plt.legend(fontsize=FONTSIZE,loc='upper left')
    plt.xlabel("$y$", fontsize=FONTSIZE+2)
    plt.yscale('log')
    plt.yticks(fontsize=FONTSIZE-2)
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xticks([-1.5,0,1.5],fontsize=FONTSIZE)
    
    plt.tight_layout()
    plt.show()

def experiment41():
    ''' combined process and distribution slice plot '''
    # retrieving trained warped GP
    model, conf = load_experiment('experiment20')
    # figure
    FONTSIZE = 35
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1])
    # left plot
    plt.subplot(gs[0])
    ## data
    x = np.linspace(-np.pi, np.pi, conf['NUM_OBSERVATIONS'])
    y = []
    for _ in range(conf['NUM_REALIZATIONS']):
        for xi in x:
            yi = np.cbrt( np.sin(xi) + np.random.normal(0, conf['true_sigma']) )
            y.append(yi)
    y = np.array(y)
    plt.plot(np.tile(x, conf['NUM_REALIZATIONS']), y, marker='.', color='lightgray', linestyle='none', alpha=0.1,zorder=0, rasterized=True)
    ## evaluation
    NUM_EVAL = 401
    x_grid = np.linspace(-np.pi, np.pi, NUM_EVAL)
    ### original
    zor = np.sin(x_grid)
    yor_median = np.cbrt(zor)
    yor_up = np.cbrt(zor + 1.96*conf['true_sigma'])
    yor_down = np.cbrt(zor - 1.96*conf['true_sigma'])
    plt.plot(x_grid, yor_median, ':k', label=r'$p(\boldsymbol{y})$')
    plt.plot(x_grid, yor_up, ':k')
    plt.plot(x_grid, yor_down, ':k')
    ### warped GP
    f_med = []
    f_var = []
    for xi in x_grid:
        f_med.append( model.compute_mean(xi) )
        f_var.append( model.compute_variance(xi) )
    f_med = np.array( f_med )
    f_var = np.array( f_var )
    f_std = np.sqrt( np.array(f_var) + conf['model_sigma']**2 )
    #### (in) observation
    f_obs = model.warping.inverse_transform( f_med )
    f_up = model.warping.inverse_transform( f_med + 1.96*f_std )
    f_down = model.warping.inverse_transform( f_med - 1.96*f_std )
    plt.plot(x_grid, f_obs, 'k')
    plt.plot(x_grid, f_up, 'k')
    plt.plot(x_grid, f_down, 'k')
    ### axis
    xtick_locs = [-np.pi, -np.pi/2, -np.pi/12, np.pi/2, np.pi]
    xtick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{12}$', r'$\frac{\pi}{2}$', r'$\pi$']
    plt.xticks(xtick_locs, xtick_labels, fontsize=FONTSIZE)
    ytick_locs = [-1, 0, 1]
    ytick_labels = ['$-1$', '$0$', '$1$']
    plt.yticks(ytick_locs, ytick_labels, fontsize=FONTSIZE-4)
    ## slice
    plt.axvline(x=-np.pi/12, color='gray', linestyle=(0, (5, 5))) # linestyle = (offset, (dash_length, gap_length))
    plt.subplot(gs[1])
    y_grid = np.linspace(-1.5, 1.5, 500)
    eval_x = -np.pi/12
    ### true generating distribution (z = y^3)
    true_mean_z = np.sin(eval_x) 
    true_sigma = conf['true_sigma'] 
    z_true = y_grid**3
    dg_dy_true = 3 * y_grid**2
    true_pdf = (1.0 / (true_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((z_true - true_mean_z) / true_sigma)**2)
    true_pdf = true_pdf * np.abs(dg_dy_true)
    ### learned distribution
    learned_mu = model.compute_mean(eval_x)
    learned_sigma = np.sqrt( model.compute_variance(eval_x) + conf['model_sigma']**2 )
    z = model.warping.transform(y_grid)
    dg_dy = model.warping._dg_dy(y_grid)
    learned_pdf = (1.0 / (learned_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((z - learned_mu) / learned_sigma)**2)
    learned_pdf = learned_pdf * np.abs(dg_dy)
    #### plot(s)
    plt.plot(y_grid, true_pdf, ':k')
    plt.plot(y_grid, learned_pdf, '-k')
    xtick_locs = [-1, 0, 1]
    xtick_labels = ['$-1$', '$0$', '$1$']
    plt.xticks(xtick_locs, xtick_labels, fontsize=FONTSIZE-4)
    plt.yticks([0],['$0$'],fontsize=FONTSIZE-4)
    plt.ylim([0,1.95])    
    ##### modify all 4 borders (spines)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('gray')
        # spine.set_linestyle('--')
        spine.set_linestyle((0, (5, 5)))
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.show()


""" MAIN """
def main(args):
    # REPRODUCIBILITY
    np.random.seed(args.seed)
    # EXPERIMENT
    globals()['experiment' + args.id]()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-id',type=str,help='id number of the experiment (from 00 to 99)')
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    main(args)

