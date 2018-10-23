import numpy as np
# modelling
from lmfit import Model
from scipy import signal
import lmfit

def residuals(y_true, y_model,x, logscaled=False):
    if logscaled:
        return np.abs(y_true-y_model)*(1/(1+np.log(x)))
    else:
        return np.abs(y_true-y_model)
def RSS(y_true, y_model, x, logscaled=False):
    return np.sum(residuals(y_true, y_model,x)**2)
def AIC(N_data,N_params, y_true, y_model,x, logscaled=False):
    return N_data*np.log(RSS(y_true, y_model, x, logscaled=False)/N_data) + 2*N_params
def log_likelihood(N_data, y_true, y_model,x):
    return -(N_data/2) * np.log(RSS(y_true, y_model,x)/N_data)
def AICc(N_data, N_params, y_true, y_model,x, logscaled=False):
    return AIC(N_data, N_params, y_true, y_model,x, logscaled=False) + (2*N_params * (N_params+1))/ (N_data-N_params-1)
def delta_AIC(AICs):
    return AICs - np.min(AICs)
def relative_likelihood(delta_AIC):
    return np.exp(-.5*delta_AIC)
def Prob_model_Given_data_and_models(model_relative_likelihoods):
    """ probability of the model given data and the other models
    """
    return model_relative_likelihoods/np.sum(model_relative_likelihoods)
def evidence_ratios(prob_1, prob_2):
    return prob_1/prob_2
def r2(y_true, y_model,x):
    ss_res = RSS(y_true, y_model,x)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot


### model fitting
def powerlaw_decay(p, x):
    return p['aa'] * x**(p['pb']) + p['c_p']

def powerlaw_decay_res(arg, x,y, fit):
    if fit == 'lin': return residuals(y, powerlaw_decay(arg,x),x)
    else: return residuals(y,powerlaw_decay(arg,x),x, logscaled=True)

def exp_decay(p, x):
    return p['a'] * np.exp(-x*p['tau']) + p['c_exp']

def exp_decay_res(arg, x,y, fit):
    if fit == 'lin': return residuals(y, exp_decay(arg,x),x)
    else: return residuals(y,exp_decay(arg,x),x, logscaled=True)

def concat_decay(p, x):
    return p['aa']*(x**(p['pb'])) + p['a']*(np.exp(-x*p['tau']))

def concat_decay_res(arg,x,y, fit):
    if fit == 'lin': return residuals(y, concat_decay(arg,x),x)
    else: return residuals(y,concat_decay(arg,x),x, logscaled=True)


def fit_models(sig, distances,parameters, fit):
    """ Fit an exponential, powerlaw, and concatenative model to data
    """
    # if the fit is logarithmic, get rid of values less than or equal to zero
    if fit == 'log':
        mask = sig > 0
    p_concat, p_power, p_exp = parameters
    # power
    results_power_min = lmfit.Minimizer(powerlaw_decay_res, p_power, fcn_args = (distances,sig, fit), nan_policy='omit')
    results_power = results_power_min.minimize(method='leastsq')

    # exponential
    results_exp_min = lmfit.Minimizer(exp_decay_res, p_exp, fcn_args = (distances,sig, fit), nan_policy='omit')
    results_exp = results_exp_min.minimize(method='leastsq')

    # concatenative
    results_concat_min = lmfit.Minimizer(concat_decay_res, p_concat, fcn_args = (distances,sig, fit), nan_policy='omit')
    results_concat = results_concat_min.minimize(method='leastsq')


    return results_power, results_exp, results_concat
