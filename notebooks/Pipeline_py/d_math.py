from numba import njit
import math 
import numba_scipy.special 
from scipy.special import betainc, gammainc
import numpy as np

#####################################
# Math helper functions
#####################################
@njit(inline='always')
def safe_log(prob, config_nt):
    if not math.isfinite(prob) or prob < 0 or prob > 1:
        return np.nan
    return prob

### Helper functions to stop overflow
@njit(inline='always')
def logsumexp2(a, b):
    ''' 
    Two term logsumexp helper
    '''
    if a == b:
        return a + math.log(2.0)
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

log_0_5 = -math.log(2.0)

@njit(inline='always')
def log1minexp(log_p):
    '''
    Stable log(1 - exp(log_p))
    '''

    if log_p < log_0_5:
        return math.log1p(-math.exp(log_p))
    else:
        return math.log(-math.expm1(log_p))

#####################################
# LPMF functions
#####################################
   
@njit 
def poisson_lpmf(x, mu):
    ''' 
    Poisson log pmf
    '''
    return x*math.log(mu) - mu - math.lgamma(x+1)

@njit
def neg_bin_lpmf(x, mu, sigma2):
    p = mu/sigma2
    r = (mu*p) / (1-p)
    return math.lgamma(x+r) - math.lgamma(r) - math.lgamma(x+1) + r*math.log(p) + x*math.log(1-p)

@njit 
def get_nb_lpmf_val(x, mu, sigma2, config_nt):    
    if sigma2 / mu <= config_nt.min_mean_var_ratio:
        return poisson_lpmf(x, mu)
    else: 
        return neg_bin_lpmf(x, mu, sigma2)

@njit 
def hurdle_lpmf(x, mu, sigma2, p, config_nt):
    '''
    Note the poisson fallback is handled as we use the nb functions with poisson integrated
    '''
    if x == 0:
        return math.log1p(-p)
    else:
        return math.log(p) + get_nb_lpmf_val(x - 1, mu, sigma2, config_nt)
    

@njit(inline='always')
def get_lpmf_val(x, mu, sigma2, p, config_nt):
    ''' 
    Gets the LPMF value for the count
    '''
    if config_nt.hurdle_model:
        return hurdle_lpmf(x, mu, sigma2, p, config_nt)
    else:
        return get_nb_lpmf_val(x, mu, sigma2, config_nt)
    
#####################################
# Calibration tail values
#####################################

@njit 
def poisson_log_upper_tail(x, mu, config_nt):
    ''' 
    Gets p(X>= x) for a poisson distribution 
    '''
    if x == 0:
        return 0
    else:
        upper_tail = gammainc(float(x), mu)
        return safe_log(upper_tail, config_nt)

@njit 
def neg_bin_log_upper_tail(x, mu, sigma2, config_nt):
    ''' 
    Gets p(X>= x) for a negative binomial distribution 
    '''
    if x == 0:
        return 0
    
    p = mu/sigma2
    r = (mu*p) / (1-p)

    upper_tail = betainc(float(x), r, 1 - p)
    return safe_log(upper_tail, config_nt)

@njit 
def get_nb_upper_tail_value(x, mu, sigma2, config_nt):
    ''' 
    Gets the NB upper tail
    Note:
    We dont need edge case checks here as the other function is called with the same mu and sigma
    '''
    if sigma2/mu <= config_nt.min_mean_var_ratio:
        return poisson_log_upper_tail(x, mu, config_nt)
    else: 
        return neg_bin_log_upper_tail(x, mu, sigma2, config_nt)

@njit 
def hurdle_upper_tail(x, mu, sigma2, p, config_nt):
    ''' 
    Gets the upper tail value for the hurdle model
    '''

    # Special case x = 0
    if x == 0:
        return 0

    return safe_log(p, config_nt) + get_nb_upper_tail_value(x - 1, mu, sigma2, config_nt)
    
@njit(inline='always')
def get_upper_tail_value(x, mu, sigma2, p, config_nt):
    if config_nt.hurdle_model:
        return hurdle_upper_tail(x, mu, sigma2, p, config_nt)
    else:
        return get_nb_upper_tail_value(x, mu, sigma2, config_nt)

@njit(inline='always')
def get_randomised_log_upper_tail(strict_log_upper_tail, lpmf):
    '''
    Gets the randomised upper tail
    '''
    random_value = np.random.random()

    if random_value == 0:
        return strict_log_upper_tail
    else:
        return logsumexp2(strict_log_upper_tail, math.log(random_value) + lpmf)