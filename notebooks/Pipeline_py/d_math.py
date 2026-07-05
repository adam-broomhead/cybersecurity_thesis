from numba import njit
import math 

#####################################
# Math helper functions
#####################################

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

@njit(inline="always")
def log1minexp(log_p):
    """
    Stable log(1 - exp(log_p))
    """

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
    if mu <= 0:
        raise ValueError('Mu < 0')
    if sigma2 <= 0:
        raise ValueError('Sigma^2 < 0')
    
    mu = max(mu, config_nt.mean_min)
    sigma2 = max(sigma2, config_nt.var_min)
    if sigma2 <= mu + config_nt.min_mean_var_diff:
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
        return math.log(p) + get_nb_lpmf_val(x, mu, sigma2, config_nt) - log1minexp(get_nb_lpmf_val(0, mu, sigma2, config_nt))


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
def poisson_log_upper_tail(x, mu):
    ''' 
    Gets p(X>= x) for a poisson distribution 
    '''
    if x == 0:
        return 0
    
    # Looping over k and getting prob x = k and adding to lower tail
    log_prob_k = -mu
    lower_tail = log_prob_k
    for k in range(1,x):
        log_prob_k = log_prob_k + math.log(mu) - math.log(k)
        lower_tail = logsumexp2(lower_tail, log_prob_k)

    return log1minexp(lower_tail)

@njit 
def neg_bin_log_upper_tail(x, mu, sigma2):
    ''' 
    Gets p(X>= x) for a negative binomial distribution 
    '''
    if x == 0:
        return 0
    
    p = mu/sigma2
    r = (mu*p) / (1-p)

    # Looping over k and getting prob x = k and adding to lower tail
    log_prob_k = r * math.log(p)
    log_lower_tail = log_prob_k
    for k in range(1,x):
        log_prob_k = log_prob_k+ math.log((k-1)+r) - math.log(k) + math.log(1-p)
        log_lower_tail =  logsumexp2(log_lower_tail, log_prob_k)

    return log1minexp(log_lower_tail)

@njit 
def get_nb_upper_tail_value(x, mu, sigma2, config_nt):
    ''' 
    We dont need edge case checks here as the other function is called with the same mu and sigma
    '''
    mu = max(mu, config_nt.mean_min)
    sigma2 = max(sigma2, config_nt.var_min)
    if sigma2 <= mu + config_nt.min_mean_var_diff:
        return poisson_log_upper_tail(x, mu)
    else: 
        return neg_bin_log_upper_tail(x, mu, sigma2)

@njit 
def hurdle_upper_tail(x, mu, sigma2, p, config_nt):
    if x == 0:
        return 0
    else:
        return math.log(p) + get_nb_upper_tail_value(x, mu, sigma2, config_nt) - log1minexp(get_nb_lpmf_val(0, mu, sigma2, config_nt))
    
@njit(inline='always')
def get_upper_tail_value(x, mu, sigma2, p, config_nt):
    if config_nt.hurdle_model:
        return hurdle_upper_tail(x, mu, sigma2, p, config_nt)
    else:
        return get_nb_upper_tail_value(x, mu, sigma2, config_nt)