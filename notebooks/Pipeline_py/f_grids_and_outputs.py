import numpy as np 
from numba import njit
import math 
import d_math as d


#####################################
# Alpha Grid update
#####################################

def init_alpha_grid(n_counts_init, constant_alpha, smooth_a, smooth_t, fb_per_cb):
    '''
    Init a grid of alpha values that are used in smoothing
    '''
    # Return a constant grid if we are using constant shrinkage else make a count dependent alpha
    if constant_alpha:
        return np.full_like(n_counts_init, smooth_a, dtype='float64')
    else:
        return fb_per_cb * smooth_t / ((1 - smooth_t) * n_counts_init + fb_per_cb * smooth_t)
    
@njit
def get_alpha_val(n_counts, constant_alpha, smooth_a, smooth_t, fb_per_cb):
    '''
    Gets a value of alpha fron n_counts similarly to `init_alpha_grid`
    '''
    if constant_alpha:
        return smooth_a
    else:
        return fb_per_cb * smooth_t / ((1 - smooth_t) * n_counts + fb_per_cb * smooth_t)

@njit
def update_n_counts_and_alpha_grids(n_counts, alpha_mu_grid, alpha_sigma2_grid, crnt_user_id, usr_updt_n_counts, w, fb_per_cb, config_nt):
    '''
    Updates alpha grids used for msoothing and observed counts
    '''
    n_coarse_bins = n_counts.shape[1]

    for coarse_bin in range(n_coarse_bins):
        n_counts[crnt_user_id, coarse_bin] = (1 - w) * n_counts[crnt_user_id, coarse_bin] + w * usr_updt_n_counts[coarse_bin]
        current_n = n_counts[crnt_user_id, coarse_bin]

        alpha_mu_grid[crnt_user_id, coarse_bin] = get_alpha_val(current_n, config_nt.constant_alpha, config_nt.smooth_a_mu, config_nt.smooth_t_mu, fb_per_cb)
        alpha_sigma2_grid[crnt_user_id, coarse_bin] = get_alpha_val(current_n, config_nt.constant_alpha, config_nt.smooth_a_sigma2, config_nt.smooth_t_sigma2, fb_per_cb)

#####################################
# Collecting and updating u, v, p grids
#####################################

@njit
def collect_temp_grid(usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_n_cnts, crnt_coarse_bin, x, mu_t, sigma_2_t, p_t, w, config_nt):
    '''
    As data comes in we update the interpolated mu values by combining with incoming data as per lambert and liu formula
    Also has an update version for hurdle model using ll formula + new formula for p
    '''
    # updating the number of pos observations
    if x > 0:
        usr_updt_n_cnts[crnt_coarse_bin] += 1
    
    # Update logic for the hurdle model
    if config_nt.hurdle_model:
        # Init a binary variable that detects x >0 
        if x > 0:
            z= 1
        else:
            z = 0

        # Updating p value and sum of p vals in the bin   
        p_new = (1 - w) * p_t + w * z
        usr_updt_p_sum[crnt_coarse_bin] += p_new

        # Update other values if x > 0
        if x > 0:
            mu_new = (1 - w) * mu_t + w * (x - 1)
            sigma2_new = (1 - w) * sigma_2_t + w * (x - 1 - mu_t) * (x - 1 - mu_new)

            usr_updt_u_sum[crnt_coarse_bin] += mu_new
            usr_updt_v_sum[crnt_coarse_bin] += sigma2_new

    # Update logic for the NB model
    else: 
        mu_new = (1-w)*mu_t + w*x
        sigma2_new = (1-w)*sigma_2_t + w*(x-mu_t)*(x-mu_new)

        usr_updt_u_sum[crnt_coarse_bin] += mu_new
        usr_updt_v_sum[crnt_coarse_bin] += sigma2_new

@njit
def update_grid(u, v, p, crnt_user_id, usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_n_cnts, fine_bins_per_coarse_bin, config_nt):
    ''' 
    Replaces the grid values using the temporary grid as data comes in
    '''
    # Update hurdle model grid
    if config_nt.hurdle_model:
        n_coarse_bins = u.shape[1]

        for coarse_bin in range(n_coarse_bins):
            p_new = usr_updt_p_sum[coarse_bin] / fine_bins_per_coarse_bin
            p[crnt_user_id, coarse_bin] = p_new

            # Only update grid if we have a positive count in that coaraes bin
            if usr_updt_n_cnts[coarse_bin] > 0:
                u[crnt_user_id, coarse_bin] = usr_updt_u_sum[coarse_bin] / usr_updt_n_cnts[coarse_bin]
                v[crnt_user_id, coarse_bin] = usr_updt_v_sum[coarse_bin] / usr_updt_n_cnts[coarse_bin]
                
    # Update the grid for the NB model 
    else:
        u[crnt_user_id, : ] = usr_updt_u_sum/fine_bins_per_coarse_bin
        v[crnt_user_id, : ] = usr_updt_v_sum/fine_bins_per_coarse_bin

    
#####################################
# Updating output metrics
#####################################

@njit
def update_outputs(time_period_int, output_metrics, output_idx_nt, lpmf):
    '''
    Accumulates likelihood sum and number of scored bins overall
    '''
    if time_period_int == 0 or time_period_int == 1:
        output_metrics[time_period_int, output_idx_nt.n_bins_scored] += 1
        output_metrics[time_period_int, output_idx_nt.non_degen_ll_sum] += lpmf

@njit(inline='always')
def update_user_outputs(user_id, user_output_metrics, lpmf):
    '''
    Accumulates likelihood sum and number of scored bins for a single user
    '''
    user_output_metrics[user_id, 0] += 1
    user_output_metrics[user_id, 1] += lpmf

@njit
def update_calibration_outputs(user_id, lpmf, capped_lpmf, log_strict_upper_tail, log_calibration_thresholds, 
                               breakdown_groups, calibration_output):
    '''
    Updates calibration outputs with one observation
    Uses randomisation to smooth discrete calibration cutoffs
    '''
    rdm_upper_tail = d.get_randomised_log_upper_tail(log_strict_upper_tail, lpmf)

    for breakdown_type_idx in range(breakdown_groups.shape[0]):
        breakdown_group_idx = int(breakdown_groups[breakdown_type_idx, user_id])
        calibration_output[breakdown_type_idx, breakdown_group_idx, 0] += 1
        calibration_output[breakdown_type_idx, breakdown_group_idx, 1] += capped_lpmf

        for threshold_idx in range(log_calibration_thresholds.shape[0]):
            if rdm_upper_tail < log_calibration_thresholds[threshold_idx]:
                calibration_output[breakdown_type_idx, breakdown_group_idx, 2 + threshold_idx,] += 1

    return rdm_upper_tail