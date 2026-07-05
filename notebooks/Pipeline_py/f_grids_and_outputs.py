import numpy as np 
from numba import njit
import math 


#####################################
# Alpha Grid update
#####################################

def init_alpha_grid(n_counts_init, config_dict):
    ''' 
    Init a grid of alpha values that are used in smoothing
    '''
    # Return a constant grid if we are using linear smoothing else make a count dependent alpha
    if config_dict['linear_smooth']:
        return np.full_like(n_counts_init, config_dict['smooth_a'], dtype='float64')
    else: 
        return config_dict['smooth_k'] / (np.log1p(n_counts_init) + config_dict['smooth_k'])
    
@njit
def update_alpha_grid(n_counts, config_nt):
    ''' 
    Gets a value of alpha fron n_counts similarly to `init_alpha_grid`
    '''
    if config_nt.linear_smooth:
        return config_nt.smooth_a
    else: 
        return config_nt.smooth_k / (math.log1p(n_counts) + config_nt.smooth_k)

@njit
def update_n_counts_and_alpha_grid(n_counts, alpha_grid, crnt_user_id, usr_updt_n_counts, config_nt):
    ''' 
    Runs at the end of every week in the ll runner
    '''
    n_coarse_bins = n_counts.shape[1]

    # Iterating over the grid adding the weekly sums and updating the grid
    for coarse_bin in range(n_coarse_bins):
        n_counts[crnt_user_id, coarse_bin] += usr_updt_n_counts[coarse_bin]

        alpha_grid[crnt_user_id, coarse_bin] = update_alpha_grid(n_counts[crnt_user_id, coarse_bin], config_nt)

#####################################
# Collecting and updating u, v, p grids
#####################################

@njit
def collect_temp_grid(usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_pos_sum, usr_updt_cnt_sum, crnt_coarse_bin, x, mu_t, sigma_2_t, p_t, config_nt):
    '''
    As data comes in we update the interpolated mu values by combining with incoming data as per lambert and liu formula
    returns nothing as we modify in place. Also has an update version for hurdle model using ll formula + new formula for p
    '''
    # updating the count sum
    usr_updt_cnt_sum[crnt_coarse_bin] += x
    
    # Update logic for the hurdle model
    if config_nt.hurdle_model:
        # Init a binary variable that detects x >0 
        if x > 0:
            z= 1
        else:
            z = 0

        # Updating p value and sum of p vals in the bin   
        p_new = (1 - config_nt.w) * p_t + config_nt.w * z
        p_new = min(max(p_new, config_nt.p_min), config_nt.p_max)
        usr_updt_p_sum[crnt_coarse_bin] += p_new

        # Update other values if x > 0
        if x > 0:
            mu_new = (1-config_nt.w)*mu_t + config_nt.w*x
            sigma2_new = (1-config_nt.w)*sigma_2_t + config_nt.w*(x-mu_t)*(x-mu_new)

            mu_new = max(mu_new, config_nt.mean_min)
            sigma2_new = max(sigma2_new, config_nt.var_min)

            usr_updt_u_sum[crnt_coarse_bin] += mu_new
            usr_updt_v_sum[crnt_coarse_bin] += sigma2_new
            usr_updt_pos_sum[crnt_coarse_bin] += 1

    # Update logic for the NB model
    else: 
        mu_new = (1-config_nt.w)*mu_t + config_nt.w*x
        sigma2_new = (1-config_nt.w)*sigma_2_t + config_nt.w*(x-mu_t)*(x-mu_new)

        mu_new = max(mu_new, config_nt.mean_min)
        sigma2_new = max(sigma2_new, config_nt.var_min)

        usr_updt_u_sum[crnt_coarse_bin] += mu_new
        usr_updt_v_sum[crnt_coarse_bin] += sigma2_new

@njit
def update_grid(u, v, p, crnt_user_id, usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_pos_sum, fine_bins_per_coarse_bin, config_nt):
    ''' 
    Replaces the grid values using the temporary grid as data comes in
    '''
    # Update hurdle model grid
    if config_nt.hurdle_model:
        n_coarse_bins = u.shape[1]

        for coarse_bin in range(n_coarse_bins):
            p_new = usr_updt_p_sum[coarse_bin] / fine_bins_per_coarse_bin
            p[crnt_user_id, coarse_bin] = min(max(p_new, config_nt.p_min), config_nt.p_max)

            # Only update grid if we have a positive count in that coaraes bin
            if usr_updt_pos_sum[coarse_bin] > 0:
                u[crnt_user_id, coarse_bin] = max(usr_updt_u_sum[coarse_bin] / usr_updt_pos_sum[coarse_bin], config_nt.mean_min)
                v[crnt_user_id, coarse_bin] = max(usr_updt_v_sum[coarse_bin] / usr_updt_pos_sum[coarse_bin], config_nt.var_min)

    # Update the grid for the NB model 
    else:
        u[crnt_user_id, : ] = usr_updt_u_sum/fine_bins_per_coarse_bin
        v[crnt_user_id, : ] = usr_updt_v_sum/fine_bins_per_coarse_bin

    
#####################################
# Updating output metrics
#####################################

@njit
def update_outputs(time_period_int, output_metrics, calibration_output, output_idx_nt, model_idx_nt,
                  log_calibration_thresholds, log_upper_tail_raw, log_upper_tail_smoothed, lpmf_raw, lpmf_smoothed):
    ''' 
    Function used for updating the output and calibration threshold output
    Args:
        x : the count observed
        time_period_int : number that states whether we are in test train or validation
        output_metrics : Np array where we will store our outputs
        calibration_output : output of how many p values fall below each threshold in train and validation for each model
        output_idx_nt : named tuple that tells us which index of `output_metrics` each metric lives in
        model_idx_nt :  named tuple that tells us which index of `calibration_output` each model lives in

        log_raw_degen_threshold : threshold that tells us whether we are in a degenerate bin or not (defined by P(X=0))
        log_calibration_thresholds : calibration thresholds we are monitoring (how many p values fall below each threshold)

        log_p0_raw  + smoothed: the log prob of 0 is compared to the degen threshold in the function
        log_upper_tail_raw + smooth : the log_probability of observing a value greater that or equal to x. compared to the log calibration thresholds

        lpmf_raw + smoothed : the log likelihood values we observe
    '''

    # If we are in train or validation update the metrics
    if time_period_int == 0 or time_period_int == 1:
        output_metrics[time_period_int, output_idx_nt.n_bins_scored] += 1
        output_metrics[time_period_int, output_idx_nt.non_degen_ll_sum] += lpmf_raw
        output_metrics[time_period_int, output_idx_nt.non_degen_smoothed_ll_sum] += lpmf_smoothed

        # Add a point for each calibration threshold we are less than 
        # 1 row of output for raw model one for smoothed model
        for calib_threshold_idx in range(log_calibration_thresholds.shape[0]):

            if log_upper_tail_raw < log_calibration_thresholds[calib_threshold_idx]:
                calibration_output[time_period_int, calib_threshold_idx, model_idx_nt.raw_model_calib_index] += 1

            if log_upper_tail_smoothed < log_calibration_thresholds[calib_threshold_idx]:
                calibration_output[time_period_int, calib_threshold_idx, model_idx_nt.smoothed_model_calib_index] += 1