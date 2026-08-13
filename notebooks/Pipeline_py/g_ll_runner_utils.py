import numpy as np
from numba import njit
import math 

import d_math as d
import e_smoothing as e

#####################################
# Init function
#####################################
    
@njit(inline='always')
def _init_user_count_table_pointer(n_users, user_interactions_nt, user_counts_nt, train_test_nt):
    ''' 
    Initialises a pointer which points to the first non train row in user_counts for each user.
    '''
    usr_frst_rw = user_interactions_nt.user_first_index.copy()

    for user_id in range(n_users):
        cnt_tbl_idx = user_interactions_nt.user_first_index[user_id]
        usr_lst_idx = user_interactions_nt.user_last_index[user_id]
        
        while cnt_tbl_idx <= usr_lst_idx and user_counts_nt.fine_bin_id[cnt_tbl_idx] < train_test_nt.burn_in_start:
            cnt_tbl_idx +=1
        usr_frst_rw[user_id] = cnt_tbl_idx
    return usr_frst_rw

#####################################
# Location helpers
#####################################

@njit
def get_time_period(fine_bin_id, validation_start, validation_end, test_start, test_end):
    ''' 
    Returns:
        0 if we are in the validation data
        1 if we are in the test data
        -1 otherwise (train + burn in and any unused data)
    '''
    if validation_start <= fine_bin_id and fine_bin_id < validation_end:
        return 0
    elif test_start <= fine_bin_id and fine_bin_id < test_end:
        return 1
    else: 
        return -1

@njit(inline='always')
def _bin_computations(bin_metric_nt, fine_bin_idx):
    ''' 
    Computes bin computations needed within the lambert liu runner
    '''
    fine_bin_pos_in_cycle = fine_bin_idx % bin_metric_nt.fine_bins_per_cycle
    crnt_coarse_bin = fine_bin_pos_in_cycle // bin_metric_nt.fine_bins_per_coarse_bin
    crnt_fine_bin_within_coarse_pos = fine_bin_pos_in_cycle % bin_metric_nt.fine_bins_per_coarse_bin

    return crnt_coarse_bin, crnt_fine_bin_within_coarse_pos

#####################################
# Getting count, params and lpmf
#####################################

@njit(inline='always')
def _get_user_count(cnt_tbl_idx, user_counts_nt, usr_end_idx, fine_bin_idx):
    ''' 
    Gets the count for that bin and moves the pointer if needed
    '''
    if cnt_tbl_idx <= usr_end_idx and user_counts_nt.fine_bin_id[cnt_tbl_idx] == fine_bin_idx:
        x = user_counts_nt.count[cnt_tbl_idx]
        cnt_tbl_idx += 1
    else:
        x = 0
    return x, cnt_tbl_idx

@njit(inline='always')
def _get_smoothed_and_unsmoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, cluster_groups, 
                                        user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                        alpha_mu_grid, alpha_sigma2_grid, alpha_p_grid, alpha_zero_grid, degen_mask, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt):
    ''' 
    Getting the smoothed and unsmoothed params mu, sigma and p
    '''
    # Getting the smoothed params
    mu_t, sigma_2_t, p_t = e.get_smoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, cluster_groups, 
                                            user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                            alpha_mu_grid, alpha_sigma2_grid, alpha_p_grid, degen_mask, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)
    

    # Getting unsmoothed but interpolated params and using that for updates (difference from above call is passing smoothing strength 0):
    mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t = e.get_smoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, cluster_groups, 
                                            user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                            alpha_zero_grid, alpha_zero_grid, alpha_zero_grid, degen_mask, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)

    return mu_t, sigma_2_t, p_t, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t


@njit(inline='always')
def _get_lpmf_and_upper_tail(x, mu, sigma2, p, calc_calibration, config_nt):

    # Get LPMF
    lpmf = d.get_lpmf_val(x, mu, sigma2, p, config_nt)

    # Get upper tail value if needed
    log_strict_upper_tail = 0
    if calc_calibration:
        log_strict_upper_tail = d.get_upper_tail_value(x + 1, mu, sigma2, p, config_nt)

    return lpmf, log_strict_upper_tail

#####################################
# Error handling
#####################################
@njit(inline='always')
def get_parameter_errors(mu, sigma2, p, hurdle_model, scoring):
    '''
    Takes the parameters we use for scoring a count
    Uses these parameter to a bitmask used for rasing errors
    '''
    # Init the error
    error = 0

    # Invalid mean error
    if not np.isfinite(mu) or mu < 0:
        error |= 2

    # Invalid var error
    if not np.isfinite(sigma2) or sigma2 < 0:
        error |= 4

    # Invalid p error
    if hurdle_model and (not np.isfinite(p) or p < 0 or p > 1):
        error |=8

    # Test and validation only errors
    if scoring:
        if mu == 0:
            error |= 2
        # Infinite mean var ario
        elif not np.isfinite(sigma2 / mu):
            error |= 16
    return error

@njit(inline='always')
def raise_parameter_errors(n_threads, thread_errors):
    '''
    Raising errors giving by that function
    '''
    combined_error = 0
    for thread_id in range(n_threads):
        combined_error |= thread_errors[thread_id]

    if combined_error & 2:
        raise ValueError('Invalid mu')

    if combined_error & 4:
        raise ValueError('Invalid var')

    if combined_error & 8:
        raise ValueError('Invalid p')

    if combined_error & 16:
        raise ValueError('Invalid mean or var ratio')

    if combined_error & 1:
        raise ValueError('prob is infinite or nan')

@njit(inline='always')
def cap_quadratic_params(mu, sigma2, p, hurdle_model):
    mu = max(mu, 2e-16)
    sigma2 = max(sigma2, 0)

    if hurdle_model:
        p = min(max(p, 0), 1)

    return mu, sigma2, p

#####################################
# End of loop updates
#####################################
@njit
def combine_threads(output_metrics, thread_output_metrics, calibration_output, thread_calibration_output, output_idx_nt, 
                    time_period_int, n_threads, config_nt):
    '''
    Runs at the end of the loop to update outputs from the threads we have
    Modifies in place
    '''

    if time_period_int >= 0:
        for thread_id in range(n_threads):
            for output_idx in range(len(output_idx_nt)):
                output_metrics[time_period_int, output_idx] += thread_output_metrics[thread_id, time_period_int, output_idx]
                thread_output_metrics[thread_id, time_period_int, output_idx] = 0

    if time_period_int == 1 and not config_nt.nll_only:
        for thread_id in range(n_threads):
            for breakdown_type_idx in range(calibration_output.shape[0]):
                for breakdown_group_idx in range(calibration_output.shape[1]):
                    for output_idx in range(calibration_output.shape[2]):
                        calibration_output[breakdown_type_idx, breakdown_group_idx, output_idx] += (
                            thread_calibration_output[thread_id, breakdown_type_idx, breakdown_group_idx, output_idx])
                        thread_calibration_output[thread_id, breakdown_type_idx, breakdown_group_idx, output_idx] = 0

@njit 
def get_new_clustering_means(cluster_groups, u, v, p):
    ''' 
    Calculates cluster centres u, v and p values for each cluster group and each time bin
    Used when using l2 or malhanobis metric
    '''

    n_users, n_coarse_bins = u.shape 
    n_clusters = cluster_groups.max() + 1

    # Init mean vectors
    cluster_mean_u = np.zeros((n_clusters, n_coarse_bins), dtype='float64')
    cluster_mean_v = np.zeros((n_clusters, n_coarse_bins), dtype='float64')
    cluster_mean_p = np.zeros((n_clusters, n_coarse_bins), dtype='float64')

    users_per_cluster = np.zeros(n_clusters, dtype='float64')

    # Summing u and v contributions in each cluster
    # Extract the cluster assignment for each user and then add their parameters to each bin
    for user_id in range(n_users):
        cluster_assignment = int(cluster_groups[user_id])

        cluster_mean_u[cluster_assignment, :] += u[user_id, :]
        cluster_mean_v[cluster_assignment, :] += v[user_id, :]
        cluster_mean_p[cluster_assignment, :] += p[user_id, :]

        users_per_cluster[cluster_assignment] += 1

    # Dividing through to get the averages in each cluster
    for cluster_assignment in range(n_clusters):
        if users_per_cluster[cluster_assignment] > 0:
            cluster_mean_u[cluster_assignment, :] /= users_per_cluster[cluster_assignment]
            cluster_mean_v[cluster_assignment, :] /= users_per_cluster[cluster_assignment]
            cluster_mean_p[cluster_assignment, :] /= users_per_cluster[cluster_assignment]

    return cluster_mean_u, cluster_mean_v, cluster_mean_p

@njit
def get_param_cluster_medians(cluster_groups, param_grid):
    '''
    Gets cluster medians to use for smoothing
    '''
    n_users, n_coarse_bins = param_grid.shape
    n_clusters = cluster_groups.max() + 1
    users_per_cluster = np.zeros(n_clusters, dtype='int64')

    for user_id in range(n_users):
        users_per_cluster[cluster_groups[user_id]] += 1

    cluster_medians = np.zeros((n_clusters, n_coarse_bins), dtype='float64',)

    for cluster_id in range(n_clusters):
        for coarse_bin in range(n_coarse_bins):
            values = np.zeros(users_per_cluster[cluster_id], dtype='float64')
            value_idx = 0

            for user_id in range(n_users):
                if int(cluster_groups[user_id]) == cluster_id:
                    values[value_idx] = param_grid[user_id, coarse_bin]
                    value_idx += 1

            cluster_medians[cluster_id, coarse_bin] = np.median(values)

    return cluster_medians

@njit(inline='always')
def get_new_cluster_centres(cluster_groups, u, v, p, distance_metric):
    '''
    Returns either the cluster mean or the cluster median depending on the distance metric this run is for
    '''
    if distance_metric=='l1':
        return get_param_cluster_medians(cluster_groups, u), get_param_cluster_medians(cluster_groups, v), get_param_cluster_medians(cluster_groups, p)
    else:
        return get_new_clustering_means(cluster_groups, u, v, p)

#####################################
# Detection
#####################################

@njit
def init_detection_outputs(n_users, n_test_fbs, n_attack_sizes, n_attack_fbs):
    '''
    Inits arrays needed for the attack experiment
    '''

    observed_p_vals = np.full((n_users, n_test_fbs), np.nan, dtype='float64')
    attack_p_vals = np.full((n_users, n_attack_sizes, n_attack_fbs), np.nan, dtype='float64')

    # inputs are x, mu sigma2, and p for the attack bin
    attack_bin_inputs = np.full((n_users, n_attack_fbs, 4), np.nan, dtype='float64')

    return observed_p_vals, attack_p_vals, attack_bin_inputs

@njit(inline='always')
def store_attack_bin_inputs(attack_bin_inputs, fine_bin, attack_start_fb, x, mu, sigma2, p):
    '''
    Stores the attack bin inputs in an array
    '''
    # Compute postition relative to attack start if position is less than 12 we store
    attack_fb = fine_bin - attack_start_fb
    if 0 <= attack_fb < 12:
        attack_bin_inputs[attack_fb, 0] = x
        attack_bin_inputs[attack_fb, 1] = mu
        attack_bin_inputs[attack_fb, 2] = sigma2
        attack_bin_inputs[attack_fb, 3] = p

@njit
def get_attack_p_values(attack_p_vals, attack_bin_inputs, attack_sizes, config_nt):
    '''
    Updates the observed p values for the day with the attack p values
    '''

    n_attack_sizes = attack_sizes.shape[0]
    n_attack_fbs = attack_bin_inputs.shape[0]

    # All non attack bins are changed for that day
    for attack_size_idx in range(n_attack_sizes):
        added_counts_per_fb = attack_sizes[attack_size_idx] // n_attack_fbs

        for attack_fb in range(n_attack_fbs):
            attack_count = attack_bin_inputs[attack_fb, 0] + added_counts_per_fb
            lpmf, log_strict_upper_tail = _get_lpmf_and_upper_tail(attack_count, attack_bin_inputs[attack_fb, 1], 
                                                                   attack_bin_inputs[attack_fb, 2], attack_bin_inputs[attack_fb, 3], True, config_nt)
            log_p = d.get_randomised_log_upper_tail(log_strict_upper_tail, lpmf)
            attack_p_vals[attack_size_idx, attack_fb] = math.exp(log_p)