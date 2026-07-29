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
    Computes coarse bin and fine bin within the week and coarse bin
    '''
    fine_bin_pos_in_day = fine_bin_idx % bin_metric_nt.fine_bins_per_day
    crnt_coarse_bin = fine_bin_pos_in_day // bin_metric_nt.fine_bins_per_coarse_bin
    crnt_fine_bin_within_coarse_pos = fine_bin_pos_in_day % bin_metric_nt.fine_bins_per_coarse_bin

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
                                        alpha_mu_grid, alpha_sigma2_grid, alpha_p_grid, alpha_zero_grid, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt):
    ''' 
    Getting the values of mu and sigma for both the raw and smoothed model
    '''
    # Getting the smoothed params and capping them at the minimal value
    mu_t, sigma_2_t, p_t = e.get_smoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, cluster_groups, 
                                            user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                            alpha_mu_grid, alpha_sigma2_grid, alpha_p_grid, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)
    

    # Getting unsmoothed but interpolated params and using that for updates (difference from above call is passing smoothing strength 0):
    mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t = e.get_smoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, cluster_groups, 
                                            user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                            alpha_zero_grid, alpha_zero_grid, alpha_zero_grid, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)
    
    # Capping values
    mu_t = max(mu_t, config_nt.mean_min)
    sigma_2_t = max(sigma_2_t, config_nt.var_min)
    mu_unsmth_t = max(mu_unsmth_t, config_nt.mean_min)
    sigma_unsmth_2_t = max(sigma_unsmth_2_t, config_nt.var_min)
    p_t = min(max(p_t, config_nt.p_min), config_nt.p_max)
    p_unsmth_t = min(max(p_unsmth_t, config_nt.p_min), config_nt.p_max)

    return mu_t, sigma_2_t, p_t, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t


@njit(inline='always')
def _get_log_p0_lpmf_and_upper_tail(x, mu_t, sigma_2_t, p_t, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, 
                                    calculate_calibration, config_nt):
    '''
    Returns log p0, the lpmf, and strict upper tail values for the smoothed and raw models
    '''
    # Get LPMF
    lpmf_smoothed = d.get_lpmf_val(x, mu_t, sigma_2_t, p_t, config_nt)
    lpmf_raw = d.get_lpmf_val(x, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, config_nt)

    if not calculate_calibration:
        return lpmf_raw, lpmf_smoothed, 0, 0
    else:
        log_strict_upper_tail_smoothed = d.get_upper_tail_value(x + 1, mu_t, sigma_2_t, p_t, config_nt)
        return lpmf_raw, lpmf_smoothed, 0, log_strict_upper_tail_smoothed

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
    Calculates mean u and v values for each cluster group and each time bin
    Args:
        cluster_groups a n_users length vector of cluster assignments
        u : the calculated vector of u parameter means
        v : the calculated vector of v parameter variances
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

