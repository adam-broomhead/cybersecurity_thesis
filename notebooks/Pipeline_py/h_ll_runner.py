from numba import njit 
import numpy as np 

@njit
def run_lambert_liu(u_init, v_init, p_init, cluster_u_init, cluster_v_init, cluster_p_init, cluster_groups, n_counts_init, 
                    alpha_grid_init, degen_mask, user_counts_nt, user_interactions_nt, interpolation_weights,
                    train_test_nt, bin_metric_nt, config_nt,
                    output_idx_nt, model_idx_nt):
    ''' 
    Runs the lambert liu algorithm
    Args:
        u_init + v_init : inital parameter grids
        cluster_u_init + cluster_v_init : inital cluster parameters
        cluster groups : inital cluster assignments 1 row per user id
        smooth_a : parameter (experiment to vary)
        user_counts_nt: is a named tuple version of user_counts has columns user_id, fine_bin_id, count
        user_interactions_nt: is a named tuple verion of user_interactions has columns user id and first and last interaction index in user_counts
        interpolation_weights : precalculated weights for parameter interpolation
        train_test_nt, bin_metric_nt, config_dt : named tuple versions of dicts train_test_dict, config_dict and bin_metric_dict
        output_idx_nt : a named tuple containing the names and indicies of the outputs we want to store
        model_idx_nt : a named tuple containing the names and indicies where we store each model outputs
    '''
    ###
    # Initialising grids where we will keep track of parameters and cluster mean parameters
    u = u_init.copy()
    v = v_init.copy()
    p = p_init.copy()

    cluster_u = cluster_u_init.copy()
    cluster_v = cluster_v_init.copy()
    cluster_p = cluster_p_init.copy()

    n_counts = n_counts_init.copy()
    alpha_grid = alpha_grid_init.copy()
    zero_alpha_grid = np.zeros_like(alpha_grid)


    # Calculating needed values
    n_users, n_coarse_bins = u.shape

    log_calibration_thresholds = np.log(config_nt.calibration_thresholds)

    burn_in_first_week = train_test_nt.burn_in_start // bin_metric_nt.fine_bins_per_week
    test_last_week = (train_test_nt.test_end-1)// bin_metric_nt.fine_bins_per_week

    # Init outputs
    output_metrics = np.zeros((2, len(output_idx_nt)), dtype='float64')
    calibration_output = np.zeros((2, log_calibration_thresholds.shape[0], len(model_idx_nt)), dtype='float64')

    # Init pointer for user interactions
    usr_frst_rw = _init_user_count_table_pointer(n_users, user_interactions_nt, user_counts_nt, train_test_nt)

    for week in range(burn_in_first_week, test_last_week + 1):
        
        week_start = week * bin_metric_nt.fine_bins_per_week
        week_end = (week + 1) * bin_metric_nt.fine_bins_per_week

        if week_end > train_test_nt.test_end:
            week_end = train_test_nt.test_end

        # Getting grid totals for seperating rate from shape
        user_u_totals = get_grid_row_sums(u)
        user_v_totals = get_grid_row_sums(v)
        user_p_totals = get_grid_row_sums(p)

        cluster_u_totals = get_grid_row_sums(cluster_u)
        cluster_v_totals = get_grid_row_sums(cluster_v)
        cluster_p_totals = get_grid_row_sums(cluster_p)

        # For each week iterate over the users and init the pointers
        for user_id in range(n_users):

            cnt_tbl_idx = usr_frst_rw[user_id]
            usr_end_idx = user_interactions_nt.user_last_index[user_id]

            # Init numpy vectors for calculating the user sums
            usr_updt_u_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            usr_updt_v_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            usr_updt_p_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            
            usr_updt_pos_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            usr_updt_cnt_sum = np.zeros(n_coarse_bins, dtype=np.float64)

            for fine_bin in range(week_start, week_end):
                
                x, cnt_tbl_idx = _get_user_count(cnt_tbl_idx, user_counts_nt, usr_end_idx, fine_bin)
                
                crnt_coarse_bin, crnt_fine_bin_within_coarse_pos = _bin_computations(bin_metric_nt, fine_bin)

                mu_t, sigma_2_t, p_t, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t = _get_smoothed_and_unsmoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, 
                                                                                    cluster_groups, user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                                                                    alpha_grid, zero_alpha_grid, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)

                time_period_int = get_time_period(fine_bin, train_test_nt.validation_start, train_test_nt.validation_end, 
                                                                train_test_nt.test_start, train_test_nt.test_end)
                
                if (time_period_int == 0 or time_period_int == 1) and not degen_mask[user_id, crnt_coarse_bin]: 
                    lpmf_raw, lpmf_smoothed, log_upper_tail_raw, log_upper_tail_smoothed = _get_log_p0_lpmf_and_upper_tail(x, mu_t, sigma_2_t, p_t, 
                                                                                                                            mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, config_nt)
                    # Updating outputs
                    update_outputs(time_period_int, output_metrics, calibration_output, output_idx_nt, model_idx_nt, log_calibration_thresholds, log_upper_tail_raw, log_upper_tail_smoothed, lpmf_raw, lpmf_smoothed)

                collect_temp_grid(usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_pos_sum, usr_updt_cnt_sum, crnt_coarse_bin, x, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, config_nt)
                            
            # Updating the users first row (for the next week) and the parameter grid and alpha grid
            usr_frst_rw[user_id] = cnt_tbl_idx
            update_grid(u, v, p, user_id, usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_pos_sum, bin_metric_nt.fine_bins_per_coarse_bin, config_nt)
            update_n_counts_and_alpha_grid(n_counts, alpha_grid, user_id, usr_updt_cnt_sum, config_nt)

        cluster_u, cluster_v, cluster_p = get_new_clustering_means(cluster_groups, u, v, p)

    return output_metrics, calibration_output, u, v, p, cluster_u, cluster_v, cluster_p, n_counts, alpha_grid