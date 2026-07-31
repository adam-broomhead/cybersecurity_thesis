from numba import njit, prange, get_num_threads, get_thread_id
import numpy as np 
import g_ll_runner_utils as g
import e_smoothing as e
import f_grids_and_outputs as f

@njit(parallel=True)
def run_lambert_liu(u_init, v_init, p_init, cluster_u_init, cluster_v_init, cluster_p_init, cluster_groups, 
    n_counts_init, alpha_mu_grid_init, alpha_sigma2_grid_init, alpha_p_grid_init, degen_mask, user_counts_nt, user_interactions_nt, 
    interpolation_weights, train_test_nt, bin_metric_nt, config_nt, output_idx_nt, model_idx_nt, breakdown_groups):
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

    # Init n counts and alpha grids
    n_counts = n_counts_init.copy()
    alpha_mu_grid = alpha_mu_grid_init.copy()
    alpha_sigma2_grid = alpha_sigma2_grid_init.copy()
    alpha_p_grid = alpha_p_grid_init.copy()
    zero_alpha_grid = np.zeros_like(alpha_mu_grid)

    # Calculating needed values
    n_users, n_coarse_bins = u.shape

    if config_nt.nll_only:
        log_calibration_thresholds = np.empty(0, dtype='float64')
        calibration_output = np.empty((0, 0, 0), dtype='float64')

    else:
        log_calibration_thresholds = np.log(config_nt.calibration_thresholds)
        n_groups = int(breakdown_groups.max()) + 1
        n_breakdowns = breakdown_groups.shape[0]

        # n_bins, non_degen_ll and calibration threshold counts
        n_calibration_outputs = 2 + log_calibration_thresholds.shape[0]
        calibration_output = np.zeros((n_breakdowns, n_groups, n_calibration_outputs), dtype='float64')

    burn_in_first_day = train_test_nt.burn_in_start // bin_metric_nt.fine_bins_per_day
    test_last_day = (train_test_nt.test_end - 1) // bin_metric_nt.fine_bins_per_day

    # Init outputs
    output_metrics = np.zeros((2, len(output_idx_nt)), dtype='float64')
    user_output_metrics = np.zeros((n_users, 2), dtype='float64')

    # Init thread level outputs for paralellisation
    n_threads = get_num_threads()
    thread_errors = np.zeros(n_threads, dtype='uint8')
    thread_output_metrics = np.zeros((n_threads, 2, len(output_idx_nt)), dtype='float64')
    if config_nt.nll_only:
        thread_calibration_output = np.empty( (0, 0, 0, 0), dtype='float64')
    else:
        thread_calibration_output = np.zeros((n_threads, n_breakdowns, n_groups, n_calibration_outputs), dtype='float64')

    # Init pointer for user interactions
    usr_frst_rw = g._init_user_count_table_pointer(n_users, user_interactions_nt, user_counts_nt, train_test_nt)

    # Initialise days passed
    days_passed = 0

    for day in range(burn_in_first_day, test_last_day + 1):
        days_passed += 1
        w = config_nt.w_inf + (1 - config_nt.w_inf) / (1 + days_passed)

        day_start = day * bin_metric_nt.fine_bins_per_day
        day_end = (day + 1) * bin_metric_nt.fine_bins_per_day

        if day_end > train_test_nt.test_end:
            day_end = train_test_nt.test_end

        # Getting the train test valid
        time_period_int = g.get_time_period(day_start, train_test_nt.validation_start, train_test_nt.validation_end, 
                                                train_test_nt.test_start, train_test_nt.test_end)

        calc_calibration = time_period_int == 1 and not config_nt.nll_only

        # Getting grid totals for seperating rate from shape
        user_u_totals = e.get_grid_row_sums(u)
        user_v_totals = e.get_grid_row_sums(v)
        user_p_totals = e.get_grid_row_sums(p)

        cluster_u_totals = e.get_grid_row_sums(cluster_u)
        cluster_v_totals = e.get_grid_row_sums(cluster_v)
        cluster_p_totals = e.get_grid_row_sums(cluster_p)

        # For each week iterate over the users and init the pointers
        for user_id in prange(n_users):
            thread_id = get_thread_id()
            if calc_calibration:
                np.random.seed(config_nt.sampling_seed + day * n_users + user_id)

            cnt_tbl_idx = usr_frst_rw[user_id]
            usr_end_idx = user_interactions_nt.user_last_index[user_id]

            # Init numpy vectors for calculating the user sums
            usr_updt_u_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            usr_updt_v_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            usr_updt_p_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            
            usr_updt_pos_sum = np.zeros(n_coarse_bins, dtype=np.float64)
            usr_updt_cnt_sum = np.zeros(n_coarse_bins, dtype=np.float64)

            for fine_bin in range(day_start, day_end):
                
                x, cnt_tbl_idx = g._get_user_count(cnt_tbl_idx, user_counts_nt, usr_end_idx, fine_bin)
                
                crnt_coarse_bin, crnt_fine_bin_within_coarse_pos = g._bin_computations(bin_metric_nt, fine_bin)

                mu_t, sigma_2_t, p_t, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t = g._get_smoothed_and_unsmoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, 
                                                                                    cluster_groups, user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                                                                    alpha_mu_grid, alpha_sigma2_grid, alpha_p_grid, zero_alpha_grid, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)

                
                if (time_period_int == 0 or time_period_int == 1) and not degen_mask[user_id, crnt_coarse_bin]: 
                    lpmf_raw, lpmf_smoothed, _, log_strict_upper_tail_smoothed = g._get_log_p0_lpmf_and_upper_tail(x, mu_t, sigma_2_t, p_t, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, calc_calibration, config_nt)

                    # Updating outputs
                    if ((not np.isfinite(lpmf_raw) or not np.isfinite(lpmf_smoothed)) 
                        or (calc_calibration and not np.isfinite(log_strict_upper_tail_smoothed))):
                        thread_errors[thread_id] = 1

                    else:
                        f.update_outputs(time_period_int, thread_output_metrics[thread_id], output_idx_nt, lpmf_raw, lpmf_smoothed)

                        if time_period_int == 1:
                            f.update_user_outputs(user_id=user_id, user_output_metrics=user_output_metrics, lpmf_smoothed=lpmf_smoothed)
        
                        if calc_calibration:
                            f.update_calibration_outputs(user_id=user_id, lpmf_smoothed=lpmf_smoothed, 
                                log_strict_upper_tail_smoothed=log_strict_upper_tail_smoothed, log_calibration_thresholds=log_calibration_thresholds, 
                                breakdown_groups=breakdown_groups, calibration_output=thread_calibration_output[thread_id])
                            
                f.collect_temp_grid(usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_pos_sum, usr_updt_cnt_sum, crnt_coarse_bin, x, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, w, config_nt)
            
            # Updating the users first row (for the next week) and the parameter grid and alpha grid
            usr_frst_rw[user_id] = cnt_tbl_idx
            f.update_grid(u, v, p, user_id, usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_pos_sum, bin_metric_nt.fine_bins_per_coarse_bin, config_nt)
            f.update_n_counts_and_alpha_grids(n_counts, alpha_mu_grid, alpha_sigma2_grid, user_id, usr_updt_cnt_sum, w, bin_metric_nt.fine_bins_per_coarse_bin, config_nt)
            
        for thread_id in range(n_threads):
            if thread_errors[thread_id] != 0:
                raise ValueError('infinite or nan probability identified')

        g.combine_threads(output_metrics, thread_output_metrics, calibration_output, thread_calibration_output, output_idx_nt, time_period_int, n_threads, config_nt)
        cluster_u, cluster_v, cluster_p = g.get_new_cluster_centres(cluster_groups, u, v, p, config_nt.distance_metric)

    return output_metrics, calibration_output, user_output_metrics