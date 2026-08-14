from numba import njit, prange, get_num_threads, get_thread_id
import numpy as np 
import g_ll_runner_utils as g
import e_smoothing as e
import f_grids_and_outputs as f
import d_math as d

@njit(parallel=True)
def run_lambert_liu(u_init, v_init, p_init, cluster_u_init, cluster_v_init, cluster_p_init, cluster_groups, 
    n_counts_init, alpha_mu_grid_init, alpha_sigma2_grid_init, alpha_p_grid_init, degen_mask, user_counts_nt, user_interactions_nt, 
    interpolation_weights, train_test_nt, bin_metric_nt, config_nt, output_idx_nt, breakdown_groups,
    quadratic_interpolation, attack_start_fb, attack_sizes):
    ''' 
    Runs the lambert liu algorithm
    Args:
        u_init + v_init : inital parameter grids
        cluster_u_init + cluster_v_init : inital cluster parameters
        cluster groups : inital cluster assignments 1 row per user id
        user_counts_nt: is a named tuple version of user_counts has columns user_id, fine_bin_id, count
        user_interactions_nt: is a named tuple verion of user_interactions has columns user id and first and last interaction index in user_counts
        interpolation_weights : precalculated weights for parameter interpolation
        train_test_nt, bin_metric_nt, config_dt : named tuple versions of dicts train_test_dict, config_dict and bin_metric_dict
        output_idx_nt : a named tuple containing the names and indicies of the outputs we want to store
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
        log_calibration_thresholds = np.empty(0)
        calibration_output = np.empty((0, 0, 0))

    else:
        log_calibration_thresholds = np.log(config_nt.calibration_thresholds)
        n_groups = breakdown_groups.max() + 1
        n_breakdowns = breakdown_groups.shape[0]

        # n_bins, non_degen_ll and calibration threshold counts
        n_calibration_outputs = 2 + log_calibration_thresholds.shape[0]
        calibration_output = np.zeros((n_breakdowns, n_groups, n_calibration_outputs))

    burn_in_first_cycle = train_test_nt.burn_in_start // bin_metric_nt.fine_bins_per_cycle
    test_last_cycle = (train_test_nt.test_end - 1) // bin_metric_nt.fine_bins_per_cycle

    # Init outputs
    output_metrics = np.zeros((2, len(output_idx_nt)))
    user_output_metrics = np.zeros((n_users, 2))

    # Get min p and likelihood values
    log_min_likelihood = np.log(config_nt.min_likelihood)

    # Init thread level outputs for paralellisation
    n_threads = get_num_threads()
    thread_errors = np.zeros(n_threads, dtype='int8')
    thread_output_metrics = np.zeros((n_threads, 2, len(output_idx_nt)))
    if config_nt.nll_only:
        thread_calibration_output = np.empty( (0, 0, 0, 0))
        observed_p_vals = np.empty((0, 0))
        attack_p_vals = np.empty((0, 0, 0))
        attack_bin_inputs = np.empty((0, 0, 0))
    else:
        thread_calibration_output = np.zeros((n_threads, n_breakdowns, n_groups, n_calibration_outputs))
        observed_p_vals, attack_p_vals, attack_bin_inputs = g.init_detection_outputs(n_users, train_test_nt.test_end - train_test_nt.test_start, attack_sizes.shape[0], bin_metric_nt.fine_bins_per_coarse_bin)

    # Init pointer for user interactions
    usr_frst_rw = g._init_user_count_table_pointer(n_users, user_interactions_nt, user_counts_nt, train_test_nt)

    # Initialise cycles passed
    cycles_passed = 0

    for cycle in range(burn_in_first_cycle, test_last_cycle + 1):
        cycles_passed += 1
        w = config_nt.w_inf + (1 - config_nt.w_inf) / (1 + cycles_passed)

        cycle_start = cycle * bin_metric_nt.fine_bins_per_cycle
        cycle_end = (cycle + 1) * bin_metric_nt.fine_bins_per_cycle

        if cycle_end > train_test_nt.test_end:
            cycle_end = train_test_nt.test_end

        # Getting the train test valid
        time_period_int = g.get_time_period(cycle_start, train_test_nt.validation_start, train_test_nt.validation_end, 
                                                train_test_nt.test_start, train_test_nt.test_end)

        calc_calibration = time_period_int == 1 and not config_nt.nll_only

        # Getting grid totals for seperating rate from shape
        user_u_totals = e.get_grid_row_sums(u)
        user_v_totals = e.get_grid_row_sums(v)
        user_p_totals = e.get_grid_row_sums(p)

        cluster_u_totals = e.get_grid_row_sums(cluster_u)
        cluster_v_totals = e.get_grid_row_sums(cluster_v)
        cluster_p_totals = e.get_grid_row_sums(cluster_p)

        # Iterate over the users and init the pointers to that users row
        for user_id in prange(n_users):
            thread_id = get_thread_id()
            if calc_calibration:
                np.random.seed(config_nt.seed + cycle * n_users + user_id)

            cnt_tbl_idx = usr_frst_rw[user_id]
            usr_end_idx = user_interactions_nt.user_last_index[user_id]

            is_attack_day = calc_calibration and cycle_start <= attack_start_fb[user_id] < cycle_end

            # Init numpy vectors for calculating the user sums
            usr_updt_u_sum = np.zeros(n_coarse_bins)
            usr_updt_v_sum = np.zeros(n_coarse_bins)
            usr_updt_p_sum = np.zeros(n_coarse_bins)
            usr_updt_n_cnts = np.zeros(n_coarse_bins)

            for fine_bin in range(cycle_start, cycle_end):
                
                x, cnt_tbl_idx = g._get_user_count(cnt_tbl_idx, user_counts_nt, usr_end_idx, fine_bin)
                
                crnt_coarse_bin, crnt_fine_bin_within_coarse_pos = g._bin_computations(bin_metric_nt, fine_bin)
                degen_mask_bin_idx = (fine_bin // bin_metric_nt.fine_bins_per_coarse_bin) % degen_mask.shape[1]

                mu_t, sigma_2_t, p_t, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t = g._get_smoothed_and_unsmoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, 
                                                                                    cluster_groups, user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                                                                                    alpha_mu_grid, alpha_sigma2_grid, alpha_p_grid, zero_alpha_grid, degen_mask, user_id, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)

                if quadratic_interpolation:
                    mu_t, sigma_2_t, p_t = g.cap_quadratic_params(mu_t, sigma_2_t, p_t, config_nt.hurdle_model)
                    mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t = g.cap_quadratic_params(mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, config_nt.hurdle_model)

                # Getting errors
                update_error = g.get_parameter_errors(mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, config_nt.hurdle_model, False)
                thread_errors[thread_id] |= update_error

                # Working out scored bins
                if ((time_period_int == 0 or time_period_int == 1) and not degen_mask[user_id, degen_mask_bin_idx]):
                    scoring_error = g.get_parameter_errors(mu_t, sigma_2_t, p_t, config_nt.hurdle_model, True)
                    thread_errors[thread_id] |= scoring_error

                    if scoring_error == 0:
                        lpmf, log_strict_upper_tail = g._get_lpmf_and_upper_tail(x, mu_t, sigma_2_t, p_t, calc_calibration, config_nt)

                        if np.isnan(lpmf) or lpmf > 0 or (calc_calibration and (np.isnan(log_strict_upper_tail) or log_strict_upper_tail > 0)):
                            thread_errors[thread_id] |= 1

                        # Updating outputs and test outputs
                        else:
                            capped_lpmf = max(lpmf, log_min_likelihood)
                            f.update_outputs(time_period_int, thread_output_metrics[thread_id], output_idx_nt, capped_lpmf)

                            if time_period_int == 1:
                                f.update_user_outputs(user_id=user_id, user_output_metrics=user_output_metrics, lpmf=capped_lpmf)

                            if calc_calibration:
                                observed_log_p_vals = f.update_calibration_outputs(user_id=user_id, lpmf=lpmf, capped_lpmf=capped_lpmf, log_strict_upper_tail=log_strict_upper_tail, log_calibration_thresholds=log_calibration_thresholds, breakdown_groups=breakdown_groups, calibration_output=thread_calibration_output[thread_id])
                                observed_p_vals[user_id, fine_bin - train_test_nt.test_start] = np.exp(observed_log_p_vals)
                                if is_attack_day:
                                    g.store_attack_bin_inputs(attack_bin_inputs[user_id], fine_bin, attack_start_fb[user_id], x, mu_t, sigma_2_t, p_t)
                if update_error == 0:
                    f.collect_temp_grid(usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_n_cnts, crnt_coarse_bin, x, mu_unsmth_t, sigma_unsmth_2_t, p_unsmth_t, w, config_nt)

            # Scoring attacks
            if is_attack_day:
                g.get_attack_p_values(attack_p_vals[user_id], attack_bin_inputs[user_id], attack_sizes, config_nt)

            # Updating the users first row and the parameter grid and alpha grid
            usr_frst_rw[user_id] = cnt_tbl_idx
            f.update_grid(u, v, p, user_id, usr_updt_u_sum, usr_updt_v_sum, usr_updt_p_sum, usr_updt_n_cnts, bin_metric_nt.fine_bins_per_coarse_bin, config_nt)
            f.update_n_counts_and_alpha_grids(n_counts, alpha_mu_grid, alpha_sigma2_grid, user_id, usr_updt_n_cnts, w, bin_metric_nt.fine_bins_per_coarse_bin, config_nt)
            
        g.raise_parameter_errors(n_threads, thread_errors)

        g.combine_threads(output_metrics, thread_output_metrics, calibration_output, thread_calibration_output, output_idx_nt, time_period_int, n_threads, config_nt)
        cluster_u, cluster_v, cluster_p = g.get_new_cluster_centres(cluster_groups, u, v, p, config_nt.distance_metric)

    return output_metrics, calibration_output, user_output_metrics, observed_p_vals, attack_p_vals