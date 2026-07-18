from numba import njit
import math 
import d_math as d
import polars as pl
import numpy as np 
import g_ll_runner_utils as g

#####################################
# Benchmark init
#####################################

def get_user_poisson_rates(user_counts, n_users, train_test_dict, config_dict):

    # Getting the counts per user in the period
    period_df = user_counts.filter((pl.col('fine_bin_id') >= train_test_dict['train_start']) & (pl.col('fine_bin_id') < train_test_dict['burn_in_end']))
    period_df = period_df.group_by('user_id').agg(pl.sum('count').alias('sum_cnt'))

    # Init the poisson rates for each user
    poisson_rates = np.zeros(n_users, dtype='float64')

    # Assinging the mean counts to the users
    users_to_assign = period_df['user_id'].to_numpy()
    poisson_rates[users_to_assign] = period_df['sum_cnt'].to_numpy() / (train_test_dict['burn_in_end'] - train_test_dict['train_start'])
    poisson_rates = np.maximum(poisson_rates, config_dict['mean_min'])

    return poisson_rates

def get_user_nb_params(user_counts, n_usrs, period_start, period_end, config_dict):
    '''
    Static mean and varaice estimator. Estimates parameters between period start to period end
    '''

    n_fine_bins = period_end - period_start

    # Getting counts and count 2 for the period to calc mean and variacne
    period_df = user_counts.filter((pl.col('fine_bin_id') >= period_start) & (pl.col('fine_bin_id') < period_end))
    period_df = period_df.with_columns((pl.col('cnt') ** 2).alias('cnt_2')).group_by('user_id').agg(
        pl.sum('cnt').alias('sum_cnt'), pl.sum('cnt_2').alias('sum_cnt_2'))

    # Init mean and variance vectors
    usr_means = np.zeros(n_usrs, dtype='float64')
    usr_variances = np.zeros(n_usrs, dtype='float64')

    # Calculating the means for each user and then assigning that mean to user means
    usr_means = period_df['sum_cnt'].to_numpy() / n_fine_bins
    usr_variances = (period_df['sum_cnt_2'].to_numpy() - (period_df['sum_cnt'].to_numpy() ** 2) / n_fine_bins) / (n_fine_bins - 1)
    usrs = period_df['usr_id'].to_numpy()
    usr_means[usrs] = usr_means
    usr_variances[usrs] = usr_variances

    # Capping mean and variance values
    usr_means = np.maximum(usr_means, config_dict['mean_min'])
    usr_variances = np.maximum(usr_variances, config_dict['var_min'])

    return usr_means, usr_variances

#####################################
# Benchmark runner
#####################################

@njit
def run_poisson_benchmarks(user_poisson_rates, user_coarse_poisson_rates, degen_mask, 
                           user_counts_nt, user_interactions_nt, train_test_nt, bin_metric_nt, config_nt, poisson_benchmark_idx_nt):
    ''' 
    Runs simple benchmarks one which just fits a poisson model for each user and one which fits a poisson model for each user x coarse bin
    '''
    # Init varaibles and outputs
    n_users = user_poisson_rates.shape[0]
    2 = len(poisson_benchmark_idx_nt)
    log_calibration_thresholds = np.log(config_nt.calibration_thresholds)

    # Init outputs
    output_metrics = np.zeros((2, 2, 2), dtype='float64')
    calibration_output = np.zeros((2, log_calibration_thresholds.shape[0], 2), dtype='float64')

    # Init pointer for user interactions
    usr_frst_rw = g._init_user_count_table_pointer(n_users, user_interactions_nt, user_counts_nt, train_test_nt)

    # Iterating over the weeks
    burn_in_first_week = train_test_nt.burn_in_start // bin_metric_nt.fine_bins_per_week
    test_last_week = (train_test_nt.test_end - 1) // bin_metric_nt.fine_bins_per_week

    for week in range(burn_in_first_week, test_last_week + 1):

        week_start = week * bin_metric_nt.fine_bins_per_week
        week_end = (week + 1) * bin_metric_nt.fine_bins_per_week

        if week_end > train_test_nt.test_end:
            week_end = train_test_nt.test_end

        # Iterating over the users and init the pointers
        for user_id in range(n_users):

            cnt_tbl_idx = usr_frst_rw[user_id]
            usr_end_idx = user_interactions_nt.user_last_index[user_id]

            for fine_bin in range(week_start, week_end):

                # Get count and update pointer if necessary get time period and counts coarse bin
                x, cnt_tbl_idx = g._get_user_count(cnt_tbl_idx, user_counts_nt, usr_end_idx, fine_bin)

                crnt_coarse_bin, _ = g._bin_computations(bin_metric_nt, fine_bin)

                time_period_int = g.get_time_period(fine_bin, train_test_nt.validation_start, train_test_nt.validation_end, train_test_nt.test_start, train_test_nt.test_end)

                # Compute metrics for non degen bins
                if ((time_period_int == 0 or time_period_int == 1) and not degen_mask[user_id, crnt_coarse_bin]):
                    
                    # Update lambdas and get pmf and upper tail
                    lambda_user = max(user_poisson_rates[user_id], config_nt.mean_min)
                    lambda_user_coarse = max(user_coarse_poisson_rates[user_id, crnt_coarse_bin], config_nt.mean_min,)

                    lpmf_user = d.poisson_lpmf(x, lambda_user)
                    lpmf_user_coarse = d.poisson_lpmf(x, lambda_user_coarse)

                    log_upper_tail_user = d.poisson_log_upper_tail(x, lambda_user)
                    log_upper_tail_user_coarse = d.poisson_log_upper_tail(x, lambda_user_coarse)

                    # Storing output metrics
                    user_idx = poisson_benchmark_idx_nt.user_poisson
                    user_coarse_idx = poisson_benchmark_idx_nt.user_coarse_poisson

                    output_metrics[time_period_int, user_idx, 0] += 1
                    output_metrics[time_period_int, user_idx, 1] += lpmf_user

                    output_metrics[time_period_int, user_coarse_idx, 0] += 1
                    output_metrics[time_period_int, user_coarse_idx, 1] += lpmf_user_coarse

                    # Storing calibration metrics
                    for calib_threshold_idx in range(log_calibration_thresholds.shape[0]):

                        if log_upper_tail_user < log_calibration_thresholds[calib_threshold_idx]:
                            calibration_output[time_period_int, calib_threshold_idx, user_idx] += 1

                        if log_upper_tail_user_coarse < log_calibration_thresholds[calib_threshold_idx]:
                            calibration_output[time_period_int, calib_threshold_idx, user_coarse_idx] += 1

            usr_frst_rw[user_id] = cnt_tbl_idx

    return output_metrics, calibration_output

@njit
def run_nb_benchmarks(evaluation_counts, user_means, user_variances, user_hour_means, user_hour_variances, calibration_thresholds, config_nt, bin_metric_nt, degen_mask):
    '''
    Runs the user static negative binomial benchmark and the user x coarse bin negative binomial benchmark
    '''
    # Creating log likelihood scores and calibration outputs
    user_log_likelihood = 0
    user_hour_log_likelihood = 0

    user_calibration = np.zeros(calibration_thresholds.shape[0])
    user_hour_calibration = np.zeros(calibration_thresholds.shape[0])

    # Iterating over the rows 
    for row_idx in range(evaluation_counts.shape[0]):
        # Getting the table info as vars
        user_id = evaluation_counts[row_idx, 0]
        fine_bin = evaluation_counts[row_idx, 1]
        count = evaluation_counts[row_idx, 2]
        coarse_bin = (fine_bin % bin_metric_nt.fine_bins_per_week) // bin_metric_nt.fine_bins_per_coarse_bin

        # Skipping degen bins
        if degen_mask[user_id, coarse_bin]:
            continue

        # Getting model parameters
        user_mu = user_means[user_id]
        user_sigma2 = user_variances[user_id]
        user_hour_mu = user_hour_means[user_id, coarse_bin]
        user_hour_sigma2 = user_hour_variances[user_id, coarse_bin]

        # Scoring counts and getting tail values
        user_log_likelihood += d.get_nb_lpmf_val(count, user_mu, user_sigma2, config_nt)
        user_hour_log_likelihood += d.get_nb_lpmf_val(count, user_hour_mu, user_hour_sigma2, config_nt)
        user_log_tail = d.get_nb_upper_tail_value(count, user_mu, user_sigma2, config_nt)
        user_hour_log_tail = d.get_nb_upper_tail_value(count, user_hour_mu, user_hour_sigma2, config_nt)
        n_scored += 1

        # Updating calibration outputs
        for threshold_idx in range(calibration_thresholds.shape[0]):
            log_threshold = math.log(calibration_thresholds[threshold_idx])

            if user_log_tail <= log_threshold:
                user_calibration[threshold_idx] += 1

            if user_hour_log_tail <= log_threshold:
                user_hour_calibration[threshold_idx] += 1

    return user_log_likelihood, user_hour_log_likelihood, user_calibration, user_hour_calibration, n_scored

#####################################
# Benchmark output rows
#####################################

def make_poisson_benchmark_output_rows(poisson_output_metrics, test_valid, poisson_benchmark_idx):

    # Gets the period index
    if test_valid == 'valid':
        period_idx = 0
    elif test_valid == 'test':
        period_idx = 1

    # Itertaing over the models
    output = []
    for model_name, model_idx in poisson_benchmark_idx.items():

        # Getting metrics and updating outputs
        n_bins_scored = poisson_output_metrics[period_idx, model_idx, 0]
        ll_sum = poisson_output_metrics[period_idx, model_idx, 1]
        output.append({'smoothed_model_name': model_name, 'test_valid': test_valid, 'non_degen_ll': ll_sum / n_bins_scored})

    return output

def make_poisson_benchmark_calibration_rows(poisson_output_metrics, poisson_calibration_outputs, config_dict, test_valid, poisson_benchmark_idx):
    
    # Gets the period index
    if test_valid == 'valid':
        period_idx = 0
    elif test_valid == 'test':
        period_idx = 1

    output = []
    for model_name, model_idx in poisson_benchmark_idx.items():

        n_bins_scored = poisson_output_metrics[period_idx, model_idx, 0]

        for threshold_idx in range(config_dict['calibration_thresholds'].shape[0]):
            output.append({'smoothed_model_name': model_name, 
                           'test_valid': test_valid, 
                           'threshold': config_dict['calibration_thresholds'][threshold_idx], 
                           'model_calibration': poisson_calibration_outputs[period_idx, threshold_idx, model_idx] / n_bins_scored})

    return output