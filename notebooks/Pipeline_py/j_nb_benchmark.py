import g_ll_runner_utils as g
import f_grids_and_outputs as f
import c_clustering as c
import utils as ut
from numba import njit, prange, get_num_threads, get_thread_id
import numpy as np 
import d_math as d
import polars as pl
import math

metric_breakdowns = ut.load_json5('metric_breakdowns')['breakdowns']
hurdle_benchmark_names = ('static_user_hurdle', 'static_user_hour_hurdle')

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

    return poisson_rates

def get_user_nb_params(user_counts, n_usrs, period_start, period_end, config_dict):
    '''
    Static mean and varaice estimator. Estimates parameters between period start to period end
    '''

    n_fine_bins = period_end - period_start

    # Getting counts and count 2 for the period to calc mean and variacne
    period_df = user_counts.filter((pl.col('fine_bin_id') >= period_start) & (pl.col('fine_bin_id') < period_end))
    period_df = period_df.with_columns( cnt=pl.col('cnt').cast(pl.Float64), cnt_2=pl.col('cnt').cast(pl.Float64) ** 2)
    period_df = period_df.group_by('user_id').agg(pl.sum('cnt').alias('sum_cnt'), pl.sum('cnt_2').alias('sum_cnt_2'))

    # Init mean and variance vectors
    usr_means = np.zeros(n_usrs, dtype='float64')
    usr_variances = np.zeros(n_usrs, dtype='float64')

    # Calculating the means for each user and then assigning that mean to user means
    means_to_assign = period_df['sum_cnt'].to_numpy() / n_fine_bins
    vars_to_assign = (period_df['sum_cnt_2'].to_numpy() - (period_df['sum_cnt'].to_numpy() ** 2) / n_fine_bins) / (n_fine_bins - 1)
    usrs = period_df['user_id'].to_numpy()
    usr_means[usrs] = means_to_assign
    usr_variances[usrs] = vars_to_assign

    return usr_means, usr_variances

def get_user_hurdle_params(user_counts, n_users, period_start, period_end, config_dict):
    '''
    Static p, and positive mean and varaice estimator. Estimates parameters between period start to period end
    '''

    n_fine_bins = period_end - period_start

    # Getting counts and count 2 for the period to calc mean and variacne
    period_df = user_counts.filter((pl.col('fine_bin_id') >= period_start) & (pl.col('fine_bin_id') < period_end))
    period_df = period_df.with_columns(count=pl.col('count').cast(pl.Float64), count_2=pl.col('count').cast(pl.Float64) ** 2)
    period_df = period_df.group_by('user_id').agg(pl.sum('count').alias('sum_cnt'), pl.sum('count_2').alias('sum_cnt_2'), pl.len().alias('n_bins'))

    # Init p, mean and variance vectors
    usr_means = np.zeros(n_users, dtype='float64')
    usr_variances = np.zeros(n_users, dtype='float64')
    usr_p = np.zeros(n_users, dtype='float64')

    # Getting sum of counts -1 which are used to find the hurdle mean and variance
    usrs = period_df['user_id'].to_numpy()
    n_bins = period_df['n_bins'].to_numpy()
    sum_cnt_minus_1 = period_df['sum_cnt'].to_numpy() - n_bins
    sum_cnt_2_minus_1 = period_df['sum_cnt_2'].to_numpy() - (2 * period_df['sum_cnt'].to_numpy()) + n_bins

    # This gets the sum of the counts 
    usr_p[usrs] = n_bins / n_fine_bins
    usr_means[usrs] = sum_cnt_minus_1 / n_bins
    usr_variances[usrs] = (sum_cnt_2_minus_1- (sum_cnt_minus_1 ** 2 / n_bins)) / (n_bins - 1)

    return usr_means, usr_variances, usr_p

#####################################
# Benchmark runner
#####################################

@njit
def run_nb_benchmarks(evaluation_counts, user_means, user_variances, user_hour_means, user_hour_variances, calibration_thresholds, config_nt, bin_metric_nt, degen_mask):
    '''
    Runs the user static negative binomial benchmark and the user x coarse bin negative binomial benchmark
    '''
    # Creating log likelihood scores and calibration outputs
    user_log_likelihood = 0
    user_hour_log_likelihood = 0
    n_scored = 0

    user_calibration = np.zeros(calibration_thresholds.shape[0])
    user_hour_calibration = np.zeros(calibration_thresholds.shape[0])

    # Iterating over the rows 
    for row_idx in range(evaluation_counts.shape[0]):
        # Getting the table info as vars
        user_id = evaluation_counts[row_idx, 0]
        fine_bin = evaluation_counts[row_idx, 1]
        count = evaluation_counts[row_idx, 2]
        coarse_bin_id = (fine_bin % bin_metric_nt.fine_bins_per_day) // bin_metric_nt.fine_bins_per_coarse_bin

        # Skipping degen bins
        if degen_mask[user_id, coarse_bin_id]:
            continue

        # Getting model parameters
        user_mu = user_means[user_id]
        user_sigma2 = user_variances[user_id]
        user_hour_mu = user_hour_means[user_id, coarse_bin_id]
        user_hour_sigma2 = user_hour_variances[user_id, coarse_bin_id]

        # Scoring counts and getting tail values
        user_log_likelihood += d.get_nb_lpmf_val(count, user_mu, user_sigma2, config_nt)
        user_hour_log_likelihood += d.get_nb_lpmf_val(count, user_hour_mu, user_hour_sigma2, config_nt)
        user_log_tail = d.get_nb_upper_tail_value(count, user_mu, user_sigma2, config_nt)
        user_hour_log_tail = d.get_nb_upper_tail_value(count, user_hour_mu, user_hour_sigma2, config_nt)
        n_scored += 1

        # Updating calibration outputs
        for threshold_idx in range(calibration_thresholds.shape[0]):
            log_threshold = math.log(calibration_thresholds[threshold_idx])

            if user_log_tail < log_threshold:
                user_calibration[threshold_idx] += 1

            if user_hour_log_tail < log_threshold:
                user_hour_calibration[threshold_idx] += 1

    return user_log_likelihood, user_hour_log_likelihood, user_calibration, user_hour_calibration, n_scored

@njit(parallel=True)
def run_hurdle_benchmarks(user_counts_nt, user_interactions_nt, user_means, user_variances, user_p, user_hour_means, 
        user_hour_variances, user_hour_p, period_start, period_end, breakdown_groups, config_nt, bin_metric_nt, degen_mask):
    '''
    Runs the user static hurdle benchmark and the user x coarse bin hurdle benchmark
    '''
    n_users = user_means.shape[0]
    results = np.zeros((2, 2), dtype='float64')

    if config_nt.nll_only:
        log_calibration_thresholds = np.empty(0, dtype='float64')
        full_results = np.empty((0, 0, 0, 0), dtype='float64')
    else:
        log_calibration_thresholds = np.log(config_nt.calibration_thresholds)
        n_breakdown_types = breakdown_groups.shape[0]
        n_breakdown_groups = int(breakdown_groups.max()) + 1

        n_full_results = 2 + log_calibration_thresholds.shape[0]
        full_results = np.zeros((2, n_breakdown_types, n_breakdown_groups, n_full_results), dtype='float64')

    n_threads = get_num_threads()
    thread_results = np.zeros((n_threads, 2, 2), dtype='float64')

    if config_nt.nll_only:
        thread_full_results = np.empty((0, 0, 0, 0, 0), dtype='float64')
    else:
        thread_full_results = np.zeros((n_threads, 2, n_breakdown_types, n_breakdown_groups, n_full_results), dtype='float64')

    for user_id in prange(n_users):
        thread_id = get_thread_id()

        if not config_nt.nll_only:
            np.random.seed(config_nt.seed + user_id)

        cnt_tbl_idx = user_interactions_nt.user_first_index[user_id]
        user_end_idx = user_interactions_nt.user_last_index[user_id]

        while cnt_tbl_idx <= user_end_idx and user_counts_nt.fine_bin_id[cnt_tbl_idx] < period_start: 
            cnt_tbl_idx += 1

        for fine_bin in range(period_start, period_end):

            count, cnt_tbl_idx = g._get_user_count(cnt_tbl_idx, user_counts_nt, user_end_idx, fine_bin)
            coarse_bin_id = (fine_bin % bin_metric_nt.fine_bins_per_day) // bin_metric_nt.fine_bins_per_coarse_bin

            if degen_mask[user_id, coarse_bin_id]:
                continue

            user_lpmf = d.get_lpmf_val(count, user_means[user_id], user_variances[user_id], user_p[user_id], config_nt)
            user_hour_lpmf = d.get_lpmf_val(count, user_hour_means[user_id, coarse_bin_id], 
                        user_hour_variances[user_id, coarse_bin_id], user_hour_p[user_id, coarse_bin_id], config_nt)

            thread_results[thread_id, 0, 0] += 1
            thread_results[thread_id, 0, 1] += user_lpmf
            thread_results[thread_id, 1, 0] += 1
            thread_results[thread_id, 1, 1] += user_hour_lpmf

            if not config_nt.nll_only:
                user_strict_upper_tail = d.get_upper_tail_value(count + 1, user_means[user_id], user_variances[user_id], user_p[user_id], config_nt)

                user_hour_strict_upper_tail = d.get_upper_tail_value(count + 1, user_hour_means[user_id, coarse_bin_id], 
                    user_hour_variances[user_id, coarse_bin_id], user_hour_p[user_id, coarse_bin_id], config_nt)

                f.update_calibration_outputs(user_id=user_id, lpmf_smoothed=user_lpmf, log_strict_upper_tail_smoothed=user_strict_upper_tail, 
                                             log_calibration_thresholds=log_calibration_thresholds, breakdown_groups=breakdown_groups,
                                               calibration_output=thread_full_results[thread_id, 0])

                f.update_calibration_outputs(user_id=user_id, lpmf_smoothed=user_hour_lpmf, 
                    log_strict_upper_tail_smoothed=user_hour_strict_upper_tail, log_calibration_thresholds=log_calibration_thresholds, 
                    breakdown_groups=breakdown_groups, calibration_output=thread_full_results[thread_id, 1])

    for thread_id in range(n_threads):
        for model_idx in range(2):
            for output_idx in range(2):
                results[model_idx, output_idx] += thread_results[thread_id, model_idx, output_idx]

        if not config_nt.nll_only:
            for model_idx in range(2):
                for breakdown_type_idx in range(n_breakdown_types):
                    for breakdown_group_idx in range(n_breakdown_groups):
                        for output_idx in range(n_full_results):
                            full_results[model_idx, breakdown_type_idx, breakdown_group_idx, output_idx] += thread_full_results[thread_id, model_idx, breakdown_type_idx, breakdown_group_idx, output_idx]

    return results, full_results

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
        output.append({'model_name': model_name, 'test_valid': test_valid, 'non_degen_ll': ll_sum / n_bins_scored})

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
            output.append({'model_name': model_name, 
                           'test_valid': test_valid, 
                           'threshold': config_dict['calibration_thresholds'][threshold_idx], 
                           'model_calibration': poisson_calibration_outputs[period_idx, threshold_idx, model_idx] / n_bins_scored})

    return output

def get_hurdle_benchmark_base_output(model_name, config_dict, test_valid):
    '''
    Creates feilds shared between both outputs
    '''
    return {
        'model_name': model_name,
        'experiment_name': 'benchmark',
        'seed': config_dict['seed'],
        'hurdle_model': True,
        'test_valid': test_valid}


def make_hurdle_benchmark_output_rows(results, config_dict, test_valid):
    '''
    Makes output rows for hurde benchmakr
    '''
    output = []

    for model_idx, model_name in enumerate(hurdle_benchmark_names):
        n_bins_scored = results[model_idx, 0]

        output_row = get_hurdle_benchmark_base_output(model_name=model_name, config_dict=config_dict, test_valid=test_valid)

        output_row['non_degen_ll'] = results[model_idx, 1] / n_bins_scored
        output_row['non_degen_smoothed_ll'] = None

        output.append(output_row)

    return output


def make_full_hurdle_benchmark_output_rows(full_results, config_dict):
    '''
    Makes full test benchmark outputs
    '''
    output = []

    for model_idx, model_name in enumerate(hurdle_benchmark_names):
        for breakdown_type_idx, breakdown_config in enumerate(metric_breakdowns):
            breakdown_type = breakdown_config['type']

            for breakdown_group_idx, breakdown_group in enumerate(breakdown_config['groups']):
                group_output = full_results[model_idx, breakdown_type_idx, breakdown_group_idx]

                if group_output[0] == 0:
                    continue

                output_row = get_hurdle_benchmark_base_output(model_name=model_name, config_dict=config_dict, test_valid='test')

                output_row['breakdown_type'] = breakdown_type
                output_row['breakdown_group'] = breakdown_group
                output_row['n_bins_scored'] = group_output[0]
                output_row['non_degen_ll_sum'] = group_output[1]

                for threshold_idx, threshold in enumerate(config_dict['calibration_thresholds']):
                    threshold_name = format(float(threshold), 'f')
                    output_row[f'calibration_count_{threshold_name}'] = group_output[2 + threshold_idx]

                output.append(output_row)

    return output