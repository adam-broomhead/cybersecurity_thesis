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

@njit(parallel=True)
def run_hurdle_benchmarks(user_counts_nt, user_interactions_nt, user_means, user_variances, user_p, user_hour_means, 
        user_hour_variances, user_hour_p, period_start, period_end, breakdown_groups, config_nt, bin_metric_nt, degen_mask):
    '''
    Runs the user static hurdle benchmark and the user x coarse bin hurdle benchmark
    '''
    log_min_likelihood = np.log(config_nt.min_likelihood)

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

            # Getting the count and updating the count table index
            count, cnt_tbl_idx = g._get_user_count(cnt_tbl_idx, user_counts_nt, user_end_idx, fine_bin)
            coarse_bin_id = (fine_bin % bin_metric_nt.fine_bins_per_day) // bin_metric_nt.fine_bins_per_coarse_bin

            if degen_mask[user_id, coarse_bin_id]:
                continue

            # Getting the lmpf values for the two counts
            user_lpmf = d.get_lpmf_val(count, user_means[user_id], user_variances[user_id], user_p[user_id], config_nt)

            user_hour_lpmf = d.get_lpmf_val(count, user_hour_means[user_id, coarse_bin_id],  user_hour_variances[user_id, coarse_bin_id], user_hour_p[user_id, coarse_bin_id], config_nt)

            # Capping LPMF values
            user_capped_lpmf = max(user_lpmf, log_min_likelihood)
            user_hour_capped_lpmf = max(user_hour_lpmf, log_min_likelihood)

            # Updating the outputs for the two models
            thread_results[thread_id, 0, 0] += 1
            thread_results[thread_id, 0, 1] += user_capped_lpmf
            thread_results[thread_id, 1, 0] += 1
            thread_results[thread_id, 1, 1] += user_hour_capped_lpmf

            if not config_nt.nll_only:
                user_strict_upper_tail = d.get_upper_tail_value(count + 1, user_means[user_id], user_variances[user_id], user_p[user_id], config_nt)

                user_hour_strict_upper_tail = d.get_upper_tail_value(count + 1, user_hour_means[user_id, coarse_bin_id], 
                    user_hour_variances[user_id, coarse_bin_id], user_hour_p[user_id, coarse_bin_id], config_nt)

                f.update_calibration_outputs(user_id=user_id, lpmf=user_lpmf, capped_lpmf=user_capped_lpmf, log_strict_upper_tail=user_strict_upper_tail, 
                                             log_calibration_thresholds=log_calibration_thresholds, breakdown_groups=breakdown_groups,
                                               calibration_output=thread_full_results[thread_id, 0])

                f.update_calibration_outputs(user_id=user_id, lpmf=user_hour_lpmf, capped_lpmf=user_hour_capped_lpmf,
                    log_strict_upper_tail=user_hour_strict_upper_tail, log_calibration_thresholds=log_calibration_thresholds, 
                    breakdown_groups=breakdown_groups, calibration_output=thread_full_results[thread_id, 1])

    # Combiging thread outputs together
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
        output.append(output_row)

    return output


def make_full_hurdle_benchmark_output_rows(full_results, config_dict):
    '''
    Makes full test benchmark outputs
    '''
    output = []

    # Iterates over models and breakdown types and groups
    for model_idx, model_name in enumerate(hurdle_benchmark_names):
        for breakdown_type_idx, breakdown_config in enumerate(metric_breakdowns):
            breakdown_type = breakdown_config['type']

            for breakdown_group_idx, breakdown_group in enumerate(breakdown_config['groups']):

                # Retrieves the relevant metric from the results table
                group_output = full_results[model_idx, breakdown_type_idx, breakdown_group_idx]

                # Ignore default outputs
                if group_output[0] == 0:
                    continue

                output_row = get_hurdle_benchmark_base_output(model_name=model_name, config_dict=config_dict, test_valid='test')

                # Adding to the output table
                output_row['breakdown_type'] = breakdown_type
                output_row['breakdown_group'] = breakdown_group
                output_row['n_bins_scored'] = group_output[0]
                output_row['non_degen_ll_sum'] = group_output[1]

                # Adding calubrtion results to the output table
                for threshold_idx, threshold in enumerate(config_dict['calibration_thresholds']):
                    threshold_name = format(float(threshold), 'f')
                    output_row[f'calibration_count_{threshold_name}'] = group_output[2 + threshold_idx]

                output.append(output_row)
    return output