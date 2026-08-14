import utils as ut
import b_run_staging as b
from i_valid_and_test_runs import Tuner
import numpy as np
import polars as pl
import pandas as pd
import os

##########################
# Weekly variant
##########################

def init_weekly_grids(weekly_user_counts, n_users, train_test_dict, bin_metric_dict):
    ''' 
    Forms our initial parameter grids for the weekly variant
    '''
    # Init list which stores grid parameters for that day
    weekly_u_days = []
    weekly_v_days = []
    weekly_p_days = []

    # Loop over the days and init that days grid
    for day_idx in range(7):
        period_start = train_test_dict['train_start'] + day_idx * bin_metric_dict['fine_bins_per_day']
        period_end = period_start + bin_metric_dict['fine_bins_per_day']

        day_u, day_v, day_p = b.init_grid_hurdle(user_counts=weekly_user_counts, n_users=n_users, 
                coarse_bins_per_day=bin_metric_dict['coarse_bins_per_day'], period_start=period_start, period_end=period_end, bin_metric_dict=bin_metric_dict)

        weekly_u_days.append(day_u)
        weekly_v_days.append(day_v)
        weekly_p_days.append(day_p)

    # Concat daily grids to form a weekly grid
    weekly_u_pos_init = np.concatenate(weekly_u_days, axis=1)
    weekly_v_pos_init = np.concatenate(weekly_v_days, axis=1)
    weekly_p_init = np.concatenate(weekly_p_days, axis=1)

    return weekly_u_pos_init, weekly_v_pos_init, weekly_p_init


def get_weekly_degen_mask(weekly_user_counts, n_users, train_test_dict, bin_metric_dict, static_configs):
    '''
    Gets the degen mask for the weekly model
    '''

    # Getting the number of rows with over 1 count in each coarse bin in train + burn in
    weekly_non_degen_counts = weekly_user_counts.filter((pl.col('fine_bin_id') >= train_test_dict['train_start']) & (pl.col('fine_bin_id') < train_test_dict['burn_in_end']) & (pl.col('count') > 1))
    weekly_non_degen_counts = weekly_non_degen_counts.with_columns(((pl.col('fine_bin_id') % bin_metric_dict['fine_bins_per_week']) // bin_metric_dict['fine_bins_per_coarse_bin']).alias('weekly_coarse_bin'))
    weekly_non_degen_counts = weekly_non_degen_counts.group_by(['user_id', 'weekly_coarse_bin']).agg(pl.len().alias('n_bins')).collect(engine='streaming')

    # Get the degen mask using the cut off per coarse bin
    weekly_degen_mask = np.ones((n_users, bin_metric_dict['coarse_bins_per_day'] * 7), dtype='bool')
    non_degen_rows = weekly_non_degen_counts['n_bins'].to_numpy() >= static_configs['degen_threshold']

    # Assigning false to non degen counts
    entries_to_assign = weekly_non_degen_counts.select(['user_id', 'weekly_coarse_bin']).to_numpy()
    weekly_degen_mask[entries_to_assign[non_degen_rows, 0], entries_to_assign[non_degen_rows, 1]] = False

    return weekly_degen_mask

def get_weekly_configs(bin_metric_dict, train_test_dict):
    '''
    Updates train test dict and config nt to work with weekly parameters
    '''
    weekly_bin_metric_dict = bin_metric_dict.copy()

    # Updating fine bins per cycle default to fine bins per week
    weekly_bin_metric_dict['fine_bins_per_cycle'] = bin_metric_dict['fine_bins_per_week']
    weekly_bin_metric_nt = b.dictionary_to_named_tuple_class('weekly_bin_metric_nt', weekly_bin_metric_dict)(**weekly_bin_metric_dict)

    # Using a 1 week train period rather than 1 day
    weekly_train_test_dict = train_test_dict.copy()
    weekly_train_test_dict['train_end'] = bin_metric_dict['fine_bins_per_week']
    weekly_train_test_dict['burn_in_start'] = bin_metric_dict['fine_bins_per_week']

    return weekly_bin_metric_nt, weekly_train_test_dict


def run_weekly_variant(t, user_counts, user_mapping, train_test_dict, bin_metric_dict, static_configs, base_config, hyperparams):
    '''
    Runs the weekly cycle variant experiment
    '''
    experiment_name = 'variant_weekly'
    hurdle_model = True
    n_users = user_mapping.shape[0]
    weekly_user_counts = user_counts.lazy()

    # Init grids and configs
    weekly_u_pos_init, weekly_v_pos_init, weekly_p_init = init_weekly_grids(weekly_user_counts, n_users, train_test_dict, bin_metric_dict)
    weekly_bin_metric_nt, weekly_train_test_dict = get_weekly_configs(bin_metric_dict, train_test_dict)

    # Init degen mask and n_counts
    weekly_degen_mask = get_weekly_degen_mask(weekly_user_counts, n_users, train_test_dict, bin_metric_dict, static_configs)
    weekly_n_counts_init = np.zeros_like(weekly_u_pos_init)

    # Creating weekly tuner and running variant 
    weekly_t = Tuner(weekly_u_pos_init, weekly_v_pos_init, weekly_p_init, weekly_u_pos_init, weekly_v_pos_init,
        weekly_u_pos_init, weekly_v_pos_init, weekly_u_pos_init, weekly_v_pos_init, weekly_p_init,
        weekly_n_counts_init, t.user_counts_nt, t.user_interactions_nt, t.interpolation_weights, weekly_bin_metric_nt, 
        t.output_idx_nt, t.train_test_nt_class, t.user_type_groups)

    weekly_results = weekly_t.tune_models(experiment_name=experiment_name, hurdle_model=hurdle_model, hyperparams=hyperparams, 
        train_test_dict=weekly_train_test_dict, config_dict=base_config, degen_mask=weekly_degen_mask, run_name=experiment_name)

    return weekly_results, weekly_degen_mask


def run_daily_weekly_mask_variant(t, weekly_degen_mask, hyperparams, train_test_dict, base_config):
    ''' 
    Runs the best LL model with the week level degen mask for consistency of comparison
    '''
    experiment_name = 'variant_daily_weekly_mask'
    hurdle_model = True

    daily_weekly_mask_results = t.tune_models(experiment_name=experiment_name, hurdle_model=hurdle_model, hyperparams=hyperparams, train_test_dict=train_test_dict, config_dict=base_config, degen_mask=weekly_degen_mask, run_name=experiment_name)
    return daily_weekly_mask_results

##########################
# Other variants
##########################

def run_quadratic_variant(t, quadratic_interpolation_weights, hyperparams, train_test_dict, base_config, degen_mask):
    '''
    Runs the quadratic variant experiment
    '''
    experiment_name = 'variant_quadratic'
    hurdle_model = True

    quad_t = Tuner(t.u_init, t.v_init, t.p_init, t.u_pos_init, t.v_pos_init, t.u_clustering, t.v_clustering, t.u_pos_clustering, t.v_pos_clustering, t.p_pos_clustering, 
              t.n_counts_init, t.user_counts_nt, t.user_interactions_nt, quadratic_interpolation_weights, t.bin_metric_nt, t.output_idx_nt, t.train_test_nt_class, t.user_type_groups, quadratic_interpolation=True)
    results = quad_t.tune_models(experiment_name=experiment_name, hurdle_model=hurdle_model,  hyperparams=hyperparams, 
                        train_test_dict=train_test_dict, config_dict=base_config, degen_mask=degen_mask, run_name=experiment_name)
    return results


def run_nb_variant(t, hyperparams, train_test_dict, base_config, degen_mask):
    ''' 
    Runs the negative binomial variant experiment
    '''
    experiment_name = 'variant_nb'
    hurdle_model = False
    results = t.tune_models(experiment_name=experiment_name, hurdle_model=hurdle_model,  hyperparams=hyperparams, 
                        train_test_dict=train_test_dict, config_dict=base_config, degen_mask=degen_mask, run_name=experiment_name)
    return results

##########################
# Storing final comparison
##########################

def store_variant_results():
    ''' 
    Calculates the difference from best performing unsmoothed model and stores results
    '''
    outputs_dir = f'{ut.project_root}/outputs'
    os.makedirs(outputs_dir, exist_ok=True)

    variant_comparison = pd.DataFrame({
        'Change': ['Ordinary NB', 'Quadratic interpolation', 'Weekly cycle',],
        'variant_ll': [
            pd.read_parquet(f'{ut.results_dir}/variant_nb/nll_only/')['non_degen_ll'].max(),
            pd.read_parquet(f'{ut.results_dir}/variant_quadratic/nll_only/')['non_degen_ll'].max(),
            pd.read_parquet(f'{ut.results_dir}/variant_weekly/nll_only/')['non_degen_ll'].max(),
        ],
        'comparison_ll': [
            pd.read_parquet(f'{ut.results_dir}/LL_runner/nll_only/')['non_degen_ll'].max(),
            pd.read_parquet(f'{ut.results_dir}/LL_runner/nll_only/')['non_degen_ll'].max(),
            pd.read_parquet(f'{ut.results_dir}/variant_daily_weekly_mask/nll_only/')['non_degen_ll'].max()]})

    variant_comparison['$Log-Likelihood $\Delta$'] = (variant_comparison['variant_ll'] - variant_comparison['comparison_ll'])
    variant_comparison[['Change', '$Log-Likelihood $\Delta$']].to_csv(f'{outputs_dir}/variants.csv', index=False)