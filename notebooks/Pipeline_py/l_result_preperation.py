import os

import numpy as np
import pandas as pd


ll_model_names = (
    'no_smoothing',
    'global_smoothing',
    'cluster_smoothing')

model_labels = {
        'no_smoothing': 'No shrinkage',
        'global_smoothing': 'Global shrinkage',
        'cluster_smoothing': 'Cluster shrinkage',
        'static_user_hurdle': 'User level benchmark',
        'static_user_hour_hurdle': 'Hourly user benchmark'}

def safe_read(directory, expected_files=None):
    '''
    Reads parquet files in a directory
    '''
    if expected_files is not None:
        assert len(os.listdir(directory)) == expected_files

    return pd.read_parquet(directory)

def load_test_results(results_dir, n_test_seeds):
    '''
    Loads the full results and concats them into a big dataframe adds the mean log likelihood column
    '''

    # Creating three long dataframes with the reruns in them
    ll_full_results = pd.concat([safe_read(directory=f'{results_dir}/test/{model}/full', expected_files=n_test_seeds).assign(model=model)
                                 for model in ll_model_names], ignore_index=True)

    ll_user_results = pd.concat([safe_read(directory=f'{results_dir}/test/{model}/user', expected_files=(n_test_seeds if model == 'cluster_smoothing' else 1)).assign(model=model) 
                                 for model in ll_model_names],ignore_index=True)

    benchmark_results = safe_read(directory=f'{results_dir}/benchmarks/full', expected_files=1).rename(columns={'model_name': 'model'})

    # Adding in mean log likelihood column
    ll_full_results['mean_ll'] = ll_full_results['non_degen_ll_sum'] / ll_full_results['n_bins_scored']
    ll_user_results['mean_ll'] = ll_user_results['non_degen_ll_sum'] / ll_user_results['n_bins_scored']

    benchmark_results['mean_ll'] = benchmark_results['non_degen_ll_sum'] / benchmark_results['n_bins_scored']

    return ll_full_results, ll_user_results, benchmark_results


def summarise_decile_improvements(run_results, decile_column):
    '''
    Calculates mean and sd improvement for each model for the two deciles:
    activity and distance
    '''
    # Pivot to have the models on the column
    results_pvt = run_results.pivot(index=['seed', decile_column], columns='model', values='mean_ll').reset_index()

    # Calculating improvements
    improvements = results_pvt[['seed', decile_column]].copy().assign(
            global_smoothing=results_pvt['global_smoothing'] - results_pvt['no_smoothing'],
            cluster_smoothing=results_pvt['cluster_smoothing'] - results_pvt['no_smoothing'])

    # Unpivot and cacluate mean and sd
    improvements = improvements.melt(id_vars=['seed', decile_column], var_name='model', value_name='improvement')
    improvements = improvements.groupby([decile_column, 'model'], as_index=False).agg(
        mean_improvement=('improvement', 'mean'), seed_sd=('improvement', 'std'))
    return improvements

def get_activity_decile_improvement(ll_full_results):
    '''
    gets improvement by activity decile
    '''
    run_results = ll_full_results.loc[ll_full_results['breakdown_type'] == 'activity_decile', 
                                      ['seed', 'breakdown_group', 'model', 'mean_ll']]
    run_results = run_results.rename(columns={'breakdown_group': 'activity_decile'})

    output = summarise_decile_improvements(run_results=run_results, decile_column='activity_decile')

    # Altering output form
    output.insert(0, 'breakdown', 'activity')
    output = output.rename(columns={'activity_decile': 'decile'})

    return output

def get_distance_decile_improvement(ll_user_results):
    '''
    Gets the distance improvement over the baseline model by decile for the two models:
        returns both mean and sd across seeds
    '''
    # Get the results for cluster smoothing
    cluster_smoothing_results = ll_user_results.loc[ll_user_results['model'] == 'cluster_smoothing'].copy()

    # Ranking the users for each seed and getting deciles
    distance_rank = cluster_smoothing_results.groupby('seed')['cluster_distance'].rank(method='first')
    users_per_seed = cluster_smoothing_results.groupby('seed')['cluster_distance'].transform('size')
    cluster_smoothing_results['distance_decile'] = (((distance_rank - 1) * 10 // users_per_seed) + 1).astype('int8')

    # For the no cluster models creating one row per seed for each user with the distance decile they are assigned to
    distance_groups = cluster_smoothing_results[['seed', 'user_idx', 'distance_decile']]
    global_and_static_results = ll_user_results.loc[ll_user_results['model'] != 'cluster_smoothing'].drop(columns=['seed', 'cluster_distance', 'mean_ll'])
    global_and_static_results = global_and_static_results.merge(distance_groups, on='user_idx', how='inner')

    # Concat together two results
    cluster_smoothing_results = cluster_smoothing_results[['model', 'seed', 'user_idx', 'distance_decile', 'non_degen_ll_sum', 'n_bins_scored']]
    all_results = pd.concat([global_and_static_results, cluster_smoothing_results], ignore_index=True)

    # Calculate the totals for each decile and get mean and sd
    all_results = all_results.groupby(['seed', 'distance_decile', 'model'], as_index=False).agg(
        non_degen_ll_sum=('non_degen_ll_sum', 'sum'), n_bins_scored=('n_bins_scored', 'sum'))
    all_results = all_results.assign(mean_ll=lambda data: data['non_degen_ll_sum'] / data['n_bins_scored'])
    output = summarise_decile_improvements(run_results=all_results, decile_column='distance_decile')

    # Altering output form
    output.insert(0, 'breakdown', 'distance')
    output = output.rename(columns={'distance_decile': 'decile'})

    return output

def get_calibration_summary(ll_full_results):
    '''
    Gets average and sd of calibraiton results across different seeds
    '''

    # Getting overall results
    overall_results = ll_full_results.loc[ll_full_results['breakdown_type'] == 'overall'].copy()

    # Unpivoting the calibration count columns and getting threshold and great
    calibration_columns = [column for column in overall_results.columns if column.startswith('calibration_count_')]
    calibration_output = overall_results.melt(id_vars=['model', 'seed', 'n_bins_scored'], value_vars=calibration_columns, var_name='calibration_column', value_name='calibration_count')
    calibration_output = calibration_output.assign(threshold=lambda data: (data['calibration_column'].str.replace('calibration_count_', '', regex=False).astype(float)),
            observed_rate=lambda data: (data['calibration_count'] / data['n_bins_scored']))

    # Getting mean and sd of calibration exceed
    calibration_output = calibration_output.groupby(['model', 'threshold'], as_index=False, sort=False).agg(
            observed_rate_mean=('observed_rate', 'mean'), seed_sd=('observed_rate', 'std'))
    
    return calibration_output

def get_user_type_summary(ll_full_results):
    '''
    Gets mean and seed sd of user log likelihood by model
    '''
    output = ll_full_results.loc[ll_full_results['breakdown_type'] == 'user_type', ['model', 'seed', 'breakdown_group', 'mean_ll']]
    output = output.rename(columns={'breakdown_group': 'user_type'})
    output = output.groupby(['user_type', 'model'], as_index=False, sort=False).agg(mean_ll=('mean_ll', 'mean'), seed_sd=('mean_ll', 'std'))
    output.loc[output['model'] != 'cluster_smoothing', 'seed_sd'] = np.nan

    return output

def get_all_model_performance_table(ll_full_results, benchmark_results):
    '''
    Table of mean log likelihood for 3 models and benchmarks
    '''
    ll_performance = ll_full_results.loc[ll_full_results['breakdown_type'] == 'overall', ['model', 'mean_ll']]
    ll_performance = ll_performance.groupby('model', as_index=False, sort=False).agg(mean_log_likelihood=('mean_ll', 'mean'), seed_sd=('mean_ll', 'std'))

    benchmark_performance = benchmark_results.loc[benchmark_results['breakdown_type'] == 'overall', ['model', 'mean_ll']]
    benchmark_performance = benchmark_performance.groupby('model', as_index=False, sort=False).agg(mean_log_likelihood=('mean_ll', 'mean'))
    benchmark_performance = benchmark_performance.assign(seed_sd=np.nan)

    overall_performance = pd.concat([ll_performance, benchmark_performance], ignore_index=True)
    overall_performance.loc[overall_performance['model'] != 'cluster_smoothing', 'seed_sd'] = np.nan

    return overall_performance

def prepare_model_comparisons(ll_user_results,overall_performance):
    '''
    Looks at log likelihood improvement by model and how that is distributed across the users
    '''

    # Getting model pairs for comparison
    model_pairs = pd.DataFrame({'m1': ['global_smoothing', 'cluster_smoothing', 'cluster_smoothing'],
                                'm2': ['no_smoothing', 'no_smoothing', 'global_smoothing']})

    # Getting mean log likelihood an number of bins scored by model
    user_model_results = ll_user_results.groupby(['user_idx', 'model'], as_index=False, sort=False).agg(
                                                    mean_log_likelihood=('mean_ll', 'mean'))

    # Getting two copies of the model table ad joining onto model pairs
    m1_scores = user_model_results.rename(columns={'model' : 'm1', 'mean_log_likelihood': 'm1_mean_log_likelihood'})
    m2_scores = user_model_results.rename(columns={'model': 'm2', 'mean_log_likelihood': 'm2_mean_log_likelihood'})
    model_pairs = model_pairs.merge(m1_scores[['user_idx', 'm1', 'm1_mean_log_likelihood']], on='m1', how='left')
    model_pairs = model_pairs.merge(m2_scores[['user_idx', 'm2', 'm2_mean_log_likelihood']], on=['user_idx', 'm2'], how='left')

    # Getting the users of bins that imporve
    model_pairs = model_pairs.assign(user_ll_difference=lambda data: data['m1_mean_log_likelihood'] - data['m2_mean_log_likelihood'],
                                     user_improved=lambda data: data['user_ll_difference'] > 0)
    model_pairs = model_pairs.groupby(['m1', 'm2'], as_index=False, sort=False).agg(
                                                    users_improved_pct=('user_improved', 'mean'),
                                                    median_user_ll_difference=('user_ll_difference', 'median'))
    model_pairs = model_pairs.assign(users_improved_pct=lambda data: 100 * data['users_improved_pct'])

    # Getting the log likelihood difference on a model level
    m1_ll = overall_performance[['model', 'mean_log_likelihood']].rename(columns={'model': 'm1', 'mean_log_likelihood': 'm1_mean_log_likelihood'})
    m2_ll = overall_performance[['model', 'mean_log_likelihood']].rename(columns={'model': 'm2', 'mean_log_likelihood': 'm2_mean_log_likelihood'})

    model_pairs = model_pairs.merge(m1_ll, on='m1', how='left').merge(m2_ll, on='m2', how='left')
    model_pairs = model_pairs.assign(mean_log_likelihood_difference=lambda data: data['m1_mean_log_likelihood'] - data['m2_mean_log_likelihood'])
    model_pairs = model_pairs[['m1', 'm2', 'mean_log_likelihood_difference', 'median_user_ll_difference', 'users_improved_pct']]

    return model_pairs


#####################################
# Final output prep
#####################################

def get_extreme_calibration_output(calibration):
    '''
    Gets the extreme calibration output table
    '''

    extreme_calibration = calibration.loc[calibration['threshold'].isin([1e-4, 1e-3, 1e-2])].sort_values('threshold')
    calibration_means = extreme_calibration.pivot_table(index='threshold', columns='model', values='observed_rate_mean', sort=False).add_suffix('_mean')
    calibration_sds = extreme_calibration.pivot_table(index='threshold', columns='model', values='seed_sd', sort=False).add_suffix('_sd')
    output = pd.concat([calibration_means, calibration_sds],axis=1).reset_index()
    output.columns.name = None

    return output

def get_user_type_output(user_type_summary):
    '''
    Pivots the user type table ready for output
    '''
    mean_log_likelihood = user_type_summary.pivot_table(index='user_type', columns='model', values='mean_ll', sort=False)
    cluster_seed_sd = user_type_summary.loc[user_type_summary['model'] == 'cluster_smoothing'].set_index('user_type')['seed_sd'].rename('cluster_smoothing_seed_sd')
    output = pd.concat([mean_log_likelihood, cluster_seed_sd], axis=1).reset_index()
    output.columns.name = None
    output['user_type'] = output['user_type'].str.title()

    return output

def get_overall_performance_output(overall_performance):
    '''
    Maps the label onto the performance table
    '''
    output = overall_performance.copy()
    output['model'] = output['model'].map(model_labels)
    return output

def get_model_comparison_output(model_comparisons):
    '''
    Gets the final model comparison table ready for output by adding model names
    '''
    output = model_comparisons.copy()
    output.insert(0, 'model_comparison', (output['m1'].map(model_labels) + ' - ' + output['m2'].map(model_labels)))

    return output.drop(columns=['m1', 'm2'])

#####################################
# Storage and loading
#####################################

def store_results(results, results_dir, filename):
    processed_dir = f'{results_dir}/processed'
    os.makedirs(processed_dir, exist_ok=True)
    results.to_parquet(f'{processed_dir}/{filename}.parquet',index=False)


def load_results(results_dir, filename):
    return pd.read_parquet(f'{results_dir}/processed/{filename}.parquet')