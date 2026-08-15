import os

import numpy as np
import pandas as pd


ll_model_names = ('no_smoothing', 'global_smoothing', 'cluster_smoothing')

model_labels = {
        'no_smoothing': 'No shrinkage',
        'global_smoothing': 'Global shrinkage',
        'cluster_smoothing': 'Cluster shrinkage',
        'static_user_hurdle': 'User level benchmark',
        'static_user_hour_hurdle': 'Hourly user benchmark'}

def read_all(directory, expected_files=None):
    '''
    Reads all parquet files in a directory
    '''
    if expected_files is not None:
        assert len(os.listdir(directory)) == expected_files
    return pd.read_parquet(directory)

def load_test_results(results_dir, n_test_seeds):
    '''
    Loads all the results and concats them into a big dataframe adds the mean log likelihood column
    '''
    # Creating three long dataframes with the reruns in them
    ll_calibration_results = pd.concat([read_all(directory=f'{results_dir}/test/{model}/calibration', expected_files=n_test_seeds).assign(model=model)
                                 for model in ll_model_names], ignore_index=True)

    ll_user_results = pd.concat([read_all(directory=f'{results_dir}/test/{model}/user', expected_files=(n_test_seeds if model == 'cluster_smoothing' else 1)).assign(model=model) 
                                 for model in ll_model_names],ignore_index=True)

    benchmark_results = read_all(directory=f'{results_dir}/benchmarks/calibration', expected_files=1).rename(columns={'model_name': 'model'})

    # Adding in mean log likelihood column
    ll_calibration_results['mean_ll'] = ll_calibration_results['non_degen_ll_sum'] / ll_calibration_results['n_bins_scored']
    ll_user_results['mean_ll'] = ll_user_results['non_degen_ll_sum'] / ll_user_results['n_bins_scored']

    benchmark_results['mean_ll'] = benchmark_results['non_degen_ll_sum'] / benchmark_results['n_bins_scored']

    return ll_calibration_results, ll_user_results, benchmark_results

def summarise_decile_improvements(run_results, decile_column):
    '''
    Calculates mean and sd improvement for each model for the two deciles:
    activity and distance
    '''
    # Getting the ll run results and merging onto the othe rmodels
    no_smoothing = run_results.loc[run_results['model'] == 'no_smoothing', ['seed', decile_column, 'mean_ll']].rename(columns={'mean_ll': 'no_smoothing'})
    improvements = run_results.loc[run_results['model'] != 'no_smoothing', ['seed', decile_column, 'model', 'mean_ll']].merge(no_smoothing, on=['seed', decile_column])

    improvements['improvement'] = improvements['mean_ll'] - improvements['no_smoothing']
    improvements['relative_improvement'] = (100 * improvements['improvement'] / improvements['no_smoothing'].abs())
    improvements = improvements.groupby([decile_column, 'model'], as_index=False).agg(
        mean_improvement=('improvement', 'mean'),
        seed_sd=('improvement', 'std'),
        mean_relative_improvement=('relative_improvement', 'mean'),
        relative_seed_sd=('relative_improvement', 'std'))
    return improvements

def get_activity_decile_improvement(ll_calibration_results):
    '''
    gets improvement by activity decile
    '''
    run_results = ll_calibration_results.loc[ll_calibration_results['breakdown_type'] == 'activity_decile', 
                                      ['seed', 'breakdown_group', 'model', 'mean_ll']]
    run_results = run_results.rename(columns={'breakdown_group': 'activity_decile'})

    output = summarise_decile_improvements(run_results=run_results, decile_column='activity_decile')

    # Altering output form
    output.insert(0, 'breakdown', 'activity')
    output = output.rename(columns={'activity_decile': 'decile'})
    output['decile'] = output['decile'].astype('int8')

    return output

def get_calibration_summary(ll_calibration_results):
    '''
    Gets average and sd of calibraiton results across different seeds
    '''

    # Getting overall results
    overall_results = ll_calibration_results.loc[ll_calibration_results['breakdown_type'] == 'overall'].copy()

    # Unpivoting the calibration count columns and getting threshold and great
    calibration_columns = [column for column in overall_results.columns if column.startswith('calibration_count_')]
    calibration_output = overall_results.melt(id_vars=['model', 'seed', 'n_bins_scored'], value_vars=calibration_columns, var_name='calibration_column', value_name='calibration_count')
    calibration_output['threshold'] = calibration_output['calibration_column'].str.replace('calibration_count_', '', regex=False).astype(float)
    calibration_output['observed_rate'] = calibration_output['calibration_count'] / calibration_output['n_bins_scored']

    # Getting mean and sd of calibration exceed
    calibration_output = calibration_output.groupby(['model', 'threshold'], as_index=False, sort=False).agg(
            observed_rate_mean=('observed_rate', 'mean'), seed_sd=('observed_rate', 'std'))
    
    return calibration_output

def get_all_model_performance_table(ll_calibration_results, benchmark_results):
    '''
    Table of mean log likelihood for 3 models and benchmarks
    '''
    ll_performance = ll_calibration_results.loc[ll_calibration_results['breakdown_type'] == 'overall', ['model', 'mean_ll']]
    ll_performance = ll_performance.groupby('model', as_index=False, sort=False).agg(mean_log_likelihood=('mean_ll', 'mean'), seed_sd=('mean_ll', 'std'))

    benchmark_performance = benchmark_results.loc[benchmark_results['breakdown_type'] == 'overall', ['model', 'mean_ll']]
    benchmark_performance = benchmark_performance.groupby('model', as_index=False, sort=False).agg(mean_log_likelihood=('mean_ll', 'mean'))
    benchmark_performance['seed_sd'] = np.nan

    overall_performance = pd.concat([ll_performance, benchmark_performance], ignore_index=True)
    overall_performance.loc[overall_performance['model'] != 'cluster_smoothing', 'seed_sd'] = np.nan

    return overall_performance

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

def format_numeric_cols(table, columns):
    '''
    Rounds selected cols to 5pd keeping trailing zeros
    '''
    for column in columns:
        table[column] = table[column].apply(lambda value: f'{value:.5f}')
    return table


def format_mean_and_sd(mean, sd):
    '''
    Combines a mean with \pm sd if aplicable
    '''
    if pd.notna(sd):
        return f'{mean:.5f} $\pm$ {sd:.5f}'
    else: 
        return mean

def get_overall_performance_output(overall_performance, user_type_summary):
    '''
    Gets the initial summary table
    '''
    # Getting human and machine results for each model
    human_results = user_type_summary.loc[user_type_summary['user_type'] == 'human', ['model', 'mean_ll', 'seed_sd']]
    human_results = human_results.rename(columns={'mean_ll': 'human_mean_log_likelihood', 'seed_sd': 'human_seed_sd'})

    machine_results = user_type_summary.loc[user_type_summary['user_type'] == 'machine', ['model', 'mean_ll', 'seed_sd']]
    machine_results = machine_results.rename(columns={'mean_ll': 'machine_mean_log_likelihood', 'seed_sd': 'machine_seed_sd'})

    output = overall_performance.merge(human_results, on='model', how='left').merge(machine_results, on='model', how='left')
    output = output.sort_values(by='mean_log_likelihood', ascending=False).reset_index(drop=True)
    output.columns.name = None

    # Get the raw model score and add the difference
    unsmoothing_ll_lpmf = output.loc[output['model'] == 'no_smoothing', 'mean_log_likelihood'].iloc[0]
    output['$\Delta$ LL'] = output['mean_log_likelihood'] - unsmoothing_ll_lpmf
    output['$\Delta$ LL'] = output['$\Delta$ LL'].map(lambda x: f'{x:+.5f}')

    output.loc[output['model'] == 'no_smoothing', '$\Delta$ LL'] = '-'
    output['model'] = output['model'].map(model_labels)

    output['Overall LL'] = output.apply(lambda row: format_mean_and_sd(row['mean_log_likelihood'], row['seed_sd']), axis=1)
    output['Human LL'] = output.apply(lambda row: format_mean_and_sd(row['human_mean_log_likelihood'], row['human_seed_sd']), axis=1)
    output['Machine LL'] = output.apply(lambda row: format_mean_and_sd(row['machine_mean_log_likelihood'], row['machine_seed_sd']), axis=1)

    output = output[['model', 'Overall LL', '$\Delta$ LL', 'Human LL', 'Machine LL']].rename(columns={'model': 'Model'})
    return output

#####################################
# Detection outputs
#####################################

def get_attack_detection_summary(results_dir, n_test_seeds):
    '''
    Calculatates detection results mean and sd cross multiple seeds
    '''
    detection_results = pd.concat([read_all(directory=f'{results_dir}/test/{model}/detection', expected_files=n_test_seeds) 
                                   for model in ll_model_names], ignore_index=True)

    output = detection_results.copy()
    output['detection_rate'] = output['n_detected'] / output['n_attacks']
    output = output.rename(columns={'experiment_name': 'model'})

    output = output.groupby(['model', 'alert_w', 'fpr_rate', 'attack_size'], as_index=False, sort=False).agg(
        threshold_mean=('threshold', 'mean'), threshold_sd=('threshold', 'std'), 
        detection_mean=('detection_rate', 'mean'), detection_sd=('detection_rate', 'std'))

    return output

def get_mean_threshold_differences(detection_results):
    '''
    get the threshold differences from the unsmoothed ll model
    '''
    # Dropping the attack strenght col
    output = detection_results[['model', 'alert_w', 'fpr_rate', 'threshold_mean']].drop_duplicates()
    no_shrink_threshold = output.loc[output['model'] == 'no_smoothing', ['alert_w', 'fpr_rate', 'threshold_mean']].rename(columns={'threshold_mean': 'll_threshold'})
    output = output.loc[output['model'] != 'no_smoothing'].merge(no_shrink_threshold, on=['alert_w', 'fpr_rate'])
    output['threshold_difference'] = output['threshold_mean'] - output['ll_threshold']
    output = output.drop(columns='threshold_mean')
    return output

def get_user_pct_ll_improvement(ll_user_results):
    '''
    gets the % improvement in loglikelihood for each user
    '''
    # Getting mean ll across seeds
    ll_user_results = ll_user_results.groupby(['user_idx', 'model'], as_index=False, 
                                              sort=False).agg(mean_ll=('mean_ll', 'mean'))
    
    # getting the mean ll for the the no shrinkage ll model
    no_shrinkage_df = ll_user_results.loc[ll_user_results['model'] == 'no_smoothing', ['user_idx', 'mean_ll']]
    no_shrinkage_df = no_shrinkage_df.rename(columns={'mean_ll': 'unsmooth_ll'})

    # Getting pct imprvement
    ll_user_results = ll_user_results.merge(no_shrinkage_df, on='user_idx', how='left')
    ll_user_results = ll_user_results[ll_user_results['model'] != 'no_smoothing'].copy()
    ll_user_results['pct_improvement'] = ((ll_user_results['mean_ll'] - ll_user_results['unsmooth_ll'])/ ll_user_results['unsmooth_ll'].abs() * 100)

    return ll_user_results[['user_idx', 'model', 'pct_improvement']]

#####################################
# Storage and loading
#####################################

def store_results(results, results_dir, filename):
    processed_dir = f'{results_dir}/processed'
    os.makedirs(processed_dir, exist_ok=True)
    results.to_parquet(f'{processed_dir}/{filename}.parquet',index=False)


def load_results(results_dir, filename):
    return pd.read_parquet(f'{results_dir}/processed/{filename}.parquet')