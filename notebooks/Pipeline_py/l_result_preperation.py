import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ll_model_names = ('no_smoothing', 'global_smoothing', 'cluster_smoothing')

benchmark_names = ('static_user_hurdle', 'static_user_hour_hurdle')

plot_labels = {'no_smoothing': 'No shrinkage', 
               'global_smoothing': 'Global shrinkage', 
               'cluster_smoothing': 'Cluster shrinkage', 
               'static_user_hurdle': 'User level benchmark', 
               'static_user_hour_hurdle': 'User hourly level benchmark'}


def safe_read(directory, expected_files=None):
    '''
    Reads parquet files in a directory
    '''
    if expected_files is not None:
        assert len(os.listdir(directory)) == expected_files

    return pd.read_parquet(directory)


def load_test_results(results_dir, n_test_seeds):
    '''
    Created dictionaries with the test results in them
    '''
    ll_model_results = {'no_smoothing': {'full': safe_read(directory=(f'{results_dir}/test/no_smoothing/full'), expected_files=n_test_seeds),
                                         'user': safe_read(directory=(f'{results_dir}/test/no_smoothing/user'), expected_files=1)},
                        'global_smoothing': {'full': safe_read(directory=(f'{results_dir}/test/global_smoothing/full'), expected_files=n_test_seeds),
                                             'user': safe_read(directory=(f'{results_dir}/test/global_smoothing/user'), expected_files=1)},
                        'cluster_smoothing': {'full': safe_read(directory=(f'{results_dir}/test/cluster_smoothing/full'), expected_files=n_test_seeds),
                                              'user': safe_read(directory=(f'{results_dir}/test/cluster_smoothing/user'), expected_files=n_test_seeds)}}

    # Splitting out the benchmark results to have the same structure as the three models
    benchmark_results = safe_read(directory=f'{results_dir}/benchmarks/full', expected_files=1)
    benchmark_results = {benchmark_name: benchmark_results.loc[benchmark_results['model_name'] == benchmark_name].copy() 
                        for benchmark_name in benchmark_names}

    return ll_model_results, benchmark_results


def add_mean_ll(results):
    '''
    Adds mean log likelihood to a pandas dataframe
    '''
    results = results.copy()
    results['mean_ll'] = results['non_degen_ll_sum'] / results['n_bins_scored']

    return results


def get_decile_improvements(run_results, decile):
    '''
    Gets the decile improvements and the mean and sd of improvements across seeds
    '''
    # Pivoting data to get global and cluster smoothing model side by side
    results_pvt = run_results.pivot(index=['seed', decile], columns='model', values='mean_ll').reset_index()

    # Create model difference to no smoothing
    improvements = results_pvt[['seed', decile]].copy()
    improvements['global_smoothing'] = results_pvt['global_smoothing'] - results_pvt['no_smoothing']
    improvements['cluster_smoothing'] = results_pvt['cluster_smoothing'] - results_pvt['no_smoothing']

    # Unpivot and get mean and sd of improvements
    improvements = improvements.melt(id_vars=['seed', decile], var_name='model', value_name='improvement')
    improvements = improvements.groupby([decile, 'model'], as_index=False,)['improvement'].agg(mean='mean', sd='std')

    return improvements


def prepare_overall_performance(ll_models, benchmarks):
    '''
    Prepares overall performance for the three LL models and
    two benchmarks.
    '''
    model_order = {
        'no_smoothing': 0,
        'global_smoothing': 1,
        'cluster_smoothing': 2,
        'static_user_hurdle': 3,
        'static_user_hour_hurdle': 4}

    # Init the output with the model and benchmark names
    output = pd.DataFrame({'model': list(ll_models.keys()) + list(benchmarks.keys())})
    output['label'] = output['model'].map(plot_labels)

    # Concat the benchmark and model results together and compute log likelihood stats
    log_likelihood_stats = pd.concat([data['full'].loc[data['full']['breakdown_type'] == 'overall'].assign(model=name)
                            for name, data in {**ll_models, **benchmarks}.items()], ignore_index=True)
    log_likelihood_stats = log_likelihood_stats.groupby('model')['mean_ll'].agg(mean_ll='mean', seed_sd='std').reset_index()

    # Join stats on to output
    output = output.merge(log_likelihood_stats, on='model', how='left')
    output['seed_sd'] = np.where(output['model'] == 'cluster_smoothing', output['seed_sd'], np.nan)

    # Add an order column
    output['order'] = output['model'].map(model_order)
    output = output.sort_values('order').drop(columns='order').reset_index(drop=True)
    return output

def get_paried_bootstrap_differences(ll_models, n_bootstrap_replicates, seed):
    '''
    Gets the paired bootstrap diffferences of:
    global - no
    cluster - no 
    cluster - global
    '''
    models = {}

    for name in ['no_smoothing', 'global_smoothing', 'cluster_smoothing']:

        # Iterate over the models and compute the log likelihood and sort by user noting we have multiple seeds for the cluster code
        user_results = ll_models[name]['user'].copy()
        user_results['mean_ll'] = user_results['non_degen_ll_sum'] / user_results['n_bins_scored']
        if name == 'cluster_smoothing':
            user_results = user_results.groupby('user_idx', as_index=False)[['mean_ll', 'n_bins_scored']].mean()
        models[name] = user_results.sort_values('user_idx')

    # Getting the values of user log liklihood and the weights for bootstrapping
    n_bins = models['no_smoothing']['n_bins_scored'].to_numpy()
    mean_ll = {name: model['mean_ll'].to_numpy() for name, model in models.items()}
    n_users = len(n_bins)

    rng = np.random.default_rng(seed)

    bootstrap_users = rng.choice(n_users, size=(n_bootstrap_replicates, n_users))
    
    weights = np.vstack([
        np.ones(n_users, dtype='int64'),
        weights,
    ])

    weighted_bins = weights * n_bins
    denom = weighted_bins.sum(axis=1)

    ll = {
        name: (weighted_bins @ values) / denom
        for name, values in mean_ll.items()
    }

    diffs = pd.DataFrame({
        'Global - no shrinkage':
            ll['global_smoothing'] - ll['no_smoothing'],
        'Cluster - no shrinkage':
            ll['cluster_smoothing'] - ll['no_smoothing'],
        'Cluster - global shrinkage':
            ll['cluster_smoothing'] - ll['global_smoothing'],
    })

    return pd.DataFrame({
        'comparison': diffs.columns,
        'estimate': diffs.iloc[0].to_numpy(),
        'ci_lower': diffs.iloc[1:].quantile(0.025).to_numpy(),
        'ci_upper': diffs.iloc[1:].quantile(0.975).to_numpy(),
    })