import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from l_result_preperation import model_labels
import seaborn as sns
from l_result_preperation import model_labels, ll_model_names, format_mean_and_sd
import numpy as np
import pandas as pd


def declile_log_likelihood_plot(decile_results, x_label, error_bars=True):
    '''
    Log likelihood decile plot used for both activity and distance
    '''
    fig, ax = plt.subplots(figsize=(6, 4.9))

    for model in ('global_smoothing', 'cluster_smoothing'):

        # Getting results for that model
        results = decile_results.loc[decile_results['model'] == model].sort_values('decile')

        if model == 'cluster_smoothing':
            ax.errorbar(results['decile'], results['mean_relative_improvement'], linestyle='none', 
                            yerr=results['relative_seed_sd'], marker='o', capsize=2, label=model_labels[model])
        else: 
            ax.plot(results['decile'], results['mean_relative_improvement'], linestyle='none', marker='o', label=model_labels[model])

    ax.axhline(0, linewidth=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel('CMLL Change (%)')
    ax.set_xticks(range(1, 11))
    add_legend(ax)

    fig.tight_layout()

    return fig, ax


def plot_calibration(calibration_df, extreme, error_bars=True):
    '''
    Creates calibration and extreme calibration plots
    '''
    if extreme:
        relevant_thresholds = calibration_df.loc[calibration_df['threshold'] < 0.1]
    else:
        relevant_thresholds = calibration_df.loc[calibration_df['threshold'] >= 0.1]

    fig, ax = plt.subplots(figsize=(6, 4.9))

    for model in ('no_smoothing', 'global_smoothing', 'cluster_smoothing'):
        results = relevant_thresholds.loc[relevant_thresholds['model'] == model].sort_values('threshold')

        if error_bars:
            ax.errorbar(results['threshold'], results['observed_rate_mean'], 
                        yerr=results['seed_sd'], marker='o', capsize=2, label=model_labels[model])
        else:
            ax.plot(results['threshold'], results['observed_rate_mean'], marker='o', label=model_labels[model])

    # Perfect calibration line
    if extreme:
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Getting lower and upper limit for the straight line
        lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
        upper = max(ax.get_xlim()[1], ax.get_ylim()[1])

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.plot([lower, upper], [lower, upper], linestyle='--', linewidth=0.8, label='Perfect calibration')
    else: 
        ax.plot([0, 1], [0, 1], linestyle='--', linewidth=0.8, label='Perfect calibration')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_xlabel('P-Value')
    ax.set_ylabel('Observed rate')
    add_legend(ax)

    fig.tight_layout()

    return fig, ax

def add_legend(ax, position='best'):
    legend_args = {'frameon': True, 'fancybox': False, 'edgecolor': 'black'}
    if position == 'outside':
        ax.legend( loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, **legend_args)
    else:
        ax.legend(loc=position, **legend_args)

def plot_full_attack_detection(detection_results, runtime_configs, error_bars=True):
    '''
    Plots the attack detection rates
    '''
    fig, axes_array = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey='row')

    for row, fpr_rate in enumerate(runtime_configs['fpr_rates']):
        for col, alert_w in enumerate(runtime_configs['alert_w_vals']):
            ax = axes_array[row, col]
            for model_name in ('no_smoothing', 'global_smoothing', 'cluster_smoothing'):
                    results = detection_results.loc[(detection_results['model'] == model_name) & 
                                                  (detection_results['alert_w'] == alert_w) & 
                                                  (detection_results['fpr_rate'] == fpr_rate)]
                    results = results.sort_values('attack_size')
                    results['attack_size'] = results['attack_size'] // 12

                    if error_bars:
                        ax.errorbar(results['attack_size'], results['detection_mean'], yerr=results['detection_sd'], 
                                    marker='o', capsize=2, label=model_labels[model_name])
                    else:
                        ax.plot(results['attack_size'], results['detection_mean'], marker='o', label=model_labels[model_name])

            if row == 0:
                ax.set_title(f'\u03BB={alert_w}')
            if col == 0:
                ax.set_ylabel(f'FPR = {format_fpr_exponent(fpr_rate)}')
            if row == 2:
                ax.set_xlabel('Additional Counts')
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xscale('log', base=2)
    
    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95 ,wspace=0.12, hspace=0.12)
    axes_array[2, 0].set_xticks(results['attack_size'].unique())
    axes_array[2,0].set_xticklabels(results['attack_size'].unique())

    return fig, axes_array

def plot_small_attack_detection(detection_results, runtime_configs):
    '''
    Plots the attack detection rates
    '''
    fig, axes_array = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey='row')
    results = detection_results.copy()
    results['attack_size'] = results['attack_size'] // 12
    results = results.loc[results['attack_size'].isin([4, 32])]
    x = np.array([1,2])
    bar_width = 0.25

    for row, fpr_rate in enumerate(runtime_configs['fpr_rates']):
        for col, alert_w in enumerate(runtime_configs['alert_w_vals']):
            ax = axes_array[row, col]
            for model_name, bar_offset in zip(('no_smoothing', 'cluster_smoothing', 'global_smoothing'), 
                                              (-bar_width, 0, bar_width)):
                    
                    model_results = results.loc[(results['model'] == model_name) & 
                                                  (results['alert_w'] == alert_w) & 
                                                  (results['fpr_rate'] == fpr_rate)]
                    model_results = model_results.sort_values('attack_size')

                    bars = ax.bar(x + bar_offset, model_results['detection_mean'], 
                                bar_width, label=model_labels[model_name], edgecolor='black')
                    labels = [f'{value * 100:.2g}%' for value in model_results['detection_mean']]
                    ax.bar_label(bars, labels=labels, padding=2, fontsize=8, rotation=45)
            if row == 0:
                ax.set_title(f'\u03BB={alert_w}')
            if col == 0:
                ax.set_ylabel(f'FPR = {format_fpr_exponent(fpr_rate)}')
            if row == 2:
                ax.set_xlabel('Additional Counts')

        axes_array[row, 0].yaxis.set_major_formatter(PercentFormatter(1))

    axes_array[0, 0].set_ylim((0, 0.92))
    axes_array[1, 0].set_ylim((0, 0.23))
    axes_array[2, 0].set_ylim((0, 0.031))

                
    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95 ,wspace=0.12, hspace=0.12)
    axes_array[2, 0].set_xticks(x)
    axes_array[2,0].set_xticklabels([4, 32])

    return fig, axes_array


def plot_threshold_heatmap(threshold_differences, model):
    '''
    heatmap of differences of EWMA thresholds from the no smoothing ll model.
    '''
    fig, ax = plt.subplots(figsize=(4, 3.5))
    vmax = threshold_differences['threshold_difference'].abs().max()

    # Get the results for that model and and the unsmoothed ll model 
    results = threshold_differences.loc[threshold_differences['model'] == model]
    threshold_difference = results.pivot(index='fpr_rate', columns='alert_w', values='threshold_difference').sort_index(ascending=False)
    ll_threshold = results.pivot(index='fpr_rate', columns='alert_w', values='ll_threshold').sort_index(ascending=False)
    heatmap_cell_annot = threshold_difference.copy().astype(str)

    # Creating heatmap cell annotations
    for fpr in threshold_difference.index:
        for w in threshold_difference.columns:
            heatmap_cell_annot.loc[fpr, w] = f'{threshold_difference.loc[fpr, w]:.2f} \n No = {ll_threshold.loc[fpr, w]:.2f}'

    sns.heatmap(threshold_difference, ax=ax, annot=heatmap_cell_annot, fmt='', cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                cbar_kws={'label' : 'Threshold \u0394'})

    ax.set_xlabel('\u03BB')
    ax.set_ylabel('FPR')
    fig.tight_layout()
    ax.set_yticklabels([format_fpr_exponent(fpr) for fpr in threshold_difference.index])

    return fig, ax

def plot_user_pct_ll_improvemet(user_improvements):
    '''
    % improvement in log likelihood plot by users
    '''
    shrinkage_models = ll_model_names[1:]
    fig, ax = plt.subplots(figsize=(6, 4.9))
    for model in shrinkage_models:
        results = user_improvements.loc[user_improvements['model'] == model]
        results = results.sort_values('pct_improvement', ascending=False).reset_index(drop=True)

        user_prop_rank = (results.index + 1) / len(results)

        ax.plot(user_prop_rank, results['pct_improvement'], label=model_labels[model])

    ax.axhline(0, linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-5, 10)

    ax.set_xlabel('Proportion of users')
    ax.set_ylabel('CMLL Change (%)')

    add_legend(ax)

    fig.tight_layout()

    return fig, ax

def format_fpr_exponent(fpr):
    exponent = int(f'{fpr:.0e}'.split('e')[1])
    return rf'$10^{{{exponent}}}$'

def get_overall_performance_output(ll_calibration_results, benchmark_results):
    '''
    Gets the initial summary table
    '''
    results = pd.concat([ll_calibration_results[['model', 'breakdown_group', 'mean_ll', 'breakdown_type']],
                         benchmark_results[['model', 'breakdown_group', 'mean_ll', 'breakdown_type']]])
    overall_performance = results.loc[results['breakdown_type'] == 'overall', ['model', 'mean_ll']]
    overall_performance = overall_performance.groupby('model', as_index=False, sort=False).agg(mean_log_likelihood=('mean_ll', 'mean'), seed_sd=('mean_ll', 'std'))
    overall_performance.loc[overall_performance['model'] != 'cluster_smoothing', 'seed_sd'] = np.nan

    
    # Getting human and machine results for each model
    human_machine_results = results.loc[results['breakdown_type'] == 'user_type', ['model', 'breakdown_group', 'mean_ll']]
    human_machine_results = human_machine_results.groupby(['model', 'breakdown_group'], as_index=False).agg(mean_ll=('mean_ll', 'mean'), seed_sd=('mean_ll', 'std'))

    human_results = human_machine_results.loc[human_machine_results['breakdown_group'] == 'human', ['model', 'mean_ll', 'seed_sd']]
    human_results = human_results.rename(columns={'mean_ll': 'human_mean_log_likelihood', 'seed_sd': 'human_seed_sd'})
    machine_results = human_machine_results.loc[human_machine_results['breakdown_group'] == 'machine', ['model', 'mean_ll', 'seed_sd']]
    machine_results = machine_results.rename(columns={'mean_ll': 'machine_mean_log_likelihood', 'seed_sd': 'machine_seed_sd'})

    output = overall_performance.merge(human_results, on='model', how='left').merge(machine_results, on='model', how='left')
    output = output.sort_values(by='mean_log_likelihood', ascending=False).reset_index(drop=True)
    output.columns.name = None

    # Get the raw model score and add the difference
    unsmoothing_ll_lpmf = output.loc[output['model'] == 'no_smoothing', 'mean_log_likelihood'].iloc[0]
    output['$\Delta$ CMLL'] = output.apply(lambda row: format_mean_and_sd(row['mean_log_likelihood'] - unsmoothing_ll_lpmf, row['seed_sd']), axis=1)

    output.loc[output['model'] == 'no_smoothing', '$\Delta$ CMLL'] = '-'
    output['model'] = output['model'].map(model_labels)

    output['Overall CMLL'] = output.apply(lambda row: format_mean_and_sd(row['mean_log_likelihood'], row['seed_sd']), axis=1)
    output['Human CMLL'] = output.apply(lambda row: format_mean_and_sd(row['human_mean_log_likelihood'], row['human_seed_sd']), axis=1)
    output['Machine CMLL'] = output.apply(lambda row: format_mean_and_sd(row['machine_mean_log_likelihood'], row['machine_seed_sd']), axis=1)

    output = output[['model', 'Overall CMLL', '$\Delta$ CMLL', 'Human CMLL', 'Machine CMLL']].rename(columns={'model': 'Model'})
    return output