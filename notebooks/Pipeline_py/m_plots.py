import matplotlib.pyplot as plt
from l_result_preperation import model_labels
import seaborn as sns


def declile_log_likelihood_plot(decile_results, x_label, relative):
    '''
    Log likelihood decile plot used for both activity and distance
    '''
    fig, ax = plt.subplots(figsize=(6, 4.9))

    # Setting config and label
    if relative:
        mean_col = 'mean_relative_improvement'
        sd_col = 'relative_seed_sd'
        y_label = 'Mean Log-Likelihood Improvement (%)'
    else:
        mean_col = 'mean_improvement'
        sd_col = 'seed_sd'
        y_label = 'Mean Log-Likelihood Improvement'

    for model in ('global_smoothing', 'cluster_smoothing'):

        # Getting results for that model
        results = decile_results.loc[decile_results['model'] == model].sort_values('decile')

        ax.errorbar(results['decile'], results[mean_col], yerr=results[sd_col], fmt='o', 
                    linestyle='none', capsize=2, label=model_labels[model])

    ax.axhline(0, linewidth=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(1, 11))
    add_legend(ax)

    fig.tight_layout()

    return fig, ax


def plot_calibration(calibration_df, extreme):
    '''
    Creates calibration and extreme calibration plots
    '''
    if extreme:
        relevant_thresholds = calibration_df.loc[calibration_df['threshold'].isin([1e-4, 1e-3, 1e-2])]
    else:
        relevant_thresholds = calibration_df.loc[calibration_df['threshold'] >= 0.1]

    fig, ax = plt.subplots(figsize=(6, 4.9))

    for model in ('no_smoothing', 'global_smoothing', 'cluster_smoothing'):
        results = relevant_thresholds.loc[relevant_thresholds['model'] == model].sort_values('threshold')

        ax.errorbar(results['threshold'], results['observed_rate_mean'], 
                    yerr=results['seed_sd'], marker='o', markersize=2, capsize=4, label=model_labels[model])

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

    ax.set_xlabel('Nominal rate')
    ax.set_ylabel('Observed rate')
    add_legend(ax)

    fig.tight_layout()

    return fig, ax

def add_legend(ax, position='best'):
    legend_args = {'frameon': True, 'fancybox': False, 'edgecolor': 'black'}
    if position == 'outside':
        ax.legend( loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, **legend_args)
    else:
        ax.legend(loc=position, **legend_args)

def plot_attack_detection(detection_results):
    '''
    Plots the attack detection rates
    '''
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey='row')

    for row, fpr_rate in enumerate(runtime_configs['fpr_rates']):
        for col, alert_w in enumerate(runtime_configs['alert_ws']):
            for model_name in ('no_smoothing', 'global_smoothing', 'cluster_smoothing'):
                    results = detection_results.loc[(detection_results['model'] == model_name) & 
                                                  (detection_results['alert_w'] == alert_w) & 
                                                  (detection_results['fpr_rate'] == fpr_rate)]
                    results = results.sort_values('attack_size')

                    ax.errorbar(results['attack_size'], results['detection_mean'], yerr=results['detection_sd'], 
                                marker='o', capsize=2, label=model_labels[model_name])

            if row == 0:
                ax.set_title(f'w={alert_w}')
            if col == 0:
                ax.set_ylabel(f'FPR = {fpr_rate}')
            if row == 2:
                ax.set_xlabel('Attack Strength')
            ax.set_ylim(bottom=0)
    add_legend(axes[0, 0])
    fig.tight_layout()

    return fig, axes

def plot_threshold_heatmap(threshold_differences):
    '''
    heatmap of differences of EWMA thresholds from the no smoothing ll model.
    '''

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    vmax = threshold_differences['threshold_difference'].abs().max()

    for ax, model in zip(axes, ('global_smoothing', 'cluster_smoothing')):
        # Get the results for that model and and the unsmoothed ll model 
        results = threshold_differences.loc[threshold_differences['model'] == model]
        threshold_difference = results.pivot(index='fpr_rate', columns='alert_w', values='threshold_difference').loc[runtime_configs['fpr_rates'], runtime_configs['alert_ws']]
        ll_threshold = results.pivot(index='fpr_rate', columns='alert_w', values='ll_threshold').loc[runtime_configs['fpr_rates'], runtime_configs['alert_ws']]
        heatmap_cell_annot = threshold_difference.copy().astype(str)

        for fpr in runtime_configs['fpr_rates']:
            for w in runtime_configs['alert_ws']:
                heatmap_cell_annot.loc[fpr, w] = (f'{threshold_difference.loc[fpr, w]} No = {ll_threshold.loc[fpr, w]}')

        sns.heatmap(threshold_difference, ax=ax, annot=heatmap_cell_annot, fmt='', cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax)

        ax.set_title(model_labels[model])
        ax.set_xlabel('w')
        ax.set_ylabel('FPR')

    fig.tight_layout()

    return fig, axes

def plot_user_pct_ll_improvemet(user_improvements):
    '''
    % improvement in log likelihood plot by users
    '''
    fig, ax = plt.subplots(figsize=(6, 4.9))
    for model in ('global_smoothing', 'cluster_smoothing'):
        results = user_improvements.loc[user_improvements['model'] == model]
        results = results.sort_values('pct_improvement', ascending=False).reset_index(drop=True)

        user_prop_rank = (results.index + 1) / len(results)

        ax.plot(user_prop_rank, results['pct_improvement'], label=model_labels[model])

        ax.axhline(0, linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-10, 20)

        ax.set_xlabel('Proportion of users')
        ax.set_ylabel('Percentage Improvement in Log-Likelihood')

        add_legend(ax)

        fig.tight_layout()

        return fig, ax

