import matplotlib.pyplot as plt
from l_result_preperation import model_labels


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
        model_results = decile_results.loc[decile_results['model'] == model].sort_values('decile')

        ax.errorbar(model_results['decile'], model_results[mean_col], yerr=model_results[sd_col], fmt='o', 
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
    Plots the calibration at deciles 0.1-0.9
    '''
    if extreme:
        relevant_thresholds = calibration_df.loc[calibration_df['threshold'].isin([1e-4, 1e-3, 1e-2])]
    else:
        relevant_thresholds = calibration_df.loc[calibration_df['threshold'] >= 0.1]

    fig, ax = plt.subplots(figsize=(6, 4.9))

    for model in ('no_smoothing', 'global_smoothing', 'cluster_smoothing'):
        model_results = relevant_thresholds.loc[relevant_thresholds['model'] == model].sort_values('threshold')

        # Adds in error bars for uniform randomness
        ax.errorbar(model_results['threshold'], model_results['observed_rate_mean'], 
                    yerr=model_results['seed_sd'], marker='o', markersize=2, capsize=4, label=model_labels[model])

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
        ax.legend(loc=position, **legend_args)# fig.savefig(
#     f'{outputs_dir}/activity_decile_relative.pdf',
#     bbox_inches='tight',
#     pad_inches=0.02
# )

plt.show()