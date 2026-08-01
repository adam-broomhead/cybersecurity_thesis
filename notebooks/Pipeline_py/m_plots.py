import matplotlib.pyplot as plt


def declile_log_likelihood_plot(decile_results, x_label):
    '''
    Log likelihood decile plot used for both activity
    '''
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for model in ('global_smoothing', 'cluster_smoothing',):
        model_results = decile_results.loc[decile_results['model'] == model].sort_values('decile')

        # Adding in error bars for the clustered model
        ax.errorbar(model_results['decile'], model_results['mean_improvement'], yerr=model_results['seed_sd'], 
                    fmt='o', linestyle='none', capsize=2, label=model_labels[model])

    # Adding in 0 improvement line
    ax.axhline(0, linewidth=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Mean Log-Likelihood Improvement')
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False)
    fig.tight_layout()

    return fig, ax


def plot_calibration(calibration_df):
    '''
    Plots the calibration at deciles 0.1-0.9
    '''
    plot_results = calibration_df.loc[calibration_df['threshold'] >= 0.1]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for model in ('no_smoothing', 'global_smoothing', 'cluster_smoothing'):
        model_results = plot_results.loc[plot_results['model'] == model].sort_values('threshold')

        # Adds in error bars for uniform randomness
        ax.errorbar(model_results['threshold'], model_results['observed_rate_mean'], 
                    yerr=model_results['seed_sd'], marker='o', capsize=2, label=model_labels[model])

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=0.8, label='Perfect calibration')

    ax.set_xlabel('Nominal rate')
    ax.set_ylabel('Observed rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    fig.tight_layout()

    return fig, ax