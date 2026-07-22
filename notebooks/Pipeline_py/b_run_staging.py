import numpy as np 
import polars as pl 
import utils as ut 
from collections import namedtuple

static_configs = ut.load_json5('static_configs')

#####################################
# Getting inital grids
#####################################

def get_period_sums(user_counts, period_start, period_end):
    ''' 
    Gets the sums of a df in the given period, the 

    This data is used to get the initial parameter estimates
    It is also used to create the grids for clustering
    '''

    period_df = user_counts.filter((pl.col('fine_bin_id') >= period_start) & (pl.col('fine_bin_id') < period_end))

    period_df = period_df.with_columns(count=pl.col('count').cast(pl.Float64))
    period_df = period_df.with_columns(count_2 = pl.col('count') ** 2)
    period_df = period_df.group_by(['user_id', 'coarse_bin_id']).agg(pl.sum('count').alias('sum_cnt'), 
                                                                    pl.sum('count_2').alias('sum_cnt_2'),
                                                                    pl.len().alias('n_bins'))
    
    return period_df 

def get_fine_bins_per_cb(period_start, period_end, bin_metric_dict):
    ''' 
    Returns the number of fine bins per coarse bin in the period
    '''
    return ((period_end - period_start)//bin_metric_dict['fine_bins_per_week']) * bin_metric_dict['fine_bins_per_coarse_bin']


def init_grid_NB(user_counts, n_users, coarse_bins_per_week, period_start, period_end, bin_metric_dict):
    ''' 
    Creates the u and v init grids from the training data
    '''
    # Getting the sum of counts and sum of counts squared needed for mean and variance calculations
    train_df = get_period_sums(user_counts, period_start, period_end)
    fb_per_cb = get_fine_bins_per_cb(period_start, period_end, bin_metric_dict)
    
    # Init a grid of parmeters to use
    u_init = np.zeros((n_users, coarse_bins_per_week), dtype='float64')
    v_init = np.zeros((n_users, coarse_bins_per_week), dtype='float64')

    # Extrating the entries to assign and assigning them to the df
    entries_to_assign = train_df.select(['user_id', 'coarse_bin_id']).to_numpy()

    u_init[entries_to_assign[:,0], entries_to_assign[:,1]] = train_df['sum_cnt'].to_numpy() / fb_per_cb
    v_init[entries_to_assign[:,0], entries_to_assign[:,1]] = (train_df['sum_cnt_2'].to_numpy() - 
                                                            ((train_df['sum_cnt'] **2) / fb_per_cb))/(fb_per_cb - 1)

    # Capping the min values of u_init and v_init
    u_init = np.maximum(u_init, static_configs['mean_min'])
    v_init = np.maximum(v_init, static_configs['var_min'])

    return u_init, v_init

def init_grid_hurdle(user_counts, n_users, coarse_bins_per_week, period_start, period_end, bin_metric_dict):

    # Getting the sum of counts and sum of counts squared needed for mean and variance calculations
    train_df = get_period_sums(user_counts, period_start, period_end)
    fb_per_cb = get_fine_bins_per_cb(period_start, period_end, bin_metric_dict)
    
    # Init a grid of parmeters to use
    u_init = np.zeros((n_users, coarse_bins_per_week), dtype='float64')
    v_init = np.zeros((n_users, coarse_bins_per_week), dtype='float64')
    p_init = np.zeros((n_users, coarse_bins_per_week), dtype='float64')

    # Extrating the entries to assign and assigning them to the df
    entries_to_assign = train_df.select(['user_id', 'coarse_bin_id']).to_numpy()

    # Getting columns as numpy arrays
    n_bins = train_df['n_bins'].to_numpy()
    sum_cnt = train_df['sum_cnt'].to_numpy()
    sum_cnt_2 = train_df['sum_cnt_2'].to_numpy()

    # shifted sums (as hurdle model is 1+ nb)
    sum_cnt_shifted = sum_cnt - n_bins
    sum_cnt_2_shifted = sum_cnt_2 - (2 * sum_cnt) + n_bins

    # Creating a mask we use to assign values
    mean_mask = n_bins > 0
    var_mask = n_bins > 1

    p_init[entries_to_assign[:,0], entries_to_assign[:,1]] = n_bins / fb_per_cb
    u_init[entries_to_assign[mean_mask, 0], entries_to_assign[mean_mask, 1]] = sum_cnt_shifted[mean_mask]/ n_bins[mean_mask]
    v_init[entries_to_assign[var_mask, 0], entries_to_assign[var_mask, 1]] = (sum_cnt_2_shifted[var_mask] - (sum_cnt_shifted[var_mask] ** 2 / n_bins[var_mask])
                                                                              ) / (n_bins[var_mask] - 1)

    # Capping the min values of u_init and v_init
    u_init = np.maximum(u_init, static_configs['mean_min'])
    v_init = np.maximum(v_init, static_configs['var_min'])
    p_init = np.minimum(np.maximum(p_init, static_configs['p_min']), static_configs['p_max'])

    return u_init, v_init, p_init

def init_n_counts_grid(user_counts, n_users, coarse_bins_per_week, period_start, period_end):
    '''
    Counts how many counts were observed in the period needed for the smoothing equation that relies on n counts
    '''

    train_df = get_period_sums(user_counts, period_start, period_end)

    # Init a grid and get entries to assign and assigning the number of counts
    n_counts = np.zeros((n_users, coarse_bins_per_week), dtype='float64')
    entries_to_assign = train_df.select(['user_id', 'coarse_bin_id']).to_numpy()
    n_counts[entries_to_assign[:, 0], entries_to_assign[:, 1]] = train_df['n_bins'].to_numpy()

    return n_counts

#####################################
# Degen mask and interpolation rates
#####################################

def get_degen_mask(user_counts, n_users, n_coarse_bins, static_configs, train_test_dict):
    ''' 
    Creates a grid of user x coarse bin combinations that have less than `static_configs.degen_threshold` 
    counts in them. These bins are excluded from scoring.
    '''
    # Get the number of counts for each user in the period
    train_df = get_period_sums(user_counts, train_test_dict['train_start'], train_test_dict['burn_in_end'])

    # Init the output mask with all bins as degen
    degen_mask = np.ones((n_users, n_coarse_bins), dtype='bool')

    # Finding which rows are not degen and setting the mask to false
    non_degen_rows = train_df['n_bins'].to_numpy() >= static_configs['degen_threshold']
    entries_to_assign = train_df.select(['user_id', 'coarse_bin_id']).to_numpy()
    degen_mask[entries_to_assign[non_degen_rows, 0], entries_to_assign[non_degen_rows, 1]] = False

    return degen_mask


def get_quadratic_interpolation_weights(bin_metric_dict):
    '''
    A function that returns the weights we apply when interpolating.
    To understand the computation steps see pages 11 and 12 of lambert liu
    returns:
        weights a numpy array which will be applied as w-1 U-1 + w0 U0 + w1 U1
        the weights array has a row for every fine bin in the coarse bin and 3 columns where each column is the weight being applies to Uis
    '''

    # Getting the m (fine bin number, q and r (defined in lambert liu))
    M = bin_metric_dict['fine_bins_per_coarse_bin']
    m = np.arange(1, M+1)
    q = (m - 1)/M
    r = m/M

    # Getting the two terms used in all calculations
    term_1 = r**2 + r*q + q**2
    term_2 = r + q

    # Computing the weights using the lambert and liu formula
    # these come from rearranging the fomula at the top of page 12 for U-1 U0 and U1
    weights = np.zeros((M, 3))

    weights[:, 0] = term_1/6 - term_2/2 + 1/3
    weights[:, 1] = -term_1/3 + term_2/2 + 5/6
    weights[:, 2] = term_1/ 6 - 1/6

    return weights

def get_linear_interpolation_weights(bin_metric_dict):
    '''
    Gets the interpolate weights for the linear interpolation correction
    '''

    # getting M and the fraction through the coarse bin
    M = bin_metric_dict['fine_bins_per_coarse_bin']
    position_fraction = (np.arange(M, dtype='float64') + 0.5) / M

    # Init the weights and array that tells us if we interpolate with the left or right value
    weights = np.zeros((M, 3), dtype='float64')

    # Obtain linear interpolation weights
    weights[:, 0] = np.maximum(0, 0.5 - position_fraction)
    weights[:, 2] = np.maximum(0, position_fraction - 0.5)
    weights[:, 1] = 1 - weights[:, 0] - weights[:, 2]

    return weights

#####################################
# Named tuple functions
#####################################

def dictionary_to_named_tuple_class(name : str, dictionary : dict):
    ''' 
    Converts a dict to a named tuple with the given name.
    Can be used downstream in the njit functions
    '''
    if name in globals():
        raise ValueError(f'Named tuple: {name} already exists')
    else:
        return namedtuple(name, dictionary.keys())
        
    
def df_to_nt(name, df):
    ''' 
    Converts a dataframe to a Namedtuple of numpy arrays for use in the numba runner
    '''
    table_dict = {col : df[col].to_numpy().astype('int64') for col in df.columns}

    return dictionary_to_named_tuple_class(name, table_dict)(**table_dict)

def get_model_and_output_idx_nt():
    # Creating dictionaries of names of outputs and oder they appear in
    output_names = ['n_bins_scored', 'non_degen_ll_sum','non_degen_smoothed_ll_sum']
    model_names = ['raw_model_calib_index', 'smoothed_model_calib_index']

    output_idx_dict = {name : idx for idx, name in enumerate(output_names)}
    model_idx_dict = {name : idx for idx, name in enumerate(model_names)}

    # For other dicts we dont need to save the class
    output_idx_nt = dictionary_to_named_tuple_class('output_idx_nt', output_idx_dict)(**output_idx_dict)
    model_idx_nt = dictionary_to_named_tuple_class('model_idx_nt', model_idx_dict)(**model_idx_dict)

    return output_idx_nt, model_idx_nt

def converting_dicts_to_nt(static_configs, train_test_dict, bin_metric_dict):
    ''' 
    Converts the config dictionaries to named tuples
    '''
    # Converting userful info to named tuples
    ## Creating a seperate config_nt_class and train_test_nt_class as they are reused in the tuning loop
    config_nt_class = dictionary_to_named_tuple_class('config_nt', static_configs)
    config_nt = config_nt_class(**static_configs)
    train_test_nt_class = dictionary_to_named_tuple_class('train_test_nt', train_test_dict)
    train_test_nt = train_test_nt_class(**train_test_dict)
    bin_metric_nt = dictionary_to_named_tuple_class('bin_metric_nt', bin_metric_dict)(**bin_metric_dict)

    return config_nt_class, config_nt, train_test_nt_class, train_test_nt, bin_metric_nt

