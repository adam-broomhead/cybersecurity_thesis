#####################################
# Imports and variables
#####################################

import polars as pl 
import json5
import numpy as np
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
data_dir = f'{project_root}/data/processed/intermediate'
results_dir = f'{project_root}/data/processed/results'
configs_dir = f'{project_root}/notebooks/configs'
input_data = 'train_df'

# Used only in 100
raw_data_dir = f'{project_root}/data/raw'
temp_data_dir = f'{project_root}/data/temp'

#####################################
# Load and store data functions
#####################################

# Creating functions for storing and reading data
def store_data(data, filename, csv=False, results=False, data_dir=data_dir, results_dir=results_dir):

    if results:
        output_dir = Path(f'{results_dir}/output')
    else:
        output_dir = Path(f'{data_dir}/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pl.LazyFrame):
        if csv:
            data.sink_csv(f'{output_dir}/{filename}.csv')
        else:
            data.sink_parquet(f'{output_dir}/{filename}.parquet')

    elif isinstance(data, pl.DataFrame):
        if csv:
            data.write_csv(f'{output_dir}/{filename}.csv')
        else:
            data.write_parquet(f'{output_dir}/{filename}.parquet')

    elif isinstance(data, np.ndarray):
        np.save(f'{output_dir}/{filename}.npy', data)

    else:
        raise TypeError('cant pass this datatype')

def load_data(filename, data_type, results=False, data_dir=data_dir, results_dir=results_dir):
    '''
    Args:
        filename: saved file name
        data_type: one of 'lazy', 'np', or 'df'
    '''
    if results:
        output_dir = Path(f'{results_dir}/output')
    else:
        output_dir = Path(f'{data_dir}/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_type == 'lazy':
        return pl.scan_parquet(f'{output_dir}/{filename}.parquet')

    if data_type == 'np':
        return np.load(f'{output_dir}/{filename}.npy')

    if data_type == 'df':
        return pl.read_parquet(f'{output_dir}/{filename}.parquet')

    raise TypeError('cant pass this datatype')

################

def store_run_results(results, dir, run_name, calibration_results=None, results_dir=results_dir):
    '''
    Writes run results to results folder
    '''

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = Path(f'{results_dir}/{dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Converting results to polars df if needed
    if isinstance(results, pl.DataFrame):
        results_df = results
    else:
        results_df = pl.DataFrame(results)

    # Forcing data types to be consistent
    result_dtypes = {
        'linear_smooth': pl.Boolean,
        'hurdle_model': pl.Boolean,
        'seed': pl.Int32,
        'cluster_param': pl.Int16,
        'smoothing_target': pl.Int16,
        'w_inf': pl.Float32,
        'smooth_a_mu': pl.Float32,
        'smooth_a_sigma2': pl.Float32,
        'smooth_a_p': pl.Float32,
        'smooth_t_mu': pl.Float32,
        'smooth_t_sigma2': pl.Float32,
        'smooth_t_p': pl.Float32,
        'cluster_inertia': pl.Float32,
        'threshold': pl.Float32,
        'raw_tail_rate': pl.Float32,
        'smoothed_tail_rate': pl.Float32,
        'non_degen_ll': pl.Float32,
        'non_degen_smoothed_ll': pl.Float32,
        'n_bins_scored': pl.Int64,
        'non_degen_ll_sum': pl.Float64,
        'user_idx': pl.Int32,
        'cluster_distance': pl.Float64}

    results_df = results_df.with_columns([pl.col(column).cast(dtype) for column, dtype in result_dtypes.items() if column in results_df.columns])

    calibration_count_columns = [column for column in results_df.columns if column.startswith('calibration_count_')]
    if calibration_count_columns:
        results_df = results_df.with_columns([pl.col(column).cast(pl.Int64) for column in calibration_count_columns])

    # Writing results
    results_df = results_df.with_columns(pl.lit(timestamp).alias('run_timestamp'))
    results_df.write_parquet(f'{output_dir}/{run_name}_{timestamp}.parquet')

#####################################
# Json 5 stuff
#####################################

def load_json5(filename):
    with open(f'{configs_dir}/{filename}.json5') as f:
        return json5.load(f)

def dump_json5(dict, filename):
    with open(f'{configs_dir}/{filename}.json5', 'w') as f:
        json5.dump(dict, f)

#####################################
# Merging configs
#####################################

def merge_configs(*configs):
    '''
    Merges config dicts
    '''
    config_dict = {}

    # Insert configs into the config_dict
    for config in configs:
        config_dict.update(config)

    # Converting lists to np arrays
    if 'calibration_thresholds' in config_dict:
        config_dict['calibration_thresholds'] = np.array(config_dict['calibration_thresholds'], dtype='float64')

    return config_dict

