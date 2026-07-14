import polars as pl 
import json5
import numpy as np
from datetime import datetime
import os

data_path = '/home/ma/a/alb25/Project/thesis_code/data/processed/intermediate'
results_path = '/home/ma/a/alb25/Project/thesis_code/data/processed/results'
input_data = 'train_df'

#####################################
# Load and store data functions
#####################################

# Creating functions for storing and reading data
def store_data(data, filename, csv=False, results=False, data_path=data_path, results_path=results_path):
    if results:
        data_path = results_path
    if isinstance(data, pl.LazyFrame):
        if csv == True:
            data.sink_csv(f'{data_path}/output/{filename}.csv')
        else: 
            data.sink_parquet(f'{data_path}/output/{filename}.parquet')
    elif isinstance(data, pl.DataFrame):
        if csv == True:
            data.write_csv(f'{data_path}/output/{filename}.csv')
        else:
            data.write_parquet(f'{data_path}/output/{filename}.parquet')
    elif isinstance(data, np.ndarray):
        np.save(f'{data_path}/output/{filename}.npy', data)
    else:
        raise TypeError('Function doesnt support this data type')

def load_data(filename, data_type, results=False, data_path=data_path, results_path=results_path):
    ''' 
    Args:
        filename: the saved file name
        data_type: ['lazy', 'np', 'df'] the type of data we want to load in
    '''
    if results:
        data_path = results_path
    if data_type == 'lazy':
        data = pl.scan_parquet(f'{data_path}/output/{filename}.parquet')
    elif data_type == 'np':
        data = np.load(f'{data_path}/output/{filename}.npy')
    elif data_type == 'df':
        data = pl.read_parquet(f'{data_path}/output/{filename}.parquet')
    else:
        raise TypeError('Function doesnt support this data type')
    return data

def store_run_results(results, calibration_results, stage, run_name, results_path=results_path):
    '''
    Writes run results to results folder
    '''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Creating dirs if needed
    os.makedirs(f'{results_path}/{stage}/summary', exist_ok=True)
    os.makedirs(f'{results_path}/{stage}/calibration', exist_ok=True)

    # Turning into polars dataframes
    if isinstance(results, pl.DataFrame):
        results_df = results
    else:
        results_df = pl.DataFrame(results)
    if isinstance(calibration_results, pl.DataFrame):
        calibration_df = calibration_results
    else:
        calibration_df = pl.DataFrame(calibration_results)

    # Adding run timestamp column
    results_df = results_df.with_columns(pl.lit(timestamp).alias('run_timestamp'))
    calibration_df = calibration_df.with_columns(pl.lit(timestamp).alias('run_timestamp'))

    # Writing results to the df
    results_df.write_parquet(f'{results_path}/{stage}/summary/{run_name}_{timestamp}.parquet')
    calibration_df.write_parquet(f'{results_path}/{stage}/calibration/{run_name}_{timestamp}.parquet')
    
#####################################
# Json 5 stuff
#####################################

def load_json5(filename):
    with open(f'/home/ma/a/alb25/Project/thesis_code/notebooks/configs/{filename}.json5') as f:
        return json5.load(f)
    
def dump_json5(dict, filename):
    with open(f'/home/ma/a/alb25/Project/thesis_code/notebooks/configs/{filename}.json5', 'w') as f:
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