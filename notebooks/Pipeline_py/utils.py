import polars as pl 
import json5
import numpy as np

data_path = '/home/ma/a/alb25/Project/thesis_code/data/processed/intermediate'
input_data = 'train_df'

#####################################
# Load and store data functions
#####################################

# Creating functions for storing and reading data
def store_data(data, filename, data_path=data_path, csv=False):
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
    

def load_data(filename, data_type, data_path=data_path):
    ''' 
    Args:
        filename: the saved file name
        data_type: ['lazy', 'np', 'df'] the type of data we want to load in
    '''
    if data_type == 'lazy':
        data = pl.scan_parquet(f'{data_path}/output/{filename}.parquet')
    elif data_type == 'np':
        data = np.load(f'{data_path}/output/{filename}.npy')
    elif data_type == 'df':
        data = pl.read_parquet(f'{data_path}/output/{filename}.parquet')
    else:
        raise TypeError('Function doesnt support this data type')
    return data

#####################################
# Json 5 stuff
#####################################

def load_json5(filename):
    with open(f'/home/ma/a/alb25/Project/thesis_code/notebooks/configs/{filename}.json5') as f:
        return json5.load(f)
    
def dump_json5(dict, filename):
    with open(f'/home/ma/a/alb25/Project/thesis_code/notebooks/configs/{filename}.json5', 'w') as f:
        json5.dump(dict, f)
