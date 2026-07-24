import polars as pl 
import utils as ut

static_configs = ut.load_json5('static_configs')

# Helper functions outputing dicitonaries
def get_bin_metrics(static_configs=static_configs):
    ''' 
    Returns a dictionary containing all information needed about bin length these numbers are needed throughout the pipeline
    '''
    # Check that bin length divides number of days
    assert 24 * 60 % static_configs['fine_bin_mins'] == 0
    assert 24 * 60 % static_configs['coarse_bin_mins'] == 0
    assert static_configs['coarse_bin_mins'] % static_configs['fine_bin_mins'] == 0

    output_dict = {'fine_bins_per_day' : 24 * 60 // static_configs['fine_bin_mins']}
    output_dict['fine_bins_per_week'] = output_dict['fine_bins_per_day'] * 7
    output_dict['fine_bins_per_coarse_bin'] = static_configs['coarse_bin_mins'] // static_configs['fine_bin_mins']
    output_dict["coarse_bins_per_day"] = output_dict["fine_bins_per_day"] // output_dict["fine_bins_per_coarse_bin"]
    output_dict['fine_bin_seconds'] = static_configs['fine_bin_mins'] * 60 

    return output_dict

def get_train_test_split(static_configs=static_configs):
    ''' 
    Calculates the fine bin index of the train test and validation period start and end
    Returns a dictionary with all this information in
    '''

    fine_bins_per_day = 24 * 60 // static_configs['fine_bin_mins']

    train_bins = fine_bins_per_day * static_configs['train_days']
    burn_in_bins = fine_bins_per_day * static_configs['burn_in_days']
    validation_bins = fine_bins_per_day * static_configs['validation_days']
    test_bins = fine_bins_per_day * static_configs['test_days']

    # Constructing a dict for the output
    output_dict = {'train_start' : 0,
                   'train_end' : train_bins}
    
    output_dict['burn_in_start'] = output_dict['train_end']
    output_dict['burn_in_end'] = output_dict['burn_in_start'] + burn_in_bins

    output_dict['validation_start'] = output_dict['burn_in_end']
    output_dict['validation_end'] = output_dict['validation_start'] + validation_bins

    output_dict['test_start'] = output_dict['validation_end']
    output_dict['test_end'] = output_dict['test_start'] + test_bins
    
    return output_dict
    
# Adding the fine_bins_per_coarse_bin to the train_test_split_dict
def add_training_denom(bin_metric_dict, static_configs=static_configs):
    ''' 
    Finds how many fine bins are used for the estimate of the iniatial parameter.
    This is the denominator on the mean estimate as it is counts/bins and is similarly used in the variance estimate.

    This fucntion assumes train days is a multiple of 7 to work

    Returns:
        An updated bin metric dict with a new column train denom and cluster denom
    '''
    bin_metric_dict["train_denom"] = bin_metric_dict["fine_bins_per_coarse_bin"] * static_configs["train_days"]
    return bin_metric_dict


####

# Creating a fine bin and identifying users with some counts in the bin

def create_counts_data(df, bin_metric_dict):
    ''' 
    Gets the counts per fine bin x source user from the data
    '''
    df = df.with_columns(time = pl.col('time').dt.total_seconds())
    df = df.with_columns(fine_bin_id = pl.col('time') // bin_metric_dict['fine_bin_seconds'])
    user_x_fine_bin_cnts = df.group_by(['source_user@domain', 'fine_bin_id']).agg(pl.len().alias('count'))

    return user_x_fine_bin_cnts

def create_user_to_id_mapping(users_df, mapping_file_name):
    ''' 
    Creates a table with user to id mapping
    '''
    # Creating a user lookup table and storing it
    user_mapping = users_df.select('source_user@domain').unique().sort(by='source_user@domain').with_row_index('user_id')
    user_mapping = user_mapping.with_columns(source_user_type = pl.when(pl.col('source_user@domain').str.contains(r"^U\d+@")).then(pl.lit("human")
                                            ).when(pl.col('source_user@domain').str.contains(r"^C\d+\$@")).then(pl.lit("machine"))).collect(engine='streaming')
    ut.store_data(user_mapping, mapping_file_name)
    
    # Joining on the lookup table and dropping columns
    users_df = users_df.join(user_mapping.lazy(), on='source_user@domain', how='inner')
    users_df = users_df.select(['user_id', 'fine_bin_id', 'count']).sort(['user_id', 'fine_bin_id'])

    return users_df, user_mapping

def create_coarse_bins(users_df, bin_metric_dict):
    ''' 
    takes a DF and creates two new columns:
        coarse_bin_id
        fine_bin_within_coarse_pos
    '''
    # Creating columns needed fr
    users_df = users_df.with_columns(fine_bin_pos_in_day=(pl.col("fine_bin_id") % bin_metric_dict["fine_bins_per_day"]))
    users_df = users_df.with_columns(coarse_bin_id= pl.col("fine_bin_pos_in_day") // bin_metric_dict["fine_bins_per_coarse_bin"])

    return users_df

def create_first_last_interaction_arrays(user_counts):
    '''
    Creates an output table with user first and last interaction indicies within df counts
    Note this is not a fine bin index but a row index
    '''
    # Creating a lookup table for numba for the first and last entry of users 
    user_interactions = user_counts.with_row_index().group_by('user_id').agg(
        user_first_index = pl.min('index'),
        user_last_index = pl.max('index')).sort(by='user_id')
    
    return user_interactions