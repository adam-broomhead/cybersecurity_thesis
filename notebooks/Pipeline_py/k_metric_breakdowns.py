import numpy as np 
import c_clustering as c

    #####################################
    # Decile Creation
    #####################################

def get_activity_deciles(user_counts_nt, n_users, period_start, period_end):
    '''
    Filter activity to pe in period then groupby user id and get counts
    '''
    period_filter = ((user_counts_nt.fine_bin_id >= period_start) & (user_counts_nt.fine_bin_id < period_end))

    return make_rank_deciles(np.bincount(user_counts_nt.user_id[period_filter], weights=user_counts_nt.count[period_filter]))

def make_rank_deciles(val_to_rank):
    '''
    Assigns users to deciles based on val_to_rank
    '''
    # Create and rank assignments 
    n_users = val_to_rank.shape[0]
    ranked_groups = (np.arange(n_users, dtype=np.int64) * 10) // n_users

    # Init output and assign users
    output = np.empty(n_users, dtype='int8')
    output[np.argsort(val_to_rank, kind='stable')] = ranked_groups
    return output

def get_metric_breakdown(user_counts_nt, user_type_groups, model, config_dict, u_clustering, v_clustering, p_clustering, train_test_dict):
    '''
    Creates the breakdowns for the run in the test set
    '''

    n_users = u_clustering.shape[0]
    all_user_group = np.zeros(n_users, dtype='int8')

    activity_deciles = get_activity_deciles(user_counts_nt=user_counts_nt, n_users=n_users, 
                                period_start=train_test_dict['train_start'], period_end=train_test_dict['burn_in_end'])

    matrix_to_cluster = c.make_clustering_matrix(u_clustering, v_clustering, p_clustering, config_dict)
    cluster_distances = c.get_user_cluster_distances(matrix_to_cluster=matrix_to_cluster, 
        cluster_assignments=model['cluster_assignments'], cluster_centres=model['cluster_centres'], distance_metric=model['distance_metric'])
    distance_deciles = make_rank_deciles(cluster_distances)
    

    return np.vstack((all_user_group, activity_deciles, distance_deciles, user_type_groups)).astype('int8')
