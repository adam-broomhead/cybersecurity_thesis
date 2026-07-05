import numpy as np
import polars as pl
from sklearn.cluster import KMeans
import utils as ut

runtime_configs = ut.load_json5('runtime_configs')

#####################################
# Creating functions that return the correct vector for clustering
#####################################
# Matrix construction functions
def make_u_matrix(u, v, p): return u

def make_log_u_matrix(u, v, p): return np.log(u)

def make_v_matrix(u, v, p): return v

def make_normalised_u_clustering_matrix(u, v, p): return u / u.sum(axis=1)[:, None]


# Wrapper function
def make_clustering_matrix(u, v, p, runtime_configs): 
    ''' 
    Constructs the matrix used for clustering. 
    The construction used depends on `clustering_matrix_name` found within `runtime_configs`
    '''
    if runtime_configs['clustering_matrix_name'] == 'u':
        return make_u_matrix(u, v, p)
    elif runtime_configs['clustering_matrix_name'] == 'log_u':
        return make_log_u_matrix(u, v, p)
    elif runtime_configs['clustering_matrix_name'] == 'v':
        return make_v_matrix(u, v, p)
    elif runtime_configs['clustering_matrix_name'] == 'normalised_u':
        return make_normalised_u_clustering_matrix(u, v, p)
    else: 
        raise ValueError("runtime_configs['clustering_matrix_name'] invalid")
    
#####################################
# Functions for getting cluster assignments
#####################################

def get_k_means_assignments(k, random_state, matrix_to_cluster):
    '''
    Performs k means clustering on a numpy array and returns a vector of cluster assignments
    '''

    k_means_model = KMeans(n_clusters=k, random_state=random_state)
    clusters = k_means_model.fit_predict(matrix_to_cluster).astype(np.int64)

    return clusters, k_means_model

def get_cluster_assignments(cluster_param, matrix_to_cluster, runtime_configs : dict):
    ''' 
    Generic clustering runner that can be scaled to include multiple algorithms
    '''
    if runtime_configs['clustering_method'] == 'k_means':
        return get_k_means_assignments(k=cluster_param, random_state=runtime_configs['seed'], matrix_to_cluster=matrix_to_cluster)
    else:
        raise ValueError('Clustering method not Reckognised')
    
#####################################
# Algorithm performance metrics
#####################################

def get_centroid_distance(cluster_centres):
    ''' 
    For each cluster computes l2 distances and returns:
        - Average distance to other clusters
        - Min distance to other clusters
    '''
    output = {}
    for i in range(cluster_centres.shape[0]):
        cluster_distances = []
        for j in range(cluster_centres.shape[0]):
            
            # Calculate l2 distance
            if i != j:
                cluster_distances.append(np.sqrt(np.sum((cluster_centres[i] - cluster_centres[j]) ** 2)))

        # Get outputs for cluster
        if len(cluster_distances) == 0:
            output[i] = {'nearest_centroid_dist' : np.nan, 'avg_centroid_dist' : np.nan}
        else:
            output[i] = {'nearest_centroid_dist' : min(cluster_distances), 'avg_centroid_dist' : sum(cluster_distances)/len(cluster_distances)}

    return output

def create_cluster_summary_df(model, user_mapping, runtime_configs):
    ''' 
    Creates an output df which summarises cluster quality metrics
    '''

    # Getting the number of clusters
    if runtime_configs['clustering_method'] == 'k_means':
        n_clusters=model['cluster_param']
    else:
        raise ValueError('Clustering method not Reckognised')
    
    output = []
    cluster_centroid_dict = get_centroid_distance(model['cluster_centres'])

    for cluster_id in range(n_clusters):

        users_in_cluster = user_mapping.filter(pl.col('user_id').is_in(np.where(model['cluster_assignments'] == cluster_id)[0]))
        n_users_in_cluster = users_in_cluster.shape[0]
        human_users_in_cluster = users_in_cluster.filter(pl.col('source_user_type') == 'human').shape[0]
        machine_users_in_cluster = users_in_cluster.filter(pl.col('source_user_type') == 'machine').shape[0]

        output.append({
            'cluster_id' : cluster_id,
            'cluster_param' : model['cluster_param'],
            'seed' : model['seed'],
            'clustering_matrix_name' : model['clustering_matrix_name'],

            # Cluster content metrics
            'n_users' : n_users_in_cluster,
            'n_humans' : human_users_in_cluster,
            'n_machine_users' : machine_users_in_cluster,

            # Performance metrics
            'inertia' : model['cluster_inertia'],
            'nearest_centroid_dist' : cluster_centroid_dict[cluster_id]['nearest_centroid_dist'],
            'avg_centroid_dist' : cluster_centroid_dict[cluster_id]['avg_centroid_dist'],
        })

    return pl.DataFrame(output)

#####################################
# Model + cluster mean creation
#####################################


def get_param_cluster_mean(cluster_groups, param_grid):
    ''' 
    Helper function which calculates the cluster param given a param grid (u, v or p)
    '''
    n_users, n_coarse_bins = param_grid.shape 
    n_clusters = cluster_groups.max() + 1

    # Init mean vectors
    cluster_mean = np.zeros((n_clusters, n_coarse_bins), dtype='float64')
    users_per_cluster = np.zeros(n_clusters, dtype='float64')

    # Summing u or v or p contributions in each cluster
    # Extract the cluster assignment for each user and then add their parameters to each bin
    for user_id in range(n_users):
        cluster_assignment = int(cluster_groups[user_id])
        cluster_mean[cluster_assignment, :] += param_grid[user_id, :]
        users_per_cluster[cluster_assignment] += 1

    # Dividing through to get the averages in each cluster
    for cluster_assignment in range(n_clusters):
        if users_per_cluster[cluster_assignment] > 0:
            cluster_mean[cluster_assignment, :] /= users_per_cluster[cluster_assignment]

    return cluster_mean

def get_cluster_means(cluster_groups, u_init, v_init, p_init):
    ''' 
    Calculates mean u and v and p values for each cluster group and each time bin
    Used within the get clustering model function
    Args:
        cluster_groups a n_users length vector of cluster assignments
        u_init : the calculated vector of parameter means (or non zero mean if a hurdle model)
        v_init : the calculated vector of initial parameter variances (or non zero variances if hurdle model)
        v_init : the calculated vector of initial parameter activation probs (if a hurdle model)
    '''
    return get_param_cluster_mean(cluster_groups, u_init), get_param_cluster_mean(cluster_groups, v_init), get_param_cluster_mean(cluster_groups, p_init)

def make_cluster_model(cluster_param, runtime_configs, u_init, v_init, p_init=None):
    ''' 
    Creates the clustering model dictionary 
    '''

    # Init p and matrix to cluster
    if p_init is None:
        p_init = np.zeros_like(u_init)
    matrix_to_cluster = make_clustering_matrix(u_init, v_init, p_init, runtime_configs)

    # Global pooling defaults
    if cluster_param == 1:
        cluster_assignments = np.zeros(u_init.shape[0], dtype=np.int64)
        cluster_centres = matrix_to_cluster.mean(axis=0, keepdims=True)
        # Dont care about inertia for 1 cluster
        cluster_inertia = 0

    # Getting cluster assignments and cluster centres
    else:
        cluster_assignments, clustering_model = get_cluster_assignments(cluster_param=cluster_param, matrix_to_cluster=matrix_to_cluster, runtime_configs=runtime_configs)
        cluster_centres = clustering_model.cluster_centers_
        cluster_inertia = clustering_model.inertia_

    # Get mean cluster values
    cluster_mean_u, cluster_mean_v, cluster_mean_p = get_cluster_means(cluster_groups=cluster_assignments, u_init=u_init, v_init=v_init, p_init=p_init)
    

    output = {
        # Clustering configs
        'name' : f"{runtime_configs['clustering_method']}",
        'clustering_matrix_name' : runtime_configs['clustering_matrix_name'],
        'seed' : runtime_configs['seed'],
        'cluster_param' : cluster_param,

        # Identified values
        'cluster_mean_u' : cluster_mean_u,
        'cluster_mean_v' : cluster_mean_v,
        'cluster_mean_p' : cluster_mean_p,
        'cluster_assignments' : cluster_assignments,
        
        # Cluster quality metrics
        'cluster_inertia' : cluster_inertia, 
        'cluster_centres' : cluster_centres,
        }

    return output 