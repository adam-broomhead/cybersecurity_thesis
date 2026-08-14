import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedians import kmedians
from sklearn.preprocessing import StandardScaler
import utils as ut

runtime_configs = ut.load_json5('runtime_configs')

#####################################
# Creating functions that return the correct vector for clustering
#####################################
# Wrapper function
def make_clustering_matrix(u, v, p, runtime_configs): 
    ''' 
    Constructs the matrix used for clustering. 
    The construction used depends on `runtime_configs`
    '''
    # Select the parameter cols
    if runtime_configs['clustering_matrix_name'] == 'u':
        matrix_cols = (u,)

    elif runtime_configs['clustering_matrix_name'] == 'u_p':
        matrix_cols = (u, p)

    elif runtime_configs['clustering_matrix_name'] == 'all':
        matrix_cols = (u, v, p)

    else:
        raise ValueError('invalid matrix name')

    # Transform the cols according to the scaling
    transformed_cols = []
    for matrix_col in matrix_cols:
        if runtime_configs['clustering_transformation'] == 'none':
            transformed_col = matrix_col
        elif runtime_configs['clustering_transformation'] == 'log':
            transformed_col = np.log(np.maximum(matrix_col, 1e-12))                 
        elif (runtime_configs['clustering_transformation'] == 'normalise'):
            row_totals = matrix_col.sum(axis=1, keepdims=True)
            transformed_col = np.divide(matrix_col, row_totals, out=np.zeros_like(matrix_col, dtype='float64'), where=row_totals !=0)
        else:
            raise ValueError('invalid matrix transformation')
        transformed_cols.append(transformed_col)

    # Sandardise columns for diagonal malhanobis 
    matrix_to_cluster = np.concatenate(transformed_cols, axis=1)
    if runtime_configs['distance_metric'] == 'standardised_l2':
        return StandardScaler().fit_transform(matrix_to_cluster)
    return matrix_to_cluster
    
#####################################
# Functions for getting cluster assignments
#####################################

def get_k_means_assignments(k, random_state, matrix_to_cluster):
    '''
    Performs k means clustering on a numpy array and returns a vector of cluster assignments
    '''
    k_means_model = KMeans(n_clusters=k, random_state=random_state)

    clusters = k_means_model.fit_predict(matrix_to_cluster).astype(np.int64)

    return clusters, k_means_model.cluster_centers_, k_means_model.inertia_


def get_k_medians_assignments(k, random_state, matrix_to_cluster):
    ''' 
    Performc k medians clustering and returns a vector of cluster assignments
    '''
    # Init centres
    rng = np.random.default_rng(random_state) 
    unique_vectors = np.unique(matrix_to_cluster, axis=0)
    initial_centres = unique_vectors[rng.choice(unique_vectors.shape[0], size=k, replace=False)]

    # Run the algo and get centres
    model = kmedians(matrix_to_cluster.tolist(), initial_centres.tolist()).process()
    cluster_centres = np.asarray(model.get_medians())

    # Creating a vecotr of cluster centres and getting interita
    clusters = np.zeros(matrix_to_cluster.shape[0], dtype=np.int64)
    for cluster_id, row_indices in enumerate(model.get_clusters()):
        clusters[row_indices] = cluster_id

    distances = np.abs(matrix_to_cluster - cluster_centres[clusters]).sum(axis=1)
    cluster_inertia = distances.sum()

    return clusters, cluster_centres, cluster_inertia

def get_cluster_assignments(cluster_param, matrix_to_cluster, runtime_configs : dict):
    ''' 
    Generic clustering runner that can be scaled to include multiple algorithms
    '''
    if runtime_configs['clustering_method'] == 'k_clusters':
        if runtime_configs['distance_metric'] in ('l2', 'standardised_l2'):
            return get_k_means_assignments(k=cluster_param, random_state=runtime_configs['seed'], matrix_to_cluster=matrix_to_cluster)

        elif runtime_configs['distance_metric'] == 'l1':
            return get_k_medians_assignments(k=cluster_param, random_state=runtime_configs['seed'], matrix_to_cluster=matrix_to_cluster)

    
#####################################
# Algorithm performance metrics
#####################################

def get_centroid_distance(cluster_centres, distance_metric):
    '''
    For each cluster computes distances and returns:
        - Average distance to other clusters
        - Min distance to other clusters
    '''
    output = {}
    for i in range(cluster_centres.shape[0]):
        cluster_distances = []
        for j in range(cluster_centres.shape[0]):
            
            # Calculate distance
            if i != j:
                if distance_metric == 'l1':
                    distance = np.abs(cluster_centres[i] - cluster_centres[j]).sum()
                elif distance_metric in ('l2', 'standardised_l2'):
                    distance = np.sqrt(np.sum((cluster_centres[i] - cluster_centres[j]) ** 2))

                cluster_distances.append(distance)

        # Get outputs for cluster
        if len(cluster_distances) == 0:
            output[i] = {'nearest_centroid_dist' : np.nan, 'avg_centroid_dist' : np.nan}
        else:
            output[i] = {'nearest_centroid_dist' : min(cluster_distances), 'avg_centroid_dist' : sum(cluster_distances)/len(cluster_distances)}

    return output

def get_user_cluster_distances(matrix_to_cluster, cluster_assignments, cluster_centres, distance_metric):
    '''
    Calculates user distances from cluster centres
    '''
    vec_to_centre = matrix_to_cluster - cluster_centres[cluster_assignments]

    if distance_metric == 'l1':
        return np.abs(vec_to_centre).sum(axis=1)

    if distance_metric in ('l2', 'standardised_l2'):
        return np.sqrt(np.square(vec_to_centre).sum(axis=1))

def create_cluster_summary_df(model, user_mapping, runtime_configs):
    ''' 
    Creates an output df which summarises cluster quality metrics
    '''

    # Getting the number of clusters
    if runtime_configs['clustering_method'] == 'k_clusters':
        n_clusters = model['cluster_param']
    else:
        raise ValueError('Clustering method not recognised')
        
    output = []
    cluster_centroid_dict = get_centroid_distance(model['cluster_centres'], model['distance_metric'])
    for cluster_id in range(n_clusters):

        users_in_cluster = user_mapping.filter(pl.col('user_id').is_in(np.where(model['cluster_assignments'] == cluster_id)[0]))
        n_users_in_cluster = users_in_cluster.shape[0]
        human_users_in_cluster = users_in_cluster.filter(pl.col('source_user_type') == 'human').shape[0]
        machine_users_in_cluster = users_in_cluster.filter(pl.col('source_user_type') == 'machine').shape[0]

        output.append({
            'cluster_id' : cluster_id,
            'cluster_param' : model['cluster_param'],
            'seed' : model['seed'],
            'clustering_matrix_name': model['clustering_matrix_name'],
            'clustering_transformation': model['clustering_transformation'],
            'distance_metric': model['distance_metric'],

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
#  Cluster center creation
#####################################

def get_param_cluster_centre(cluster_groups, param_grid, distance_metric):
    '''
    Gets the cluster centre for each cluster
    '''
    # Getting number of clusters and bins and init a vector of centre points
    n_clusters = cluster_groups.max() + 1
    n_coarse_bins = param_grid.shape[1]
    cluster_centre = np.zeros((n_clusters, n_coarse_bins), dtype='float64')

    for cluster_id in range(n_clusters):
        # Identify the rows in the clsuter and get the centre
        cluster_rows = param_grid[cluster_groups == cluster_id]
        if distance_metric == 'l1':
            cluster_centre[cluster_id] = np.median(cluster_rows, axis=0)
        else:
            cluster_centre[cluster_id] = cluster_rows.mean(axis=0)

    return cluster_centre


def get_cluster_centres(cluster_groups, u_init, v_init, p_init, distance_metric):
    ''' 
    Gets the cluster centre this is either the median (l1 distance) or the mean (l2 distance)
    '''
    return (get_param_cluster_centre(cluster_groups, u_init, distance_metric), get_param_cluster_centre(cluster_groups, v_init, distance_metric), 
            get_param_cluster_centre(cluster_groups, p_init, distance_metric))

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
        p_init : the calculated vector of initial parameter activation probs (if a hurdle model)
    '''
    return get_param_cluster_mean(cluster_groups, u_init), get_param_cluster_mean(cluster_groups, v_init), get_param_cluster_mean(cluster_groups, p_init)

#####################################
#  Making the cluster model
#####################################

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

        if runtime_configs['distance_metric'] == 'l1':
            cluster_centres = np.median(matrix_to_cluster, axis=0, keepdims=True)
        else:
            cluster_centres = matrix_to_cluster.mean(axis=0, keepdims=True)

        cluster_inertia = 0.0

    # Getting cluster assignments and cluster centres
    else:
        cluster_assignments, cluster_centres, cluster_inertia = get_cluster_assignments(cluster_param=cluster_param, matrix_to_cluster=matrix_to_cluster, runtime_configs=runtime_configs)

    # Get mean cluster values
    cluster_centre_u, cluster_centre_v, cluster_centre_p = get_cluster_centres(cluster_groups=cluster_assignments, u_init=u_init, v_init=v_init, p_init=p_init, 
                                                                               distance_metric=runtime_configs['distance_metric'])

    output = {
        # Clustering configs
        'name' : runtime_configs['clustering_method'],
        'distance_metric': runtime_configs['distance_metric'],
        'clustering_matrix_name' : runtime_configs['clustering_matrix_name'],
        'clustering_transformation': runtime_configs['clustering_transformation'],
        'seed' : runtime_configs['seed'],
        'cluster_param' : cluster_param,

        # Cluster centres and assignments
        'cluster_centre_u' : cluster_centre_u,
        'cluster_centre_v' : cluster_centre_v,
        'cluster_centre_p' : cluster_centre_p,
        'cluster_assignments' : cluster_assignments,
        
        # Cluster quality outputs
        'cluster_inertia' : cluster_inertia, 
        'cluster_centres' : cluster_centres}
    return output 