from numba import njit
import numpy as np


#####################################
# Interpolation and smoothing logic
#####################################

@njit(inline='always')
def smoothing_function(alpha, user_parameter, target_parameter):
    return (1.0 - alpha) * user_parameter + alpha * target_parameter


@njit 
def interpolate_values(v_neg_1, v_0, v_1, fine_bin_within_coarse_pos, interpolation_weights):
    '''
    Applies the interpolation between the left middle and right bin values
    '''
    # Extract the weights we will use for the interpolation
    w_neg_1, w_0, w_1 = interpolation_weights[fine_bin_within_coarse_pos]
    return w_neg_1 * v_neg_1 + w_0 * v_0 + w_1 * v_1

#####################################
# Getting target value to smooth towards
#####################################

@njit
def get_grid_row_sums(grid):
    ''' 
    Gets the sum of a parameter grid
    Needed for seperating the shape from the rate and vice versa
    '''
    # Extract the number of rows and init the ouptu gird
    n_rows, n_cols = grid.shape
    output = np.zeros(n_rows, dtype='float64')

    # Loop over and count element contributions
    for row_idx in range(n_rows):
        total = 0
        for col_idx in range(n_cols):
            total += grid[row_idx, col_idx]
        output[row_idx] = total
    return output


@njit 
def get_cluster_target(user_grid, cluster_grid, cluster_assignments, user_totals, cluster_totals, crnt_user, crnt_coarse_bin, config_nt):
    ''' 
    Returns a variable to pass to smooth values for the cluster value
    '''
    # Get the assignment val for user and cluster
    cluster_id = cluster_assignments[crnt_user]
    user_val = user_grid[crnt_user, crnt_coarse_bin]
    cluster_val = cluster_grid[cluster_id, crnt_coarse_bin]

    user_total = user_totals[crnt_user]
    cluster_total =cluster_totals[cluster_id] 

    # Absolute smoothing (we smooth towards tehc cluster value)
    if config_nt.smoothing_target == 0:
        return cluster_val

    # Shape smoothing (we smooth towards the custer shape while keeping the user rate)
    if config_nt.smoothing_target == 1:
        cluster_shape = cluster_val / cluster_total
        return user_total * cluster_shape

    # Rate smoothing we smooth towards cluster rate keeping the users shape
    if config_nt.smoothing_target == 2:
        user_shape = user_val / user_total
        return cluster_total * user_shape
    
#####################################
# Smoothing parameter functions
#####################################

@njit 
@njit 
def smooth_params(user_param_grid, cluster_param_grid, cluster_assignments, user_totals, cluster_totals, 
                  alpha_grid, degen_mask, crnt_user, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt):
    '''
    Interpolated params with the interpolation weights
    Smooths params if applicable
    '''

    # Get the 3 values to interpolate
    n_coarse_bins = user_param_grid.shape[1]
    neg_1_coarse_bin = (crnt_coarse_bin -1) % n_coarse_bins
    _1_coarse_bin = (crnt_coarse_bin + 1)% n_coarse_bins

    # Getting the values of alpha for smoothing
    alpha_neg_1 = alpha_grid[crnt_user, neg_1_coarse_bin]
    alpha_0 = alpha_grid[crnt_user, crnt_coarse_bin]
    alpha_1 = alpha_grid[crnt_user, _1_coarse_bin]

    # Getting the cluster targets (values we smooth towards)
    target_neg_1 = get_cluster_target(user_param_grid, cluster_param_grid, cluster_assignments, user_totals, cluster_totals, crnt_user, neg_1_coarse_bin, config_nt)
    target_0 = get_cluster_target(user_param_grid, cluster_param_grid, cluster_assignments, user_totals, cluster_totals, crnt_user, crnt_coarse_bin, config_nt)
    target_1 = get_cluster_target(user_param_grid, cluster_param_grid, cluster_assignments, user_totals, cluster_totals, crnt_user, _1_coarse_bin, config_nt)

    ## Smooth the 3 values we will later interpolate if alpha is non zero
    if alpha_neg_1 == 0:
        v_neg_1 = user_param_grid[crnt_user, neg_1_coarse_bin]
    else:
        v_neg_1 = smoothing_function(alpha_neg_1, user_param_grid[crnt_user, neg_1_coarse_bin], target_neg_1)
    if alpha_0 == 0:
        v_0 = user_param_grid[crnt_user, crnt_coarse_bin]
    else:
        v_0 = smoothing_function(alpha_0, user_param_grid[crnt_user, crnt_coarse_bin], target_0)
    if alpha_1 == 0:
        v_1 =  user_param_grid[crnt_user, _1_coarse_bin]
    else:
        v_1 = smoothing_function(alpha_1, user_param_grid[crnt_user, _1_coarse_bin], target_1)

    # Do not interpolate towards a degenerate positive count distribution
    if degen_mask is not None:
        if degen_mask[crnt_user, neg_1_coarse_bin]:
            v_neg_1 = v_0
        if degen_mask[crnt_user, _1_coarse_bin]:
            v_1 = v_0

    # Interpolate the values 
    return interpolate_values(v_neg_1, v_0, v_1, crnt_fine_bin_within_coarse_pos, interpolation_weights)

@njit
def get_smoothed_params(u, v, p, cluster_u, cluster_v, cluster_p, cluster_assignments, 
                        user_u_totals, user_v_totals, user_p_totals, cluster_u_totals, cluster_v_totals, cluster_p_totals, 
                        alpha_mu_grid, alpha_sigma2_grid, alpha_p_grid, degen_mask, crnt_user, crnt_coarse_bin, 
                        crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt):
    '''
    Smooths mu sigma2 and p
    '''
    
    mu = smooth_params(u, cluster_u, cluster_assignments, user_u_totals, cluster_u_totals, 
                       alpha_mu_grid, degen_mask, crnt_user, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)

    sigma2 = smooth_params(v, cluster_v, cluster_assignments, user_v_totals, cluster_v_totals, 
                           alpha_sigma2_grid, degen_mask, crnt_user, crnt_coarse_bin, crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)

    if config_nt.hurdle_model:
        p_val = smooth_params(p, cluster_p, cluster_assignments, user_p_totals, cluster_p_totals, 
                              alpha_p_grid, None, crnt_user, crnt_coarse_bin, 
                              crnt_fine_bin_within_coarse_pos, interpolation_weights, config_nt)

        # Clipping values above 1 if we shape or rate smooth
        if config_nt.smoothing_target == 1 or config_nt.smoothing_target == 2:
            p_val = min(p_val, 1.0)
    else:
        p_val = 0

    return mu, sigma2, p_val

