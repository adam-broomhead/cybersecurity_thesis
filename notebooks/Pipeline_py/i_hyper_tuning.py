import h_ll_runner as h
import numpy as np
import c_clustering as c
import f_grids_and_outputs as f
from itertools import product

class Tuner:

    def __init__(self, u_init, v_init, p_init, u_pos_init, v_pos_init, 
                u_clustering, v_clustering, u_pos_clustering, v_pos_clustering, p_pos_clustering, n_counts_init, 
                user_counts_nt, user_interactions_nt, interpolation_weights, bin_metric_nt, output_idx_nt, model_idx_nt, 
                config_nt_class, train_test_nt_class):
            
            self.u_init = u_init
            self.v_init = v_init
            self.p_init = p_init
            self.u_pos_init = u_pos_init
            self.v_pos_init = v_pos_init
            self.u_clustering = u_clustering
            self.v_clustering = v_clustering
            self.u_pos_clustering = u_pos_clustering
            self.v_pos_clustering = v_pos_clustering
            self.p_pos_clustering = p_pos_clustering
            self.n_counts_init = n_counts_init
            self.user_counts_nt = user_counts_nt
            self.user_interactions_nt = user_interactions_nt
            self.interpolation_weights = interpolation_weights
            self.bin_metric_nt = bin_metric_nt
            self.output_idx_nt = output_idx_nt
            self.model_idx_nt = model_idx_nt
            self.config_nt_class = config_nt_class
            self.train_test_nt_class = train_test_nt_class

    #####################################
    # ll single iteration runner
    #####################################

    def get_ll_param_grids(self, config_dict):
        '''
        If hurdle model creates a grid of 0s for p else just returns grids
        '''
        if config_dict['hurdle_model']:
            return self.u_pos_init, self.v_pos_init, self.p_init, self.u_pos_clustering, self.v_pos_clustering, self.p_pos_clustering
        else:
            return self.u_init, self.v_init, np.zeros_like(self.u_init), self.u_clustering, self.v_clustering, np.zeros_like(self.u_clustering)
        
    def run_pipeline_ll(self, model, config_nt, train_test_nt, config_dict, degen_mask):
        ''' 
        Makes a call to the numba lambert liu runner
        '''
        u, v, p, _, _, _ = self.get_ll_param_grids(config_dict)

        return h.run_lambert_liu(
            u_init=u,
            v_init=v,
            p_init=p,
            cluster_u_init=model['cluster_mean_u'],
            cluster_v_init=model['cluster_mean_v'],
            cluster_p_init=model['cluster_mean_p'],
            cluster_groups=model['cluster_assignments'],
            n_counts_init=self.n_counts_init,
            alpha_grid_init=f.init_alpha_grid(self.n_counts_init, config_dict),
            degen_mask=degen_mask,
            user_counts_nt=self.user_counts_nt,
            user_interactions_nt=self.user_interactions_nt,
            interpolation_weights=self.interpolation_weights,
            train_test_nt=train_test_nt,
            bin_metric_nt=self.bin_metric_nt,
            config_nt=config_nt,
            output_idx_nt=self.output_idx_nt,
            model_idx_nt=self.model_idx_nt)

    #####################################
    # Output row creation
    #####################################


    def make_output_table_row(self, model, output_metrics, config_dict, test_valid, experiment_name):
        ''' 
        Transforms the output row into a readable dictionary that can be used as an output df row.
        Args:
            test_valid : Can take values `test` and `valid` tells us what period we are in

        '''

        # Init variables
        if test_valid == 'valid':
            period_idx = 0
        elif test_valid == 'test':
            period_idx = 1
        else:
            raise ValueError('test_valid must either be `test` or `valid`')
        n_non_degen_bins = output_metrics[period_idx, self.output_idx_nt.n_bins_scored]
        

        output = {
            # Row descriptions
            'smoothed_model_name': model['name'],
            'experiment_name' : experiment_name,
            'w': config_dict['w'],
            'cluster_param': model['cluster_param'],
            'smooth_a': config_dict['smooth_a'],
            'smooth_k': config_dict['smooth_k'],
            'linear_smooth': config_dict['linear_smooth'],
            'smoothing_target': config_dict['smoothing_target'],
            'hurdle_model': config_dict['hurdle_model'],
            'test_valid' : test_valid,

            # Log likelihood for non degenerate bins
            'non_degen_ll': output_metrics[period_idx, self.output_idx_nt.non_degen_ll_sum] / n_non_degen_bins,
            'non_degen_smoothed_ll': output_metrics[period_idx, self.output_idx_nt.non_degen_smoothed_ll_sum] / n_non_degen_bins,
            
            # Clustering metrics
            'clustering_matrix_name': model['clustering_matrix_name'],
            'seed': model['seed'],
            'cluster_inertia': model['cluster_inertia']}

        return output


    def make_calibration_output_rows(self, model, output_metrics, calibration_outputs, test_valid, config_dict, experiment_name):
        ''' 
        Creates a table of calibration outputs
        '''

        # Init variables and outputs
        if test_valid == 'valid':
            period_idx = 0
        elif test_valid == 'test':
            period_idx = 1
        else:
            raise ValueError('test_valid must either be `test` or `valid`')
        
        output = []
        n_non_degen_bins = output_metrics[period_idx, self.output_idx_nt.n_bins_scored]

        for threshold_idx in range(config_dict['calibration_thresholds'].shape[0]):

            ## Appending a row to output
            output.append({
                # Row descriptions
                'smoothed_model_name': model['name'],
                'experiment_name' : experiment_name,
                'w': config_dict['w'],
                'cluster_param': model['cluster_param'],
                'smooth_a': config_dict['smooth_a'],
                'smooth_k': config_dict['smooth_k'],
                'linear_smooth': config_dict['linear_smooth'],
                'smoothing_target': config_dict['smoothing_target'],
                'hurdle_model': config_dict['hurdle_model'],
                'clustering_matrix_name': model['clustering_matrix_name'],
                'test_valid': test_valid,

                # Calibration metrics
                'threshold': config_dict['calibration_thresholds'][threshold_idx],
                'raw_tail_rate': calibration_outputs[period_idx, threshold_idx, self.model_idx_nt.raw_model_calib_index] / n_non_degen_bins,
                'smoothed_tail_rate': calibration_outputs[period_idx, threshold_idx, self.model_idx_nt.smoothed_model_calib_index] / n_non_degen_bins,
            })

        return output

    #####################################
    # Tuning loop
    #####################################

    def join_configs_and_hypers(self, config_dict, w, cluster_param, hurdle_model, smoothing_target, linear_smooth, smooth_a, smooth_k, clustering_matrix_name):
        ''' 
        Adds a set of hyperparameters to a temporary copy of the config dict
        '''
        temp_config = config_dict.copy()

        temp_config['w'] = w
        temp_config['cluster_param'] = cluster_param
        temp_config['hurdle_model'] = hurdle_model
        temp_config['smoothing_target'] = smoothing_target
        temp_config['linear_smooth'] = linear_smooth
        temp_config['smooth_a'] = smooth_a
        temp_config['smooth_k'] = smooth_k
        temp_config['clustering_matrix_name'] = clustering_matrix_name

        return temp_config


    def create_tuning_dict(self, dict):
        ''' 
        Returns a dictionary without a test period used for hyperparameter tuning
        '''
        validation_only_dict = dict.copy()
        validation_only_dict['test_start'] = validation_only_dict['validation_end']
        validation_only_dict['test_end'] = validation_only_dict['validation_end']
        return validation_only_dict

    def sample_configs(self, experiment_name, hyperparams):
        '''
        Samples a random set of hyperparameters for the given experiment
        '''
        # If we dont smooth just iterate over w values and have the rest filled with defaults
        if experiment_name == 'no_smoothing':
            return [
                {'w': w, 'cluster_param': 1, 'smoothing_target': 0, 'linear_smooth': True,
                'smooth_a': 0.0, 'smooth_k': hyperparams['smoothing_k_vals'][0], 'clustering_matrix_name': 'u'} for w in hyperparams['w_vals']]

        hypers_list = []

        # Iterate over the smooth a configs
        for w, smoothing_target, smooth_a in product(hyperparams['w_vals'], hyperparams['smoothing_target'], hyperparams['smoothing_a_vals']):
            hyper_row = {'w': w, 'smoothing_target': smoothing_target, 'linear_smooth': True, 'smooth_a': smooth_a, 'smooth_k': hyperparams['smoothing_k_vals'][0]}
            hypers_list.append(hyper_row)

        # Iterate over the smooth k configs
        for w, smoothing_target, smooth_k in product(hyperparams['w_vals'], hyperparams['smoothing_target'], hyperparams['smoothing_k_vals']):
            hyper_row = { 'w': w, 'smoothing_target': smoothing_target, 'linear_smooth': False, 'smooth_a': hyperparams['smoothing_a_vals'][0], 'smooth_k': smooth_k}
            hypers_list.append(hyper_row)

        ## Adding cluster configs to smoothing configs
        if experiment_name == 'global_smoothing':
            hyper_list = [{**hyper_dict, 'cluster_param': 1, 'clustering_matrix_name': 'u'} for hyper_dict in hypers_list]

        elif experiment_name == 'cluster_smoothing':
            for hyper_dict, cluster_param, clustering_matrix_name in product(hypers_list, hyperparams['cluster_param_vals'], hyperparams['clustering_matrix_name_vals']):
                hyper_list = [{**hyper_dict, 'cluster_param': cluster_param, 'clustering_matrix_name': clustering_matrix_name}]
        else:
            raise ValueError('Invalid experimental name')

        # Sample from created lists 
        rng = np.random.default_rng(hyperparams['seed'])
        sampled_indices = rng.choice(len(hyper_list), size=min(hyperparams['n_random_configs'], len(hyper_list)), replace=False)

        return [hyper_list[idx] for idx in sampled_indices]


    def tune_models(self, experiment_name, hurdle_model, hyperparams, train_test_dict, config_dict, degen_mask):
        '''
        Runs the hyperparameter tuning for one fixed experiment name
        '''

        # Creates a dict that allows the runner to not run on test
        validation_only_dict = self.create_tuning_dict(train_test_dict)
        validation_only_nt = self.train_test_nt_class(**validation_only_dict)

        sampled_configs = self.sample_configs(experiment_name, hyperparams)

        # Creating output lists
        results = []
        calibration_results = []
        hypers_tested = []

        # Iterate over the sample configs
        for sampled_config in sampled_configs:

            # Making a copy of the config dict with our sampled hyperparameters and turning it into nt
            temp_config = self.join_configs_and_hypers(config_dict=config_dict, w=sampled_config['w'], cluster_param=sampled_config['cluster_param'],
                                                hurdle_model=hurdle_model, smoothing_target=sampled_config['smoothing_target'], linear_smooth=sampled_config['linear_smooth'], 
                                                smooth_a=sampled_config['smooth_a'], smooth_k=sampled_config['smooth_k'], clustering_matrix_name=sampled_config['clustering_matrix_name'])
            temp_config_nt = self.config_nt_class(**temp_config)

            # Getting the model and the cluster grids
            _, _, _, u_cluster, v_cluster, p_cluster = self.get_ll_param_grids(temp_config)
            model = c.make_cluster_model(cluster_param=temp_config['cluster_param'], runtime_configs=temp_config, u_init=u_cluster, v_init=v_cluster, p_init=p_cluster)

            # Running the LL pipeline
            output_metrics, calibration_outputs, *_ = self.run_pipeline_ll(model=model, config_nt=temp_config_nt, train_test_nt=validation_only_nt, config_dict=temp_config, degen_mask=degen_mask)

            # Updating outputs
            results.append(self.make_output_table_row(model=model, output_metrics=output_metrics, config_dict=temp_config, test_valid='valid', 
                                                      experiment_name=experiment_name))
            calibration_results.extend(self.make_calibration_output_rows(model=model, output_metrics=output_metrics, calibration_outputs=calibration_outputs, 
                                                                         test_valid='valid', config_dict=temp_config, experiment_name=experiment_name))
            hypers_tested.append(temp_config)

        return results, calibration_results, hypers_tested
