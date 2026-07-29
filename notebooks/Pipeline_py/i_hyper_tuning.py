import h_ll_runner as h
import numpy as np
import c_clustering as c
import f_grids_and_outputs as f
import b_run_staging as b
from time import perf_counter
import gc 
import utils as ut

metric_breakdowns = ut.load_json5('metric_breakdowns')['breakdowns']

class Tuner:

    def __init__(self, u_init, v_init, p_init, u_pos_init, v_pos_init, 
                u_clustering, v_clustering, u_pos_clustering, v_pos_clustering, p_pos_clustering, n_counts_init, 
                user_counts_nt, user_interactions_nt, interpolation_weights, bin_metric_nt, output_idx_nt, model_idx_nt, train_test_nt_class, user_type_groups):
            
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
            self.train_test_nt_class = train_test_nt_class
            self.user_type_groups = user_type_groups

    #####################################
    # Decile Creation
    #####################################

    def get_activity_deciles(self, user_counts_nt, n_users, period_start, period_end):
        '''
        Filter activity to pe in period then groupby user id and get counts
        '''
        period_filter = ((user_counts_nt.fine_bin_id >= period_start) & (user_counts_nt.fine_bin_id < period_end))

        return self.make_rank_deciles(np.bincount(user_counts_nt.user_id[period_filter], weights=user_counts_nt.count[period_filter]))

    def make_rank_deciles(self, val_to_rank):
        '''
        Assigns users to deciles based on val_to_rank
        '''
        # Create and rank assignments 
        n_users = val_to_rank.shape[0]
        ranked_groups = (np.arange(n_users, dtype=np.int64) * 10) // n_users

        # Init output and assign users
        output = np.empty(n_users, dtype='int8')
        output[np.argsort(val_to_rank, kind='stable')] = np.minimum(ranked_groups, 9)
        return output

    def get_metric_breakdown(self, model, config_dict, u_clustering, v_clustering, p_clustering, train_test_dict):
        '''
        Creates the breakdowns for the run in the test set
        '''

        n_users = u_clustering.shape[0]
        all_user_group = np.zeros(n_users, dtype='int8')

        activity_deciles = self.get_activity_deciles(user_counts_nt=self.user_counts_nt, n_users=n_users, 
                                    period_start=train_test_dict['train_start'], period_end=train_test_dict['burn_in_end'])

        matrix_to_cluster = c.make_clustering_matrix(u_clustering, v_clustering, p_clustering, config_dict)
        cluster_distances = c.get_user_cluster_distances(matrix_to_cluster=matrix_to_cluster, 
            cluster_assignments=model['cluster_assignments'], cluster_centres=model['cluster_centres'], distance_metric=model['distance_metric'])
        distance_deciles = self.make_rank_deciles(cluster_distances)
        

        return np.vstack((all_user_group, activity_deciles, distance_deciles, self.user_type_groups)).astype('int8')

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
        
    def run_pipeline_ll(self, model, config_nt, train_test_nt, config_dict, degen_mask, breakdown_groups=None):
        ''' 
        Makes a call to the numba lambert liu runner
        '''

        # Init an empty array if we dont have breakdown requirement
        if breakdown_groups is None:
            breakdown_groups = np.empty((0, 0), dtype='int8')

        alpha_mu_grid_init = f.init_alpha_grid(self.n_counts_init, config_dict['linear_smooth'], config_dict['smooth_a_mu'], config_dict['smooth_t_mu'], self.bin_metric_nt.fine_bins_per_coarse_bin)
        alpha_sigma2_grid_init = f.init_alpha_grid(self.n_counts_init, config_dict['linear_smooth'], config_dict['smooth_a_sigma2'], config_dict['smooth_t_sigma2'], self.bin_metric_nt.fine_bins_per_coarse_bin)
        n_counts_p = np.full_like(self.n_counts_init, self.bin_metric_nt.fine_bins_per_coarse_bin, dtype='float64')
        alpha_p_grid_init = f.init_alpha_grid(n_counts_p, config_dict['linear_smooth'], config_dict['smooth_a_p'], config_dict['smooth_t_p'], self.bin_metric_nt.fine_bins_per_coarse_bin)

        u, v, p, _, _, _ = self.get_ll_param_grids(config_dict)
        cluster_u_init, cluster_v_init, cluster_p_init = c.get_cluster_centres(cluster_groups=model['cluster_assignments'], u_init=u, v_init=v, p_init=p, distance_metric=config_dict['distance_metric'])

        return h.run_lambert_liu(
            u_init=u,
            v_init=v,
            p_init=p,
            cluster_u_init=cluster_u_init,
            cluster_v_init=cluster_v_init,
            cluster_p_init=cluster_p_init,
            cluster_groups=model['cluster_assignments'],
            n_counts_init=self.n_counts_init,
            alpha_mu_grid_init=alpha_mu_grid_init,
            alpha_sigma2_grid_init=alpha_sigma2_grid_init,
            alpha_p_grid_init=alpha_p_grid_init,
            degen_mask=degen_mask,
            user_counts_nt=self.user_counts_nt,
            user_interactions_nt=self.user_interactions_nt,
            interpolation_weights=self.interpolation_weights,
            train_test_nt=train_test_nt,
            bin_metric_nt=self.bin_metric_nt,
            config_nt=config_nt,
            output_idx_nt=self.output_idx_nt,
            model_idx_nt=self.model_idx_nt,
            breakdown_groups=breakdown_groups)

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
            'sampling_seed' : config_dict['sampling_seed'],
            'w_inf': config_dict['w_inf'],
            'cluster_param': model['cluster_param'],
            'smooth_a_mu': config_dict['smooth_a_mu'],
            'smooth_a_sigma2': config_dict['smooth_a_sigma2'],
            'smooth_a_p': config_dict['smooth_a_p'],
            'smooth_t_mu': config_dict['smooth_t_mu'],
            'smooth_t_sigma2': config_dict['smooth_t_sigma2'],
            'smooth_t_p': config_dict['smooth_t_p'],
            'linear_smooth': config_dict['linear_smooth'],
            'smoothing_target': config_dict['smoothing_target'],
            'hurdle_model': config_dict['hurdle_model'],
            'test_valid' : test_valid,

            # Log likelihood for non degenerate bins
            'non_degen_ll': output_metrics[period_idx, self.output_idx_nt.non_degen_ll_sum] / n_non_degen_bins,
            'non_degen_smoothed_ll': output_metrics[period_idx, self.output_idx_nt.non_degen_smoothed_ll_sum] / n_non_degen_bins,
            
            # Clustering metrics
            'clustering_matrix_name': model['clustering_matrix_name'],
            'clustering_transformation' : model['clustering_transformation'],
            'distance_metric' : model['distance_metric'],
            'clustering_seed': model['clustering_seed'],
            'cluster_inertia': model['cluster_inertia']}

        return output


    def make_full_output_rows(self, model, calibration_outputs, config_dict, experiment_name):
        '''
        Makes outputs rows for full test runs
        '''
        output = []

        for breakdown_type_idx, breakdown_config in enumerate(metric_breakdowns):
            breakdown_type = breakdown_config['type']
            for breakdown_group_idx, breakdown_group in enumerate(breakdown_config['groups']):
                group_output = calibration_outputs[breakdown_type_idx, breakdown_group_idx]

                n_bins_scored = group_output[0]

                if n_bins_scored == 0:
                    continue

                output_row = {
                    'smoothed_model_name': model['name'],
                    'experiment_name': experiment_name,
                    'sampling_seed': config_dict['sampling_seed'],
                    'w_inf': config_dict['w_inf'],
                    'cluster_param': model['cluster_param'],
                    'smooth_a_mu': config_dict['smooth_a_mu'],
                    'smooth_a_sigma2': config_dict['smooth_a_sigma2'],
                    'smooth_a_p': config_dict['smooth_a_p'],
                    'smooth_t_mu': config_dict['smooth_t_mu'],
                    'smooth_t_sigma2': config_dict['smooth_t_sigma2'],
                    'smooth_t_p': config_dict['smooth_t_p'],
                    'linear_smooth': config_dict['linear_smooth'],
                    'smoothing_target': config_dict['smoothing_target'],
                    'hurdle_model': config_dict['hurdle_model'],
                    'test_valid': 'test',

                    'clustering_matrix_name': model['clustering_matrix_name'],
                    'clustering_transformation': model['clustering_transformation'],
                    'distance_metric': model['distance_metric'],
                    'clustering_seed': model['clustering_seed'],
                    'cluster_inertia':  model['cluster_inertia'],
                    'breakdown_type': breakdown_type,
                    'breakdown_group': breakdown_group,

                    'n_bins_scored': n_bins_scored,
                    'non_degen_ll_sum': group_output[1],
                }

                for threshold_idx, threshold in enumerate(config_dict['calibration_thresholds']):
                    threshold_name = format(float(threshold), 'f')
                    output_row[f'calibration_count_'f'{threshold_name}'] = group_output[2 + threshold_idx]
                output.append(output_row)

        return output

    #####################################
    # Config Sampler
    #####################################
    def sample_configs(self, experiment_name, hyperparams, hurdle_model):
        '''
        Samples a random set of hyperparameters for the given experiment
        '''

        # Fill in w inf and use defaults for the rest
        if experiment_name == 'no_smoothing' or experiment_name == 'model_selection':
            return [{'w_inf': w_inf, 
                     'cluster_param': 1, 'smoothing_target': 0, 'clustering_matrix_name': 'u', 'clustering_transformation': 'none', 'distance_metric': 'l2', 'linear_smooth': True, 'smooth_a_mu': 0, 'smooth_a_sigma2': 0, 'smooth_a_p': 0, 'smooth_t_mu': 0, 'smooth_t_sigma2': 0, 'smooth_t_p': 0}
                for w_inf in hyperparams['w_inf_vals']]

        # Init rng and seen configs
        rng = np.random.default_rng(hyperparams['sampling_seed'])
        output_list = []
        seen_configs = set()

        while len(output_list) < hyperparams['n_hypers_sampled']:

            w_inf = rng.choice(hyperparams['w_inf_vals'])


            # Setting clustering hyperparameters or defaults
            if experiment_name == 'global_smoothing':
                cluster_param = 1
                clustering_matrix_name = 'u'
                clustering_transformation = 'none'
                distance_metric = rng.choice(hyperparams['distance_metric_vals'])
            else:
                clustering_matrix_name = rng.choice(hyperparams['clustering_matrix_name_vals'])
                clustering_transformation = rng.choice(hyperparams['clustering_transformation_vals'])
                distance_metric = rng.choice(hyperparams['distance_metric_vals'])
                cluster_param = rng.choice(hyperparams['cluster_param_vals'])

            linear_smooth = rng.choice(hyperparams['linear_smooth_vals'])
            smoothing_target = rng.choice(hyperparams['smoothing_target_vals'])

            if linear_smooth:
                smooth_a_mu = rng.choice(hyperparams['alpha_vals'])
                smooth_a_sigma2 = rng.choice(hyperparams['alpha_vals'])
                if hurdle_model:
                    smooth_a_p = rng.choice(hyperparams['alpha_vals'])
            # Setting defaults
                else:
                    smooth_a_p = 0
                smooth_t_mu = 0
                smooth_t_sigma2 = 0
                smooth_t_p = 0
            else:
                smooth_t_mu = rng.choice(hyperparams['tau_vals'])
                smooth_t_sigma2 = rng.choice(hyperparams['tau_vals'])
                if hurdle_model:
                    smooth_t_p = rng.choice(hyperparams['tau_vals'])
            # Setting defaults
                else:
                    smooth_t_p = 0
                smooth_a_mu = 0
                smooth_a_sigma2 = 0
                smooth_a_p = 0

            # Creating the row and adding it to the outputs
            hyper_row = {'w_inf': w_inf, 'cluster_param': cluster_param, 'smoothing_target': smoothing_target, 
                         'clustering_matrix_name': clustering_matrix_name, 'clustering_transformation': clustering_transformation, 
                         'distance_metric': distance_metric, 'linear_smooth': linear_smooth, 
                         'smooth_a_mu': smooth_a_mu, 'smooth_a_sigma2': smooth_a_sigma2, 'smooth_a_p': smooth_a_p, 'smooth_t_mu': smooth_t_mu, 'smooth_t_sigma2': smooth_t_sigma2, 'smooth_t_p': smooth_t_p}

            config_tuple = tuple(hyper_row.values())
            if config_tuple not in seen_configs:
                seen_configs.add(config_tuple)
                output_list.append(hyper_row)

        return output_list

    #####################################
    # Tuning loop
    #####################################

    def join_configs_and_hypers(self, config_dict, hurdle_model, sampled_config, sampling_seed):
        '''
        Adds a set of hyperparameters to a temporary copy of the config dict
        '''
        temp_config = config_dict.copy()
        temp_config.update(sampled_config)

        temp_config['hurdle_model'] = hurdle_model
        temp_config['sampling_seed'] = sampling_seed

        return temp_config


    def create_tuning_dict(self, dict):
        ''' 
        Returns a dictionary without a test period used for hyperparameter tuning
        '''
        validation_only_dict = dict.copy()
        validation_only_dict['test_start'] = validation_only_dict['validation_end']
        validation_only_dict['test_end'] = validation_only_dict['validation_end']
        return validation_only_dict


    def tune_models(self, experiment_name, hurdle_model, hyperparams, train_test_dict, config_dict, degen_mask, run_name):
        '''
        Runs the hyperparameter tuning for one fixed experiment name
        '''

        # Creates a dict that allows the runner to not run on test
        validation_only_dict = self.create_tuning_dict(train_test_dict)
        validation_only_nt = self.train_test_nt_class(**validation_only_dict)


        seen_configs = self.sample_configs(experiment_name, hyperparams, hurdle_model)
        first_sampled_config = seen_configs[0]

        # Using the first config in loop to create a nt class
        first_config = self.join_configs_and_hypers(config_dict=config_dict, hurdle_model=hurdle_model, sampled_config=first_sampled_config, 
                                                    sampling_seed=hyperparams['sampling_seed'])
        first_config['nll_only'] = True
        config_nt_class = (b.dictionary_to_named_tuple_class('config_nt', first_config))


        # Creating output lists
        results = []

        # Iterate over the sample configs
        for config_idx, sampled_config in enumerate(seen_configs, start=1):

            # Recording start of config
            config_start = perf_counter()

            # Making a copy of the config dict with our sampled hyperparameters and turning it into nt
            temp_config = self.join_configs_and_hypers(config_dict=config_dict, hurdle_model=hurdle_model, sampled_config=sampled_config, sampling_seed=hyperparams['sampling_seed'])
            temp_config['nll_only'] = True
            temp_config_nt = config_nt_class(**temp_config)

            # Getting the model and the cluster grids
            _, _, _, u_cluster, v_cluster, p_cluster = self.get_ll_param_grids(temp_config)
            model = c.make_cluster_model(cluster_param=temp_config['cluster_param'], runtime_configs=temp_config, u_init=u_cluster, v_init=v_cluster, p_init=p_cluster)

            # Running the LL pipeline
            output_metrics, calibration_outputs, *_ = self.run_pipeline_ll(model=model, config_nt=temp_config_nt, train_test_nt=validation_only_nt, config_dict=temp_config, degen_mask=degen_mask)

            # Updating outputs
            result_row = self.make_output_table_row(model=model, output_metrics=output_metrics, config_dict=temp_config, test_valid='valid', experiment_name=experiment_name)
            results.append(result_row)


            # Storing completed configs
            if run_name is not None:
                ut.store_run_results(results=[result_row], dir=f'{run_name}/nll_only', run_name=run_name)

            print(f'finished_config {config_idx}/{len(seen_configs)} in {perf_counter() - config_start:.1f}s')

            # cleanup
            del model, output_metrics, calibration_outputs, temp_config, temp_config_nt, u_cluster, v_cluster, p_cluster
            gc.collect()

        return results

    #####################################
    # Test set runner
    #####################################

    def test_run(self, experiment_name, hurdle_nb_model, selected_config, train_test_dict, base_config, degen_mask, bin_metric_dict, nll_only=False):
        '''
        Runs the best config on the test set depending on the experiment name
        '''
        if selected_config is None:
            raise ValueError(f'Config not found for {experiment_name}')

        if hurdle_nb_model['hurdle_model'] is None:
            raise ValueError('Should only be run after selecting hurdle or NB model')

        # Create one complete runnable configuration
        best_config = ut.merge_configs(base_config, hurdle_nb_model, selected_config)
        best_config['nll_only'] = nll_only

        _, config_nt, _, train_test_nt, _ = b.converting_dicts_to_nt(best_config, train_test_dict, bin_metric_dict)

        _, _, _, u_cluster, v_cluster, p_cluster = self.get_ll_param_grids(best_config)
        test_model = c.make_cluster_model(cluster_param=best_config['cluster_param'], runtime_configs=best_config, u_init=u_cluster, v_init=v_cluster, p_init=p_cluster,)

        # Getting deciles for breakdown if needed
        if best_config['nll_only']:
            breakdown_groups = None
        else:
            breakdown_groups = self.get_metric_breakdown(model=test_model, config_dict=best_config, u_clustering=u_cluster, 
                v_clustering=v_cluster, p_clustering=p_cluster, train_test_dict=train_test_dict)

        output_metrics, calibration_outputs, *_ = self.run_pipeline_ll(model=test_model, config_nt=config_nt, 
            train_test_nt=train_test_nt, config_dict=best_config, degen_mask=degen_mask, breakdown_groups=breakdown_groups)

        if best_config['nll_only']:
            test_results = [self.make_output_table_row(model=test_model, output_metrics=output_metrics, config_dict=best_config, test_valid='test', experiment_name=experiment_name)]
        else:
            test_results = self.make_full_output_rows(model=test_model, calibration_outputs=calibration_outputs, config_dict=best_config, experiment_name=experiment_name)
        return test_results