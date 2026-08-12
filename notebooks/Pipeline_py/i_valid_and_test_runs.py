import h_ll_runner as h
import numpy as np
import c_clustering as c
import f_grids_and_outputs as f
import b_run_staging as b
from time import perf_counter
import gc 
import utils as ut
import k_metric_breakdowns as k
from scipy.special import ndtri
from numba import njit, prange

metric_breakdowns = ut.load_json5('metric_breakdowns')['breakdowns']

class Tuner:

    def __init__(self, u_init, v_init, p_init, u_pos_init, v_pos_init, 
                u_clustering, v_clustering, u_pos_clustering, v_pos_clustering, p_pos_clustering, n_counts_init, 
                user_counts_nt, user_interactions_nt, interpolation_weights, bin_metric_nt, output_idx_nt, model_idx_nt, 
                train_test_nt_class, user_type_groups, quadratic_interpolation=False):
            
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
            self.quadratic_interpolation = quadratic_interpolation
    
    #####################################
    # ll single iteration runner
    #####################################

    def get_ll_param_grids(self, config_dict):
        '''
        If not a hurdle model creates a grid of 0s, else just returns grids from the config dict
        '''
        if config_dict['hurdle_model']:
            return self.u_pos_init, self.v_pos_init, self.p_init, self.u_pos_clustering, self.v_pos_clustering, self.p_pos_clustering
        else:
            return self.u_init, self.v_init, np.zeros_like(self.u_init), self.u_clustering, self.v_clustering, np.zeros_like(self.u_clustering)
        
    def run_pipeline_ll(self, model, config_nt, train_test_nt, config_dict, degen_mask, breakdown_groups=None, attack_start_fb=None, attack_sizes=None):
        ''' 
        Makes a call to the numba lambert liu runner
        '''

        # Init empty defaults is args are not passed for breakdown or attacks
        if breakdown_groups is None:
            breakdown_groups = np.empty((0, 0), dtype='int8')
        if attack_start_fb is None:
            attack_start_fb = np.empty(0, dtype='int64')
        if attack_sizes is None:
            attack_sizes = np.empty(0, dtype='int64')

        # Init grids needed for the ll runner
        alpha_mu_grid_init = f.init_alpha_grid(self.n_counts_init, config_dict['constant_alpha'], config_dict['smooth_a_mu'], config_dict['smooth_t_mu'], self.bin_metric_nt.fine_bins_per_coarse_bin)
        alpha_sigma2_grid_init = f.init_alpha_grid(self.n_counts_init, config_dict['constant_alpha'], config_dict['smooth_a_sigma2'], config_dict['smooth_t_sigma2'], self.bin_metric_nt.fine_bins_per_coarse_bin)
        n_counts_p = np.full_like(self.n_counts_init, self.bin_metric_nt.fine_bins_per_coarse_bin, dtype='float64')
        alpha_p_grid_init = f.init_alpha_grid(n_counts_p, config_dict['constant_alpha'], config_dict['smooth_a_p'], config_dict['smooth_t_p'], self.bin_metric_nt.fine_bins_per_coarse_bin)
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
            breakdown_groups=breakdown_groups,
            quadratic_interpolation=self.quadratic_interpolation,
            attack_start_fb=attack_start_fb,
            attack_sizes=attack_sizes)

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
            # Configs
            'smoothed_model_name': model['name'],
            'experiment_name' : experiment_name,
            'seed' : config_dict['seed'],
            'w_inf': config_dict['w_inf'],
            'cluster_param': model['cluster_param'],
            'smooth_a_mu': config_dict['smooth_a_mu'],
            'smooth_a_sigma2': config_dict['smooth_a_sigma2'],
            'smooth_a_p': config_dict['smooth_a_p'],
            'smooth_t_mu': config_dict['smooth_t_mu'],
            'smooth_t_sigma2': config_dict['smooth_t_sigma2'],
            'smooth_t_p': config_dict['smooth_t_p'],
            'constant_alpha': config_dict['constant_alpha'],
            'smoothing_target': config_dict['smoothing_target'],
            'hurdle_model': config_dict['hurdle_model'],
            'test_valid' : test_valid,

            # Output metrics
            'non_degen_ll': output_metrics[period_idx, self.output_idx_nt.non_degen_ll_sum] / n_non_degen_bins,
            
            # Clustering metrics
            'clustering_matrix_name': model['clustering_matrix_name'],
            'clustering_transformation' : model['clustering_transformation'],
            'distance_metric' : model['distance_metric'],
            'cluster_inertia': model['cluster_inertia']}

        return output


    def make_full_output_rows(self, model, calibration_outputs, config_dict, experiment_name):
        '''
        Makes outputs rows for full test runs involving breakdowns and calibration
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
                    # Configs
                    'smoothed_model_name': model['name'],
                    'experiment_name': experiment_name,
                    'seed': config_dict['seed'],
                    'w_inf': config_dict['w_inf'],
                    'cluster_param': model['cluster_param'],
                    'smooth_a_mu': config_dict['smooth_a_mu'],
                    'smooth_a_sigma2': config_dict['smooth_a_sigma2'],
                    'smooth_a_p': config_dict['smooth_a_p'],
                    'smooth_t_mu': config_dict['smooth_t_mu'],
                    'smooth_t_sigma2': config_dict['smooth_t_sigma2'],
                    'smooth_t_p': config_dict['smooth_t_p'],
                    'constant_alpha': config_dict['constant_alpha'],
                    'smoothing_target': config_dict['smoothing_target'],
                    'hurdle_model': config_dict['hurdle_model'],
                    'test_valid': 'test',

                    # Clustering hypers
                    'clustering_matrix_name': model['clustering_matrix_name'],
                    'clustering_transformation': model['clustering_transformation'],
                    'distance_metric': model['distance_metric'],
                    'cluster_inertia':  model['cluster_inertia'],

                    # Breakdown groups
                    'breakdown_type': breakdown_type,
                    'breakdown_group': breakdown_group,

                    # Output metrics
                    'n_bins_scored': n_bins_scored,
                    'non_degen_ll_sum': group_output[1],
                }

                # First two rows are log likelihood + n bins rest are calibration
                for threshold_idx, threshold in enumerate(config_dict['calibration_thresholds']):
                    threshold_name = format(float(threshold), 'f')
                    output_row[f'calibration_count_{threshold_name}'] = group_output[2 + threshold_idx]
                output.append(output_row)

        return output

    def make_user_output_table(self, model, user_output_metrics, experiment_name, config_dict, u_clustering, v_clustering, p_clustering):
        '''
        Creates the user-level log likelihood output table
        '''
        n_users = user_output_metrics.shape[0]

        output = {'user_idx': np.arange(n_users, dtype=np.int32),
                  'seed': np.full(n_users, model['seed'], dtype=np.int32), 
                  'non_degen_ll_sum': user_output_metrics[:, 1], 
                  'n_bins_scored': user_output_metrics[:, 0]}

        if experiment_name == 'cluster_smoothing':
            matrix_to_cluster = c.make_clustering_matrix(u=u_clustering, v=v_clustering, p=p_clustering, runtime_configs=config_dict)

            output['cluster_distance'] = c.get_user_cluster_distances(matrix_to_cluster=matrix_to_cluster, cluster_assignments=model['cluster_assignments'], 
                                            cluster_centres=model['cluster_centres'], distance_metric=model['distance_metric'])
        return output

    #####################################
    # Config Sampler
    #####################################
    def sample_configs(self, experiment_name, hyperparams, hurdle_model, seed):
        '''
        Samples a random set of hyperparameters for the given experiment
        '''

        # Fill in w inf and use defaults for the rest
        if experiment_name == 'no_smoothing' or experiment_name == 'LL_runner' or experiment_name.startswith('ablation_'):
            return [{'w_inf': w_inf, 
                     'cluster_param': 1, 'smoothing_target': 0, 'clustering_matrix_name': 'u', 'clustering_transformation': 'none', 'distance_metric': 'l2', 'constant_alpha': True, 'smooth_a_mu': 0, 'smooth_a_sigma2': 0, 'smooth_a_p': 0, 'smooth_t_mu': 0, 'smooth_t_sigma2': 0, 'smooth_t_p': 0}
                for w_inf in hyperparams['w_inf_vals']]

        # Init rng and seen configs
        rng = np.random.default_rng(seed)
        output_list = []
        seen_configs = set()

        # Keep sampling until we hit the required number to sample
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

            constant_alpha = rng.choice(hyperparams['constant_alpha_vals'])
            smoothing_target = rng.choice(hyperparams['smoothing_target_vals'])

            if constant_alpha:
                smooth_a_mu = rng.choice(hyperparams['alpha_mu_vals'])
                smooth_a_sigma2 = rng.choice(hyperparams['alpha_sigma2_vals'])
                if hurdle_model:
                    smooth_a_p = rng.choice(hyperparams['alpha_p_vals'])
                # Setting defaults
                else:
                    smooth_a_p = 0
                smooth_t_mu = 0
                smooth_t_sigma2 = 0
                smooth_t_p = 0
            else:
                smooth_t_mu = rng.choice(hyperparams['tau_mu_vals'])
                smooth_t_sigma2 = rng.choice(hyperparams['tau_sigma2_vals'])
                if hurdle_model:
                    smooth_t_p = rng.choice(hyperparams['tau_p_vals'])
            # Setting defaults
                else:
                    smooth_t_p = 0
                smooth_a_mu = 0
                smooth_a_sigma2 = 0
                smooth_a_p = 0

            # Creating the row and adding it to the outputs
            hyper_row = {'w_inf': w_inf, 'cluster_param': cluster_param, 'smoothing_target': smoothing_target, 
                         'clustering_matrix_name': clustering_matrix_name, 'clustering_transformation': clustering_transformation, 
                         'distance_metric': distance_metric, 'constant_alpha': constant_alpha, 
                         'smooth_a_mu': smooth_a_mu, 'smooth_a_sigma2': smooth_a_sigma2, 'smooth_a_p': smooth_a_p, 'smooth_t_mu': smooth_t_mu, 'smooth_t_sigma2': smooth_t_sigma2, 'smooth_t_p': smooth_t_p}

            # Storing hyper values in seen configs and adding to output
            config_tuple = tuple(hyper_row.values())
            if config_tuple not in seen_configs:
                seen_configs.add(config_tuple)
                output_list.append(hyper_row)
        return output_list

    #####################################
    # Tuning loop
    #####################################
    def join_configs_and_hypers(self, config_dict, hurdle_model, sampled_config):
        '''
        Adds a set of hyperparameters to a temporary copy of the config dict
        '''
        temp_config = config_dict.copy()
        temp_config.update(sampled_config)
        temp_config['hurdle_model'] = hurdle_model

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


        seen_configs = self.sample_configs(experiment_name, hyperparams, hurdle_model, seed=config_dict['seed'])
        first_sampled_config = seen_configs[0]

        # Using the first config in loop to create a nt class
        first_config = self.join_configs_and_hypers(config_dict=config_dict, hurdle_model=hurdle_model, sampled_config=first_sampled_config)
        first_config['nll_only'] = True
        config_nt_class = (b.dictionary_to_named_tuple_class('config_nt', first_config))


        # Creating output lists
        results = []

        # Iterate over the sample configs
        for config_idx, sampled_config in enumerate(seen_configs, start=1):

            # Recording start of config
            config_start = perf_counter()

            # Making a copy of the config dict with our sampled hyperparameters and turning it into nt
            temp_config = self.join_configs_and_hypers(config_dict=config_dict, hurdle_model=hurdle_model, sampled_config=sampled_config)
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


    # ####################################
    # Attack simulation
    # ####################################

    def get_attack_hour(degen_mask, train_test_nt, bin_metric_nt, seed):
        '''
        Selects an hour to attack in the second week of test for each user
        '''
        rng = np.random.default_rng(seed)
        n_users = degen_mask.shape[0]
        n_hours_in_mask = degen_mask.shape[1]

        # Init attack time array
        attack_start_fb = np.full(n_users, -1, dtype='int64')
        hours_in_wk_2 = np.arange(7 * 24, 14 * 24)

        for user_id in range(n_users):
            eligible_hours = hours_in_wk_2[~degen_mask[user_id, hours_in_wk_2 % n_hours_in_mask]]

            if eligible_hours.size > 0:
                attack_hour = rng.choice(eligible_hours)
                attack_start_fb[user_id] = train_test_nt.test_start + attack_hour * bin_metric_nt.fine_bins_per_coarse_bin

        return attack_start_fb

    @njit(parallel=True)
    def get_ewma_scores(z_scores, alert_w, initial_scores):
        '''
        Updates EWMA scores using z scores
        '''
        ewma_scores = np.empty_like(z_scores)
        for usr_row_idx in prange(z_scores.shape[0]):
            current_score = initial_scores[usr_row_idx]

            for fine_bin_idx in range(z_scores.shape[1]):
                z = z_scores[usr_row_idx, fine_bin_idx]
                # Update non degens
                if not np.isnan(z):
                    current_score = (1 - alert_w) * current_score + alert_w * z
                ewma_scores[usr_row_idx, fine_bin_idx] = current_score

        return ewma_scores

    def get_z_scores(p):
        '''
        Gets the z scores for all non-degenerate bins
        '''
        z = np.full_like(p, np.nan)
        mask = ~np.isnan(p)
        z[mask] = -ndtri(p[mask])
        return z

    def get_attack_max_scores(attack_z_vals, observed_ewma, attack_start_fb, alert_w, train_test_nt):
        '''
        Gets the highest EWMA score during the attack for each fb
        '''

        # Init variables
        n_users, n_attack_sizes, n_attack_fbs = attack_z_vals.shape
        initial_scores = np.zeros(n_users, dtype='float64')

        # Getting initial scored of EWMA before bin starts
        usrs_possible_to_attack = attack_start_fb >= 0
        attackable_user_idxs = np.where(usrs_possible_to_attack)[0]
        attack_test_relative_fb = attack_start_fb - train_test_nt.test_start
        initial_scores[attackable_user_idxs] = observed_ewma[attackable_user_idxs, attack_test_relative_fb[attackable_user_idxs] - 1]

        # Running EWMA and taking the max
        attack_ewma = get_ewma_scores(attack_z_vals.reshape(n_users * n_attack_sizes, n_attack_fbs), alert_w, np.repeat(initial_scores, n_attack_sizes))
        attack_ewma = attack_ewma.reshape(n_users, n_attack_sizes, n_attack_fbs)
        attack_max = np.max(attack_ewma, axis=2)

        return attack_max, usrs_possible_to_attack

    def get_detection_results(observed_p_vals, attack_p_vals, attack_start_fb, attack_sizes, alert_w_vals, fpr_rates,
                            train_test_nt, bin_metric_nt, seed, experiment_name):
        '''
        Gets the results for the injected attack detection from the observed p values over the two weeks 
        and the attack p vals for a given hour
        '''
        # Get degen from p vals
        non_degen_fb_mask = ~np.isnan(observed_p_vals)

        # Get the observed z score for thresholding and attack p vals
        observed_z_vals = get_z_scores(observed_p_vals)
        attack_z_vals = get_z_scores(attack_p_vals)

        n_users = observed_p_vals.shape[0]
        results = []

        # first 6 hours to test end calibration period and shrink the non degen mask to this period
        threshold_calibration_start = 6 * bin_metric_nt.fine_bins_per_coarse_bin
        threshold_calibration_end = bin_metric_nt.fine_bins_per_week
        non_degen_fb_mask = non_degen_fb_mask[:, threshold_calibration_start:threshold_calibration_end]

        for alert_w in alert_w_vals:
            observed_ewma = get_ewma_scores(observed_z_vals, alert_w, np.zeros(n_users))
            calibration_ewma = observed_ewma[:, threshold_calibration_start:threshold_calibration_end]
            threshold_scores = calibration_ewma[non_degen_fb_mask]

            # Get thesholds and use them to compute max EWMA during attack
            thresholds = np.quantile(threshold_scores, 1 - np.asarray(fpr_rates))
            attack_max, users_attacked = get_attack_max_scores(attack_z_vals, observed_ewma, attack_start_fb, alert_w, train_test_nt)

            # Compare detection threshold and attack
            for fpr_rate, threshold in zip(fpr_rates, thresholds):
                for attack_size_idx, attack_size in enumerate(attack_sizes):
                    attack_scores = attack_max[users_attacked, attack_size_idx]

                    results.append({'experiment_name': experiment_name,
                                    'seed': seed,
                                    'alert_w': alert_w,
                                    'fpr_rate': fpr_rate,
                                    'attack_size': attack_size,

                                    'threshold': threshold,
                                    'n_attacks': attack_scores.size,
                                    'n_detected': np.count_nonzero(attack_scores > threshold)})
        return results

    #####################################
    # Test set runner
    #####################################

    def test_run(self, experiment_name, hurdle_nb_model, selected_config, train_test_dict, base_config, degen_mask, bin_metric_dict):
        '''
        Single seed test runner, using selected config
        Runs the best config on the test set depending on the experiment name
        '''
        
        if selected_config is None:
            raise ValueError(f'Config not found for {experiment_name}')

        if hurdle_nb_model['hurdle_model'] is None:
            raise ValueError('Should only be run after selecting hurdle or NB model')

        # Create one complete runnable configuration
        best_config = ut.merge_configs(base_config, hurdle_nb_model, selected_config)
        best_config['nll_only'] = False

        # Creating nts and param grids and model
        _, config_nt, _, train_test_nt, _ = b.converting_dicts_to_nt(best_config, train_test_dict, bin_metric_dict)
        _, _, _, u_cluster, v_cluster, p_cluster = self.get_ll_param_grids(best_config)
        test_model = c.make_cluster_model(cluster_param=best_config['cluster_param'], runtime_configs=best_config, u_init=u_cluster, v_init=v_cluster, p_init=p_cluster,)

        # Getting breakdown groups and attack groups if needed
        breakdown_groups = k.get_metric_breakdown(user_counts_nt=self.user_counts_nt, user_type_groups=self.user_type_groups,
                                model=test_model, config_dict=best_config, u_clustering=u_cluster, v_clustering=v_cluster, 
                                p_clustering=p_cluster, train_test_dict=train_test_dict)
        attack_sizes = np.asarray(best_config['attack_sizes'], dtype='int64')
        attack_start_fb = get_attack_hour(degen_mask, train_test_nt, self.bin_metric_nt, best_config['seed'])

        # Run LL pipeline
        _, calibration_outputs, user_output_metrics, observed_p_vals, attack_p_vals = self.run_pipeline_ll(model=test_model, config_nt=config_nt, 
            train_test_nt=train_test_nt, config_dict=best_config, degen_mask=degen_mask, breakdown_groups=breakdown_groups, attack_start_fb=attack_start_fb, attack_sizes=attack_sizes)

        # Make results table
        test_results = self.make_full_output_rows(model=test_model, calibration_outputs=calibration_outputs, config_dict=best_config, experiment_name=experiment_name)

        # Multi user nll and attack detection outputs
        test_user_results = self.make_user_output_table(model=test_model, user_output_metrics=user_output_metrics, experiment_name=experiment_name, config_dict=best_config, u_clustering=u_cluster, v_clustering=v_cluster, p_clustering=p_cluster)
        detection_results = get_detection_results(observed_p_vals, attack_p_vals, attack_start_fb, attack_sizes, best_config['alert_w_vals'], best_config['fpr_rates'], train_test_nt, self.bin_metric_nt, best_config['seed'], experiment_name)
                
        return test_results, test_user_results, detection_results



    def run_test_seeds(self, experiment_name, hurdle_nb_model, selected_config, train_test_dict, base_config, degen_mask, bin_metric_dict):
        '''
        Runs test model over multiple seeds
        '''
        test_seeds = np.random.default_rng(base_config['seed']).choice(10_000, size=base_config['n_test_seeds'], replace=False)
        
        for seed_number, seed in enumerate(test_seeds):
            seed_start_time = perf_counter()
            run_config = selected_config.copy()
            run_config['seed'] = seed

            test_results, user_results, detection_results = self.test_run(experiment_name=experiment_name, hurdle_nb_model=hurdle_nb_model,
                                    selected_config=run_config, train_test_dict=train_test_dict, base_config=base_config, 
                                    degen_mask=degen_mask, bin_metric_dict=bin_metric_dict)

            ut.store_run_results(results=test_results, dir=f'test/{experiment_name}/full', run_name=experiment_name)
            ut.store_run_results(results=detection_results, dir=f'test/{experiment_name}/detection', run_name=f'{experiment_name}_detection')

            # We only have seed variation in clustering and calibration therefore the log liklihood table only varies for cluster_smoothing
            if experiment_name == 'cluster_smoothing' or seed_number == 0:
                ut.store_run_results(results=user_results, dir=f'test/{experiment_name}/user', run_name=f'{experiment_name}_users')


            print(f'finished_seed {seed_number + 1}/{len(test_seeds)} in {perf_counter() - seed_start_time:.1f}s')