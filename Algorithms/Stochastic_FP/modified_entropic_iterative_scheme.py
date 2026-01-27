import numpy as np
import os
from tqdm import tqdm

from Algorithms.Stochastic_FP.entropic_estimate_OT import *
from Algorithms.data_manage import *
from Experiments.Synthetic_Generation.metrics_to_compare import W2_pot

class modified_entropic_iterative_scheme:
    r'''
    Python class for implementing the entropic iterative scheme for approximating the fixed point of the V-operator
    '''
   
    def __init__(self, 
                 num_iter : int,
                 input_sampler, 
                 rand_state : np.random.RandomState, 
                 init_gauss : dict, 
                 truncate_radius : float, 
                 sinkhorn_impl: str,
                 sample_size_scheme : list, 
                 warm_start : bool, 
                 bary_sample_collection : dict,
                 input_sampler_for_evaluation):
        r'''
        Constructor
        Inputs: 
            num_iter: number of iterations to run the iterative scheme
            input_sampler: object used for generating samples from the input measures in a reproducible way
            rand_state: numpy.random.RandomState object used for sampling the initial Gaussian measure
            init_gauss: dictionary containing keys "mean" and "cov" that corresponds to the mean and covariance of the initial Gaussian measure
            truncate_radius: radius for truncating the image of the G-operator at each iteration
            sinkhorn_impl: ["ott", "geomloss"] the choice of implementation of the Sinkhorn algorithm
            sample_size_scheme: list with len(sample_size_scheme) = num_iter containing the number of samples generated in each iteration of the scheme for estimating the OT maps
            warm_start: boolean indicating whether to use a warm-start strategy when running the Sinkhorn algorithm for estimating OT maps
            bary_sample_collection: dictionary containing samples from the reference barycenter/true barycenter for evaluation
            input_sampler_for_evaluation: object used for generating another independent stream of samples from the input measures that are used for approximately computing V-values for evaluation
        '''

        self.num_iter = num_iter
        self.input_sampler = input_sampler
        self.dim = input_sampler.dim
        self.num_measures = input_sampler.num_measures
        self.rand_state = rand_state
        self.init_gauss = init_gauss
        self.truncate_radius = truncate_radius
        self.sinkhorn_impl = sinkhorn_impl
        self.sample_size_scheme = sample_size_scheme
        self.warm_start = warm_start
        self.bary_sample_collection = bary_sample_collection
        self.input_sampler_for_evaluation = input_sampler_for_evaluation

        # dictionary for storing computed OT maps
        self.OT_collections = {}

        # data structures for storing evaluation metrics and diagnostics
        self.G_samples_dict = {}
        self.V_values_dict = {}
        self.W2_to_bary_dict = {}

    def iterative_sampling(self, iter, num_samples, sample_logger = None):
        '''
        Sample from the pushforward measure by the V-operator at each iteration based on the current OT map estimators
        '''
        count = 0
        accepted = np.zeros((num_samples, self.dim))

        with tqdm(total=num_samples, desc=f"sampling from the pushforward measure by V-operator at iteration_{iter}") as pbar:
            while count < num_samples:
                log_info(sample_logger,
                        f"\n########## Sampling started at Iteration_{iter} for sample_{count} ##########\n")

                sample = self.rand_state.multivariate_normal(self.init_gauss["mean"], self.init_gauss["cov"])

                for t in range(iter):
                    sum_sample = np.zeros(self.dim)
                    for measure_index in range(self.num_measures):
                        OT_map_estimator: entropic_OT_map_estimate = self.OT_collections[(t, measure_index)]
                        sub_sample = OT_map_estimator.regularize_entropic_OT_map(0.5 * self.truncate_radius**2, sample)
                        log_info(sample_logger,
                                f"\n####### Pushforward sample to Measure_{measure_index} at Round_{t} #######\n"
                                f"Pushforward sample: {sub_sample}\n")
                        sum_sample += sub_sample

                    sample = sum_sample / self.num_measures
                    log_info(sample_logger,
                            f"\n####### Averaged sample at Round_{t} #######\n"
                            f"Averaged sample: {sample}\n")

                if np.linalg.norm(sample) <= self.truncate_radius:
                    accepted[count, :] = sample
                    count += 1
                    pbar.update(1)  # update progress bar by one
                
        log_info(sample_logger, f"\n"
                                f"########## Sampling completed at Iteration_{iter} ##########\n"
                                )
                            
        return accepted
    
    def V_value_compute(self, bary_samples, input_samples_collection: dict):
        '''
        bary_samples denotes the samples from the true/approximated barycenter measure
        input_samples_collection is a dictionary with k keys, each key corresponds to the samples from the k-th input measure.
        '''
        V_value = 0
        for measure_index in tqdm(range(self.num_measures), desc = "V-value computation"):
            input_samples = np.array(input_samples_collection[measure_index])
            V_value += W2_pot(input_samples, bary_samples)
        V_value /= self.num_measures
        return V_value
    
    def map_construct(self, iter, accepted_samples, input_samples_collection: dict, epsilon, map_logger = None, warm_start = False):
        '''
        Construct OT map estimators from the current measure to each of the input measures
        based on the generated samples after iterations;
        Will be envoked each time after iterative_sampling() is called when a new (empirical) G(\mu) measure is obtained.
        '''
        for measure_index in tqdm(range(self.num_measures), desc = "OT map construction"):
            input_measure_samples = np.array(input_samples_collection[measure_index])
            log_info(map_logger, f"\n"
                                f"################################################################\n"
                                f"Current teration: {iter}\n"
                                f"OT map estimation for Measure_{measure_index}\n"
                                f"################################################################\n"
                                )
                
            # Store the V-value (i.e.,\@ the weighted sum of the Wasserstein distances between the input measures and the generated samples)

            OT_map_estimator = entropic_OT_map_estimate(accepted_samples, input_measure_samples)

            prev_key = (iter - 1, measure_index)

            # Warm-starting the Sinkhorn algorithm using the previous OT map estimator
            if prev_key in self.OT_collections and iter > 1 and warm_start:
                print("Warm-starting the Sinkhorn algorithm using the previous OT map estimator")
                prev_estimator = self.OT_collections[prev_key]
                customized_potential_initializer = prev_estimator.customize_initializers()
                OT_map_estimator.get_dual_potential(epsilon = epsilon, initializer= customized_potential_initializer)
            else: 
                print("No warm-starting")
                OT_map_estimator.get_dual_potential(epsilon = epsilon)

            # store the OT map estimator (python class) in the OT_collctions dictionary
            self.OT_collections[(iter, measure_index)] = OT_map_estimator
            log_info(map_logger, f"\n"
                                f"################################################################\n"
                                f"OT map estimation to Measure_{measure_index} at Iteration_{iter} completed\n"
                                f"################################################################\n"
                                )

    def converge(self, 
                 bary_sample_collection,
                 input_sampler_for_evaluation,
                 max_iter, 
                 num_samples, 
                 epsilon, 
                 MC_size : int, # Monte Carlo sample size at each iteration
                 logger : dict = {logger: None for logger in ['sample_logger', 'map_logger']},
                 data_dir: str = None,
                 warm_start: bool = True):
        '''
        Main function to run the entropic iterative scheme for approximating the V-operator fixed point;
        Inputs:
              bary_samples: samples from the initialized barycenter measure \mu_0
              input_samples_collection: a dictionary with k keys, each key corresponds to the samples from the k-th input measure.
              max_iter: maximum number of iterations for the entropic iterative scheme
              num_samples: number of samples to generate at each iteration
              epsilon: entropic regularization parameter for the OT map estimation at each iteration
              MC_size: Monte Carlo sample size at each iteration
              logger: a dictionary containing the loggers for sampling and OT map construction
              data_dir: directory path for saving the logged data
        Outputs:
              V_values_dict: a dictionary containing the logged V-values at each iteration
              W2_to_bary_dict: a dictionary containing the logged Wasserstein distances to the barycenter at each iteration
              G_samples_dict: a dictionary containing the logged generated samples at each iteration
        ''' 
        
        # Set-up for logging and data saving
        sample_logger, map_logger = logger.get('sample_logger'), logger.get('map_logger')
        V_values_dir = os.path.join(data_dir, "V_values")
        W2_to_bary_dir = os.path.join(data_dir, "W2_to_bary")
        G_samples_dir = os.path.join(data_dir, "G_samples")
        os.makedirs(V_values_dir, exist_ok=True)
        os.makedirs(W2_to_bary_dir, exist_ok=True)
        os.makedirs(G_samples_dir, exist_ok=True)

        # Start the iterations
        iter = 0
        while True:
            V_values_list = []
            W2_to_bary_list = []
            accepted_samples_list = []
            for i in tqdm(range(MC_size), desc = f"Monte Carlo Sampling at iteration {iter}"): # Monte carlo sample size
                bary_samples = bary_sample_collection[str(i)]
                accepted_samples = self.iterative_sampling(iter, num_samples, sample_logger)
                input_samples_collection: dict = input_sampler_for_evaluation.sample(num_samples)
                V_value = self.V_value_compute(accepted_samples, input_samples_collection)
                W2_to_bary = self.W2_to_bary_compute(bary_samples, accepted_samples)
                accepted_samples_list.append(accepted_samples.tolist())
                V_values_list.append(V_value)
                W2_to_bary_list.append(W2_to_bary)
            
            self.V_values_dict[f"iteration_{iter}"] = V_values_list
            self.W2_to_bary_dict[f"iteration_{iter}"] = W2_to_bary_list
            self.G_samples_dict[f"iteration_{iter}"] = accepted_samples_list
            save_json(self.V_values_dict, V_values_dir, f"V_values_iter{iter}.json")
            save_json(self.W2_to_bary_dict, W2_to_bary_dir, f"W2_to_bary_iter{iter}.json")
            save_json(self.G_samples_dict, G_samples_dir, f"G_samples_iter{iter}.json")
            
            if iter >= max_iter:
                break

            self.map_construct(iter, accepted_samples, input_samples_collection, epsilon, map_logger, warm_start = warm_start)
            iter += 1
