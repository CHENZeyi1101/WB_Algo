import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from Algorithms.Stochastic_FP.entropic_estimate_OT_ott import entropic_OT_map_estimate_ott
from Algorithms.Stochastic_FP.entropic_estimate_OT_geomloss import entropic_OT_map_estimate_geomloss
from Algorithms.data_manage import *
from Experiments.Synthetic_Generation.metrics_to_compare import evaluate_zipped

class entropic_iterative_scheme:
    r'''
    Python class for implementing the entropic iterative scheme for approximating the fixed point of the G-operator
    '''
   
    def __init__(self, 
                 dim : int,
                 num_iters : int,
                 input_sampler, 
                 rand_state : np.random.RandomState, 
                 init_method : dict, 
                 truncate_radius : float, 
                 sinkhorn_impl: str,
                 sample_size_scheme : list, 
                 reg_param_scheme : list,
                 warm_start : dict | None, 
                 bary_samples_collection : dict,
                 input_samples_for_evaluation : dict,
                 eval_num_samples : int,
                 eval_MC_size : int,
                 num_parallel : int = 10):
        r'''
        Constructor
        Inputs: 
            num_iters: number of iterations to run the iterative scheme
            input_sampler: object used for generating samples from the input measures in a reproducible way
            rand_state: numpy.random.RandomState object used for sampling the initial Gaussian measure
            init_method: dictionary containing information for setting the initial Gaussian measure
            truncate_radius: radius for truncating the image of the G-operator at each iteration
            sinkhorn_impl: ["ott", "geomloss"] the choice of implementation of the Sinkhorn algorithm
            sample_size_scheme: list with len(sample_size_scheme) = num_iter containing the number of samples generated in each iteration of the scheme for estimating the OT maps
            reg_param_scheme: list with len(sample_size_scheme) = num_iter containing the regularization parameters used in each iteration of the scheme for estimating the OT maps
            warm_start: boolean indicating whether to use a warm-start strategy when running the Sinkhorn algorithm for estimating OT maps
            bary_sample_collection: dictionary containing samples from the reference barycenter/true barycenter for evaluation
            input_sampler_for_evaluation: object used for generating another independent stream of samples from the input measures that are used for approximately computing V-values for evaluation
            eval_num_samples: number of samples used when evaluating the V-values and the W2 distances to the true barycenter
            eval_MC_size: number of Monte Carlo repetitions when evaluating the V-values and the W2 distances to the true barycenter
            num_parallel: number of parallel processes when evaluating the V-values and the W2 distances to the true barycenter
        '''

        self.num_iters = num_iters
        self.input_sampler = input_sampler
        self.dim = dim
        self.num_measures = input_sampler.num_measures
        self.rand_state = rand_state
        self.truncate_radius = truncate_radius
        self.sinkhorn_impl = sinkhorn_impl
        self.sample_size_scheme = sample_size_scheme
        self.reg_param_scheme = reg_param_scheme
        self.bary_samples_collection = bary_samples_collection
        self.input_samples_for_evaluation = input_samples_for_evaluation
        self.eval_num_samples = eval_num_samples
        self.eval_MC_size = eval_MC_size
        self.num_parallel = num_parallel

        assert len(self.sample_size_scheme) >= self.num_iters, "sample size scheme mis-specified"
        assert len(self.reg_param_scheme) >= self.num_iters, "regularization scheme mis-specified"
        assert self.sinkhorn_impl == "ott" or self.sinkhorn_impl == "geomloss", "unknown Sinkhorn implementation"

        if self.sinkhorn_impl == "ott":
            self.initializers = [entropic_OT_map_estimate_ott.create_initializer(warm_start) for _ in range(self.num_measures)]
        else:
            self.initializers = [None] * self.num_measures

        # dictionary for storing computed OT maps
        self.OT_collections = {}

        # data structures for storing evaluation metrics and diagnostics
        self.G_samples_dict = {}
        self.V_values_dict = {}
        self.W2_to_bary_dict = {}

        self.set_init_gauss(init_method)
    
    def set_init_gauss(self, init_method : dict):
        if init_method["type"] == "fixed":
            self.init_gauss = {"mean": init_method["mean"], "cov": init_method["cov"]}
        elif init_method["type"] == "moment":
            samps_for_init = np.vstack(list(self.input_sampler.sample(init_method["sample_size"]).values()))
            self.init_gauss = {"mean": np.mean(samps_for_init, axis = 0), "cov": np.cov(samps_for_init, rowvar=False)}
        else:
            raise ValueError("Unknown initialization method")

        mean = self.init_gauss["mean"]
        cov = self.init_gauss["cov"]
        print(f"Initial Gaussian measure:")
        print(f"mean = {mean}")
        print(f"cov = ")
        print(cov)

    def iterative_sampling(self, iter, num_samples, sample_logger = None):
        '''
        Sample from the pushforward measure by the G-operator at each iteration based on the current OT map estimators
        '''
        count = 0
        accepted_list = []
        num_samples_batch = int(num_samples * 1.1)

        while count < num_samples:
            log_info(sample_logger,
                    f"\n########## Sampling started at Iteration_{iter} for sample_{count} ##########\n")

            samples = self.rand_state.multivariate_normal(self.init_gauss["mean"], self.init_gauss["cov"], size = num_samples_batch)

            for t in range(iter):
                sum_samples = np.zeros((num_samples_batch, self.dim))
                for measure_index in range(self.num_measures):
                    OT_map_estimator = self.OT_collections[(t, measure_index)]
                    samples_pf = OT_map_estimator.compute_modified_entropic_OT_map(samples, self.truncate_radius)
                    log_info(sample_logger,
                            f"\n####### Pushforward sample to Measure_{measure_index} at Round_{t} #######\n"
                            f"Pushforward sample: {samples_pf}\n")
                    sum_samples += samples_pf

                samples = sum_samples / self.num_measures
                log_info(sample_logger,
                        f"\n####### Averaged sample at Round_{t} #######\n"
                        f"Averaged sample: {samples}\n")

            accepted_batch = samples[np.linalg.norm(samples, axis = 1) <= self.truncate_radius, :]
            accepted_list.append(accepted_batch)
            count += accepted_batch.shape[0]
        
        accepted = np.vstack(accepted_list)[:num_samples, :]
                
        log_info(sample_logger, f"\n"
                                f"########## Sampling completed at Iteration_{iter} ##########\n"
                                )
                            
        return accepted
    
    def map_construct(self, iter, accepted_samples, input_samples_collection: dict, epsilon, map_logger = None):
        '''
        Construct OT map estimators from the current measure to each of the input measures
        based on the generated samples after iterations;
        Will be envoked each time after iterative_sampling() is called when a new (empirical) G(\mu) measure is obtained.
        '''
        for measure_index in tqdm(range(self.num_measures), desc = "OT map construction", disable = self.sinkhorn_impl == "ott"):
            input_measure_samples = np.array(input_samples_collection[measure_index])
            log_info(map_logger, f"\n"
                                f"################################################################\n"
                                f"Current teration: {iter}\n"
                                f"OT map estimation for Measure_{measure_index}\n"
                                f"################################################################\n"
                                )
            
            if self.sinkhorn_impl == "ott":
                OT_map_estimator = entropic_OT_map_estimate_ott(accepted_samples, input_measure_samples, initializer = self.initializers[measure_index])
                OT_map_estimator.get_dual_potential(epsilon = epsilon)
                self.initializers[measure_index] = OT_map_estimator.get_initializer()
            elif self.sinkhorn_impl == "geomloss":
                OT_map_estimator = entropic_OT_map_estimate_geomloss(accepted_samples, input_measure_samples)
                OT_map_estimator.get_dual_potential(epsilon = epsilon)

            # store the OT map estimator (python class) in the OT_collctions dictionary
            self.OT_collections[(iter, measure_index)] = OT_map_estimator
            log_info(map_logger, f"\n"
                                f"################################################################\n"
                                f"OT map estimation to Measure_{measure_index} at Iteration_{iter} completed\n"
                                f"################################################################\n"
                                )

    def converge(self, 
                 logger : dict = {logger: None for logger in ['sample_logger', 'map_logger']},
                 data_dir: str = None):
        '''
        Main function to run the entropic iterative scheme for approximating the G-operator fixed point
        Outputs are saved to JSON files in data_dir
        Inputs:
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
            # perform the evaluation
            V_values_list = []
            W2_to_bary_list = []
            accepted_samples_list = []

            eval_samples_it = [self.iterative_sampling(iter, self.eval_num_samples, sample_logger) for _ in range(self.eval_MC_size)]
            input_measure_samples_collection_it = [{k : self.input_samples_for_evaluation[i][k][:self.eval_num_samples] for k in range(self.num_measures)} for i in range(self.eval_MC_size)]
            true_bary_samples_it = [self.bary_samples_collection[i][:self.eval_num_samples] for i in range(self.eval_MC_size)]

            for eval_samples in eval_samples_it:
                accepted_samples_list.append(eval_samples.tolist())
            
            with Pool(processes = self.num_parallel) as pool, tqdm(total = self.eval_MC_size, desc = f"Evaluation of iter {iter}") as pbar:
                for V_value, W2_to_bary in pool.imap(evaluate_zipped, 
                                        zip(eval_samples_it, 
                                            input_measure_samples_collection_it, 
                                            true_bary_samples_it)):
                    V_values_list.append(V_value)
                    W2_to_bary_list.append(W2_to_bary)
                    pbar.update(1)
                    pbar.refresh()
            
            self.V_values_dict[f"iteration_{iter}"] = {
                "mean": np.mean(V_values_list), 
                "std": np.std(V_values_list),
                "sample_size": self.eval_num_samples,
                "values": V_values_list
            }
            self.W2_to_bary_dict[f"iteration_{iter}"] = {
                "mean": np.mean(W2_to_bary_list),
                "std": np.std(W2_to_bary_list),
                "sample_size": self.eval_num_samples,
                "values": W2_to_bary_list
            }

            self.G_samples_dict[f"iteration_{iter}"] = accepted_samples_list

            save_json(self.V_values_dict, V_values_dir, f"V_values_iter{iter}.json")
            save_json(self.W2_to_bary_dict, W2_to_bary_dir, f"W2_to_bary_iter{iter}.json")
            save_json(self.G_samples_dict, G_samples_dir, f"G_samples_iter{iter}.json")
            
            if iter >= self.num_iters:
                break

            # collect samples and compute the OT maps
            accepted_samples = self.iterative_sampling(iter, self.sample_size_scheme[iter], sample_logger)
            input_samples_collection: dict = self.input_sampler.sample(self.sample_size_scheme[iter])
            self.map_construct(iter, accepted_samples, input_samples_collection, self.reg_param_scheme[iter], map_logger)

            iter += 1
