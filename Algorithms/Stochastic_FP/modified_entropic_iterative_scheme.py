import numpy as np
import os
from tqdm import tqdm
from wandb import init

from Algorithms.Stochastic_FP.entropic_estimate_OT import *
from Algorithms.Stochastic_FP.modified_entropic_estimate_OT_ott import modified_entropic_OT_map_estimate_ott
from Algorithms.data_manage import *
from Experiments.Synthetic_Generation.metrics_to_compare import W2_pot

class modified_entropic_iterative_scheme:
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
                 bary_sample_collection : dict,
                 input_sampler_for_evaluation,
                 eval_num_samples : int,
                 eval_MC_size : int):
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
            eval_num_samples: number of samples used when evaluating the V-values and the W2 distance to the true barycenter
            eval_MC_size: number of Monte Carlo repetitions when evaluating the V-values and the W2 distance to the true barycenter
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
        self.bary_sample_collection = bary_sample_collection
        self.input_sampler_for_evaluation = input_sampler_for_evaluation
        self.eval_num_samples = eval_num_samples
        self.eval_MC_size = eval_MC_size

        assert len(self.sample_size_scheme) >= self.num_iters, "sample size scheme mis-specified"
        assert len(self.reg_param_scheme) >= self.num_iters, "regularization scheme mis-specified"

        self.initializers = [modified_entropic_OT_map_estimate_ott.create_initializer(warm_start) for _ in range(self.num_measures)]

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
        accepted = np.zeros((num_samples, self.dim))

        with tqdm(total=num_samples, desc=f"Sampling from the pushforward measure by G-operator at iteration_{iter}", disable=True) as pbar:
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
        for measure_index in tqdm(range(self.num_measures), desc = "V-value computation", disable=True):
            input_samples = np.array(input_samples_collection[measure_index])
            V_value += W2_pot(input_samples, bary_samples)
        V_value /= self.num_measures
        return V_value
    
    def map_construct(self, iter, accepted_samples, input_samples_collection: dict, epsilon, map_logger = None):
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

            OT_map_estimator = modified_entropic_OT_map_estimate_ott(accepted_samples, input_measure_samples, initializer = self.initializers[measure_index])
            OT_map_estimator.get_dual_potential(epsilon = epsilon)
            self.initializers[measure_index] = OT_map_estimator.get_initializer()

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
            V_values_list = np.zeros(self.eval_MC_size)
            W2_to_bary_list = np.zeros(self.eval_MC_size)
            accepted_samples_list = []
            for i in tqdm(range(self.eval_MC_size), desc = f"Evaluation at iteration {iter} by Monte Carlo"): # Monte carlo sample size
                bary_samples = self.bary_sample_collection[str(i)][:self.eval_num_samples]
                accepted_samples = self.iterative_sampling(iter, self.eval_num_samples, sample_logger)
                input_samples_collection: dict = self.input_sampler_for_evaluation.sample(self.eval_num_samples)
                V_value = self.V_value_compute(accepted_samples, input_samples_collection)
                W2_to_bary = W2_pot(bary_samples, accepted_samples)
                accepted_samples_list.append(accepted_samples.tolist())
                V_values_list[i] = V_value
                W2_to_bary_list[i] = W2_to_bary
            
            self.V_values_dict[f"iteration_{iter}"] = {
                "mean": np.mean(V_values_list), 
                "std": np.std(V_values_list),
                "values": V_values_list.tolist()
            }
            self.W2_to_bary_dict[f"iteration_{iter}"] = {
                "mean": np.mean(W2_to_bary_list),
                "std": np.std(W2_to_bary_list),
                "values": W2_to_bary_list.tolist()
            }

            self.G_samples_dict[f"iteration_{iter}"] = accepted_samples_list

            save_json(self.V_values_dict, V_values_dir, f"V_values_iter{iter}.json")
            save_json(self.W2_to_bary_dict, W2_to_bary_dir, f"W2_to_bary_iter{iter}.json")
            save_json(self.G_samples_dict, G_samples_dir, f"G_samples_iter{iter}.json")
            
            if iter >= self.num_iters:
                break

            # collect samples and compute the OT maps
            accepted_samples = self.iterative_sampling(iter, self.sample_size_scheme[iter], sample_logger)
            input_samples_collection: dict = self.input_sampler_for_evaluation.sample(self.sample_size_scheme[iter])
            self.map_construct(iter, accepted_samples, input_samples_collection, self.reg_param_scheme[iter], map_logger)

            iter += 1
