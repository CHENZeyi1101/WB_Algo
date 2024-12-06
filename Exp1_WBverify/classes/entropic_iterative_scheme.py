import numpy as np
import gurobipy as gp
from gurobipy import GRB
import ot
import ot.plot

from true_WB import *
from input_generate_plugin import *
from entropic_estimate_OT import *
from ADMM import *
from config_log import *
from sample_plot import *

from tqdm import tqdm, tqdm_notebook


def save_data(data, pathname = None, filename = None):
    output_file = os.path.join(pathname, filename)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4) 

def read_data(pathname = None, filename = None):
    file_path = os.path.join(pathname, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def W2_pot(X, Y): # solving the OT simplex problem using POT package
    M = ot.dist(X, Y)
    a, b = np.ones((X.shape[0],)) / X.shape[0], np.ones((Y.shape[0],)) / Y.shape[0]
    W2_sq = ot.emd2(a, b, M, numItermax=1e6)
  
    return W2_sq

def W2(X, Y):
    m, n = X.shape[0], Y.shape[0]
    model = gp.Model("LP_OptCoupling")
    pi = {}
    for i in range(m):
        for j in range(n):
            pi[i, j] = model.addVar(lb=0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}")
    model.update()
    
    obj = gp.quicksum(pi[i, j] * np.linalg.norm(X[i] - Y[j])**2 for i in range(m) for j in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    for j in range(n):
        model.addConstr(gp.quicksum(pi[i, j] for i in range(m)) == 1/n)
    for i in range(m):
        model.addConstr(gp.quicksum(pi[i, j] for j in range(n)) == 1/m)
    model.optimize()
    opt_coupling = np.array([[pi[i, j].x for j in range(n)] for i in range(m)])
    W2_sq = model.objVal
    return W2_sq, opt_coupling

class entropic_iterative_scheme:

    r'''
    Python class for implementing the fixed-point iterative scheme for approximately computing the W2-barycenter
    using the (regularized) entropic OT map estimators

    Attributes:
    dim: int
        dimension of the samples
    num_measures: int
        number of input measures
    source_sampler: object
        the underlying sampler of the barycenter
    input_sample_collection: dict
        a dictionary of input samples for each measure
    G_samples: dict
        a dictionary of generated samples from the G-mapping at each iteration
    V_values: dict
        a dictionary of the (empirical) Wasserstein distance between the generated samples and the barycenter samples
    OT_collections: dict
        a dictionary of the OT map estimators from the current measure to each of the input measures

    Methods:
    iterative_sampling(iter, num_samples, truncate_radius, sample_logger)
        Generate samples (with truncation) from G(\mu) at each iteration
    G_sample_save(accepted_G_samples, iter)
        Record the generated samples from G(\mu) at each iteration
    V_value_save(accepted_G_samples, bary_samples, iter)
        Compute and record the (empirical) Wasserstein distance between the generated samples and the barycenter samples
    map_construct(accepted_samples, iter, measure_index, map_logger)
        Construct the OT map estimators from the current measure to each of the input measures
    converge(iter, num_samples, max_iter)
        Execute the iterative scheme until convergence

    '''

    def __init__(self, dim, num_measures, source_sampler, entropic_sampler, source_sampler_seed, log = False):
        self.dim = dim
        self.num_measures = num_measures
        self.source_sampler = source_sampler
        # self.input_sample_collection = input_sample_collection # a dictionary of input samples for each measure
        self.entropic_sampler = entropic_sampler
        self.G_samples = {}
        self.V_values = {}
        self.OT_collections = {}
        self.W2_to_true_bary = {}
        self.source_sampler_seed = source_sampler_seed
        self.log = log   

    def iterative_sample(self, iter, num_samples = 1000, truncate_radius = 100, sample_logger = None):
        '''
        Inputs:
              iter: current iteration number (to sample from \mu_{iter})
              num_samples: number of samples to generate at each iteration
              truncate_radius: truncation radius for the generated samples
        Outputs:
              accepted: accepted samples generated from \mu_{iter}
        '''
        
        dim, num_measures = self.dim, self.num_measures
        count = 0
        accepted = np.zeros((num_samples, dim))

        with tqdm(total=num_samples, desc=f"sampling from the pushforward measure at iteration_{iter}") as pbar:
            while count < num_samples:
                if self.log:
                    sample_logger.info(f"\n"
                                    f"########## Sampling started at Iteration_{iter} for sample_{count} ##########\n"
                                    )
                    
                # the initial sample is generated from the standard normal distribution
                sample = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))

                for t in range(iter):
                    # to be executed only when iter > 0
                    sum_sample = np.zeros(dim)
                    for measure_index in range(num_measures):
                        OT_map_estimator = self.OT_collections[(t, measure_index)]
                        sub_sample = OT_map_estimator.regularize_entropic_OT_map(truncate_radius**2, sample)
                        # the pushforward image (as the samples collected by the approximated input measure)
                        if self.log:
                                sample_logger.info(f"\n"
                                                f"####### Pushforward generation by the mapping towards Measure_{measure_index} #######\n"
                                                f"####### Round number: {t} #######\n"
                                                )   

                        # Accumulate the pushforward samples from each measure
                        sum_sample += sub_sample

                    # Average the pushforward samples from all measures
                    sample = sum_sample / num_measures
                    if self.log:
                        sample_logger.info(f"\n"
                                        f"####### G-mapping completed at Round_{t} #######\n"
                                        )

                # Check if the sample is within the truncation radius
                if np.linalg.norm(sample) < truncate_radius:
                    accepted[count, :] = sample
                    count += 1
                    if (count + 1) % 10 == 0:
                        pbar.update(10)
        
        if self.log:
            sample_logger.info(f"\n"
                                f"Sampling at iteration {iter} completed\n"
                                f"Accepted samples at iteration {iter}: {accepted}\n"
                                )
                            
        return accepted
    
    def G_sample_save(self, accepted_G_samples, iter, save_pathname = None):

        # Save the generated samples from the G-mapping at each iteration;
        # "accepted_G_samples" is the accepted samples generated from the G-mapping at the current iteration;

        self.G_samples[f"iteration_{iter}"] = accepted_G_samples
        G_samples_json = {str(k): v.tolist() for k, v in self.G_samples.items()}
        G_sample_dir = f"{save_pathname}/G_samples"
        os.makedirs(G_sample_dir, exist_ok=True)
        save_data(G_samples_json, G_sample_dir, f"G_samples.json")

    def V_value_compute(self, bary_samples, input_sample_collection, iter = None, save_pathname = None):
        # Compute the V-value (i.e.,\@ the weighted sum of the Wasserstein distances between the input measures and the generated samples)
        # Notice that when iter = None, this returns the true V_value given by the ground-truth barycenter;
        # Otherwise, it is the V_value returned by an approximated barycenter.
        # The input_sample_collection is a dictionary with k keys, each key corresponds to the samples from the k-th input measure.

        V_value = 0
        for measure_index in range(self.num_measures):
            input_samples = np.array(input_sample_collection[measure_index])
            V_value += W2_pot(input_samples, bary_samples)
        
        # normalize the V_value by the number of input measures
        V_value /= self.num_measures

        if iter is None:
            self.V_values["true_V_value"] = V_value
        else:
            self.V_values[f"iteration_{iter}"] = V_value
        if save_pathname != None:
            V_values_json = self.V_values
            V_value_dir = f"{save_pathname}/V_values"
            os.makedirs(V_value_dir, exist_ok=True)
            save_data(V_values_json, V_value_dir, f"V_values.json")


    def W2_to_true_bary_compute(self, accepted_G_samples, bary_samples, iter, save_pathname = None):

        # Compute the Wasserstein distance between the generated samples from the G-mapping
        # and the barycenter samples at each iteration;
        # "accepted_G_samples" is the accepted samples generated from the G-mapping at the current iteration;
        # "bary_samples" is the barycenter samples generated from the input measure at the current iteration;

        W2_sq = W2_pot(accepted_G_samples, bary_samples)
        self.W2_to_true_bary[f"iteration_{iter}"] = W2_sq
        W2_to_true_bary_json = self.W2_to_true_bary
        W2_to_true_bary_dir = f"{save_pathname}/W2_to_true_bary"
        os.makedirs(W2_to_true_bary_dir, exist_ok=True)
        save_data(W2_to_true_bary_json, W2_to_true_bary_dir, f"W2_to_true_bary.json")
    
    def map_construct(self, accepted_samples, iter, epsilon, save_pathname, map_logger = None):

        # Construct OT map estimators from the current measure to each of the input measures
        # based on the generated samples after iterations;
        # Will be envoked each time after iterative_sampling() is called
        # when a new (empirical) G(\mu) measure is obtained.
        num_samples = accepted_samples.shape[0]
        num_measures = self.num_measures
        # entropic_sampler = self.entropic_sampler
        input_measures_samples = self.entropic_sampler.sample(num_samples) # this is a dictionary with k keys

        # print(type(input_measures_samples))

        BX = accepted_samples

        # Compute the V_value
        self.V_value_compute(BX, input_measures_samples, iter = iter, save_pathname=save_pathname)

        for measure_index in tqdm(range(num_measures)):
            BY = np.array(input_measures_samples[measure_index])

            if self.log:
                map_logger.info(f"\n"
                                f"################################################################\n"
                                f"Current teration: {iter}\n"
                                f"OT map estimation for Measure_{measure_index}\n"
                                f"################################################################\n"
                                )
                
            # Store the V-value (i.e.,\@ the weighted sum of the Wasserstein distances between the input measures and the generated samples)

            OT_map_estimator = entropic_OT_map_estimate(BX, BY, log = True)
            OT_map_estimator.get_dual_potential(epsilon = epsilon)
            
            # store the OT map estimator (python class) in the OT_collctions dictionary
            self.OT_collections[(iter, measure_index)] = OT_map_estimator
            map_logger.info(f"\n"
                            f"################################################################\n"
                            f"OT map estimation to Measure_{measure_index} at Iteration_{iter} completed\n"
                            f"################################################################\n"
                            )
        
    def converge(self, iter, num_samples = 5000, max_iter = 20, epsilon = 1, plot = True, scatter = False):
        # source_sampler samples from the initialized measure (mixture of gaussians in our experiment).
        dim = self.dim
        source_sampler = self.source_sampler
        entropic_sampler = self.entropic_sampler
        num_measures = self.num_measures
        seed = self.source_sampler_seed

        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)
        save_pathname = f"{result_dir}/entropic_measures_{num_measures}_seed_{seed}_samples_{num_samples}_dim_{dim}_epsilon_{epsilon}"

        # source_sampler information
        source_sampler_info = {
            "dim": dim,
            "radius": source_sampler.radius,
            "seed": source_sampler.seed,
            "num_components": source_sampler.num_components
        }
        source_sampler_info_dir = f"{save_pathname}/source_sampler_info"
        os.makedirs(source_sampler_info_dir, exist_ok=True)
        save_data(source_sampler_info, source_sampler_info_dir, f"source_sampler_info.json")

        # entropic_sampler information
        entropic_sampler_info = {
            "num_measures": num_measures,
            "dim": dim,
            "n_k": entropic_sampler.n_k,
            "seed": entropic_sampler.seed
        }
        entropic_sampler_info_dir = f"{save_pathname}/entropic_sampler_info"
        os.makedirs(entropic_sampler_info_dir, exist_ok=True)
        save_data(entropic_sampler_info, entropic_sampler_info_dir, f"entropic_sampler_info.json")

        # measure visualization
        plot_dirc = f"{save_pathname}/plots"
        os.makedirs(plot_dirc, exist_ok=True)

        source_measure_samples = source_sampler.sample(num_samples)
        plot_2d_source_measures_kde(source_measure_samples, plot_dirc = plot_dirc, scatter = scatter)

        input_measure_samples = entropic_sampler.sample(num_samples)
        for measure_index in range(num_measures):
            measure_samples = np.array(input_measure_samples[measure_index])
            plot_2d_input_measure_kde(measure_samples, measure_index, scatter = scatter, plot_dirc = plot_dirc)

        # Compute the true V-value
        self.V_value_compute(source_measure_samples, input_measure_samples, iter = None, save_pathname = save_pathname)
        
        # start the iterations
        while iter < max_iter:
            log_dir = f"{save_pathname}/iteration_{iter}_logs"
            os.makedirs(log_dir, exist_ok=True)
            sample_logger, _ = configure_logging(f'iter_{iter}_sample_logger', log_dir, f'iter_{iter}_sample_logger.log')
            map_logger, _ = configure_logging(f'iter_{iter}_map_logger', log_dir, f'iter_{iter}_map_logger.log')

            accepted = self.iterative_sample(iter, num_samples, sample_logger = sample_logger)
            source_samples = source_sampler.sample(num_samples)
            self.G_sample_save(accepted, iter, save_pathname = save_pathname)   
            self.W2_to_true_bary_compute(accepted, source_samples, iter, save_pathname = save_pathname)
            if plot == 1:
                plot_2d_compare_with_source_kde(source_samples, accepted, iter, plot_dirc = plot_dirc, scatter = scatter)
                
            # construct maps
            self.map_construct(accepted, iter, epsilon, save_pathname, map_logger=map_logger)
            iter += 1

# from input_generate_entropy import *
            
# dim = 2
# num_measures = 3

# source_sampler = MixtureOfGaussians(dim)
# source_sampler.random_components(5)
# source_sampler.set_truncation(100)
    
# entropic_sampler = entropic_input_sampler(dim, num_measures, 2, source_sampler, n_k = 10)
# entropic_sampler.construct_surjective_mapping()
# entropic_sampler.generate_strong_convexity_param()
# entropic_sampler.generate_g_vectors()
# entropic_sampler.generate_Y_matrices()
# entropic_sampler.compute_theta()

# iteration = entropic_iterative_scheme(dim, num_measures, source_sampler, entropic_sampler, log = True)
# iteration.converge(0, num_samples = 100, max_iter = 20, epsilon = 1, plot = True)
