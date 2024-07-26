import numpy as np
import gurobipy as gp
from gurobipy import GRB

from true_WB import *
from input_generate import *
from estimate_OT import *
from ADMM import *
from config_log import *

def save_data(data, pathname = None, filename = None):
    output_file = os.path.join(pathname, filename)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4) 

def read_data(pathname = None, filename = None):
    file_path = os.path.join(pathname, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

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
    W2_sq = model.objVal
    return W2_sq

class iterative_scheme:
    def __init__(self, dim, num_measures, source_sampler, input_sample_collection, lambda_lower = 0.01, lambda_upper = 1000, ADMM = False, smoothing = 'BA', log = True):
        self.dim = dim
        self.num_measures = num_measures
        self.source_sampler = source_sampler
        self.input_sample_collection = input_sample_collection # a dictionary of input samples for each measure
        self.G_samples = {}
        self.V_values = {}
        self.OT_collections = {}
        self.lambda_lower = lambda_lower
        self.lambda_upper = lambda_upper
        self.ADMM = ADMM
        self.smoothing = smoothing
        self.log = log   

    def iterative_sampling(self, iter, num_samples = 100, truncate_radius = 1000, sample_logger = None, sample_logger_path = None):
        dim, num_measures = self.dim, self.num_measures
        smoothing = self.smoothing
        count = 0
        accepted = np.zeros((num_samples, dim))

        np.random.seed(100 + iter)
        while count < num_samples:
            if self.log:
                sample_logger.info(f"\n"
                                f"########## Sampling started at Iteration_{iter} for sample_{count} ##########\n"
                                )
                
            sample = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))

            for t in range(iter):
                sum_sample = np.zeros(dim)
                for measure_index in range(num_measures):
                    OT_map_estimator = self.OT_collections[(t, measure_index)]

                    if smoothing == 'KS':
                        if self.log:
                            sample_logger.info(f"\n"
                                            f"####### Pushforward generation by the mapping towards Measure_{measure_index} #######\n"
                                            f"####### Round number: {t} #######\n"
                                            f"####### Smoothing method: {smoothing} #######\n"
                                            )   
                        _, sub_sample = OT_map_estimator.KS_estimate(sample, logger = sample_logger, log_file_path = sample_logger_path)

                    if smoothing == 'BA':
                        if self.log:
                            sample_logger.info(f"\n"
                                            f"####### Pushforward generation by the mapping towards Measure_{measure_index} #######\n"
                                            f"####### Round number: {t} #######\n"
                                            f"####### Smoothing method: {smoothing} #######\n"
                                            )
                        _, sub_sample = OT_map_estimator.BA_estimate(sample, logger = sample_logger, log_file_path = sample_logger_path)

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
        
        if self.log:
            sample_logger.info(f"\n"
                                f"Sampling at iteration {iter} completed\n"
                                f"Accepted samples at iteration {iter}: {accepted}\n"
                                )
                            
        return accepted
    
    def G_sample_save(self, accepted_G_samples, iter):

        # REMARK:
        # Save the generated samples from the G-mapping at each iteration;
        # "accepted_G_samples" is the accepted samples generated from the G-mapping at the current iteration;

        self.G_samples[f"iteration_{iter}"] = accepted_G_samples
        G_samples_json = {str(k): v.tolist() for k, v in self.G_samples.items()}
        G_sample_dir = "iterations/G_samples"
        os.makedirs(G_sample_dir, exist_ok=True)
        save_data(G_samples_json, G_sample_dir, f"G_samples.json")

    def V_value_save(self, accepted_G_samples, bary_samples, iter):

        # REMARK:
        # Compute the Wasserstein distance between the generated samples from the G-mapping
        # and the barycenter samples at each iteration;
        # "accepted_G_samples" is the accepted samples generated from the G-mapping at the current iteration;
        # "bary_samples" is the barycenter samples generated from the input measure at the current iteration;

        W2_sq = W2(accepted_G_samples, bary_samples)
        self.V_values[f"iteration_{iter}"] = W2_sq
        V_values_json = self.V_values
        V_value_dir = "iterations/V_values"
        os.makedirs(V_value_dir, exist_ok=True)
        save_data(V_values_json, V_value_dir, f"V_values.json")
    
    def map_construct(self, accepted_samples, iter, measure_index, map_logger = None, map_logger_path = None):

        # REMARK: 
        # Construct OT map estimators from the current measure to each of the input measures
        # based on the generated samples after iterations;
        # Will be envoked each time after iterative_sampling() is called
        # when a new (empirical) G(\mu) measure is obtained.

        input_sample_collection = self.input_sample_collection
        num_samples = accepted_samples.shape[0]
        idx_start, idx_end = iter * num_samples, (iter + 1) * num_samples
        BX = accepted_samples
        BY = input_sample_collection[f"measure_{measure_index}"][idx_start:idx_end, :]
        
        # log_dir = f"iterations/iteration_{iter}_logs"
        # os.makedirs(log_dir, exist_ok=True)
        # map_logger, map_logger_path = configure_logging(f'iter_{iter}_map_logger', log_dir, f'iter_{iter}_map_logger.log')

        if self.log:
            map_logger.info(f"\n"
                            f"################################################################\n"
                            f"Current teration: {iter}\n"
                            f"OT map estimation for Measure_{measure_index}\n"
                            f"################################################################\n"
                            )

        lambda_lower, lambda_upper, ADMM = self.lambda_lower, self.lambda_upper, self.ADMM

        # trunc_u denotes u(\nu) bounding the gradient
        trunc_u = 10 * np.linalg.norm(BY, axis = 1).max()
        OT_map_estimator = OT_map_estimate(BX, BY, log = True)
        OT_map_estimator.solve_opt_tuples(trunc_u, 
                                          lambda_lower, 
                                          lambda_upper, 
                                          ADMM, 
                                          logger = map_logger, 
                                          log_file_path = map_logger_path)
        
        # store the OT map estimator (python class) in the OT_collctions dictionary
        self.OT_collections[(iter, measure_index)] = OT_map_estimator
        map_logger.info(f"\n"
                        f"################################################################\n"
                        f"OT map estimation to Measure_{measure_index} at Iteration_{iter} completed\n"
                        f"################################################################\n"
                        )
        
    def converge(self, iter, num_samples = 500, max_iter = 20):
        # source_sampler is the underlying sampler of the barycenter.
        num_measures = self.num_measures
        source_sampler = self.source_sampler
        
        while iter < max_iter:
            log_dir = f"iterations/iteration_{iter}_logs"
            os.makedirs(log_dir, exist_ok=True)
            sample_logger, sample_logger_path = configure_logging(f'iter_{iter}_sample_logger', log_dir, f'iter_{iter}_sample_logger.log')
            map_logger, map_logger_path = configure_logging(f'iter_{iter}_map_logger', log_dir, f'iter_{iter}_map_logger.log')

            accepted = self.iterative_sampling(iter, num_samples, sample_logger = sample_logger, sample_logger_path = sample_logger_path)
            source_samples = source_sampler.sample(num_samples, seed = iter)
            self.G_sample_save(accepted, iter)
            self.V_value_save(accepted, source_samples, iter)
            for measure_index in range(num_measures):
                self.map_construct(accepted, iter, measure_index, map_logger=map_logger, map_logger_path=map_logger_path)
            iter += 1

    

