import numpy as np
import math
from scipy.linalg import sqrtm, norm
from tqdm import tqdm
from Experiments.Synthetic_Generation.MOG import *
from Algorithms.Stochastic_FP.entropic_estimate_OT import *
import pandas as pd
import os
# from .samplers_dim2 import *

class entropic_input_sampler:
    '''
    Python class for generating samples from input measures using entropic transportation maps
    '''
    def __init__(self, 
                 dim, 
                 num_measures, 
                 auxiliary_measure_sampler_set, 
                 source_sampler, 
                 n_k = 1000, 
                 seed = 120, 
                 gamma = 0.3, 
                 manual = True, 
                 truncated_radius = 100,
                 bound_type = "eigen_bound",
                 theta = 10,
                 surjective_mapping : dict = None,
                 A_matrices_dict : dict = None):
        self.dim = dim
        self.num_measures = num_measures
        self.auxiliary_measure_sampler_set = auxiliary_measure_sampler_set
        self.tilde_K = len(auxiliary_measure_sampler_set) # 2 * tilde_K > num_measures
        self.source_sampler = source_sampler
        self.n_k = n_k # we assume that $n_k$ across 1, \dots, \tilde{K} are the same
        # num_measures < 2 * tilde_K
        self.seed = seed
        self.rng_entropy = np.random.RandomState(seed)
        self.gamma = gamma
        self.manual = manual
        self.truncated_radius = truncated_radius
        self.bound_type = bound_type
        self.grid_size = 500
        self.theta = theta
        if surjective_mapping is not None:
            self.surjective_mapping = surjective_mapping
        if A_matrices_dict is not None:
            self.A_matrices_dict = A_matrices_dict

    def generate_strong_convexity_param(self):
        r'''
        Set the strong convexity parameter for the entropic OT map estimator
        '''
        tilde_K = self.tilde_K
        self.strong_convexity_param_dict = {i: 0.0001 for i in range(tilde_K)}

    def generate_Y_matrices(self):
        r'''
        Generate the Y matrices for the entropic OT map estimator (for interpolation purpose)
        '''
        auxiliary_measure_sampler_set = self.auxiliary_measure_sampler_set
        tilde_K = self.tilde_K
        n_k = self.n_k
        Y_matrix_dict = {}
        for i in range(tilde_K):
            auxiliary_measure_sampler = auxiliary_measure_sampler_set[i]
            Y = auxiliary_measure_sampler.sample(n_k)
            Y_matrix_dict[i] = Y
            print(f"Finished generating Y matrix for auxiliary measure {i}")
        self.Y_matrix_dict = Y_matrix_dict

    def generate_g_vectors(self, epsilon = 10): # epsilon here is the entropic regularization parameter
        r'''
        Generate the g vector as the output of solving entropic OT maps out of samples from the auxiliary measures and the source measure
        (i.e., the ground-truth barycenter).
        The solver is from the ott package.
        '''
        tilde_K = self.tilde_K
        source_sampler = self.source_sampler
        # theta_dict = self.theta_dict
        n_k = self.n_k
        X = source_sampler.sample(n_k)
        Y_matrix_dict = self.Y_matrix_dict
        g_vector_dict = {}
        for i in range(tilde_K):
            Y = Y_matrix_dict[i]
            entropic_OT_map_generator = entropic_OT_map_estimate(X, Y, log = False)
            epsilon = epsilon
            entropic_OT_map_generator.get_dual_potential(epsilon = epsilon)

            # here we divide the potential by 2 because the potential returned by entropic_OT_map_estimate is optimal with respect to the cost function norm(x - y) ** 2 without the coefficient 1/2; dividing the potential by 2 makes the interpretation of the parameter theta consistent with Algorithm 3 in the paper
            g_vector_dict[i] = entropic_OT_map_generator.g_potential / 2

            print(f"Finished generating g vector for auxiliary measure {i}")
        self.g_vector_dict = g_vector_dict

    def entropic_weight_vector(self, x, Y_matrix, g_vector):
        r'''
        Compute the entropic weight vector when evaluated at x, given parameter dictionaries.
        Here, we fix tilde_K, and Y_matrix is a matrix of dimension n * d, g_vector is a vector of dimension n * 1, and theta is a scalar.
        The output vector is a vector of dimension n * 1.
        '''
        n_k = self.n_k
        x_tile = np.tile(x, (n_k, 1))

        # note that here the cost function has the coefficient 1/2
        exponent_vec = (g_vector - norm(x_tile - Y_matrix, axis = 1)**2 / 2) / self.theta
        exponent_vec_max = np.max(exponent_vec)
        exponent_vec -= exponent_vec_max

        numerator = np.exp(exponent_vec)
        denominator = np.sum(np.exp(exponent_vec))
        weight_vector = numerator / denominator

        return weight_vector
        
    def solve_maxeigen_problem(self, tilde_k):
        r'''
        We aim to maximize the maximum eigenvalue of the covariance matrix (Line~5 in Algorithm~3), corresponding to the data collected from auxiliary measure tilde_k.
        Due to the highly nonlinear structure of w(x), we traverse the grid space to find the optimal solution.
        '''
        # generate the 2d grid space spanning over -100 to 100 for each dimension
        # at each dimension, we have 100 points, thus we have 100^2 points in total
        Y_matrix = self.Y_matrix_dict[tilde_k]
        g_vector = self.g_vector_dict[tilde_k]
        theta = self.theta
        grid_size = self.grid_size
        truncate_radius = self.truncated_radius
        grid_space = np.linspace(-truncate_radius, truncate_radius, grid_size)
        max_eigenvalue = 0
        optimal_x = None

        for i in tqdm(range(grid_size), desc= f"tilde_k: {tilde_k}"):
            for j in range(grid_size): # traverse the grid space   
                x = np.array([grid_space[i], grid_space[j]])
                w_tilde_k = self.entropic_weight_vector(x, Y_matrix, g_vector)

                # # old method: slow
                # Y_tilde_k = Y_matrix.T # !!! Y_tilde_k is of dimension d * n
                # diag_w_k = np.diag(w_tilde_k)  # Creates a diagonal matrix with w_k as its diagonal
                # outer_product_w_k = np.outer(w_tilde_k, w_tilde_k)  # Outer product of w_k * w_K^T
                # matrix_diff = diag_w_k - outer_product_w_k
                # covariance_matrix = Y_tilde_k @ matrix_diff @ Y_tilde_k.T
                # max_eigenvalue_candidate = np.max(np.linalg.eigvals(covariance_matrix)) # find the maximum eigenvalue of the covariance matrix

                # new method: fast
                Y_centered = Y_matrix.T - Y_matrix.T @ w_tilde_k[:, np.newaxis]  # Center the Y_matrix using the weights
                max_eigenvalue_candidate = np.max(np.linalg.eigvals(Y_centered @ (Y_centered.T * w_tilde_k[:, np.newaxis])))

                if max_eigenvalue_candidate > max_eigenvalue:
                    max_eigenvalue = max_eigenvalue_candidate
                    optimal_x = x

        print(f"max eigenvalue for {tilde_k}: {max_eigenvalue}.")

        return max_eigenvalue, optimal_x
    
    def generate_smoothness_param(self):
        r'''
        Generate the smoothness parameter for the entropic OT map estimator
        '''
        tilde_K = self.tilde_K
        theta = self.theta
        strong_convexity_param_dict = self.strong_convexity_param_dict
        smoothness_param_dict = {}

        if self.bound_type == "eigen_bound": # only used in 2d case for visually non-trivial measures (slow)
            for tilde_k in range(tilde_K):
                max_eigenvalue, _ = self.solve_maxeigen_problem(tilde_k)
                smoothness_param = max_eigenvalue / theta + 2 * strong_convexity_param_dict[tilde_k]
                smoothness_param_dict[tilde_k] = 1.05 * smoothness_param # buffering for the maximization problem
        if self.bound_type == "norm_bound":
            for tilde_k in range(tilde_K):
                # find the max norm in row vectors of Y_matrix
                Y_matrix = self.Y_matrix_dict[tilde_k]
                max_norm = np.max(np.linalg.norm(Y_matrix, axis = 1))
                smoothness_param = max_norm / theta + 2 * strong_convexity_param_dict[tilde_k]
                smoothness_param_dict[tilde_k] = 1.05 * smoothness_param
        self.smoothness_param_dict = smoothness_param_dict

    def deterministic_mapping(self, x):
        collect_candidate_dict = self.collect_candidate_maps(x)
        num_measures = self.num_measures
        measure_samples = {}
        for measure_index in range(num_measures):
        #### combination type 2 ####
            if num_measures % 2 == 1:
                if measure_index == num_measures - 1:
                    # func_1st = raw_func_list[measure_index]
                    # func_2nd = raw_func_list[0]
                    image_1st = collect_candidate_dict[2 * measure_index]
                    image_2nd = collect_candidate_dict[0]
                    samples_generated = x + (image_1st - image_2nd)
                    measure_samples[measure_index] = samples_generated
                    
                elif measure_index < (num_measures - 1) / 2:
                    # func_1st = raw_func_list[measure_index]
                    # func_2nd = raw_func_list[measure_index + 2]
                    image_1st = collect_candidate_dict[2 * measure_index]
                    image_2nd = collect_candidate_dict[2 * measure_index + 4]
                    samples_generated = image_1st + image_2nd
                    measure_samples[measure_index] = samples_generated
                    
                elif measure_index >= (num_measures - 1) / 2:
                    # func_1st = raw_func_list[2 * measure_index - num_measures + 2]
                    # func_2nd = raw_func_list[2 * measure_index - num_measures + 3]
                    image_1st = collect_candidate_dict[(2 * measure_index - num_measures + 2) * 2]
                    image_2nd = collect_candidate_dict[(2 * measure_index - num_measures + 3) * 2]
                    samples_generated = 2 * x - (image_1st + image_2nd)
                    measure_samples[measure_index] = samples_generated

            else:
                if measure_index < num_measures / 2:
                    # func_1st = raw_func_list[measure_index]
                    # func_2nd = raw_func_list[measure_index + 2]
                    image_1st = collect_candidate_dict[2 * measure_index]
                    image_2nd = collect_candidate_dict[2 * measure_index + 4]
                    samples_generated = image_1st + image_2nd
                    measure_samples[measure_index] = samples_generated
                    
                elif measure_index >= num_measures / 2:
                    # func_1st = raw_func_list[2 * measure_index - num_measures]
                    # func_2nd = raw_func_list[2 * measure_index - num_measures + 1]
                    image_1st = collect_candidate_dict[2 * (2 * measure_index - num_measures)]
                    image_2nd = collect_candidate_dict[2 * (2 * measure_index - num_measures + 1)]
                    samples_generated = 2 * x - (image_1st + image_2nd)
                    measure_samples[measure_index] = samples_generated

        return measure_samples
    

    def collect_candidate_maps(self, x):
        # x is the input vector to be evaluated at by the mappings
        Y_matrix_dict = self.Y_matrix_dict
        g_vector_dict = self.g_vector_dict
        theta = self.theta
        n_k = self.n_k
        tilde_K = self.tilde_K
  
        strong_convexity_param_dict = self.strong_convexity_param_dict
        smoothness_param_dict = self.smoothness_param_dict
        
        candidate_map_dict = {}
        x_tile = np.tile(x, (n_k, 1))
        
        for i in range(tilde_K):
            g_vector = g_vector_dict[i]
            Y_matrix = Y_matrix_dict[i]

            # note that here the cost function has the coefficient 1/2
            exponent_vec = (g_vector - norm(x_tile - Y_matrix, axis = 1)**2 / 2) / theta
            exponent_vec_max = np.max(exponent_vec)
            exponent_vec -= exponent_vec_max # divide by the maximum value to avoid numerical instability
            expval_vec = np.exp(exponent_vec)
            numerator = Y_matrix.T @ expval_vec
            denominator = np.sum(expval_vec)

            candidate_map_plus = numerator / denominator + strong_convexity_param_dict[i] * x
            candidate_map_minus = x * smoothness_param_dict[i] - candidate_map_plus
            candidate_map_dict[2 * i] = candidate_map_plus
            candidate_map_dict[2 * i + 1] = candidate_map_minus
        return candidate_map_dict

    def generate_input_measure_sample(self, x, check_empty = False):
        r'''
        Generate the input measure sample by sampling from the candidate maps
        '''
        candidate_map_dict = self.collect_candidate_maps(x)
        num_measures = self.num_measures
        tilde_K = self.tilde_K
        surjective_mapping = self.surjective_mapping
        smoothness_param_dict = self.smoothness_param_dict
        A_matrices_dict = self.A_matrices_dict
        gamma = self.gamma
        manual = self.manual

        candidate_allocation = {k: [] for k in range(num_measures)}
        for i in range(2 * tilde_K):
            b = surjective_mapping[i]
            candidate_allocation[b].append(candidate_map_dict[i])

        # check whether there is any empty allocation
        if check_empty:
            for b in range(num_measures):
                if len(candidate_allocation[b]) == 0:
                    print(f"Empty allocation for measure {b}")

        if not manual: # uniformly assign alpha
            sum_smoothness = np.sum([smoothness_param_dict[i] for i in range(tilde_K)])
            alpha = (1 - gamma) * num_measures / sum_smoothness
            beta = 1 
            if A_matrices_dict is None:
                measure_samples_dict = {b: alpha * np.sum(candidate_allocation[b], axis = 0) for b in range(num_measures)}
            else:
                measure_samples_dict = {b: alpha * np.sum(candidate_allocation[b], axis = 0) + gamma * beta * A_matrices_dict[b] @ np.squeeze(x) for b in range(num_measures)}

        else: # we design the combination of candidates and A-matrices manually in a tailored way for nontrivial measures.

            # The below operations are reverse-engineered for the case of num_measures = 5 and tilde_K = 5. The seed for this entropic sampler is 120. 
            measure_samples_dict = {}
            lambda_list = []
            for i in range(tilde_K):
                lambda_list.append(smoothness_param_dict[i])

            # idea: concentrate all the A_matrices in the mappings containing "minus" candidate maps (which are seemingly affine due to dominant lambda_overline; we use A_matrices to further shape the ground-truth measure)
            alpha_2 = 5 * (1 - gamma) / (3* lambda_list[2])
            alpha_3 = 5 * (1 - gamma) / (3* lambda_list[3])
            alpha_4 = 5 * (1 - gamma) / (3* lambda_list[4])

            add_on_matrix = np.array([[0, 0.8], [0.8, 0]])

            measure_samples_dict[0] = 0 * candidate_allocation[0][0] + alpha_2 * candidate_allocation[0][1] + alpha_3 * candidate_allocation[0][2] + gamma * x @ (A_matrices_dict[3] + A_matrices_dict[1]+ A_matrices_dict[2] - 2 * add_on_matrix)
            # the first allocation collection contains the "minus" maps corresponding to auxiliary measures 1, 2, 3.
            # note that the indices here are the ones within candidate_allocation, not the original indices.
            measure_samples_dict[1] = 0 * candidate_allocation[1][0] + alpha_4 * candidate_allocation[1][1] + gamma * x @ (A_matrices_dict[0] + A_matrices_dict[4] + 2 * add_on_matrix)
            # the second allocation collection contains the "minus" maps corresponding to auxiliary measures 0, 4.
            measure_samples_dict[2] = 0 * candidate_allocation[2][0] + alpha_3 * candidate_allocation[2][1]
            # the third allocation collection contains the "plus" maps corresponding to auxiliary measures 0, 3.
            measure_samples_dict[3] = 0 * candidate_allocation[3][0] + alpha_2 * candidate_allocation[3][1]
            # the fourth allocation collection contains the "plus" maps corresponding to auxiliary measures 1, 2.
            measure_samples_dict[4] = alpha_4 * candidate_allocation[4][0]
            # the fifth allocation collection contains the "plus" maps corresponding to auxiliary measure 4.

        return measure_samples_dict, candidate_map_dict
    
    def sample(self, sample_size = 1000):
        r'''
        Generate the input measure samples for a given sample size
        Modified on 20260114: We use different source samples for different input measures
        '''

        batch_sample_collection = {k: np.zeros((sample_size, self.dim)) for k in range(self.num_measures)}

        for k in range(self.num_measures):
            # source_samples = self.source_sampler.sample(sample_size)
            num_samples_collected = 0
            with tqdm(total=sample_size, desc=f"Sampling input measure {k}") as pbar:
                while num_samples_collected < sample_size:
                # for i in tqdm(range(sample_size), desc= f"Generating {sample_size} input measure samples"):
                    x = self.source_sampler.sample(1)
                    measure_samples_dict, _ = self.generate_input_measure_sample(x) # a dictionary with k keys
                    if np.linalg.norm(measure_samples_dict[k]) <= self.truncated_radius: # rejection sampling with the specified radius
                        batch_sample_collection[k][num_samples_collected] = measure_samples_dict[k]
                        num_samples_collected += 1
                        pbar.update(1)
        return batch_sample_collection
    
    def compute_true_V_value(self, MC_sample_size = 1e7):
        r'''
        Approximately compute the true V-value (i.e., the minimal value of the barycenter functional) via Monte Carlo integration
        The effects of the truncation of the input measures are ignored
        '''
        # ignore the random seed
        source_samples = self.source_sampler.sample(MC_sample_size)

        distsq_mat = np.zeros((MC_sample_size, self.num_measures))
        
        for i in tqdm(range(MC_sample_size), desc= f"Evaluating samples"):
            measure_samples_dict, _ = self.generate_input_measure_sample(source_samples[i])

            for k in range(self.num_measures):
                distsq_mat[i, k] = np.sum(np.square(source_samples[i] - measure_samples_dict[k]))

        V_vec = np.mean(distsq_mat, axis=1)
        V_mean = np.mean(V_vec)
        V_std = np.std(V_vec)

        return V_mean, V_std, V_vec, distsq_mat

def reservoir_sample_csv(
    csv_filename,
    num_samples,
    skiprows=0,
    usecols=None, # selected columns corresponding to targeted coefficients
    chunksize=5000,
    seed=None,
):
    """
    Uniformly sample num_samples rows from a CSV using reservoir sampling.
    Does NOT load the full CSV into memory.
    """
    rng = np.random.default_rng(seed)

    # Count total rows for progress bar (cheap, no parsing)
    with open(csv_filename, "r") as f:
        total_rows = sum(1 for _ in f) - skiprows

    reservoir = None
    seen = 0

    reader = pd.read_csv(
        csv_filename,
        skiprows=skiprows,
        usecols=usecols,
        chunksize=chunksize,
    )

    with tqdm(
        total=total_rows,
        desc=f"Reservoir sampling {os.path.basename(csv_filename)}",
        unit="rows",
    ) as pbar: # count how many rows have been read so far
        for chunk in reader:
            arr = chunk.to_numpy()
            for row in arr:
                if seen < num_samples:
                    if reservoir is None:
                        reservoir = np.empty((num_samples, arr.shape[1]))
                    reservoir[seen] = row
                else:
                    j = rng.integers(0, seen + 1)
                    if j < num_samples:
                        reservoir[j] = row
                seen += 1
            pbar.update(len(arr))

    if seen < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples but only {seen} rows available in {csv_filename}"
        )

    return reservoir
        

class csv_input_sampler:
    '''
    A simple class to load the csv files as input measure samplers
    '''
    def __init__(self, dim, num_measures, csv_path):
        self.dim = dim
        self.num_measures = num_measures
        self.csv_path = csv_path
    
    def sample(self, sample_size, seed=None):
        batch_sample_collection = {k: [] for k in range(self.num_measures)}
        for marg_id in range(self.num_measures):
            # df = pd.read_csv(
            #     f"{self.csv_path}/input_measure_samples_{marg_id}.csv",
            #     header=None
            # ).to_numpy()
            # # Randomly choose indices
            # idx = np.random.choice(df.shape[0], size=sample_size, replace=False)

            # # For each chosen row, append a 1D array
            # selected_rows = [df[i] for i in idx]

            # batch_sample_collection[marg_id] = selected_rows

            # use reservior sampling
            df = reservoir_sample_csv(
                f"{self.csv_path}/input_measure_samples_{marg_id}.csv",
                num_samples=sample_size,
                skiprows=0,
                usecols=None,
                chunksize = 5000,
                seed=None if seed is None else seed + marg_id
            )
            
            batch_sample_collection[marg_id] = [row for row in df]

        return batch_sample_collection
    
if __name__ == "__main__":
    from .samplers import *
    from ...Algorithms.Stochastic_FP.entropic_estimate_OT import *

    dim = 2
    num_components = 5
    num_measures = 5
    truncated_radius = 150
    seed = 1009

    load_dir = "./WB_Algo/Experiments/Synthetic_Generation/dim2_data/samplers_info"

    # Load the samplers
    source_sampler = MixtureOfGaussians(dim)
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim, num_components)
    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                        num_measures = num_measures, 
                                                        auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                        source_sampler = source_sampler,
                                                        truncated_radius = truncated_radius,
                                                        manual = True)

    source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")
    entropic_sampler = load_sampler(load_dir, entropic_sampler, sampler_type="entropic")
    print("done")

    csv_path = "./WB_Algo/Experiments/Synthetic_Generation/dim2_data/input_samples/csv_files"
    csv_sampler = csv_input_sampler(dim = dim, num_measures = num_measures, csv_path = csv_path)

    entropic_sample_collection = entropic_sampler.sample(sample_size = 3)
    csv_sample_collection = csv_sampler.sample(sample_size = 3)

    print(entropic_sample_collection[0])
    print(csv_sample_collection[0])

    # checked: types and formats are the same



        