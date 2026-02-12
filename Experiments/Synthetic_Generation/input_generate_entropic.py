import numpy as np
import math
from scipy.linalg import sqrtm, norm
from tqdm import tqdm
import os
import pickle
from Experiments.Synthetic_Generation.MOG import *
from Experiments.Synthetic_Generation.metrics_to_compare import *
from Algorithms.Stochastic_FP.entropic_estimate_OT_ott import *

def construct_surjective_mapping(tilde_K, num_measures, seed = 120):
    r'''
    Construct a surjective mapping from 2 * tilde_K to num_measures
    To ensure no cancellation of mappings, we will use the following strategy:
    1. We map the maps with odd indices to the first half of the measures
    2. We map the maps with even indices to the second half of the measures
    '''
    rng_entropy = np.random.RandomState(seed)

    A = list(range(2 * tilde_K))
    B = list(range(num_measures))

    A_odd = [a for a in A if a % 2 == 1]
    A_even = [a for a in A if a % 2 == 0]

    B_1 = [b for b in B if b < num_measures // 2]
    B_2 = [b for b in B if b >= num_measures // 2]

    mapping = {a: None for a in A}

    # map the odd indices to the first half of the measures
    chosen_A_odd = rng_entropy.choice(A_odd, size=len(B_1), replace=False)
    for b, a in zip(B_1, chosen_A_odd):
        mapping[a] = b
    remaining_A_odd = [a for a in A_odd if mapping[a] is None]
    for a in remaining_A_odd:
        mapping[a] = rng_entropy.choice(B_1)

    # map the even indices to the second half of the measures
    chosen_A_even = rng_entropy.choice(A_even, size=len(B_2), replace=False)
    for b, a in zip(B_2, chosen_A_even):
        mapping[a] = b
    remaining_A_even = [a for a in A_even if mapping[a] is None]
    for a in remaining_A_even:
        mapping[a] = rng_entropy.choice(B_2)

    return mapping

def generate_A_matrices(dim, num_measures, seed = 2000):
    r'''
    We generate a bunch of psd matrices whose weighted sum is K * identity matrix. (the sum is to be further weighted by gamma)
    The main idea is that, in case the generated maps seem too similar to the ground-truth measure, this part at least imposes some location-scatter transformation (e.g., rotation) to make the generated measures differ in shape.
    In other words, we look for some middle ground between purely nonlinear transformation (but seemingly affine) and location-scatter transformation.
    It is general challenging to generate such a group of psd matrices, but we can ues the following strategy from Proposition~4.1 and Theorem~4.2 of Alvarez-Esteban et al. (2016):
    1. Generate $\Sigma_j$ for j = 1, \dots, J which are a collection of covariance matrices. (One can consider the problem of solving the W_2 barycenter of J Gaussian measures.)
    2. Apply the deterministic iterative scheme in Theorem~4.2 of Alvarez-Esteban et al. (2016) to approximate $\Sigma_0$, the covariance matrix of the Gaussian barycenter.
    3. From Proposition~4.1 we know that $H(\Sigma_0) = Id$ is a necessary and sufficient condition for $\Sigma_0$ to be a barycenter. The idea now is to use the terms without weights as the psd matrices of our interests, namely
    $\Sigma^{-\frac{1}{2}} (\Sigma^{-\frac{1}{2}} \Sigma_j \Sigma^{-\frac{1}{2}})^{\frac{1}{2}} \Sigma^{-\frac{1}{2}}$ for j = 1, \dots, J.
    '''
    if seed is None:
        return None
    # the updating function from Thm 4.2 of Alvarez-Esteban et al. (2016)
    def compute_bary_cov(covariance_list, Sigma):
        Sigma_sum = np.zeros((dim, dim))
        for i in range(len(covariance_list)):
            sub_Sigma_square = sqrtm(Sigma) @ covariance_list[i] @ sqrtm(Sigma)
            sub_Sigma = sqrtm(sub_Sigma_square)
            Sigma_sum += sub_Sigma
        Sigma_sum = Sigma_sum / len(covariance_list)
        Sigma_update = np.linalg.solve(sqrtm(Sigma), np.eye(dim)) @ Sigma_sum @ Sigma_sum @ np.linalg.solve(sqrtm(Sigma), np.eye(dim))
        return Sigma_update
    
    # compute V_value of a covariance matrix (Eq. (15) of Alvarez-Esteban et al. (2016))
    def compute_V(covariance_list, Sigma):
        trace1_list = [] # the first trace term in the equation
        trace2_list = [] # the second trace term in the equation
        for i in range(len(covariance_list)):
            trace1_list.append(np.trace(covariance_list[i]))
        for i in range(len(covariance_list)):
            sub_Sigma_square = sqrtm(Sigma) @ covariance_list[i] @ sqrtm(Sigma)
            trace2_list.append(np.trace(sqrtm(sub_Sigma_square)))
        V = np.trace(Sigma) + np.mean(trace1_list) - 2 * np.mean(trace2_list)
        return V

    # construct covariance matrices.
    rng_comp = np.random.RandomState(seed)
    num_matrices = num_measures
    covariance_list = []
    for _ in range(num_matrices):
        if dim == 2:
            cov = construct_2d_covariance_ellipsoid(3, 4, rng_comp)
        else:
            cov = construct_high_dim_covariance_ellipsoid(3, 4, dim, rng_comp)
        covariance_list.append(cov)

    # initialize Sigma
    Sigma = np.eye(dim)
    V_Sigma = compute_V(covariance_list, Sigma)
    V_list = [V_Sigma]
    difference = math.inf
    while difference > 1e-5:
        Sigma = compute_bary_cov(covariance_list, Sigma)
        V_Sigma = compute_V(covariance_list, Sigma)
        difference = abs(V_Sigma - V_list[-1])
        V_list.append(V_Sigma)

    print(f"The V_value record is {V_list}.")

    # refer to H() below Eq. (17) of Alvarez-Esteban et al. (2016)
    A_matrices_dict = {}
    for i in range(num_matrices):
        sub_Sigma_square = sqrtm(Sigma) @ covariance_list[i] @ sqrtm(Sigma)
        A_matrix = np.linalg.solve(sqrtm(Sigma), np.eye(dim)) @ sqrtm(sub_Sigma_square) @ np.linalg.solve(sqrtm(Sigma), np.eye(dim))
        A_matrices_dict[i] = A_matrix

    return A_matrices_dict
    # beta_k = 1 for all k 
    

class entropic_input_sampler:
    '''
    Python class for generating samples from input measures using entropic transportation maps
    '''
    @staticmethod
    def setup(dim,
              source_info,
              auxiliary_info,
              n_k,
              alpha_list,
              theta_list,
              gamma,
              truncated_radius,
              surjective_mapping,
              A_matrices,
              maxeig_grid_size,
              save_dir):
        source_sampler = MixtureOfGaussians(dim = dim, 
                                            master_sampling_rng = source_info["master_sampling_rng"], 
                                            component_seed = source_info["component_seed"])
        source_sampler.random_components(num_components = source_info["num_components"], 
                                         uniform_weights = True)
        source_sampler.set_truncation(truncated_radius)

        auxiliary_measure_sampler_set = []
        for auxiliary_seed in auxiliary_info["auxiliary_seeds_list"]:
            auxiliary_measure_sampler = MixtureOfGaussians(dim = dim, 
                                                           master_sampling_rng = auxiliary_info["master_sampling_rng"])
            auxiliary_measure_sampler.random_components(num_components = auxiliary_info["num_components"], 
                                                        uniform_weights = True, manual_component_seed = auxiliary_seed)
            auxiliary_measure_sampler_set.append(auxiliary_measure_sampler)
        
        num_measures = len(auxiliary_measure_sampler_set)

        if dim == 2:
            bound_type = "eigen_bound"
        else:
            bound_type = "norm_bound"
        
        entropic_sampler = entropic_input_sampler(dim=dim, 
                                              num_measures = num_measures, 
                                              auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                              source_sampler = source_sampler, 
                                              n_k = n_k, 
                                              alpha_list = alpha_list,
                                              theta_list = theta_list,
                                              gamma = gamma, 
                                              truncated_radius = truncated_radius,
                                              bound_type = bound_type,
                                              surjective_mapping = surjective_mapping,
                                              A_matrices_dict = A_matrices,
                                              maxeig_grid_size = maxeig_grid_size)
        
        # generate strong convexity parameters of the mappings.
        entropic_sampler.generate_strong_convexity_param()
        print("strong convexity parameters all set.")
        # generate Y matrices
        entropic_sampler.generate_Y_matrices()
        print("Y matrices all set.")
        # generate g vectors
        entropic_sampler.generate_g_vectors()
        print("g vectors all set.")
        # generate smoothness parameters; this involves solving max eigen for each tilde_k
        entropic_sampler.generate_smoothness_param()
        print("smoothness parameters all set.")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "entropic_sampler_info.pkl")

            state = dict(entropic_sampler.__dict__)   # shallow copy

            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Entropic sampler successfully saved to {save_dir}/entropic_sampler_info.pkl")
        
        return entropic_sampler

    @staticmethod
    def load_from_file(load_dir):
        entropic_sampler = entropic_input_sampler()
        with open(f"{load_dir}/entropic_sampler_info.pkl", "rb") as f:
            loaded_data_entropic_sampler = pickle.load(f)
            print(f"Entropic sampler successfully loaded")
            entropic_sampler.__dict__.update(loaded_data_entropic_sampler)
        return entropic_sampler

    def __init__(self, 
                 dim = None, 
                 num_measures = None, 
                 auxiliary_measure_sampler_set = None, 
                 source_sampler = None, 
                 n_k = None, 
                 alpha_list = None,
                 theta_list = None,
                 gamma = None,  
                 truncated_radius = None,
                 bound_type = None,
                 surjective_mapping = None,
                 A_matrices_dict = None,
                 maxeig_grid_size = None):
        self.dim = dim
        self.num_measures = num_measures
        self.auxiliary_measure_sampler_set = auxiliary_measure_sampler_set
        self.tilde_K = len(auxiliary_measure_sampler_set) if auxiliary_measure_sampler_set is not None else 0 # 2 * tilde_K > num_measures
        self.source_sampler = source_sampler
        self.n_k = n_k # we assume that $n_k$ across 1, \dots, \tilde{K} are the same
        self.alpha_list = alpha_list
        self.theta_list = theta_list
        self.gamma = gamma
        self.truncated_radius = truncated_radius
        self.bound_type = bound_type
        self.maxeig_grid_size = maxeig_grid_size
        self.surjective_mapping = surjective_mapping
        self.A_matrices_dict = A_matrices_dict

    def generate_strong_convexity_param(self):
        r'''
        Set the strong convexity parameter for the entropic OT map estimator
        '''
        self.strong_convexity_param_dict = {i: 0.0001 for i in range(self.tilde_K)}

    def generate_Y_matrices(self):
        r'''
        Generate the Y matrices for the entropic OT map estimator (for interpolation purpose)
        '''
        Y_matrix_dict = {}
        for i in range(self.tilde_K):
            auxiliary_measure_sampler = self.auxiliary_measure_sampler_set[i]
            Y = auxiliary_measure_sampler.sample(self.n_k)
            Y_matrix_dict[i] = Y
            print(f"Finished generating Y matrix for auxiliary measure {i}")
        self.Y_matrix_dict = Y_matrix_dict

    def generate_g_vectors(self, epsilon = 10): # epsilon here is the entropic regularization parameter
        r'''
        Generate the g vector as the output of solving entropic OT maps out of samples from the auxiliary measures and the source measure
        (i.e., the ground-truth barycenter).
        The solver is from the ott package.
        '''
        X = self.source_sampler.sample(self.n_k)
        Y_matrix_dict = self.Y_matrix_dict
        g_vector_dict = {}
        for i in range(self.tilde_K):
            Y = Y_matrix_dict[i]
            entropic_OT_map_generator = entropic_OT_map_estimate_ott(X, Y, log = False)
            entropic_OT_map_generator.get_dual_potential(epsilon = epsilon)

            # here we divide the potential by 2 because the potential returned by entropic_OT_map_estimate is optimal with respect to the cost function norm(x - y) ** 2 without the coefficient 1/2; dividing the potential by 2 makes the interpretation of the parameter theta consistent with Algorithm 3 in the paper
            g_vector_dict[i] = entropic_OT_map_generator.g_potential / 2

            print(f"Finished generating g vector for auxiliary measure {i}")
        self.g_vector_dict = g_vector_dict

    def entropic_weight_vector(self, x, Y_matrix, g_vector, theta):
        r'''
        Compute the entropic weight vector when evaluated at x, given parameter dictionaries.
        Here, we fix tilde_K, and Y_matrix is a matrix of dimension n * d, g_vector is a vector of dimension n * 1, and theta is a scalar.
        The output vector is a vector of dimension n * 1.
        '''
        x_tile = np.tile(x, (self.n_k, 1))

        # note that here the cost function has the coefficient 1/2
        exponent_vec = (g_vector - norm(x_tile - Y_matrix, axis = 1)**2 / 2) / theta
        exponent_vec_max = np.max(exponent_vec)
        exponent_vec -= exponent_vec_max

        numerator = np.exp(exponent_vec)
        return numerator / np.sum(numerator)
        
    def solve_maxeigen_problem(self, tilde_k):
        r'''
        We aim to maximize the maximum eigenvalue of the covariance matrix (Line~5 in Algorithm~3), corresponding to the data collected from auxiliary measure tilde_k.
        Due to the highly nonlinear structure of w(x), we traverse the grid space to find the optimal solution.
        '''
        # generate the 2d grid space spanning over -100 to 100 for each dimension
        # at each dimension, we have 100 points, thus we have 100^2 points in total
        Y_matrix = self.Y_matrix_dict[tilde_k]
        g_vector = self.g_vector_dict[tilde_k]
        theta = self.theta_list[tilde_k]
        grid_space = np.linspace(-self.truncated_radius, self.truncated_radius, self.maxeig_grid_size)
        max_eigenvalue = 0
        optimal_x = None

        for i in tqdm(range(self.maxeig_grid_size), desc= f"tilde_k: {tilde_k}"):
            for j in range(self.maxeig_grid_size): # traverse the grid space   
                x = np.array([grid_space[i], grid_space[j]])
                w_tilde_k = self.entropic_weight_vector(x, Y_matrix, g_vector, theta)

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

        return max_eigenvalue, optimal_x
    
    def generate_smoothness_param(self):
        r'''
        Generate the smoothness parameter for the entropic OT map estimator
        '''
        smoothness_param_dict = {}

        if self.bound_type == "eigen_bound": # only used in 2d case for visually non-trivial measures (slow)
            for tilde_k in range(self.tilde_K):
                max_eigenvalue, _ = self.solve_maxeigen_problem(tilde_k)
                print(f"max eigenvalue for {tilde_k}: {max_eigenvalue}.")
                smoothness_param = max_eigenvalue / self.theta_list[tilde_k] + 2 * self.strong_convexity_param_dict[tilde_k]
                smoothness_param_dict[tilde_k] = 1.05 * smoothness_param # buffering for the maximization problem
        if self.bound_type == "norm_bound":
            for tilde_k in range(self.tilde_K):
                # find the max norm in row vectors of Y_matrix
                Y_matrix = self.Y_matrix_dict[tilde_k]
                max_normsq = np.max(np.sum(np.square(Y_matrix), axis = 1))
                print(f"max squared norm for {tilde_k}: {max_normsq}.")
                smoothness_param = max_normsq / self.theta_list[tilde_k] + 2 * self.strong_convexity_param_dict[tilde_k]
                smoothness_param_dict[tilde_k] = 1.0 * smoothness_param
        self.smoothness_param_dict = smoothness_param_dict
    

    def collect_candidate_maps(self, x):
        # x is the input vector to be evaluated at by the mappings
  
        strong_convexity_param_dict = self.strong_convexity_param_dict
        smoothness_param_dict = self.smoothness_param_dict
        
        candidate_map_dict = {}
        x_tile = np.tile(x, (self.n_k, 1))
        
        for i in range(self.tilde_K):
            g_vector = self.g_vector_dict[i]
            Y_matrix = self.Y_matrix_dict[i]

            # note that here the cost function has the coefficient 1/2
            exponent_vec = (g_vector - norm(x_tile - Y_matrix, axis = 1)**2 / 2) / self.theta_list[i]
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

        wsum_smoothness = np.sum([self.smoothness_param_dict[i] * self.alpha_list[i] for i in range(self.tilde_K)]) / self.num_measures
        beta_list = np.repeat((1 - self.gamma) * np.array(self.alpha_list) / wsum_smoothness, 2)

        candidate_map_dict = self.collect_candidate_maps(x)

        candidate_allocation = {k: [] for k in range(self.num_measures)}
        for i in range(2 * self.tilde_K):
            b = self.surjective_mapping[i]
            candidate_allocation[b].append(candidate_map_dict[i] * beta_list[i])

        # check whether there is any empty allocation
        if check_empty:
            for b in range(self.num_measures):
                if len(candidate_allocation[b]) == 0:
                    print(f"Empty allocation for measure {b}")

        if self.A_matrices_dict is None:
            measure_samples_dict = {b: np.sum(candidate_allocation[b], axis = 0) for b in range(self.num_measures)}
        else:
            measure_samples_dict = {b: np.sum(candidate_allocation[b], axis = 0) + self.gamma * self.A_matrices_dict[b] @ np.squeeze(x) for b in range(self.num_measures)}
        
        return measure_samples_dict, candidate_map_dict
    
    def sample(self, sample_size, print_rejection = False):
        r'''
        Generate the input measure samples for a given sample size
        Modified on 20260114: We use different source samples for different input measures
        '''

        batch_sample_collection = {k: np.zeros((sample_size, self.dim)) for k in range(self.num_measures)}

        for k in range(self.num_measures):
            # source_samples = self.source_sampler.sample(sample_size)
            num_samples_collected = 0
            num_samples_rejected = 0
            with tqdm(total=sample_size, desc=f"Sampling input measure {k}") as pbar:
                while num_samples_collected < sample_size:
                # for i in tqdm(range(sample_size), desc= f"Generating {sample_size} input measure samples"):
                    x = self.source_sampler.sample(1, use_truncation = False)
                    measure_samples_dict, _ = self.generate_input_measure_sample(x) # a dictionary with k keys
                    if np.linalg.norm(measure_samples_dict[k]) <= self.truncated_radius: # rejection sampling with the specified radius
                        batch_sample_collection[k][num_samples_collected] = measure_samples_dict[k]
                        num_samples_collected += 1
                        pbar.update(1)
                    else:
                        num_samples_rejected += 1
            
            if print_rejection:
                print(f"Sampling from input measure {k} complete, {num_samples_rejected} samples rejected")
        return batch_sample_collection
    
    def compute_true_V_value(self, MC_sample_size):
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
    
    def compute_true_V_value_via_OT(self, sample_size, num_rep):
        r'''
        Approximately compute the true V-value (i.e., the minimal value of the barycenter functional) via Monte Carlo integration
        The effects of the truncation of the input measures are ignored
        '''
        # ignore the random seed
        V_vec = np.zeros(num_rep)
        
        for rep in range(num_rep):
            source_samples = self.source_sampler.sample(sample_size)
            input_samples = self.sample(sample_size)

            for k in range(self.num_measures):
                V_vec[rep] += W2_pot(source_samples, input_samples[k]) / self.num_measures
            
            print(f"V-value computed = {V_vec[rep]}")

        V_mean = np.mean(V_vec)
        V_std = np.std(V_vec)

        return V_mean, V_std, V_vec
