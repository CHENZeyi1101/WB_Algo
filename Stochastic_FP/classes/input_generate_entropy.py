import numpy as np
import math
from scipy.linalg import sqrtm, norm
from tqdm import tqdm, tqdm_notebook
from true_WB import *

from entropic_estimate_OT import *

class entropic_input_sampler:
    r'''
    Python class for generating samples from input measures using entropic transportation maps
    '''
    def __init__(self, dim, num_measures, auxiliary_measure_sampler_set, source_sampler, n_k = 1000, seed = 120):
        self.dim = dim
        self.num_measures = num_measures
        self.auxiliary_measure_sampler_set = auxiliary_measure_sampler_set
        self.tilde_K = len(auxiliary_measure_sampler_set) # 2 * tilde_K > num_measures
        self.source_sampler = source_sampler
        self.n_k = n_k # we assume that $n_k$ across 1, \dots, \tilde{K} are the same
        # num_measures < 2 * tilde_K
        self.seed = seed
        self.rng_entropy = np.random.RandomState(seed)

    def generate_strong_convexity_param(self):
        r'''
        Generate the strong convexity parameter for the entropic OT map estimator
        '''
        rng_entropy = self.rng_entropy
        tilde_K = self.tilde_K
        # lower_bound = 0.0001
        # upper_bound = 0.00001
        # strong_convexity_param = rng_entropy.uniform(low = 0, high = upper_bound, size = tilde_K)
        # make it a dictionary
        # 
        self.strong_convexity_param_dict = {i: 0.0001 for i in range(tilde_K)}

    def assign_theta(self):
        r'''
        Assign the theta values for the entropic OT map estimator.
        The value of theta is empirically estimated from the iterative scheme experiment.
        Notice that theta scales quadratically with the value of samples (i.e., the truncation radius).
        '''
        tilde_K = self.tilde_K
        theta_dict = {}
        for i in range(tilde_K):
            theta_dict[i] = 10 # this is empirically estimated from the iterative scheme experiment. 
        self.theta_dict = theta_dict

    def generate_Y_matrices(self, seed = 1005):
        r'''
        Generate the Y matrices for the entropic OT map estimator (for interpolation purpose)
        '''
        auxiliary_measure_sampler_set = self.auxiliary_measure_sampler_set
        tilde_K = self.tilde_K
        n_k = self.n_k
        rng_entropy = np.random.RandomState(seed)
        Y_matrix_dict = {}
        for i in range(tilde_K):
            auxiliary_measure_sampler = auxiliary_measure_sampler_set[i]
            seed = rng_entropy.randint(1000)
            Y = auxiliary_measure_sampler.sample(n_k, seed = seed, multiplication_factor = 1)
            Y_matrix_dict[i] = Y
            print(f"Finished generating Y matrix for auxiliary measure {i}")
        self.Y_matrix_dict = Y_matrix_dict

    def generate_g_vectors(self):
        r'''
        Generate the g vector as the output of solving entropic OT maps out of samples from the auxiliary measures and the source measure
        (i.e., the ground-truth barycenter).
        The solver is from the ott package.
        '''
        tilde_K = self.tilde_K
        source_sampler = self.source_sampler
        theta_dict = self.theta_dict
        n_k = self.n_k
        X = source_sampler.sample(n_k, multiplication_factor = 1)
        Y_matrix_dict = self.Y_matrix_dict
        g_vector_dict = {}
        for i in range(tilde_K):
            Y = Y_matrix_dict[i]
            entropic_OT_map_generator = entropic_OT_map_estimate(X, Y, log = False)
            epsilon = theta_dict[i]
            entropic_OT_map_generator.get_dual_potential(epsilon = epsilon)
            g_vector_dict[i] = entropic_OT_map_generator.g_potential
            print(f"Finished generating g vector for auxiliary measure {i}")
        self.g_vector_dict = g_vector_dict

    def entropic_weight_vector(self, x, Y_matrix, g_vector, theta):
        r'''
        Compute the entropic weight vector when evaluated at x, given parameter dictionaries.
        Here, we fix tilde_K, and Y_matrix is a matrix of dimension n * d, g_vector is a vector of dimension n * 1, and theta is a scalar.
        The output vector is a vector of dimension n * 1.
        '''
        n_k = self.n_k
        x_tile = np.tile(x, (n_k, 1))
        exponent_vec = (g_vector - norm(x_tile - Y_matrix, axis = 1)**2) / theta
        exponent_vec_max = np.max(exponent_vec)
        exponent_vec -= exponent_vec_max

        numerator = np.exp(exponent_vec)
        denominator = np.sum(np.exp(exponent_vec))
        weight_vector = numerator / denominator

        return weight_vector
        
    def solve_maxeigen_problem(self, tilde_k, grid_size = 200, truncate_radius = 100):
        r'''
        We aim to maximize the maximum eigenvalue of the covariance matrix, corresponding to the data collected from auxiliary measure tilde_k.
        Due to the highly nonlinear structure of w(x), we traverse the grid space to find the optimal solution.
        '''
        # generate the 2d grid space spanning over -100 to 100 for each dimension
        # at each dimension, we have 100 points, thus we have 100^2 points in total
        Y_matrix = self.Y_matrix_dict[tilde_k]
        g_vector = self.g_vector_dict[tilde_k]
        theta = self.theta_dict[tilde_k]
        grid_space = np.linspace(-truncate_radius, truncate_radius, grid_size)
        max_eigenvalue = 0
        optimal_x = None

        for i in tqdm(range(grid_size), desc= f"tilde_k: {tilde_k}"):
            for j in range(grid_size): # traverse the grid space   
                x = np.array([grid_space[i], grid_space[j]])
                w_tilde_k = self.entropic_weight_vector(x, Y_matrix, g_vector, theta)
                Y_tilde_k = Y_matrix.T # !!! Y_tilde_k is of dimension d * n
                
                diag_w_k = np.diag(w_tilde_k)  # Creates a diagonal matrix with w_k as its diagonal
                outer_product_w_k = np.outer(w_tilde_k, w_tilde_k)  # Outer product of w_k * w_K^T
                matrix_diff = diag_w_k - outer_product_w_k
                covariance_matrix = Y_tilde_k @ matrix_diff @ Y_tilde_k.T
                max_eigenvalue_candidate = np.max(np.linalg.eigvals(covariance_matrix)) # find the maximum eigenvalue of the covariance matrix

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
        theta_dict = self.theta_dict
        strong_convexity_param_dict = self.strong_convexity_param_dict
        smoothness_param_dict = {}
        for tilde_k in range(tilde_K):
            max_eigenvalue, _ = self.solve_maxeigen_problem(tilde_k)
            smoothness_param = max_eigenvalue / theta_dict[tilde_k] + 2 * strong_convexity_param_dict[tilde_k]
            smoothness_param_dict[tilde_k] = 1.2 * smoothness_param # buffering for the maximization problem
        self.smoothness_param_dict = smoothness_param_dict

    # def generate_Y_and_g(self):
    #     r'''
    #     We manually choose "fancy" auxiliary measures to generate the Y matrices and g vectors
    #     The auxiliary measure sampler set is a list.
    #     '''
    #     tilde_K = self.tilde_K
    #     # auxiliary_measure_sampler_set = self.auxiliary_measure_sampler_set
    #     source_sampler = self.source_sampler
    #     theta_dict = self.theta_dict
    #     n_k = self.n_k
    #     X = source_sampler.sample(n_k, multiplication_factor = 1)
    #     Y_matrix_dict = self.Y_matrix_dict
    #     g_vector_dict = {}
    #     for i in range(tilde_K):
    #         Y = Y_matrix_dict[i]
    #         entropic_OT_map_generator = entropic_OT_map_estimate(X, Y, log = False)
    #         epsilon = theta_dict[i]
    #         entropic_OT_map_generator.get_dual_potential(epsilon = epsilon)
    #         g_vector_dict[i] = entropic_OT_map_generator.g_potential
    #         print(f"Finished generating Y matrix and g vector for auxiliary measure {i}")
    #     self.g_vector_dict = g_vector_dict

    def construct_surjective_mapping(self, seed = 120):
        r'''
        Construct a surjective mapping from 2 * tilde_K to num_measures
        To ensure no cancellation of mappings, we will use the following strategy:
        1. We map the maps with odd indices to the first half of the measures
        2. We map the maps with even indices to the second half of the measures
        '''
        tilde_K = self.tilde_K
        num_measures = self.num_measures
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

        self.surjective_mapping = mapping

    # def compute_theta(self, truncated_radius = 100):
    #     r'''
    #     We assume the upper bound on the norm of Y_j, which can be taken as the truncated radius.
    #     The 
    #     '''
    #     tilde_K = self.tilde_K
    #     strong_convexity_param_dict = self.strong_convexity_param_dict
    #     auxiliary_measure_sampler_set = self.auxiliary_measure_sampler_set
    #     n_k = self.n_k

    #     theta_dict = {}
    #     Y_matrix_dict = {}

    #     for i in range(tilde_K):
    #         auxiliary_measure_sampler = auxiliary_measure_sampler_set[i]
    #         Y = auxiliary_measure_sampler.sample(n_k, multiplication_factor = 1)
    #         Y_matrix_dict[i] = Y
    #         mean_vector = np.mean(Y, axis=0)
    #         term1 = np.mean([np.outer(y, y) for y in Y], axis=0)
    #         term2 = np.outer(mean_vector, mean_vector)
    #         covariance_matrix = term1 - term2 # covariance matrix of Y
    #         # find the maximum eigenvalue of the covariance matrix
    #         max_Y_eigenvalue = np.max(np.linalg.eigvals(covariance_matrix))
    #         theta_reciprocal = (1 / tilde_K) / max_Y_eigenvalue
    #         theta_dict[i] = 1 / theta_reciprocal
    
    #     self.theta_dict = theta_dict
    #     self.Y_matrix_dict = Y_matrix_dict

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
    
    def generate_A_matrices(self, seed = 2000):
        r'''
        We generate a bunch of psd matrices whose weighted sum is K * identity matrix. (the sum is to be further weighted by gamma)
        The main idea is that, in case the generated maps seem too similar to the ground-truth measure, this part at least imposes some location-scatter transformation (e.g., rotation) to make the generated measures differ in shape.
        In other words, we look for some middle ground between purely nonlinear transformation (but seemingly affine) and location-scatter transformation.
        It is general challenging to generate such a group of psd matrices, but we can ues the following strategy from Proposition~4.1 and Theorem~4.2 of Alvarez-Esteban et al. (2019):
        1. Generate $\Sigma_j$ for j = 1, \dots, J which are a collection of covariance matrices. (One can consider the problem of solving the W_2 barycenter of J Gaussian measures.)
        2. Apply the deterministic iterative scheme in Theorem~4.2 of Alvarez-Esteban et al. (2019) to approximate $\Sigma_0$, the covariance matrix of the Gaussian barycenter.
        3. From Proposition~4.1 we know that $H(\Sigma_0) = Id$ is a necessary and sufficient condition for $\Sigma_0$ to be a barycenter. The idea now is to use the terms without weights as the psd matrices of our interests, namely
        $\Sigma^{-\frac{1}{2}} (\Sigma^{-\frac{1}{2}} \Sigma_j \Sigma^{-\frac{1}{2}})^{\frac{1}{2}} \Sigma^{-\frac{1}{2}}$ for j = 1, \dots, J.
        '''
        dim = self.dim
        num_measures = self.num_measures
        # the updating function from Thm 4.2 of Alvarez-Esteban et al. (2019)
        def compute_bary_cov(covariance_list, Sigma):
            Sigma_sum = np.zeros((dim, dim))
            for i in range(len(covariance_list)):
                sub_Sigma_square = sqrtm(Sigma) @ covariance_list[i] @ sqrtm(Sigma)
                sub_Sigma = sqrtm(sub_Sigma_square)
                Sigma_sum += sub_Sigma
            Sigma_sum = Sigma_sum / len(covariance_list)
            Sigma_update = np.linalg.solve(sqrtm(Sigma), np.eye(dim)) @ Sigma_sum @ Sigma_sum @ np.linalg.solve(sqrtm(Sigma), np.eye(dim))
            return Sigma_update
        
        # compute V_value of a covariance matrix (Eq. (15) of Alvarez-Esteban et al. (2019))
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
            cov = construct_2d_covariance_ellipsoid(3, 4, rng_comp)
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

        # refer to H() below Eq. (17) of Alvarez-Esteban et al. (2019)
        A_matrices_dict = {}
        for i in range(num_matrices):
            sub_Sigma_square = sqrtm(Sigma) @ covariance_list[i] @ sqrtm(Sigma)
            A_matrix = np.linalg.solve(sqrtm(Sigma), np.eye(dim)) @ sqrtm(sub_Sigma_square) @ np.linalg.solve(sqrtm(Sigma), np.eye(dim))
            A_matrices_dict[i] = A_matrix

        self.A_matrices_dict = A_matrices_dict
        # beta_k = 1 for all k 

    def collect_candidate_maps(self, x):
        # x is the input vector to be evaluated at by the mappings
        Y_matrix_dict = self.Y_matrix_dict
        g_vector_dict = self.g_vector_dict
        theta_dict = self.theta_dict
        n_k = self.n_k
        tilde_K = self.tilde_K
        dim = self.dim

        strong_convexity_param_dict = self.strong_convexity_param_dict
        smoothness_param_dict = self.smoothness_param_dict
        
        candidate_map_dict = {}
        x_tile = np.tile(x, (n_k, 1))
        
        for i in range(tilde_K):
            g_vector = g_vector_dict[i]
            Y_matrix = Y_matrix_dict[i]
            theta = theta_dict[i]
            exponent_vec = (g_vector - norm(x_tile - Y_matrix, axis = 1)**2) / theta
            exponent_vec_max = np.max(exponent_vec)
            exponent_vec -= exponent_vec_max # divide by the maximum value to avoid numerical instability
            numerator = Y_matrix.T @ np.exp(exponent_vec)
            denominator = np.sum(np.exp(exponent_vec))

            candidate_map_plus = numerator / denominator + strong_convexity_param_dict[i] * x
            candidate_map_minus = x * smoothness_param_dict[i] - candidate_map_plus
            candidate_map_dict[2 * i] = candidate_map_plus
            candidate_map_dict[2 * i + 1] = candidate_map_minus
        return candidate_map_dict

    def generate_input_measure_sample(self, x, gamma = 0.3, manual = True, check_empty = False):
        r'''
        Generate the input measure sample by sampling from the candidate maps
        '''
        candidate_map_dict = self.collect_candidate_maps(x)
        num_measures = self.num_measures
        tilde_K = self.tilde_K
        surjective_mapping = self.surjective_mapping
        smoothness_param_dict = self.smoothness_param_dict
        A_matrices_dict = self.A_matrices_dict

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
            # # measure_samples = {b: np.sum(candidate_allocation[b], axis = 0) for b in range(num_measures)}
            measure_samples_dict = {b: alpha * np.sum(candidate_allocation[b], axis = 0) + gamma * beta * A_matrices_dict[b] @ x for b in range(num_measures)}

        else: # we design the combination of candidates and A-matrices manually in a tailored way for "fancy" measures.

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
    
    def sample(self, sample_size = 1000, show_candidate = False, gamma = 0.3, manual = True):
        r'''
        Generate the input measure samples for a given sample size
        '''
        num_measures = self.num_measures
        source_sampler = self.source_sampler
        tilde_K = self.tilde_K

        batch_sample_collection = {k: [] for k in range(num_measures)}
        candidate_sample_collection = {k: [] for k in range(2 * tilde_K)}

        source_samples = source_sampler.sample(sample_size, multiplication_factor = 1)
        # measure_samples = self.generate_input_measure_sample(source_samples[0])
        
        for i in tqdm(range(sample_size), desc= f"Generating {sample_size} input measure samples"):
            x = source_samples[i]
            measure_samples_dict, candidate_map_dict = self.generate_input_measure_sample(x, gamma, manual=manual) # a dictionary with k keys
            for k in range(num_measures):
                batch_sample_collection[k].append(measure_samples_dict[k])
        if show_candidate:
            for k in range(2 * tilde_K):
                candidate_sample_collection[k].append(candidate_map_dict[k])
            return batch_sample_collection, candidate_sample_collection
        else:
            return batch_sample_collection
            

# Example of usage
# entropic_sampler = entropic_input_sampler(dim = 2, num_measures = 4, auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, source_sampler = source_sampler, n_k = 500, seed = 100)
# entropic_sampler.generate_strong_convexity_param()
# entropic_sampler.assign_theta()
# entropic_sampler.generate_Y_matrices()
# entropic_sampler.generate_g_vectors()
# entropic_sampler.generate_smoothness_param()
# entropic_sampler.construct_surjective_mapping()
# entropic_sampler.generate_A_matrices()
# entropic_sampler.sample(sample_size = 1000, show_candidate = False)

        
        