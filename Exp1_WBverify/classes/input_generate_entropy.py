import numpy as np
from scipy.linalg import sqrtm, norm
from tqdm import tqdm, tqdm_notebook

from entropic_estimate_OT import *

class entropic_input_sampler:
    r'''
    Python class for generating samples from input measures using entropic transportation maps
    '''
    def __init__(self, dim, num_measures, auxiliary_measure_sampler_set, source_sampler, n_k = 500, seed = 100):
        self.dim = dim
        self.num_measures = num_measures
        self.auxiliary_measure_sampler_set = auxiliary_measure_sampler_set
        self.tilde_K = len(auxiliary_measure_sampler_set)
        self.source_sampler = source_sampler
        self.n_k = n_k # we assume that $n_k$ across 1, \dots, \tilde{K} are the same
        # num_measures < 2 * tilde_K
        self.seed = seed
        self.rng_entropy = np.random.RandomState(seed)

    def generate_Y_and_g(self, epsilon):
        r'''
        We manually choose "fancy" auxiliary measures to generate the Y matrices and g vectors
        The auxiliary measure sampler set is a list.
        '''
        auxiliary_measure_sampler_set = self.auxiliary_measure_sampler_set
        source_sampler = self.source_sampler
        n_k = self.n_k
        X = source_sampler.sample(n_k)
        Y_matrix_dict = {}
        g_vector_dict = {}
        for i in range(len(auxiliary_measure_sampler_set)):
            auxiliary_measure_sampler = auxiliary_measure_sampler_set[i]
            Y = auxiliary_measure_sampler.sample(n_k)
            entropic_OT_map_generator = entropic_OT_map_estimate(X, Y, log = False)
            entropic_OT_map_generator.get_dual_potential(epsilon = epsilon)
            Y_matrix_dict[i] = Y
            g_vector_dict[i] = entropic_OT_map_generator.g_potential
            print(f"Finished generating Y matrix and g vector for auxiliary measure {i}")
        self.Y_matrix_dict = Y_matrix_dict
        self.g_vector_dict = g_vector_dict

    def construct_surjective_mapping(self):
        r'''
        Construct a surjective mapping from 2 * tilde_K to num_measures
        To ensure no cancellation of mappings, we will use the following strategy:
        1. We map the maps with odd indices to the first half of the measures
        2. We map the maps with even indices to the second half of the measures
        '''
        tilde_K = self.tilde_K
        num_measures = self.num_measures
        rng_entropy = self.rng_entropy

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

    # def construct_surjective_mapping(self):
    #     r'''
    #     Construct a surjective mapping from 2 * tilde_K to num_measures
    #     '''
    #     tilde_K = self.tilde_K
    #     num_measures = self.num_measures
    #     rng_entropy = self.rng_entropy

    #     A = list(range(2 * tilde_K))
    #     B = list(range(num_measures))

    #     mapping = {a: None for a in A}
    #     chosen_A = rng_entropy.choice(A, size=len(B), replace=False)
    #     for b, a in zip(B, chosen_A):
    #         mapping[a] = b
    #     remaining_A = [a for a in A if mapping[a] is None]
    #     weights = rng_entropy.random(len(B))
    #     weights[:len(B) // 2] += rng_entropy.random(len(B) // 2) * 2 
    #     weights = weights / np.sum(weights)
    #     for a in remaining_A:
    #         mapping[a] = rng_entropy.choice(B, p = weights)

    #     self.surjective_mapping = mapping

    def generate_strong_convexity_param(self):
        r'''
        Generate the strong convexity parameter for the entropic OT map estimator
        '''
        rng_entropy = self.rng_entropy
        tilde_K = self.tilde_K
        upper_bound = 1 / tilde_K
        strong_convexity_param = rng_entropy.uniform(low = 0, high = upper_bound / 2, size = tilde_K)
        # make it a dictionary
        self.strong_convexity_param_dict = {i: strong_convexity_param[i] for i in range(tilde_K)}

    # def generate_g_vectors(self):
    #     r'''
    #     Generate the g vector for the entropic OT map estimator (for interpolation purpose)
    #     '''
    #     rng_entropy = self.rng_entropy
    #     tilde_K = self.tilde_K
    #     n_k = self.n_k
    #     g_vector_dict = {}
    #     for i in range(tilde_K):
    #         g_vector = rng_entropy.uniform(low = -1, high = 1, size = n_k)
    #         g_vector_dict[i] = g_vector
    #     self.g_vector_dict = g_vector_dict


    # def generate_Y_matrices(self):
    #     r'''
    #     Generate the Y matrices for the entropic OT map estimator (for interpolation purpose)
    #     '''
    #     rng_entropy = self.rng_entropy
    #     dim = self.dim
    #     tilde_K = self.tilde_K
    #     source_sampler = self.source_sampler
    #     n_k = self.n_k
    #     Y_matrix_dict = {}
    #     for k in range(tilde_K):
    #     # Initialize an empty matrix
    #         matrix = np.zeros((n_k, dim))
    #         for i in range(dim):
    #             # Calculate the interval for the i-th column
    #             lower_bound = -100 + 2 * 100 * i / dim
    #             upper_bound = -100 + 2 * 100 * (i + 1) / dim
    #             matrix[:, i] = rng_entropy.uniform(lower_bound, upper_bound, size=n_k)
    #         Y_matrix_dict[k] = matrix
    #     self.Y_matrix_dict = Y_matrix_dict
        
    def compute_theta(self):
        r'''
        Compute the theta entropic parameter theta that satisfies 
        max{\|y_j\|^2} / theta + \underline{lambda} < 1 / \tilde{K}
        '''
        tilde_K = self.tilde_K
        strong_convexity_param_dict = self.strong_convexity_param_dict
        Y_matrix_dict = self.Y_matrix_dict
        theta_dict = {}
        for i in range(tilde_K):
            # find the maximum norm (over all rows) of Y_matrix_dict[i]
            max_Y_norm_sq = np.max(np.linalg.norm(Y_matrix_dict[i], axis = 1))**2
            theta_reciprocal = (1 / tilde_K - strong_convexity_param_dict[i]) / max_Y_norm_sq / 2 # half the residue for safety
            theta_dict[i] = 1 / theta_reciprocal
        # find the maximum theta
        theta_max = max(theta_dict.values())
        # update the theta_dict
        for i in range(tilde_K):
            theta_dict[i] = theta_max
        self.theta_dict = theta_dict

    def collect_candidate_maps(self, x):
        # x is the input vector to be evaluated at by the mappings
        Y_matrix_dict = self.Y_matrix_dict
        g_vector_dict = self.g_vector_dict
        theta_dict = self.theta_dict
        n_k = self.n_k
        tilde_K = self.tilde_K
        dim = self.dim
        strong_convexity_param_dict = self.strong_convexity_param_dict

        candidate_map_dict = {}
        
        for i in range(tilde_K):
            dinominator = 0
            numerator = np.zeros(dim)
            for j in range(n_k):
                exponent = (g_vector_dict[i][j] + np.dot(Y_matrix_dict[i][j], x) - 0.5 * norm(Y_matrix_dict[i][j])**2) / theta_dict[i]
                dinominator += np.exp(exponent)
                numerator += np.exp(exponent) * Y_matrix_dict[i][j]
            candidate_map_plus = numerator / dinominator + strong_convexity_param_dict[i] * x
            candidate_map_minus = x / tilde_K - candidate_map_plus
            candidate_map_dict[2 * i] = candidate_map_plus
            candidate_map_dict[2 * i + 1] = candidate_map_minus
        return candidate_map_dict
    
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

    def generate_input_measure_sample(self, x, check_empty = False):
        r'''
        Generate the input measure sample by sampling from the candidate maps
        '''
        candidate_map_dict = self.collect_candidate_maps(x)
        num_measures = self.num_measures
        tilde_K = self.tilde_K
        dim = self.dim
        surjective_mapping = self.surjective_mapping

        if tilde_K < num_measures:
            candidate_allocation = {k: [] for k in range(num_measures)}
            for i in range(2 * tilde_K):
                b = surjective_mapping[i]
                candidate_allocation[b].append(candidate_map_dict[i])

            # check whether there is any empty allocation
            if check_empty:
                for b in range(num_measures):
                    if len(candidate_allocation[b]) == 0:
                        print(f"Empty allocation for measure {b}")
            
            # measure_samples = {b: np.sum(candidate_allocation[b], axis = 0) for b in range(num_measures)}
            measure_samples = {b: num_measures * np.sum(candidate_allocation[b], axis = 0) for b in range(num_measures)}
        else:
            measure_samples = self.deterministic_mapping(x)
        return measure_samples
    
    def sample(self, sample_size = 1000):
        r'''
        Generate the input measure samples for a given sample size
        '''
        num_measures = self.num_measures
        source_sampler = self.source_sampler

        batch_sample_collection = {k: [] for k in range(num_measures)}

        source_samples = source_sampler.sample(sample_size)
        measure_samples = self.generate_input_measure_sample(source_samples[0])
        
        for i in tqdm(range(sample_size), desc= f"Generating {sample_size} input measure samples"):
            x = source_samples[i]
            measure_samples = self.generate_input_measure_sample(x) # a dictionary with k keys
            for k in range(num_measures):
                batch_sample_collection[k].append(measure_samples[k])
        return batch_sample_collection
            

# entropic_sampler = entropic_input_sampler(dim = 2, num_measures = 3, tilde_K = 3)
# entropic_sampler.construct_surjective_mapping()
# entropic_sampler.generate_strong_convexity_param()
# entropic_sampler.generate_g_vectors()
# entropic_sampler.generate_Y_matrices()
# entropic_sampler.compute_theta()
# x = np.array([1, 0.5])
# measure_samples = entropic_sampler.generate_input_measure_sample(x)
# print(measure_samples)

        
        