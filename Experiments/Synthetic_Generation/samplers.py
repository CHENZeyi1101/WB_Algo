import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
import io
import math
from scipy.linalg import sqrtm, norm

from Experiments.Synthetic_Generation.MOG import *
from Experiments.Synthetic_Generation.input_generate_entropic import entropic_input_sampler, csv_input_sampler 
from Experiments.Synthetic_Generation.sample_plot import *
from Experiments.CSV_read import csv_auxiliary_sampler_SyntheticGeneration

''' 
This module characterizes and sets up samplers for synthetic experiments in 2D.
'''

def characterize_source_sampler(dim, num_components = 5, master_sampling_rng = 42, component_seed = 42, truncated_radius = 1000, save_dir = None):
    """
    Characterize the source sampler (mixture of Gaussians) and auxiliary measure samplers for synthetic experiments.
    """
    source_sampler = MixtureOfGaussians(dim, master_sampling_rng=master_sampling_rng, component_seed=component_seed)
    source_sampler.random_components(num_components = num_components, uniform_weights = True)
    source_sampler.set_truncation(truncated_radius)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/source_sampler_info.pkl", "wb") as f:
            pickle.dump(source_sampler.__dict__, f)
        print(f"Source sampler successfully saved to {save_dir}/source_sampler_info.pkl")

    return source_sampler

def characterize_auxiliary_sampler_set(dim, num_components = 5, master_sampling_rng = 42, auxiliary_seeds_list = [1010, 1018, 1014, 1016, 1003]):
    """
    Characterize a set of auxiliary measure samplers (mixture of Gaussians) for synthetic experiments.
    """
    auxiliary_measure_sampler_set = []
    for auxiliary_seed in auxiliary_seeds_list:
        auxiliary_measure_sampler = MixtureOfGaussians(dim, master_sampling_rng=master_sampling_rng)
        auxiliary_measure_sampler.random_components(num_components = num_components, uniform_weights = True, manual_component_seed = auxiliary_seed)
        auxiliary_measure_sampler_set.append(auxiliary_measure_sampler)
    return auxiliary_measure_sampler_set

def characterize_entropic_sampler(dim, 
                                 num_measures, 
                                 auxiliary_measure_sampler_set = [], 
                                 source_sampler = None,
                                 truncated_radius = None,
                                 manual = False,
                                 bound_type = "eigen_bound",
                                 gamma = 0.3,
                                 theta = 10,
                                 surjective_mapping = None,
                                 A_matrices_dict = None):
    """
    Characterize the entropic sampler for synthetic experiments.
    """
    entropic_sampler = entropic_input_sampler(dim = dim, 
                                          num_measures = num_measures, 
                                          auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                          source_sampler = source_sampler, 
                                          n_k = 1000, 
                                          seed = 120, 
                                          gamma = gamma, 
                                          manual = manual,
                                          truncated_radius = truncated_radius,
                                          bound_type = bound_type,
                                          theta = theta,
                                          surjective_mapping = surjective_mapping,
                                          A_matrices_dict = A_matrices_dict)
    
    return entropic_sampler
    
def set_up_entropic_sampler(entropic_sampler, save_dir = None): # epsilon is the regularization parameter
    """
    Set up the entropic sampler by generating all necessary parameters and matrices. Once set up, the configuration is saved to load for future use.
    """
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
    # # construct a surjective mapping to map component maps to their respective OT maps for generating input measures.
    # entropic_sampler.construct_surjective_mapping()
    # print("surjective mapping all set.")
    # # generate A matrices
    # entropic_sampler.generate_A_matrices()
    # print("A matrices all set.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "entropic_sampler_info.pkl")

        state = dict(entropic_sampler.__dict__)   # shallow copy
        # state.pop("auxiliary_measure_sampler_set", None)
        # state.pop("source_sampler")

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


        print(f"Entropic sampler successfully saved to {save_dir}/entropic_sampler_info.pkl")

    return entropic_sampler

def load_sampler(load_dir, sampler, sampler_type = "entropic"):
    """
    Load a previously saved sampler configuration from the specified directory.
    In the argument, "sampler" must be pre-initialized to the correct class type (either entropic_input_sampler or MixtureOfGaussians).
    """
    # Load the sampler attributes
    if sampler_type == "entropic":
        with open(f"{load_dir}/entropic_sampler_info.pkl", "rb") as f:
            loaded_data_entropic_sampler = pickle.load(f)
            print(f"Entropic sampler successfully loaded")
            sampler.__dict__.update(loaded_data_entropic_sampler)

    elif sampler_type == "source":
        with open(f"{load_dir}/source_sampler_info.pkl", "rb") as f:
            loaded_data_source_sampler = pickle.load(f)
            print(f"Source sampler successfully loaded")
            sampler.__dict__.update(loaded_data_source_sampler)

    return sampler

# ------ for configuring entropic sampler ------

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
    It is general challenging to generate such a group of psd matrices, but we can ues the following strategy from Proposition~4.1 and Theorem~4.2 of Alvarez-Esteban et al. (2019):
    1. Generate $\Sigma_j$ for j = 1, \dots, J which are a collection of covariance matrices. (One can consider the problem of solving the W_2 barycenter of J Gaussian measures.)
    2. Apply the deterministic iterative scheme in Theorem~4.2 of Alvarez-Esteban et al. (2019) to approximate $\Sigma_0$, the covariance matrix of the Gaussian barycenter.
    3. From Proposition~4.1 we know that $H(\Sigma_0) = Id$ is a necessary and sufficient condition for $\Sigma_0$ to be a barycenter. The idea now is to use the terms without weights as the psd matrices of our interests, namely
    $\Sigma^{-\frac{1}{2}} (\Sigma^{-\frac{1}{2}} \Sigma_j \Sigma^{-\frac{1}{2}})^{\frac{1}{2}} \Sigma^{-\frac{1}{2}}$ for j = 1, \dots, J.
    '''
    if seed is None:
        return None
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

    # refer to H() below Eq. (17) of Alvarez-Esteban et al. (2019)
    A_matrices_dict = {}
    for i in range(num_matrices):
        sub_Sigma_square = sqrtm(Sigma) @ covariance_list[i] @ sqrtm(Sigma)
        A_matrix = np.linalg.solve(sqrtm(Sigma), np.eye(dim)) @ sqrtm(sub_Sigma_square) @ np.linalg.solve(sqrtm(Sigma), np.eye(dim))
        A_matrices_dict[i] = A_matrix

    return A_matrices_dict
    # beta_k = 1 for all k 
    

if __name__ == "__main__":
    dim = 2
    num_components = 5
    num_samples = 5000
    num_measures = 5
    truncated_radius = 150
    seed = 1009
    epsilon = 10

    save_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data_test/samplers_info"
    os.makedirs(save_dir, exist_ok=True)

    auxiliary_csv_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/auxiliary_samples/csv_files"

    source_sampler = characterize_source_sampler(dim, num_components, seed, save_dir)
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(auxiliary_csv_dir, auxiliary_seeds_list = [1010, 1018, 1014, 1016, 1003])


    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = False,
                                                     bound_type="eigen_bound",
                                                     theta = 10)
    entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir)

    
    