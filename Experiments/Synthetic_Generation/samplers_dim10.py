import sys
import os
import numpy as np
from tqdm import tqdm
import pickle

from .true_WB import *
from .input_generate_entropic import entropic_input_sampler, csv_input_sampler 
from .sample_plot import *

''' 
This module characterizes and sets up samplers for synthetic experiments in 2D.
'''

def characterize_source_sampler(dim, num_components = 5, seed = None, save_dir = None):
    """
    Characterize the source sampler (mixture of Gaussians) and auxiliary measure samplers for synthetic experiments.
    """
    source_sampler = MixtureOfGaussians(dim)
    source_sampler.random_components(num_components = num_components, uniform_weights = True, seed = seed) # seed from the measure selection
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/source_sampler_info.pkl", "wb") as f:
            pickle.dump(source_sampler.__dict__, f)
        print(f"Source sampler successfully saved to {save_dir}/source_sampler_info.pkl")

    return source_sampler

def characterize_auxiliary_sampler_set(dim, num_components = 5):
    """
    Characterize a set of auxiliary measure samplers (mixture of Gaussians) for synthetic experiments.
    """
    auxiliary_measure_sampler_set = []
    for auxiliary_seed in range(10):
        auxiliary_measure_sampler = MixtureOfGaussians(dim)
        auxiliary_measure_sampler.random_components(num_components = num_components, uniform_weights = True, seed = auxiliary_seed)
        auxiliary_measure_sampler_set.append(auxiliary_measure_sampler)

    return auxiliary_measure_sampler_set

def characterize_entropic_sampler(dim, 
                                 num_measures, 
                                 auxiliary_measure_sampler_set, 
                                 source_sampler,
                                 truncated_radius,
                                 manual = False):
    """
    Characterize the entropic sampler for synthetic experiments.
    """
    entropic_sampler = entropic_input_sampler(dim = dim, 
                                          num_measures = num_measures, 
                                          auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                          source_sampler = source_sampler, 
                                          n_k = 1000, 
                                          seed = 120, 
                                          gamma = 0.3, 
                                          manual = manual,
                                          truncated_radius = truncated_radius,
                                          bound_type = "eigen_bound")
    
    return entropic_sampler
    
def set_up_entropic_sampler(entropic_sampler, save_dir = None):
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
    # construct a surjective mapping to map component maps to their respective OT maps for generating input measures.
    entropic_sampler.construct_surjective_mapping()
    print("surjective mapping all set.")
    # generate A matrices
    entropic_sampler.generate_A_matrices()
    print("A matrices all set.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/entropic_sampler_info.pkl", "wb") as f:
            pickle.dump(entropic_sampler.__dict__, f)
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
    

if __name__ == "__main__":
    dim = 10
    num_components = 5
    num_samples = 5000
    num_measures = 10
    truncated_radius = 5000
    seed = 1009

    save_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"
    os.makedirs(save_dir, exist_ok=True)

    source_sampler = characterize_source_sampler(dim, num_components, seed, save_dir)
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim, num_components)
    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = True)
    entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir)
    
    