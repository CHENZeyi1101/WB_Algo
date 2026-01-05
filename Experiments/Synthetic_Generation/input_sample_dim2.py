from Experiments.Synthetic_Generation.samplers import *
from tqdm import tqdm
import pandas as pd
import pickle
from Experiments.CSV_read import *
import math
from scipy.linalg import sqrtm, norm

if __name__ == "__main__":
    dim = 2
    num_components = 5
    num_measures = 5
    truncated_radius = 150
    seed = 1009
    instance_theta = 2000

    num_samples_in_preparation = int(1e7)

    auxiliary_csv_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/auxiliary_samples/csv_files"
    source_csv_file = f"../../WB_data/Synthetic_Generation/dim{dim}_data/source_samples/csv_files/source_measure_samples.csv"

    auxiliary_seeds_list = [1010, 1018, 1014, 1016, 1003]

    source_sampler = csv_source_sampler_SyntheticGeneration(source_csv_file, 
                                                   multiplication_factor=1,
                                                   usecols=None,
                                                   skiprows=0)
    source_sampler.set_streamer()

    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(csv_dir = auxiliary_csv_dir, auxiliary_seeds_list = auxiliary_seeds_list)
    tilde_K = len(auxiliary_measure_sampler_set)
    surjective_mapping = construct_surjective_mapping(tilde_K = tilde_K, num_measures = num_measures, seed = 120)
    A_matrices_dict = generate_A_matrices(dim = dim, num_measures = num_measures, seed = 2000)

    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = False,
                                                     bound_type="eigen_bound",
                                                     theta = instance_theta,
                                                     surjective_mapping = surjective_mapping,
                                                     A_matrices_dict = A_matrices_dict)
    entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir = f"./Experiments/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}/samplers_info")
    print("Entropic sampler configured.")

    # Generate input samples
    csv_path = f"../../WB_data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files_InstanceTheta{instance_theta}"
    os.makedirs(csv_path, exist_ok=True)
    
    input_measure_samples = entropic_sampler.sample(num_samples_in_preparation)

    for measure_index in range(num_measures):
        measure_samples = np.array(input_measure_samples[measure_index])
        # Save measure_samples to a CSV file
        csv_filename = os.path.join(csv_path, f"input_measure_samples_{measure_index}.csv")
        pd.DataFrame(measure_samples).to_csv(csv_filename, index=False, header=False)
    print("Input samples saved to CSV files.")

    
    