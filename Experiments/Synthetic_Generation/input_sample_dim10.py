from .samplers_dim10 import *
from tqdm import tqdm
import pandas as pd
import pickle

if __name__ == "__main__":
    dim = 10
    num_components = 5
    num_measures = 10
    truncated_radius = 5000
    seed = 1009

    num_samples_in_preparation = int(1e6)

    load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"

    # Load the samplers
    source_sampler = MixtureOfGaussians(dim)
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim, num_components)
    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                        num_measures = num_measures, 
                                                        auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                        source_sampler = source_sampler,
                                                        truncated_radius = truncated_radius,
                                                        manual = False,
                                                        bound_type="norm_bound")

    source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")
    entropic_sampler = load_sampler(load_dir, entropic_sampler, sampler_type="entropic")

    # Generate input samples
    csv_path = f"../WB_data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files"
    os.makedirs(csv_path, exist_ok=True)
    
    input_measure_samples = entropic_sampler.sample(num_samples_in_preparation)
    for measure_index in range(num_measures):
        measure_samples = np.array(input_measure_samples[measure_index])
        # Save measure_samples to a CSV file
        csv_filename = os.path.join(csv_path, f"input_measure_samples_{measure_index}.csv")
        pd.DataFrame(measure_samples).to_csv(csv_filename, index=False, header=False)
    print("Input samples saved to CSV files.")