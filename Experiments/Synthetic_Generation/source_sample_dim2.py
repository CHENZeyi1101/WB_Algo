from Experiments.Synthetic_Generation.true_WB import *
from Experiments.Synthetic_Generation.samplers_dim2 import *

from tqdm import tqdm
import pandas as pd
import pickle

if __name__ == "__main__":
    dim = 2
    num_measures = 5
    num_samples = int(1e7)
    truncated_radius = 150

    source_sampler = MixtureOfGaussians(dim)
    source_sampler.random_components(num_components=5, uniform_weights = True, seed = 1009)
    source_sampler.set_truncation(truncated_radius)

    source_samples = np.asarray(source_sampler.sample(num_samples))
    csv_path = f"../../WB_data/Synthetic_Generation/dim{dim}_data/source_samples/csv_files"
    os.makedirs(csv_path, exist_ok=True)
    csv_filename = os.path.join(csv_path, f"source_measure_samples.csv")
    pd.DataFrame(source_samples).to_csv(csv_filename, index=False, header=False)
    print("Source samples saved to CSV file.")