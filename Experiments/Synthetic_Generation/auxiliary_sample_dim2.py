from Experiments.Synthetic_Generation.samplers_dim2 import *
import pandas as pd

if __name__ == "__main__":
    dim = 2
    num_samples_in_preparation = int(1e7)
    csv_path = f"../../WB_data/Synthetic_Generation/dim{dim}_data/auxiliary_samples/csv_files"
    os.makedirs(csv_path, exist_ok=True)
    num_components = 5

    for auxiliary_seed in [1010, 1018, 1014, 1016, 1003]:
        auxiliary_measure_sampler = MixtureOfGaussians(dim)
        auxiliary_measure_sampler.random_components(num_components = num_components, uniform_weights = True, seed = auxiliary_seed)
        auxiliary_samples = np.asarray(auxiliary_measure_sampler.sample(num_samples_in_preparation))
        csv_filename = os.path.join(csv_path, f"auxiliary_measure_seed_{auxiliary_seed}.csv")
        pd.DataFrame(auxiliary_samples).to_csv(csv_filename, index=False, header=False)
    print("Auxiliary samples saved to CSV files.")