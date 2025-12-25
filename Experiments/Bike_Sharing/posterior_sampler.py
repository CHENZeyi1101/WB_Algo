from .posterior_sampling import *
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


class posterior_sampler:
    def __init__(self, model_path, num_measures : int = 1, multiplication_factor = 1):
        self.num_measures = num_measures
        self.model_path = model_path
        self.multiplication_factor = multiplication_factor

    def sample(self, num_samples, save_samples = False, save_dir = None):
        if self.num_measures == 1:
            # in this case, model_path corresponds to the pickle file path
            return self.multiplication_factor * sample_from_meta(self.model_path, num_chains=1, num_samples=num_samples, save_samples=save_samples, save_dir=save_dir).T
        else: 
            # in this case, model_path corresponds to the directory containing all meta files
            batch_sample_collection = {k: [] for k in range(self.num_measures)}
            for measure_idx in range(self.num_measures):
                meta_filename = os.path.join(self.model_path, f"model_split_{measure_idx}.meta.pkl")
                measure_samples = self.multiplication_factor * sample_from_meta(meta_filename, num_chains=1, num_samples=num_samples, save_samples=save_samples, save_dir=save_dir).T
                batch_sample_collection[measure_idx] = measure_samples
            return batch_sample_collection

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

class csv_posterior_sampler:
    def __init__(self, csv_dir, num_measures: int = 1, multiplication_factor=1, type: str = "full"):
        if type not in ("full", "split"):
            raise ValueError(f"type must be 'full' or 'split', got '{type}'")
        self.num_measures = num_measures
        self.csv_dir = csv_dir
        self.multiplication_factor = multiplication_factor
        self.type = type

    def sample(self, num_samples, seed=None):
        if self.type == "full" and self.num_measures != 1:
            raise ValueError("For 'full' type, num_measures must be 1.")

        if self.type == "full":
            csv_filename = os.path.join(self.csv_dir, "posterior_full.csv")
            samples = reservoir_sample_csv(
                csv_filename,
                num_samples=num_samples,
                skiprows=52,
                usecols=range(7, 16),
                chunksize = 5000,
                seed=seed,
            )
            return self.multiplication_factor * samples

        if self.type == "split":
            batch_sample_collection = {}

            for measure_idx in range(self.num_measures):
                csv_filename = os.path.join(
                    self.csv_dir, f"posterior_split_{measure_idx}.csv"
                )
                samples = reservoir_sample_csv(
                    csv_filename,
                    num_samples=num_samples,
                    skiprows=52,
                    usecols=range(7, 16),
                    chunksize = 5000,
                    seed=None if seed is None else seed + measure_idx,
                )
                batch_sample_collection[measure_idx] = [row for row in samples]

            return batch_sample_collection

if __name__ == "__main__":
    posterior_csv_dir = f"../WB_data/Bike_Sharing"
    total_posterior_sampler = csv_posterior_sampler(csv_dir=posterior_csv_dir, num_measures=1, multiplication_factor=1, type="full")
    split_posterior_sampler = csv_posterior_sampler(csv_dir=posterior_csv_dir, num_measures=5, multiplication_factor=1, type="split")
    batch_sample_collection = split_posterior_sampler.sample(num_samples=10)
    print(np.array(batch_sample_collection[0]))
    print(np.array(batch_sample_collection[0][0]))
    