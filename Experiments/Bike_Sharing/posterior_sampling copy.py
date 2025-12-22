import os
import time
import pickle
import numpy as np
import stan  # PyStan 3


def load_data(dnm):
    data = np.load(dnm)
    X = data['X']
    Y = data['y']
    # standardize covariates; last col is intercept, so no standardization there
    m = X[:, :-1].mean(axis=0)
    V = np.cov(X[:, :-1], rowvar=False) + 1e-12 * np.eye(X.shape[1] - 1)
    X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V),
                                (X[:, :-1] - m).T).T
    assert np.isfinite(X).all()
    print("X max:", np.max(X))
    data.close()
    return X[:, :-1], Y

def meta_model_save(meta_name, data_dict, stan_code, seed=0):
    """
    Save metadata of the model for future reuse.
    """
    meta = {
        'model_name': meta_name,
        'stan_code': stan_code,
        'data_dict': data_dict,
        'seed': seed
    }
    meta_path = os.path.join(OUTPUT_MODEL_DIR, f"{meta_name}.meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved metadata to:", meta_path)

def load_model_meta(meta_filename):
    with open(meta_filename, "rb") as f:
        meta = pickle.load(f)
    return meta

def sample_from_meta(meta_filename, num_chains=1, num_samples=1000, save_samples=False, save_dir= None):
    meta = load_model_meta(meta_filename)
    posterior = stan.build(meta['stan_code'], data=meta['data_dict'], random_seed=meta['seed'])
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
    draws = fit['theta']
    if save_samples:
        samples_name = meta['model_name'] + "_samples.npy"
        samples_path = os.path.join(save_dir, samples_name)
        np.save(samples_path, draws)
        print("Saved samples to:", samples_path) # file type: npy
    return draws  # (chains, draws, d)
    
if __name__ == "__main__":
    DATA_DIR = os.path.dirname(__file__)
    print("Current working directory:", DATA_DIR)
    MODEL = 'pois'    # or 'nb'
    STAN_FILE = os.path.join(DATA_DIR, f"{MODEL}.stan")
    OUTPUT_MODEL_DIR = os.path.join(DATA_DIR, "models_meta")
    OUTPUT_SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
    
    # ensure output directories
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)

    X, Y = load_data(os.path.join(DATA_DIR, 'bike_sharing_data.npz'))
    N, d = X.shape
    print(f"Loaded data: N={N}, d={d}")

    # read Stan code
    with open(STAN_FILE, "r") as f:
        stan_code = f.read()

    # Train on full dataset
    data_full = {'x': X, 'y': Y.astype(int), 'd': d, 'n': N, 'n_rep': 1}
    meta_model_save("model_total", data_full, stan_code, seed=42)

    # Train on subsets
    n_splits = 5
    np.random.seed(0)
    idx = np.arange(N)
    np.random.shuffle(idx)
    splits = np.array_split(idx, n_splits)
    for i, part in enumerate(splits):
        Xi = X[part]
        Yi = Y[part].astype(int)
        data_i = {'x': Xi, 'y': Yi, 'd': d, 'n': len(part), 'n_rep': n_splits}
        meta_model_save(f"model_split_{i}", data_i, stan_code, seed=i)

    # sample from full model
    print("Sampling from full model...")
    full_samples = sample_from_meta(os.path.join(OUTPUT_MODEL_DIR, "model_total.meta.pkl"), num_chains=1, num_samples=10000, save_samples=True, save_dir=OUTPUT_SAMPLES_DIR) 
    print("Full model samples shape:", full_samples.shape)
    print(full_samples[0:5])  # print first 5 samples

    # sample from split models
    for i in range(n_splits):
        print(f"Sampling from split model {i}...")
        split_samples = sample_from_meta(os.path.join(OUTPUT_MODEL_DIR, f"model_split_{i}.meta.pkl"), num_chains=1, num_samples=10000, save_samples=True, save_dir=OUTPUT_SAMPLES_DIR)
        print(f"Split model {i} samples shape:", split_samples.shape)
        print(split_samples[0:5])  # print first 5 samples

