import os
import numpy as np
import json
from pathlib import Path

from ucimlrepo import fetch_ucirepo 

if __name__ == "__main__":
    # fetch dataset 
    bike_sharing = fetch_ucirepo(id=275)
    
    # data (as pandas dataframes) 
    X = bike_sharing.data.features
    y = bike_sharing.data.targets
    
    # metadata 
    print(bike_sharing.metadata)
    
    # variable information 
    print(bike_sharing.variables)

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    stan_dir = cfg_dict["stan_dir"]
    samples_dir = cfg_dict["samples_dir"]
    os.makedirs(stan_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    selected_columns = cfg_dict["params_posterior_aggregation_dim9"]["selected_columns"]

    # select the columns of interest
    X = X[selected_columns].to_numpy()
    y = np.squeeze(y.astype(int).to_numpy())

    N, d = X.shape

    # standardize covariates
    m = X.mean(axis = 0)
    V = np.cov(X, rowvar = False) + 1e-12 * np.eye(X.shape[1])
    X = np.linalg.solve(np.linalg.cholesky(V), (X - m).T).T

    # Samples in the full dataset
    data_full = {'x': X.tolist(), 'y': y.tolist(), 'd': d, 'n': N, 'n_rep': 1}
    with open(os.path.join(stan_dir, "data_full.json"), "w") as json_file:
        json.dump(data_full, json_file)

    # Train on subsets
    n_splits = 5
    rs = np.random.RandomState(seed = 0)
    idx = np.arange(N)
    rs.shuffle(idx)
    splits = np.array_split(idx, n_splits)
    for i, part in enumerate(splits):
        Xi = X[part, :]
        yi = y[part]
        data_i = {'x': Xi.tolist(), 'y': yi.tolist(), 'd': d, 'n': len(part), 'n_rep': N / len(part)}
        with open(os.path.join(stan_dir, f"data_split_{i}.json"), "w") as json_file:
            json.dump(data_i, json_file)