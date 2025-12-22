import numpy as np
from optparse import OptionParser
import stan
import time

MODEL = 'pois'  # or 'nb'

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
    print('X max:', np.max(X))
    data.close()
    return X[:, :-1], Y

def parse_args():
    parser = OptionParser()
    parser.add_option("--n_splits", type="int", dest="n_splits")
    parser.add_option("--split_idx", type="int", dest="split_idx")
    parser.add_option("--seed", type="int", dest="seed", default=0)
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    options = parse_args()
    print(options)

    n_splits = options.n_splits
    split_idx = options.split_idx
    seed = options.seed

    # Read Stan model code from file
    with open(f"{MODEL}.stan", 'r') as f:
        stan_code = f.read()

    # Load and prepare data
    dnm = 'bike_sharing_dataset/biketrips_large.npz'
    X, Y = load_data(dnm)
    n_max = X.shape[0] - (X.shape[0] % n_splits)

    if split_idx < 0:
        ll_mult = 1
        save_name = 'samples_all'
        data_idx = np.arange(n_max)
    else:
        ll_mult = n_splits
        save_name = f'samples_{split_idx}'
        np.random.seed(seed)
        idx_order = np.random.permutation(n_max)
        partition = np.array_split(idx_order, n_splits)
        data_idx = partition[split_idx]

    X_sub = X[data_idx]
    Y_sub = Y[data_idx].astype(int)

    sampler_data = {
        'x': X_sub,
        'y': Y_sub,
        'd': X_sub.shape[1],
        'n': X_sub.shape[0],
        'n_rep': ll_mult
    }

    # Compile the Stan model (build)
    posterior = stan.build(stan_code, data=sampler_data, random_seed=seed)

    # Sampling settings
    g_thin = 2
    g_iter = 220000
    warmup = 20000
    num_samples = (g_iter - warmup) // g_thin

    t0 = time.time()

    # Sample
    fit = posterior.sample(
        num_chains=1,
        num_samples=num_samples,
        num_warmup=warmup
    )

    # Extract posterior draws
    pars = ['theta'] if MODEL == 'pois' else ['beta']
    if MODEL == 'pois':
        theta_draws = fit['theta']  # shape: (chains, draws, d)
    else:
        theta_draws = fit['beta']

    # If only one chain, simplify
    theta_draws = theta_draws[0]  # Now shape (draws, d)
    np.save(f'./samples/{save_name}.npy', theta_draws)

    tf = time.time()
    print('Took', tf - t0)

