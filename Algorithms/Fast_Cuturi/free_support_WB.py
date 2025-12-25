import numpy as np
import ot

def w2_barycenter_free_support_from_samples(
    samples_list,
    k=5000,
    weights=None,
    init="kmeans",
    numItermax=1000,
    stopThr=1e-7,
    verbose=False,
    numThreads=1,
    seed=0,
    log=False,
):
    """
    Compute a 2-Wasserstein barycenter of *empirical measures given by samples*
    using POT's free-support barycenter (Cuturi & Doucet 2014 fixed-point update).

    You have:
      - N measures
      - each measure i: samples_list[i] is (m_i, d), uniform weights

    This function:
      1) optionally downsamples each measure (recommended if m is huge)
      2) uses uniform weights a_i on each measure
      3) chooses barycenter weights b uniform on k atoms
      4) optimizes barycenter support locations X (k, d)

    Parameters
    ----------
    samples_list : list[np.ndarray]
        List of N arrays, each of shape (m_i, d).
    k : int
        Number of support points (atoms) for the barycenter (tradeoff: accuracy vs runtime).
    weights : np.ndarray | None
        Barycenter weights over measures, shape (N,), sums to 1. If None => uniform.
    init : {"kmeans", "random", "subset"}
        How to initialize barycenter support locations X_init.
    numItermax, stopThr, verbose, numThreads : see POT docs.
    seed : int
        Random seed used for initialization.
    log : bool
        If True, returns (X, log_dict). Else returns X.

    Returns
    -------
    X : (k, d) np.ndarray
        Barycenter support locations.
    log_dict : dict (optional)
        Contains "displacement_square_norms".
    """

    rng = np.random.default_rng(seed)

    # --------- validate ----------
    if not isinstance(samples_list, (list, tuple)) or len(samples_list) == 0:
        raise ValueError("samples_list must be a non-empty list of arrays.")
    N = len(samples_list)
    d = np.asarray(samples_list[0]).shape[1]
    for i, Xi in enumerate(samples_list):
        Xi = np.asarray(Xi)
        if Xi.ndim != 2 or Xi.shape[1] != d:
            raise ValueError(f"samples_list[{i}] must be (m_i, {d}); got {Xi.shape}")

    # --------- weights over measures ----------
    if weights is None:
        weights = np.ones(N, dtype=float) / N
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (N,):
            raise ValueError(f"weights must have shape ({N},), got {weights.shape}")
        s = weights.sum()
        if s <= 0:
            raise ValueError("weights must sum to a positive number.")
        weights = weights / s

    # --------- each empirical measure weights (uniform) ----------
    measures_locations = [np.asarray(Xi, dtype=float) for Xi in samples_list]
    measures_weights = [np.ones(Xi.shape[0], dtype=float) / Xi.shape[0] for Xi in measures_locations]

    # --------- barycenter weights b (uniform over k atoms) ----------
    b = np.ones(k, dtype=float) / k

    # --------- initialize barycenter support X_init ----------
    X_all = np.vstack(measures_locations)  # (sum m_i, d)

    if init == "random":
        # Gaussian around global mean/cov
        mu = X_all.mean(axis=0)
        cov = np.cov(X_all.T) if d > 1 else np.array([[np.var(X_all)]])
        X_init = rng.multivariate_normal(mu, cov + 1e-9 * np.eye(d), size=k)

    elif init == "subset":
        # pick k points uniformly from the pooled samples
        idx = rng.choice(X_all.shape[0], size=k, replace=False if X_all.shape[0] >= k else True)
        X_init = X_all[idx].copy()

    elif init == "kmeans":
        # lightweight k-means (no sklearn dependency): few Lloyd steps on a subsample
        # (you can replace with sklearn KMeans if you have it)
        subsz = min(20000, X_all.shape[0])
        sub = X_all[rng.choice(X_all.shape[0], size=subsz, replace=False)]
        # init centroids from subset
        cent = sub[rng.choice(sub.shape[0], size=k, replace=False if sub.shape[0] >= k else True)].copy()
        for _ in range(10):  # 10 Lloyd iterations is usually enough for init
            # assign
            # distances squared: (subsz,k)
            dist2 = ((sub[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
            lab = dist2.argmin(axis=1)
            # update
            for j in range(k):
                mask = (lab == j)
                if np.any(mask):
                    cent[j] = sub[mask].mean(axis=0)
        X_init = cent

    else:
        raise ValueError('init must be one of {"kmeans","random","subset"}')

    # --------- run POT free-support barycenter ----------
    out = ot.lp.free_support_barycenter(
        measures_locations=measures_locations,
        measures_weights=measures_weights,
        X_init=X_init,
        b=b,
        weights=weights,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=log,
        numThreads=numThreads,
    )

    return out
