"""
Purpose: Generate samples from a mixture of Gaussian distributions, where each Gaussian component has a randomly generated mean and covariance matrix.
The covariance matrices are constructed to represent ellipsoids with random orientations and semi-axis lengths determined by inverse gamma distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, invgamma
from tqdm import tqdm, tqdm_notebook

# from tqdm import tqdm, tqdm_notebook

def construct_2d_covariance_ellipsoid(alpha = 3, beta = 4, rng_comp = None):
    """
    Constructs a covariance matrix for a 2D ellipsoid where:
    - Direction is determined by a random angle θ ~ U(0, 2π).
    - Semi-axis lengths are determined by two independent inverse gamma distributions.

    Args:
        alpha (float): Shape parameter for the inverse gamma distribution.
        beta (float): Scale parameter for the inverse gamma distribution.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: 2x2 covariance matrix representing the ellipsoid.

    Remark:
    The mean of the inverse gamma distribution is beta / (alpha - 1) for alpha > 1.
    The variance of the inverse gamma distribution is beta^2 / ((alpha - 1)^2 * (alpha - 2)) for alpha > 2.
    """
    # Sample angle θ from U(0, 2π)
    theta = rng_comp.uniform(0, 2 * np.pi)
    random_state_a = rng_comp.randint(0, 2**32 - 1, dtype=np.int64)
    random_state_b = rng_comp.randint(0, 2**32 - 1, dtype=np.int64)

    # Sample semi-axis lengths from the inverse gamma distribution
    a = invgamma.rvs(alpha, scale=beta, random_state = random_state_a) * 200 # First semi-axis length
    b = invgamma.rvs(alpha, scale=beta, random_state = random_state_b) * 200 # Second semi-axis length

    # Construct the rotation matrix R
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Construct the diagonal matrix D (semi-axis lengths)
    D = np.diag([a, b])

    # Construct the covariance matrix as R * D * R^T
    covariance_matrix = R @ D @ R.T

    return covariance_matrix

def construct_high_dim_covariance_ellipsoid(alpha = 3, beta = 4, dim = 10, rng_comp = None):
    """
    Constructs a covariance matrix for a high-dimensional ellipsoid where:
    - Direction is determined by a random orthogonal matrix.
    - Semi-axis lengths are determined by independent inverse gamma distributions.

    Args:
        alpha (float): Shape parameter for the inverse gamma distribution.
        beta (float): Scale parameter for the inverse gamma distribution.
        dim (int): Dimensionality of the covariance matrix.
        rng_comp (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Covariance matrix of shape (dim, dim).
    """
    if rng_comp is None:
        rng_comp = np.random.default_rng()

    # Sample a random orthogonal matrix R
    random_matrix = rng_comp.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(random_matrix)  # QR decomposition ensures Q is orthogonal

    # Sample semi-axis lengths from the inverse gamma distribution
    semi_axes = invgamma.rvs(alpha, scale=beta, size=dim, random_state=rng_comp) * 200
    D = np.diag(semi_axes)

    # Construct the covariance matrix: R * D * R^T
    covariance_matrix = Q @ D @ Q.T

    return covariance_matrix

class MixtureOfGaussians:
### For generating samples from a mixture of Gaussian distributions (underlying barycenter measure) ###

    def __init__(self, dim, weights=None):
        self.truncation = False
        # Default weights if not provided (equally distributed)
        if weights is None:
            self.weights = []
        else:
            self.weights = weights
            self.weights /= np.sum(self.weights)
        
        # Initialize list to record parameters for each Gaussian component
        self.gaussians = []
        self.dim = dim

    def add_gaussian(self, mean, cov):
        self.gaussians.append((mean, cov))
        
    def set_weights(self, weights):
        self.weights = weights
        self.weights /= np.sum(self.weights)

    def set_truncation(self, radius):
        self.truncation = True
        self.radius = radius

    def random_components(self, num_components, uniform_weights = True, seed = 42):
        
        self.num_components = num_components
        self.seed = seed
        dim = self.dim
        rng_comp = np.random.RandomState(seed)
        for _ in range(num_components):
            mean = (rng_comp.randn(dim)) * 30
            if dim == 2:
                cov = construct_2d_covariance_ellipsoid(3, 4, rng_comp)
            else:
                cov = construct_high_dim_covariance_ellipsoid(3, 4, dim, rng_comp)
            self.add_gaussian(mean, cov)
        if uniform_weights:
            weights = np.ones(num_components)
        else:
            weights = rng_comp.rand(num_components)
        self.set_weights(weights)


    def pdf(self, x):
        """
        Compute the probability density function (PDF) of the mixture at point(s) x.
        Args:
            x: A single point (1D array) or multiple points (2D array of shape [N, dim])
        Returns:
            PDF value(s) at x.
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]  # Convert to 2D for consistency

        # sum up the squares of all columns of x
        x_sum_squared_columns = np.sum(x**2, axis = 1)
        x_norms = np.sqrt(x_sum_squared_columns)

        if self.truncation:
            # convert x_norms into binary column vector by comparing row norms with radius
            x_norms = x_norms[:, np.newaxis]
            x_norms = np.where(x_norms > self.radius, 0, 1)
            x_norms = x_norms.flatten()

        pdf_values = np.zeros(x.shape[0])  # Initialize result array
        for weight, (mean, cov) in zip(self.weights, self.gaussians):
            pdf_values += weight * multivariate_normal.pdf(x, mean=mean, cov=cov)

        pdf_values *= x_norms  # apply truncation
        # rescale pdf values to sum to 1
        pdf_values /= np.sum(pdf_values)

        return pdf_values

    def sample(self, n, seed = None, multiplication_factor = 1):
        dim = self.dim
        count = 0
        samples = np.zeros((n, dim))
        rng_sample = np.random.RandomState(seed)
        np.random.seed(seed)

        with tqdm(total=n, desc="MOG sampling") as pbar:
            while count < n:
                choice = rng_sample.choice(len(self.gaussians), p=self.weights)
                mean, cov = self.gaussians[choice]
                sample = rng_sample.multivariate_normal(mean, cov)
                if not self.truncation or np.linalg.norm(sample) <= self.radius:
                    samples[count] = sample
                    count += 1
                    pbar.update(1)
        samples *= multiplication_factor
        return samples
    
    def visualize(self, samples, name = None):
        if self.dim != 2:
            print("Visualization only supported for 2D distributions")
        else:
            plt.figure()
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
            if name:
                plt.savefig(name)
            else:
                plt.show()


# Example usage: 
# mog = MixtureOfGaussians(2)
# mog.random_components(5)
# mog.set_truncation(50)
# print(mog.gaussians)
# mog.visualize(1000)
