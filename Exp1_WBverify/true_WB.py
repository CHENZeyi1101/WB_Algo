import numpy as np
import matplotlib.pyplot as plt

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

    def random_components(self, num_components, seed = 42):
        dim = self.dim
        np.random.seed(seed)
        for _ in range(num_components):
            mean = (np.random.rand(dim) - 0.5) * 10
            A = np.random.rand(dim, dim) - 0.5
            cov = (np.dot(A, A.T) + np.eye(dim)) * 100
            self.add_gaussian(mean, cov)
        weights = np.random.rand(num_components)
        self.set_weights(weights)

    def sample(self, n, seed = None):
        dim = self.dim
        count = 0
        samples = np.zeros((n, dim))
        np.random.seed(seed)
        while count < n:
            choice = np.random.choice(len(self.gaussians), p=self.weights)
            mean, cov = self.gaussians[choice]
            sample = np.random.multivariate_normal(mean, cov)
            if not self.truncation or np.linalg.norm(sample) <= self.radius:
                samples[count] = sample
                count += 1
        return samples
    
    def visualize(self, samples):
        if self.dim != 2:
            print("Visualization only supported for 2D distributions")
        else:
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
            plt.show()


# Example usage: 
# mog = MixtureOfGaussians(2)
# mog.random_components(5)
# mog.set_truncation(50)
# print(mog.gaussians)
# mog.visualize(1000)
