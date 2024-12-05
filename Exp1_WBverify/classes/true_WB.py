import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm, tqdm_notebook

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
        self.num_components = num_components
        self.seed = seed
        dim = self.dim
        rng_comp = np.random.RandomState(seed)
        for _ in range(num_components):
            mean = (rng_comp.rand(dim) - 0.5) * 100
            A = rng_comp.rand(dim, dim) - 0.5
            cov = (np.dot(A, A.T) + np.eye(dim)) * 100
            self.add_gaussian(mean, cov)
        weights = rng_comp.rand(num_components)
        self.set_weights(weights)

    def sample(self, n, seed = None, multiplication_factor = 1):
        dim = self.dim
        count = 0
        samples = np.zeros((n, dim))
        rng_sample = np.random.RandomState(seed)
        np.random.seed(seed)

        # with tqdm(total=n, desc="source sampling") as pbar:
        while count < n:
            choice = rng_sample.choice(len(self.gaussians), p=self.weights)
            mean, cov = self.gaussians[choice]
            sample = rng_sample.multivariate_normal(mean, cov)
            if not self.truncation or np.linalg.norm(sample) <= self.radius:
                samples[count] = sample
                count += 1
                # if (count + 1) % 1024 == 0:
                #     pbar.update(1024)
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
