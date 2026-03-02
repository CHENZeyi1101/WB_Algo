import json
from pathlib import Path
import numpy as np
from Algorithms.Stochastic_FP.entropic_iterative_scheme import entropic_iterative_scheme
import os


def sample_uniform_disk(center, eps, num_samples, rng=None):
    """
    Sample n points uniformly from the closed 2D ball (disk) B(center, eps).
    center: array-like shape (2,)
    """
    if rng is None:
        rng = np.random.default_rng()

    center = np.asarray(center, dtype=float)

    # Polar method: r = eps * sqrt(U), theta = 2pi V
    u = rng.random(num_samples) # for radius
    v = rng.random(num_samples) # for angle
    r = eps * np.sqrt(u) # monte carlo sampling for radius to ensure uniform distribution in the disk
    theta = 2 * np.pi * v

    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts + center

def sample_uniform_union_of_two_disks(c1, c2, eps, num_samples, rng=None):
    """
    Uniform measure on B(c1, eps) ∪ B(c2, eps), with mass 1/2 on each disk.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Choose which disk each point comes from (0 -> c1, 1 -> c2)
    which = rng.integers(0, 2, size=num_samples)
    n1 = np.sum(which == 0)
    n2 = num_samples - n1

    pts1 = sample_uniform_disk(c1, eps, n1, rng)
    pts2 = sample_uniform_disk(c2, eps, n2, rng)

    pts = np.empty((num_samples, 2), dtype=float)
    pts[which == 0] = pts1
    pts[which == 1] = pts2
    return pts

class KMInstance_input_sampler:
    def __init__(self, seed, mu1_centers, mu2_centers, eps):
        self.rng = np.random.default_rng(seed=seed)
        self.mu1_centers = mu1_centers
        self.mu2_centers = mu2_centers
        self.eps = eps
        self.num_measures = 2

    def sample(self, num_samples):
        input_sample_collection = {k: np.empty((num_samples, 2), dtype=float) for k in range(2)}
        input_sample_collection[0] = sample_uniform_union_of_two_disks(self.mu1_centers[0], self.mu1_centers[1], self.eps, num_samples, self.rng)
        input_sample_collection[1] = sample_uniform_union_of_two_disks(self.mu2_centers[0], self.mu2_centers[1], self.eps, num_samples, self.rng)
        return input_sample_collection
    
class KMInstance_KM_sampler:
    def __init__(self, seed, karcher_centers, eps):
        self.rng = np.random.default_rng(seed=seed)
        self.karcher_centers = karcher_centers
        self.eps = eps

    def sample(self, num_samples):
        karcher_samples = sample_uniform_union_of_two_disks(self.karcher_centers[0], self.karcher_centers[1], self.eps, num_samples, self.rng)
        return karcher_samples
    

def visualize_KM_samples(bary_samples, input_samples, mu1_centers, mu2_centers, karcher_A_centers, karcher_B_centers, eps, plot_dir=None, plot_name=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.scatter(input_samples[0][:, 0], input_samples[0][:, 1], s = 10, color='blue', alpha=0.5, label='Input Measure 1')
        plt.scatter(input_samples[1][:, 0], input_samples[1][:, 1], s = 10, color='orange', alpha=0.5, label='Input Measure 2')
        plt.scatter(bary_samples[:, 0], bary_samples[:, 1], s = 10, color='green', alpha=0.5, label='Karcher Mean Samples')
        # plot the centers of the disks
        for center in mu1_centers:
            circle = plt.Circle(center, eps, color='blue', fill=False, linestyle='--')
            plt.gca().add_artist(circle)
        for center in mu2_centers:
            circle = plt.Circle(center, eps, color='orange', fill=False, linestyle='--')
            plt.gca().add_artist(circle)
        for center in karcher_A_centers:
            circle = plt.Circle(center, eps, color='green', fill=False, linestyle='--')
            plt.gca().add_artist(circle)
        for center in karcher_B_centers:
            circle = plt.Circle(center, eps, color='red', fill=False, linestyle='--')
            plt.gca().add_artist(circle)
        plt.legend()
        plt.title('Visualization of Karcher Mean Samples and Input Measures')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.grid()
        plt.axis('equal')
        if plot_dir is not None and plot_name is not None:
            plt.savefig(f"{plot_dir}/{plot_name}.png")
        else:
            plt.show()


if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)
        
    data_dir = cfg_dict['data_dir']
    instance_dir = f"{data_dir}/Karcher_Mean"
    os.makedirs(instance_dir, exist_ok=True)

    instance_identifier = cfg_dict['params_KM_experiment']['instance_identifier']


    outputs_dir = f"{instance_dir}/KM_Example_outputs/Instance_{instance_identifier}"
    os.makedirs(outputs_dir, exist_ok=True)

    num_samples = cfg_dict['params_KM_experiment']['num_samples']
    eval_num_samples = num_samples

    # Parameters
    a = cfg_dict['params_KM_experiment']['a']
    M = cfg_dict['params_KM_experiment']['M']
    eps = cfg_dict['params_KM_experiment']['eps']
    reg = cfg_dict['params_KM_experiment']['reg'] # entropic regularizer

    with open(f"{outputs_dir}/instance_info.json", 'w') as json_file:
        json.dump(cfg_dict, json_file)  

    # Measures in Example 2.2 of Backhoff et al. 
    # mu1: uniform on B((-a, M), eps) \cup B((a, -M), eps)
    # mu2: uniform on B((-a, -M), eps) \cup B((a, M), eps)
    mu1_centers = [(-a,  M), ( a, -M)]
    mu2_centers = [(-a, -M), ( a,  M)]

    # Two Karcher-mean candidates mentioned:
    # candidate A: uniform on B((-a, 0), eps) ∪ B((a, 0), eps)
    # candidate B: uniform on B((0, M), eps) ∪ B((0, -M), eps)
    karcher_A_centers = [(-a, 0), (a, 0)] # this is the true barycenter
    karcher_B_centers = [(0, M), (0, -M)] # this is the other Karcher mean; used for initialization in the entropic iterative scheme

    # Create samplers
    input_sampler = KMInstance_input_sampler(seed=2000, mu1_centers=mu1_centers, mu2_centers=mu2_centers, eps=eps)
    karcher_sampler_A = KMInstance_KM_sampler(seed=3000, karcher_centers=karcher_A_centers, eps=eps)
    karcher_sampler_B = KMInstance_KM_sampler(seed=4000, karcher_centers=karcher_B_centers, eps=eps)

    bary_samples = karcher_sampler_A.sample(eval_num_samples)
    input_samples = input_sampler.sample(eval_num_samples)

    visualize_KM_samples(bary_samples, input_samples, mu1_centers, mu2_centers, karcher_A_centers, karcher_B_centers, eps, plot_dir=outputs_dir, plot_name="Samples from input measures and the barycenter")


    # parameters for the entropic iterative scheme
    dim = cfg_dict['params_KM_experiment']['dim']
    num_iters = cfg_dict['params_KM_experiment']['num_iters']
    rand_state = np.random.RandomState(seed = 5000) # redundant here; for input constraints of the entropic iterative scheme
    init_method = {"type": "moment", "sample_size": 10000} # redundant here
    truncate_radius = 100
    sample_size_scheme = [num_samples] * num_iters
    reg_param_scheme = [reg] * num_iters
    sinkhorn_impl = "ott"
    warm_start = None
    
    eval_MC_size = 20
    eval_input_samples_collection = {}
    eval_bary_samples_collection = {}
    for k in range(eval_MC_size):
        bary_samples = karcher_sampler_A.sample(eval_num_samples)
        eval_bary_samples_collection[k] = bary_samples
        input_samples = input_sampler.sample(eval_num_samples)
        eval_input_samples_collection[k] = {k : v for k, v in input_samples.items()}

    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(
        dim = dim,
        num_iters = num_iters,
        input_sampler = input_sampler,
        rand_state = rand_state,
        init_method = init_method,
        truncate_radius = truncate_radius,
        sinkhorn_impl = sinkhorn_impl,
        sample_size_scheme = sample_size_scheme,
        reg_param_scheme = reg_param_scheme,
        warm_start = warm_start,
        bary_samples_collection = eval_bary_samples_collection,
        input_samples_for_evaluation = eval_input_samples_collection,
        eval_num_samples = eval_num_samples,
        eval_MC_size = eval_MC_size,
        num_parallel = None,
        manual_initial_sampler=karcher_sampler_B
    )

    entropic_iterative_computer.converge(logger = {'sample_logger': None, 'map_logger': None}, data_dir = outputs_dir)


    



    