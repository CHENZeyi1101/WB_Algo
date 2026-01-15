import numpy as np
from Experiments.Synthetic_Generation.MOG import *
from Experiments.Synthetic_Generation.samplers import *
import json
from pathlib import Path

if __name__ == "__main__":
    dim = 2
    num_measures = 5
    num_samples = 10
    num_components = 5
    truncated_radius = 150
    
    SEEDS_PATH = Path(__file__).parent / "seeds.json"
    with open(SEEDS_PATH, "r") as f:
        seeds_dict = json.load(f)

    source_component_seed = seeds_dict["source_components_seed"]
    print("Source component seed:", source_component_seed)
    master_source_rng = np.random.SeedSequence(seeds_dict["master_source_sampling_seed"])
    source_sampler = MixtureOfGaussians(dim, master_sampling_rng=master_source_rng, component_seed=source_component_seed)
    source_sampler.random_components(num_components=num_components, uniform_weights = True)
    source_sampler.set_truncation(truncated_radius)

    source_samples = source_sampler.sample(num_samples)

    print("Generated source samples:")
    print(source_samples)

    load_dir = f"./Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"

    input_sampler = characterize_entropic_sampler(dim=dim, num_measures=num_measures)
    input_sampler = load_sampler(load_dir, input_sampler, sampler_type = "entropic")
    print(input_sampler.__dict__)
   

    