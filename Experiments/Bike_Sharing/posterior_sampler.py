from .posterior_sampling import *

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

