from Experiments.Synthetic_Generation.metrics_to_compare import *
from Experiments.CSV_read import *
from Experiments.Synthetic_Generation.NCVCC_Kim_run_dim2 import *


# evaluation
V_values = []
W2_distances = []
V_values_path = os.path.join(outputs_dir, f"V_values_NCVCC_Kim_MCsize{MC_size}.json")
W2_distances_path = os.path.join(outputs_dir, f"W2_distances_NCVCC_Kim_MCsize{MC_size}.json")

for mc in range(MC_size):
    print(f"Starting MC run {mc+1}/{MC_size} ...")
    bary_samples = bary_samples_collection_loaded[str(mc)]
    input_samples_collection = input_sampler_for_evaluation.sample(num_samples)
    samples = sample_from_unit_mass_grid(mu_WGHA_unit, lo, hi, num_samples=1000, seed=mc + 1000)
    # Evaluate metrics
    V_value = V_value_compute(samples, input_samples_collection)
    W2_distance = W2_to_bary_compute(bary_samples, samples)

    V_values.append(V_value)
    W2_distances.append(W2_distance)

    with open(V_values_path, 'w') as json_file:
        json.dump(V_values, json_file)
    with open(W2_distances_path, 'w') as json_file:
        json.dump(W2_distances, json_file)