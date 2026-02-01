from Experiments.Synthetic_Generation.visualize_measures_dim2 import *
from Experiments.Synthetic_Generation.MOG import *
from Experiments.Synthetic_Generation.samplers import *
from Experiments.CSV_read import *
from pathlib import Path
import json, os
from tqdm import tqdm

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    
    params = cfg_dict["params_synthetic_generation_dim2"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    alpha_list = params["alpha_list"]
    theta_list = params["theta_list"]
    gamma = params["gamma"]
    num_components = params["num_components"]

    plot_measure_selection = True
    plot_source = True

    plot_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_plots/Instance{instance_identifier}"
    os.makedirs(plot_dir, exist_ok=True)
    
    if plot_measure_selection:  # decision: component_seed = 1009
        # select measures over several random seeds
        for seed in tqdm(range(1000, 1050), desc="Plotting measures for different seeds"):
            source_sampler = characterize_source_sampler(dim = dim, 
                                                num_components = num_components, 
                                                master_sampling_rng = 42,
                                                component_seed = seed,
                                                truncated_radius = truncated_radius,
                                                save_dir = None)
            plot_name = f"seed_{seed}_measure.png"
            plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=f"{plot_dir}/measure_selection", plot_name=plot_name, title = f"Measure (Seed {seed})")

    if plot_source:
        plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=f"{plot_dir}/measure_selection", plot_name="source_measure.png", title = "Source Measure (Seed 1009)")
        print("Source measure plotted.")
        
    # source_sampler = characterize_source_sampler(dim = dim, 
    #                                             num_components = num_components, 
    #                                             master_sampling_rng = 42, # for plotting only
    #                                             component_seed = seed,
    #                                             truncated_radius = truncated_radius,
    #                                             save_dir = None)
    
    # auxiliary_seeds_list = seeds_dict["auxiliary_seeds_list"]
    # auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim = dim,
    #                                                                    num_components = num_components, 
    #                                                                    master_sampling_rng = master_auxiliary_rng_plot, 
    #                                                                    auxiliary_seeds_list = auxiliary_seeds_list)

    # tilde_K = len(auxiliary_measure_sampler_set)
    # surjective_mapping_seed = seeds_dict["surjective_mapping_seed"]
    # A_matrices_seed = seeds_dict["A_matrices_seed"]
    # surjective_mapping = construct_surjective_mapping(tilde_K = tilde_K, num_measures = num_measures, seed = surjective_mapping_seed)
    # A_matrices_dict = generate_A_matrices(dim = dim, num_measures = num_measures, seed = A_matrices_seed)

    # entropic_sampler = characterize_entropic_sampler(dim = dim, 
    #                                                  num_measures = num_measures, 
    #                                                  auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
    #                                                  source_sampler = source_sampler,
    #                                                  truncated_radius = truncated_radius,
    #                                                  manual = False,
    #                                                  bound_type="eigen_bound",
    #                                                  theta = instance_theta,
    #                                                  surjective_mapping = surjective_mapping,
    #                                                  A_matrices_dict = A_matrices_dict)

    # samplers_info_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}/samplers_info"
    # os.makedirs(samplers_info_dir, exist_ok=True)
    # entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir = samplers_info_dir)

    # # plot input measures
    # input_measure_samples = entropic_sampler.sample(1000)
    # for measure_index in tqdm(range(len(input_measure_samples)), desc="Plotting input measures"):
    #     measure_samples = np.array(input_measure_samples[measure_index])
    #     plot_2d_measures_kde(measure_samples, truncated_radius = None, scatter=False, plot_dirc=plot_dir, plot_name=f"input_measure_{measure_index}_measure.png", title=fr"Input Measure $\nu_{{{measure_index + 1}}}$")    

    
