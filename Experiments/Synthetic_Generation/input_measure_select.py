from Experiments.Synthetic_Generation.visualize_measures_dim2 import *
from Experiments.Synthetic_Generation.true_WB import *
from Experiments.Synthetic_Generation.samplers_dim2 import *
from Experiments.CSV_read import *

from tqdm import tqdm

if __name__ == "__main__":
    dim = 2
    num_measures = 5
    num_samples = 1000
    truncated_radius = 150
    theta = 2000
    plot_source = False

    plot_dir = f"./Experiments/Synthetic_Generation/dim2_plots/measure_selection/theta_{theta}"
    os.makedirs(plot_dir, exist_ok=True)

    auxiliary_csv_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/auxiliary_samples/csv_files"
    
    if plot_source:
        # select measures over several random seeds
        for seed in tqdm(range(1000, 1020), desc="Plotting measures for different seeds"):
            source_sampler = MixtureOfGaussians(dim)
            source_sampler.random_components(num_components=5, uniform_weights = True, seed = seed)
            source_sampler.set_truncation(truncated_radius)
            plot_name = f"seed_{seed}_measure.png"
            plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=plot_dir, plot_name=plot_name, title = f"Measure (Seed {seed})")
        

    # plot source measure: 1009
    # source_sampler = MixtureOfGaussians(dim)
    # source_sampler.random_components(num_components=5, uniform_weights = True, seed = 1009)
    # source_sampler.set_truncation(truncated_radius)

    source_csv_file = f"../../WB_data/Synthetic_Generation/dim{dim}_data/source_samples/csv_files/source_measure_samples.csv"
    source_sampler = csv_source_sampler_SyntheticGeneration(source_csv_file, 
                                                   multiplication_factor=1,
                                                   usecols=None,
                                                   skiprows=0)
    source_sampler.set_streamer()

    if plot_source:
        plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=plot_dir, plot_name="source_measure.png", title = "Source Measure (Seed 1009)")
        print("Source measure plotted.")

    # auxiliary measures
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(csv_dir = auxiliary_csv_dir, auxiliary_seeds_list = [1010, 1018, 1014, 1016, 1003])
    tilde_K = len(auxiliary_measure_sampler_set)
    surjective_mapping = construct_surjective_mapping(tilde_K = tilde_K, num_measures = num_measures, seed = 120)
    A_matrices_dict = generate_A_matrices(dim = dim, num_measures = num_measures, seed = 2000)

    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = False,
                                                     bound_type="eigen_bound",
                                                     theta = theta,
                                                     surjective_mapping = surjective_mapping,
                                                     A_matrices_dict = A_matrices_dict)
    
    entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir = plot_dir)

    # load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_plots/measure_selection"
    # entropic_sampler = load_sampler(load_dir, entropic_sampler, sampler_type="entropic")
    # print("Entropic sampler set up.")

    # plot input measures
    input_measure_samples = entropic_sampler.sample(1000)
    for measure_index in tqdm(range(len(input_measure_samples)), desc="Plotting input measures"):
        measure_samples = np.array(input_measure_samples[measure_index])
        plot_2d_measures_kde(measure_samples, truncated_radius = None, scatter=False, plot_dirc=plot_dir, plot_name=f"input_measure_{measure_index}_measure.png", title=fr"Input Measure $\nu_{{{measure_index + 1}}}$")    

    
