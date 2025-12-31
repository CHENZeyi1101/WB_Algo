from Experiments.Synthetic_Generation.visualize_measures_dim2 import *
from Experiments.Synthetic_Generation.true_WB import *
from Experiments.Synthetic_Generation.samplers_dim2 import *
from tqdm import tqdm

if __name__ == "__main__":
    dim = 2
    num_measures = 5
    num_samples = 1000
    truncated_radius = 150
    theta = 2000

    plot_dir = "./Experiments/Synthetic_Generation/dim2_plots/measure_selection"
    os.makedirs(plot_dir, exist_ok=True)
    
    # # select measures over several random seeds
    # for seed in tqdm(range(1000, 1020), desc="Plotting measures for different seeds"):
    #     source_sampler = MixtureOfGaussians(dim)
    #     source_sampler.random_components(num_components=5, uniform_weights = True, seed = seed)
    #     source_sampler.set_truncation(truncated_radius)
    #     plot_name = f"seed_{seed}_measure.png"
    #     plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=plot_dir, plot_name=plot_name, title = f"Measure (Seed {seed})")
    

    # plot source measure: 1009
    source_sampler = MixtureOfGaussians(dim)
    source_sampler.random_components(num_components=5, uniform_weights = True, seed = 1009)
    source_sampler.set_truncation(truncated_radius)
    # plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=plot_dir, plot_name="source_measure.png", title = "Source Measure (Seed 1009)")
    # print("Source measure plotted.")

    # auxiliary measures
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim, num_components=5)
    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = False,
                                                     bound_type="eigen_bound",
                                                     theta = theta)
    entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir = plot_dir)

    # load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_plots/measure_selection"
    # entropic_sampler = load_sampler(load_dir, entropic_sampler, sampler_type="entropic")
    # print("Entropic sampler set up.")

    # plot input measures
    input_measure_samples = entropic_sampler.sample(1000)
    for measure_index in tqdm(range(len(input_measure_samples)), desc="Plotting input measures"):
        measure_samples = np.array(input_measure_samples[measure_index])
        plot_2d_measures_kde(measure_samples, truncated_radius = None, scatter=False, plot_dirc=f"{plot_dir}/theta_{theta}", plot_name=f"input_measure_{measure_index}_measure.png", title=fr"Input Measure $\nu_{{{measure_index + 1}}}$")    

    
