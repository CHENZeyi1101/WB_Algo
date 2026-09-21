import os
from input_generate_plugin import *



def feed_samples(num_measures, 
                 num_samples,
                 dim, 
                 log = False, 
                 smoothing = "BA", 
                 input_func_logger = None, 
                 input_measure_logger = None, 
                 input_func_log_file = None,
                 plot = False,
                 ):
    
    # FUNCTIONALITY:
    # 1. "Feed" suffiently many input samples to the input_measure_sampler before the iterative scheme starts.
    # 2. To reduce dependency, the input samples of each measure would be reordered randomly.
    # 3. In each iteration of the algorithm, one may just extract samples of each input measure.

    raw_func_list = []
    for i in range(num_measures):

        # generate base samples for generating convex functions
        x_samples = np.random.uniform(low = -50, high = 50, size=(100, dim))

        # log the sample points for generating the convex function
        if log:
            input_func_logger.info(f"Sample points for CvxFunction_{i}: {x_samples}")

        cvxfunc_generator = convex_function(x_samples, num_functions = num_measures, log = log, logger = input_func_logger)

        if i % 2 == 0:
            x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sqrt()
        else:
            x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sq()
        # x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sqrt()
        # x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sq()
        if plot:
            plot_dir = "cvx_func_plots_num5"
            os.makedirs(plot_dir, exist_ok=True)
            cvxfunc_generator.plot_func(x_values, max_indices, name = f"{plot_dir}/cvx_func_{i}.png")
        cvx_otmap_generator = cvx_based_OTmap(x_samples, x_values, x_gradients, log = log)

        # initialize parameters of cvx_otmap_generator
        if log:
            input_func_logger.info(f"####### Shape and Interpolation Parameters for CvxFunction_{i} #######")
        # cvx_otmap_generator.shape_paras(seed = 5 + i, logger = input_func_logger) #4, 5
        cvx_otmap_generator.shape_paras(logger = input_func_logger)
        cvx_otmap_generator.interp_paras(logger = input_func_logger)
        raw_func_list.append(cvx_otmap_generator)

    if log:
        input_func_logger.info("######### Finished generating raw functions #########")

    # The true barycenter is defined as a truncated mixture of Gaussians
    source_sampler = MixtureOfGaussians(dim)
    source_sampler.random_components(5) # seed = 42
    source_sampler.set_truncation(100)
    source_samples = source_sampler.sample(int(num_samples))

    input_measure_sampler = input_sampler(raw_func_list, 
                                          source_samples, 
                                          log = log, 
                                          func_logger = input_func_logger, 
                                          measure_logger = input_measure_logger,
                                          func_log_file_path = input_func_log_file)
    input_measure_sampler.base_function_sample(smoothing)
    input_measure_sampler.measure_sample()

    # shuffle the samples of each measure
    for key in input_measure_sampler.sample_collection:
        np.random.shuffle(input_measure_sampler.sample_collection[key])

    return source_sampler, input_measure_sampler
