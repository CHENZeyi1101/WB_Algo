Every script is executed by running the following command in the terminal

    python -m Experiments.Synthetic_Generation.(SCRIPT_NAME)

The JSON file [cfg.json](cfg.json) contains the parameter configurations for this experiment. 

# For 2-dimensional experiments:

Note that Steps 2–5 can be executed independently after Step 1, and Steps 6–13 can be executed independently after Step 5.

- Step 0: Select the source measure (i.e., the ground-truth barycenter measure) and the auxiliary measures by executing:

        python -m Experiments.Synthetic_Generation.input_measure_select_dim2

- Step 1: Set up the input measures by executing:

        python -m Experiments.Synthetic_Generation.input_sampler_setup_dim2

- Step 2: Validate that the source measure is indeed the barycenter of the input measures (ignoring the truncation effects) by executing:

        python -m Experiments.Synthetic_Generation.input_validate_dim2

- Step 3: Visualize the source measure, the auxiliary measures, and the input measures by executing:

        python -m Experiments.Synthetic_Generation.visualize_measures_dim2

- Step 4: Generate samples from the input measures by executing:

        python -m Experiments.Synthetic_Generation.input_samples_generate_dim2
        
    Samples are saved in CSV files.

- Step 5: Generate samples from the source measure and the input measures that are used for evaluation by executing:

        python -m Experiments.Synthetic_Generation.samples_for_evaluation_generate_dim2

    Samples are saved in JSON files.

- Step 6: Compute the ground-truth V-value via Monte Carlo by executing:

        python -m Experiments.Synthetic_Generation.true_V_value_dim2

- Step 7: Compute the empirical ground-truth V-value and the 2-Wasserstein distance to the barycenter via empirical approximations:

        python -m Experiments.Synthetic_Generation.true_via_OT_dim2

- Step 8: Run and evaluate the proposed stochastic fixed-point algorithm by executing:

        python -m Experiments.Synthetic_Generation.stochastic_FP_run_dim2

        python -m Experiments.Synthetic_Generation.stochastic_FP_visualize_dim2

- Step 9: Run the algorithm of Cuturi and Doucet (2014) by executing:

        python -m Experiments.Synthetic_Generation.Fast_Cuturi_run_dim2

- Step 10: Run and evaluate the algorithm of Li et al. (2020) by executing:

        python -m Experiments.Synthetic_Generation.CWB_Li_run_dim2

        python -m Experiments.Synthetic_Generation.CWB_Li_evaluate_dim2

- Step 11: Run and evaluate the algorithm of Fan et al. (2021) by executing:

        python -m Experiments.Synthetic_Generation.ICNN_Fan_run_dim2

        python -m Experiments.Synthetic_Generation.ICNN_Fan_evaluate_dim2

- Step 12: Run and evaluate the algorithm of Korotin et al. (2022) by executing:

        python -m Experiments.Synthetic_Generation.WIN_Korotin_run_dim2

        python -m Experiments.Synthetic_Generation.WIN_Korotin_evaluate_dim2

- Step 13: Run and evaluate the algorithm of Kim et al. (2025) by executing:

        python -m Experiments.Synthetic_Generation.WDHA_Kim_run_dim2

        python -m Experiments.Synthetic_Generation.WDHA_Kim_evaluate_dim2

# For 10d experiments:

Note that Steps 1–3 can be executed independently after Step 0, and Steps 4–10 can be executed independently after Step 3.

- Step 0: Set up the input measures by executing:

        python -m Experiments.Synthetic_Generation.input_sampler_setup_dim10

- Step 1: Validate that the source measure is indeed the barycenter of the input measures (ignoring the truncation effects) by executing:

        python -m Experiments.Synthetic_Generation.input_validate_dim10

- Step 2: Generate samples from the input measures by executing:

        python -m Experiments.Synthetic_Generation.input_samples_generate_dim10
        
    Samples are saved in CSV files.

- Step 3: Generate samples from the source measure and the input measures that are used for evaluation by executing:

        python -m Experiments.Synthetic_Generation.samples_for_evaluation_generate_dim10

    Samples are saved in JSON files.

- Step 4: Compute the ground-truth V-value via Monte Carlo by executing:

        python -m Experiments.Synthetic_Generation.true_V_value_dim10

- Step 5: Compute the empirical ground-truth V-value and the 2-Wasserstein distance to the barycenter via empirical approximations:

        python -m Experiments.Synthetic_Generation.true_via_OT_dim10

- Step 6: Run and evaluate the proposed stochastic fixed-point algorithm by executing:

        python -m Experiments.Synthetic_Generation.stochastic_FP_run_dim10

        python -m Experiments.Synthetic_Generation.stochastic_FP_visualize_dim10

- Step 7: Run the algorithm of Cuturi and Doucet (2014) by executing:

        python -m Experiments.Synthetic_Generation.Fast_Cuturi_run_dim10

- Step 8: Run and evaluate the algorithm of Li et al. (2020) by executing:

        python -m Experiments.Synthetic_Generation.CWB_Li_run_dim10

        python -m Experiments.Synthetic_Generation.CWB_Li_evaluate_dim10

- Step 9: Run and evaluate the algorithm of Fan et al. (2021) by executing:

        python -m Experiments.Synthetic_Generation.ICNN_Fan_run_dim10

        python -m Experiments.Synthetic_Generation.ICNN_Fan_evaluate_dim10

- Step 10: Run and evaluate the algorithm of Korotin et al. (2022) by executing:

        python -m Experiments.Synthetic_Generation.WIN_Korotin_run_dim10

        python -m Experiments.Synthetic_Generation.WIN_Korotin_evaluate_dim10

