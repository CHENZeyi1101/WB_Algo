Every script is executed by running the following command in the terminal

    python -m Experiments.Synthetic_Generation.(SCRIPT_NAME)

# For 2-dimensional experiments:

Note that Steps 2–6 can be executed independently after Step 1, and Steps 7–12 can be executed independently after Step 5.

- Step 0: Select the source measure (i.e., the ground-truth barycenter measure) and the auxiliary measures by executing **input_measure_select_dim2**.

- Step 1: Set up the input measures by executing **input_sampler_setup_dim2**.

- Step 2: Validate that the source measure is indeed the barycenter of the input measures (ignoring the truncation effects) by executing **input_validate_dim2**.

- Step 3: Visualize the source measure, the auxiliary measures, and the input measures by executing **visualize_measures_dim2**.

- Step 4: Generate samples from the input measures by executing **input_samples_generate_dim2**.
Samples are saved in CSV files.

- Step 5: Generate samples from the source measure and the input measures that are used for evaluation by executing **samples_for_evaluation_generate_dim2**.
Samples are saved in JSON files.

- Step 6: Compute the ground-truth V-value via Monte Carlo by executing **true_V_value_dim2**.

- Step 7: Run the stochastic fixed-point algorithm by executing **stochastic_FP_run_dim2**.

- Step 8: Run the algorithm of Cuturi and Doucet (2014) by executing **Fast_Cuturi_run_dim2**.

- ...

# For 10d experiments:

- Step 1: Set up and generate samples from input measures by running input_sample_dim10.py. Samples are saved in CSV files.

- Step 2: Prepare a set of samples from the barycenter for evaluation by running samples_for_evaluation_dim10.py