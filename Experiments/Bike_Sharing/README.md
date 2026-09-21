# For the bike sharing dataset:

Note that Steps 4–9 can be executed independently after Step 3.

- Step 1: Download the dataset and split it into 5 subsets by executing:
        
        python -m Experiments.Bike_Sharing.data_prepare

- Step 2: Compile the cmdstan binary by executing:

        bash ./Experiments/Bike_Sharing/stan_compile.sh

- Step 3: Generate the posterior samples for computing the barycenter and for evaluation by executing:

        bash ./Experiments/Bike_Sharing/stan_generate_samples.sh

- Step 4: Compute the approximate ground-truth V-value attained by the full-data posterior:

        python -m Experiments.Bike_Sharing.fullpost_compute

- Step 5: Run and visualize the stochastic fixed-point algorithm by executing: 

        python -m Experiments.Bike_Sharing.stochastic_FP_run

        python -m Experiments.Bike_Sharing.stochastic_FP_visualize

- Step 6: Run the algorithm of Cuturi and Doucet (2014) by executing:

        python -m Experiments.Bike_Sharing.Fast_Cuturi_run

- Step 7: Run and evaluate the algorithm of Li et al. (2020) by executing:

        python -m Experiments.Bike_Sharing.CWB_Li_run

        python -m Experiments.Bike_Sharing.CWB_Li_evaluate

- Step 8: Run and evaluate the algorithm of Fan et al. (2021) by executing:

        python -m Experiments.Bike_Sharing.ICNN_Fan_run

        python -m Experiments.Bike_Sharing.ICNN_Fan_evaluate

- Step 9: Run and evaluate the algorithm of Korotin et al. (2022) by executing:

        python -m Experiments.Bike_Sharing.WIN_Korotin_run

        python -m Experiments.Bike_Sharing.WIN_Korotin_evaluate
