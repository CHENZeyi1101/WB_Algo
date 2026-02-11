# For the bike sharing dataset:

Note that Steps 4–6 can be executed independently after Step 3.

- Step 1: Download the dataset and split it into 5 subsets by executing:
        
        python -m Experiments.Bike_Sharing.data_prepare

- Step 2: Compile the cmdstan binary by executing:

        bash ./Experiments/Bike_Sharing/stan_compile.sh

- Step 3: Generate the posterior samples for computing the barycenter and for evaluation by executing:

        bash ./Experiments/Bike_Sharing/stan_generate_samples.sh

- Step 4: Compute the ground-truth V-value via Monte Carlo by executing:

        python -m Experiments.Bike_Sharing.true_V_value

- Step 5: Run the stochastic fixed-point algorithm by executing: 

        python -m Experiments.Bike_Sharing.stochastic_FP_run

- Step 6: Run the algorithm of Cuturi and Doucet (2014) by executing:

        python -m Experiments.Bike_Sharing.Fast_Cuturi_run

- ...