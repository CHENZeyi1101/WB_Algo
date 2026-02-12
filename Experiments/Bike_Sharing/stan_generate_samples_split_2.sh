#!/bin/bash

stan_model_path=./Experiments/Bike_Sharing/bike_sharing_PoissonGLM
input_data_dir=../../WB_data/Bike_Sharing/stan
output_data_dir=../../WB_data/Bike_Sharing/generated_samples

num_warmup=10000
num_samples_full=1000000
num_samples_split=10000000
num_samples_split_evaluation=1000000

# generate samples from the split-data posteriors (regarded as the input measures)
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split  data file=$input_data_dir/data_split_2.json output file=$output_data_dir/posterior_split_2.csv random seed=1002

# generate samples from the split_data posteriors for evaluation (when computing V-values)
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split_evaluation  data file=$input_data_dir/data_split_2.json output file=$output_data_dir/posterior_for_evaluation_split_2.csv random seed=11002