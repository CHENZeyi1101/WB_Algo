#!/bin/bash

stan_model_path=./Experiments/Bike_Sharing/bike_sharing_PoissonGLM
input_data_dir=../../WB_data/Bike_Sharing/stan
output_data_dir=../../WB_data/Bike_Sharing/generated_samples

num_warmup=10000
num_samples_full=1000000
num_samples_split=10000000
num_samples_split_evaluation=1000000

# generate samples from the full-data posterior (regarded as the true barycenter)
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_full  data file=$input_data_dir/data_full.json output file=$output_data_dir/posterior_full.csv random seed=9000