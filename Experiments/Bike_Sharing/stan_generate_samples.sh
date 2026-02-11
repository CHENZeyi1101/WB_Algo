#!/bin/bash

stan_model_path=./Experiments/Bike_Sharing/bike_sharing_PoissonGLM
input_data_dir=../../WB_data/Bike_Sharing/stan
output_data_dir=../../WB_data/Bike_Sharing/generated_samples
num_warmup=10000
num_samples_full=10000
num_samples_split=100000
num_samples_split_evaluation=10000

# generate samples from the full-data posterior (regarded as the true barycenter)
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_full  data file=$input_data_dir/data_full.json output file=$output_data_dir/posterior_full.csv random seed=9000

# generate samples from the split-data posteriors (regarded as the input measures)
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split  data file=$input_data_dir/data_split_0.json output file=$output_data_dir/posterior_split_0.csv random seed=1000
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split  data file=$input_data_dir/data_split_1.json output file=$output_data_dir/posterior_split_1.csv random seed=1001
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split  data file=$input_data_dir/data_split_2.json output file=$output_data_dir/posterior_split_2.csv random seed=1002
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split  data file=$input_data_dir/data_split_3.json output file=$output_data_dir/posterior_split_3.csv random seed=1003
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split  data file=$input_data_dir/data_split_4.json output file=$output_data_dir/posterior_split_4.csv random seed=1004

# generate samples from the split_data posteriors for evaluation (when computing V-values)
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split_evaluation  data file=$input_data_dir/data_split_0.json output file=$output_data_dir/posterior_for_evaluation_split_0.csv random seed=11000
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split_evaluation  data file=$input_data_dir/data_split_1.json output file=$output_data_dir/posterior_for_evaluation_split_1.csv random seed=11001
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split_evaluation  data file=$input_data_dir/data_split_2.json output file=$output_data_dir/posterior_for_evaluation_split_2.csv random seed=11002
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split_evaluation  data file=$input_data_dir/data_split_3.json output file=$output_data_dir/posterior_for_evaluation_split_3.csv random seed=11003
$stan_model_path sample num_warmup=$num_warmup num_samples=$num_samples_split_evaluation  data file=$input_data_dir/data_split_4.json output file=$output_data_dir/posterior_for_evaluation_split_4.csv random seed=11004