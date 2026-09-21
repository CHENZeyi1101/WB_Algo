#!/bin/bash

# cmdstan installation directory
cmdstan_dir=~/Development/cmdstan

stan_model_path=./Experiments/Bike_Sharing/bike_sharing_PoissonGLM

pwd=$(pwd)

cd $cmdstan_dir

make $pwd/$stan_model_path