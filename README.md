# Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures
+ This repository contains the Python code implementations of the paper.
+ By Zeyi Chen, Ariel Neufeld and Qikun Xiang.

## Table of Contents
- [Abstract](#abstract)
- [Descriptions of folders](#descriptions-of-folders)
  - [📁 Algorithms](#-algorithms)
  - [📁 Experiments](#-experiments)
- [Environment Setup](#environment-setup)
- [Instructions to run numerical experiments](#instructions-to-run-numerical-experiments)

# Abstract

We develop a framework utilizing statistical estimators and stochastic iterations to compute 2-Wasserstein barycenters for continuous, non-parametric probability distributions. Our work provides the first thorough convergence proof for implementable stochastic versions of an iterative method from Álvarez-Esteban et al. (2016). We establish convergence guarantees and identify conditions for geometric convergence rates under bounded optimal transport map estimation errors. We propose a computationally efficient algorithm supporting measures with Caffarelli-type regularity properties, utilizing modified entropic optimal transport map estimation. Additionally, we introduce a novel synthetic benchmark generation procedure where input distributions have meaningful characteristics and barycenters are approximately known. Experimental validation using both synthetic and real datasets demonstrates computational efficiency, estimation quality, and sampling versatility.

Keywords: Wasserstein barycenter, optimal transport, information aggregation, transportation map estimation

# Descriptions of folders

## 📁 Algorithms

The `Algorithms/` folder contains implementations of all algorithms compared in the paper, each organized in its own subfolder. See the subfolder README files for detailed descriptions.

- `Algorithms/Stochastic_FP/` — Our proposed stochastic fixed-point algorithm (Algorithm 2 in the paper).
- `Algorithms/CWB_Li/` — Implementation of the continuous Wasserstein barycenter method by Li et al.
- `Algorithms/Fast_Cuturi/` — Implementation of the fast Sinkhorn barycenter method by Cuturi and Doucet.
- `Algorithms/ICNN_Fan/` — Input-Convex Neural Network (ICNN)-based approach by Fan et al. (2021).
- `Algorithms/WIN_Korotin/` — Wasserstein Incremental Networks (WIN) method by Korotin et al. (2022).
- `Algorithms/WDHA_Kim/` — Implementation of the method by Kim et al.

## 📁 Experiments

The `Experiments/` folder contains all code and resources for running and evaluating numerical experiments. See the subfolder README files for detailed descriptions.

- `Experiments/Synthetic_Generation/` — Scripts for generating synthetic problem instances with known ground-truth barycenters, as introduced in the paper.
- `Experiments/Bike_Sharing/` — Experiments on the real-world bike sharing dataset.
- `Experiments/Karcher_Mean/` — Experiments related to Karcher mean computations.

# Environment Setup

To replicate the environment and run the project, create a new Conda environment from the provided `WB_Algo_arm.yml` file:

```bash
conda env create -f WB_Algo_arm.yml
conda activate myenv
```

# Instructions to run numerical experiments

Detailed instructions for running each experiment are provided in the README files within the respective subfolders of `Algorithms/` and `Experiments/`. 