# Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures
+ This repository contains the Python code implementations of the paper. [[arXiv](https://arxiv.org/abs/2505.24384)]
+ By Zeyi Chen, Ariel Neufeld and Qikun Xiang.

## Table of Contents
- [Abstract](#abstract)
- [Descriptions of folders](#descriptions-of-folders)
  - [📁 Algorithms](#-algorithms)
  - [📁 Experiments](#-experiments)
- [Environment Setup](#environment-setup)
- [Instructions to run numerical experiments](#instructions-to-run-numerical-experiments)

# Abstract

We develop an estimator-based stochastic fixed-point framework for approximately computing the 2-Wasserstein barycenter of continuous, non-parametric probability measures. Notably, we provide the first rigorous convergence analysis for implementable estimator-based stochastic extensions of the fixed-point iterative scheme proposed by Álvarez-Esteban, del Barrio, Cuesta-Albertos, and Matrán (2016). In particular, we establish almost sure convergence, and identify sufficient conditions for geometric rates of convergence under controlled errors in optimal transport (OT) map estimation. We subsequently propose a concrete, provably convergent, and computationally tractable stochastic algorithm that accommodates input measures satisfying Caffarelli-type regularity conditions, which form a dense subset of the Wasserstein space. This algorithm leverages a modified entropic OT map estimator to enable efficient and scalable implementation. To facilitate quantitative evaluation, we further propose a novel and efficient procedure for synthetically generating benchmark instances, in which the input measures exhibit non-trivial features and the corresponding barycenters are approximately known. Numerical experiments on both synthetic and real-world datasets demonstrate the strong computational efficiency, estimation accuracy, and sampling flexibility of our approach.

**Keywords:** Wasserstein barycenter, optimal transport, transportation map estimation, entropic regularization

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