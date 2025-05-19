# Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures
+ This repository contains the Python code implementations of the paper.
+ By Zeyi Chen, Ariel Neufeld and Qikun Xiang.

## Table of Contents

- [Abstract](#abstract)
- [Descriptions of folders and files](#descriptions-of-folders-and-files)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

# Abstract 

We propose a provably convergent algorithm for approximating the 2-Wasserstein barycenter of continuous non-parametric probability measures. Our algorithm is inspired by the fixed-point iterative scheme of Álvarez-Esteban et al. (2016), whose convergence to the 2-Wasserstein barycenter relies on obtaining exact optimal transport (OT) maps.
However, OT maps are typically only approximately computed in practice, and exact computation of OT maps between continuous probability measures is only tractable for certain restrictive parametric families. To circumvent the need to compute exact OT maps between general non-parametric measures, we develop a tailored iterative scheme that utilizes consistent estimators of the OT maps instead of the exact OT maps.
This gives rise to a computationally tractable stochastic fixed-point algorithm which is provably convergent to the 2-Wasserstein barycenter. Our algorithm remarkably does not restrict the support of the 2-Wasserstein barycenter to be any fixed finite set and can be implemented in a distributed computing environment, which makes it suitable for large-scale data aggregation problems.
In our numerical experiments, we propose a method of generating non-trivial instances of 2-Wasserstein barycenter problems where the ground-truth barycenter measure is known. The results showcase the capability of our algorithm in developing high-quality approximations of the 2-Wasserstein barycenter, as well as its superiority over state-of-the-art methods based on generative neural networks in terms of accuracy, stability and efficiency.

Keywords: Wasserstein barycenter, optimal transport, information aggregation, transportation map estimation

# Descriptions of folders

## 📁 Stochastic_FP

The `Stochastic_FP/` folder contains all files and code for our proposed **stochastic fixed-point algorithm** (Algorithm 2 in the paper), used to approximate Wasserstein barycenters.

### Subfolder Structure

- [`Stochastic_FP/classes/`](Stochastic_FP/classes/)  
  Contains essential Python classes and Python functions that implement the stochastic fixed-point algorithm logic.

- [`Stochastic_FP/Notebooks/`](Stochastic_FP/Notebooks/)  
  Includes Jupyter notebooks for configuring variables, generating problem instances and applying our algorithm with modified entropic OT map estimators, as discussed in Section 4 of the paper.

  - [`Stochastic_FP/Notebooks/results/`](Stochastic_FP/Notebooks/results/)  
    Stores numerical results, generated plots, and evaluation outputs produced by the notebooks.

## 📁 ICNN_Fan

The `ICNN_Fan/` folder contains code and resources for implementing and evaluating an **Input-Convex Neural Network (ICNN)**-based approach, as introduced in [Fan et al. (2021)](https://github.com/sbyebss/Scalable-Wasserstein-Barycenter). This module supports instance generation, model training, and evaluation for barycenter approximation and related optimal transport tasks.

### Subfolder Structure

- [`ICNN_Fan/classes/`](ICNN_Fan/classes/)  
  Contains core Python classes and helper functions used to define and train ICNN models, including architectures, loss functions, and optimizers.

- [`ICNN_Fan/Notebooks/`](ICNN_Fan/Notebooks/)  
  Jupyter notebooks used to create problem instances, visualize training behavior, and assess model performance in line with the experimental settings described in the paper.

  - [`ICNN_Fan/Notebooks/results/`](ICNN_Fan/Notebooks/results/)  
    Stores numerical outputs, plots, and evaluations generated during experiments and notebook runs.





# Environment Setup

To replicate the environment and run the project, you are encouraged to create a new Conda environment and install dependencies via `requirements.txt`:

```bash
conda create -n myenv python=3.11
conda activate myenv
pip install -r requirements.txt
```
   
# Code Structure


