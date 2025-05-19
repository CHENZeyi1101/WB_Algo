# Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures

This repository contains the code implementations of numerical experiments for the project *Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures*.

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
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

# Code Structure

## 📁 Stochastic_FP

The `Stochastic_FP/` folder contains all files and code for our proposed **stochastic fixed-point algorithm** (Algorithm 2 in the paper), used to approximate Wasserstein barycenters.

### Subfolder Structure

- [`Stochastic_FP/classes/`](Stochastic_FP/classes/)  
  Contains essential Python classes and Python functions that implement the stochastic fixed-point algorithm logic.

- [`Stochastic_FP/Notebooks/`](Stochastic_FP/Notebooks/)  
  Includes Jupyter notebooks for configuring variables, generating problem instances and applying our algorithm with modified entropic OT map estimators, as discussed in Section 4 of the paper.

  - [`Stochastic_FP/Notebooks/results/`](Stochastic_FP/Notebooks/results/)  
    Stores numerical results, generated plots, and evaluation outputs produced by the notebooks.

- [`Stochastic_FP/scripts/`](Stochastic_FP/scripts/)  
  Additional Python scripts to automate tasks, such as running experiments or post-processing.

- [`Stochastic_FP/__init__.py`](Stochastic_FP/__init__.py)  
  Initializes the folder as a Python module, enabling relative imports within the package.

---

> 📌 This folder is part of a larger project on [insert broader project focus here, e.g., "computational methods for optimal transport"].
