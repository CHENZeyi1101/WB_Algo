# Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures

This repository contains the code implementations of numerical experiments for the project *Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures*.

## Table of Contents

- [Abstract](#abstract)
- [Main Contributions](#main-contributions)
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

# Main Contributions

1. **Stochastic Fixed-Point Algorithm**  
   We provide a computationally tractable stochastic fixed-point algorithm (i.e., *Algorithm 2*) for approximately computing the \( W_2 \)-barycenter of general input measures \( \nu_1, \ldots, \nu_K \).  
   Our iterative algorithm is characterized by a tailored truncating operation and an “admissible” class of OT map estimators (see *Assumption 3.4*).  
   In particular, we do not restrict the support of the approximate \( W_2 \)-barycenter to a finite collection of points, nor do we require \( \nu_1, \ldots, \nu_K \) to be discrete or to follow specific parametric families. Only mild regularity conditions are needed (see *Assumption 3.1* and *Setting 3.6*).

2. **Theoretical Convergence Guarantee**  
   We conduct a rigorous convergence analysis to show that our algorithm converges to the true \( W_2 \)-barycenter of \( \nu_1, \ldots, \nu_K \) in an almost sure sense (see *Setting 3.13* and *Theorem 3.14*).  
   Specifically, we adapt the computationally efficient entropic OT map estimator by Pooladian and Niles-Weed [52], incorporating tailored modifications to ensure convergence (see *Corollary 4.3*).  
   To the best of our knowledge, this is the first computationally tractable extension of the fixed-point iterative scheme by Álvarez-Esteban et al. [3] with a convergence guarantee.

3. **Synthetic Benchmark Generation**  
   We propose a simple and efficient method to generate synthetic instances of the \( W_2 \)-barycenter problem, where the input measures are continuous and non-parametric, and the true barycenter is known (see *Proposition 5.2*).  
   These instances allow benchmarking and evaluating the effectiveness of \( W_2 \)-barycenter algorithms.  
   Our numerical experiments demonstrate that the algorithm is empirically accurate, efficient, and stable compared with state-of-the-art alternatives, and supports parallel/distributed computing—making it well suited for large-scale applications.

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
