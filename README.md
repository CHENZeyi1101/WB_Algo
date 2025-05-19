# Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures

This repository contains the code implementations of numerical experiments for the project *Provably Convergent Stochastic Fixed-Point Algorithm for Free-Support Wasserstein Barycenter of Continuous Non-Parametric Measures*.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 📁 Stochastic_FP

The `Stochastic_FP/` folder contains all files and code for our proposed **stochastic fixed-point algorithm** (Algorithm 2 in the paper), used to approximate Wasserstein barycenters.

### Subfolder Structure

- [`Stochastic_FP/classes/`](Stochastic_FP/classes/)  
  Contains essential Python classes and functions that implement the stochastic fixed-point algorithm logic.

- [`Stochastic_FP/Notebooks/`](Stochastic_FP/Notebooks/)  
  Includes Jupyter notebooks for generating problem instances and applying our algorithm with modified entropic OT map estimators, as discussed in Section 4 of the paper.

  - [`Stochastic_FP/Notebooks/results/`](Stochastic_FP/Notebooks/results/)  
    Stores numerical results, generated plots, and evaluation outputs produced by the notebooks.

- [`Stochastic_FP/scripts/`](Stochastic_FP/scripts/)  
  Additional Python scripts to automate tasks, such as running experiments or post-processing.

- [`Stochastic_FP/__init__.py`](Stochastic_FP/__init__.py)  
  Initializes the folder as a Python module, enabling relative imports within the package.

---

> 📌 This folder is part of a larger project on [insert broader project focus here, e.g., "computational methods for optimal transport"].
