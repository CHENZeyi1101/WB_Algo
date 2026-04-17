# Algorithms

This folder contains implementations of all algorithms compared in the paper. Each subfolder provides the implementation of one method, where the link for the source code is provided.

---

## 📁 Stochastic_FP

Implementation of the **stochastic fixed-point algorithm** proposed in our paper. This folder contains Python scripts for constructing the modified entropic optimal transport map estimator, which serves as the core building block of the iterative scheme.

> Chen, Z., Neufeld, A., and Xiang, Q. (2025). *Provably convergent stochastic fixed-point algorithm for free-support Wasserstein barycenter of continuous non-parametric measures.* arXiv:2505.24384.

---

## 📁 ICNN_Fan

Implementation of the **Input Convex Neural Network (ICNN)**-based Wasserstein barycenter algorithm by Fan et al. This method parameterizes optimal transport maps using input convex neural networks and scales to high-dimensional settings.

> Fan, J., Taghvaei, A., and Chen, Y. (2021). *Scalable Computations of Wasserstein Barycenter via Input Convex Neural Networks.* Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:1571–1581.

**Implementation:** [sbyebss/Scalable-Wasserstein-Barycenter](https://github.com/sbyebss/Scalable-Wasserstein-Barycenter)

---

## 📁 WIN_Korotin

Implementation of the **Wasserstein Iterative Networks (WIN)** method by Korotin et al. This approach iteratively estimates Wasserstein barycenters using neural network-based transport maps trained in an adversarial fashion.

> Korotin, A., Egiazarian, V., Li, L., and Burnaev, E. (2022). *Wasserstein Iterative Networks for Barycenter Estimation.* Advances in Neural Information Processing Systems (NeurIPS), 35:15672–15686.

**Implementation:** [iamalexkorotin/WassersteinIterativeNetworks](https://github.com/iamalexkorotin/WassersteinIterativeNetworks)

---

## 📁 CWB_Li

Implementation of the **Continuous Regularized Wasserstein Barycenters** method by Li et al. This approach computes barycenters of continuous probability measures using entropic regularization and kernel-based representations.

> Li, L., Genevay, A., Yurochkin, M., and Solomon, J. M. (2020). *Continuous Regularized Wasserstein Barycenters.* Advances in Neural Information Processing Systems (NeurIPS), 33:17755–17765.

**Implementation:** [lingxiaoli94/CWB](https://github.com/lingxiaoli94/CWB)

---

## 📁 WDHA_Kim

Implementation of the **Wasserstein-Descent Ḣ¹-Ascent (WDHA)** algorithm by Kim et al. This method computes exact (unregularized) Wasserstein barycenters for discretized measures via a nonconvex-concave minimax formulation, achieving nearly linear time complexity.

> Kim, K., Yao, R., Zhu, C., and Chen, X. (2025). *Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization.* Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267:30879–30899.

**Implementation:** [kaheonkim/WDHA](https://github.com/kaheonkim/WDHA)

---

## 📁 Fast_Cuturi

Implementation of the **fast Sinkhorn barycenter** algorithm by Cuturi and Doucet. This is one of the earliest scalable approaches to computing Wasserstein barycenters, based on entropic regularization and the Sinkhorn algorithm.

> Cuturi, M. and Doucet, A. (2014). *Fast Computation of Wasserstein Barycenters.* International Conference on Machine Learning (ICML), PMLR:685–693.

**Implementation:** [`ot.lp.free_support_barycenter`](https://pythonot.github.io/gen_modules/ot.lp.html) from the Python Optimal Transport (POT) library.
