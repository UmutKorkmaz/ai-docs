---
title: "Generative Ai - Probabilistic Generative Models:"
description: "## 1. Introduction to Generative Modeling. Comprehensive guide covering optimization, algorithm, regression. Part of AI documentation system with 1500+ topics."
keywords: "optimization, algorithm, regression, optimization, algorithm, regression, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Probabilistic Generative Models: Mathematical Foundations

## 1. Introduction to Generative Modeling

### 1.1 Definition and Taxonomy

**Generative Models**
Statistical models that learn the underlying data distribution P(X) to generate new samples:
```
P(X) = ∫ P(X|Z)P(Z)dZ  (for latent variable models)
```

**Key Properties**
- **Data Distribution**: Learn P(X) directly or indirectly
- **Sample Generation**: Generate new data points
- **Density Estimation**: Estimate likelihood of observations
- **Representation Learning**: Learn meaningful data representations

**Taxonomy of Generative Models**
```python
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, multivariate_normal

class GenerativeModel(ABC):
    """Abstract base class for generative models"""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit model to training data"""
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate new samples"""
        pass

    @abstractmethod
    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log likelihood of data"""
        pass

    @abstractmethod
    def density(self, X: np.ndarray) -> np.ndarray:
        """Compute probability density"""
        pass

    def evaluate_generation_quality(self, X_real: np.ndarray, X_fake: np.ndarray) -> Dict[str, float]:
        """Evaluate quality of generated samples"""
        from scipy.stats import wasserstein_distance
        from sklearn.metrics import pairwise_distances

        metrics = {}

        # Wasserstein distance
        if X_real.shape[1] <= 10:  # For low-dimensional data
            wd = wasserstein_distance(X_real.flatten(), X_fake.flatten())
            metrics['wasserstein_distance'] = wd

        # Maximum Mean Discrepancy (simplified)
        mmd = self._compute_mmd(X_real, X_fake)
        metrics['mmd'] = mmd

        # Coverage metrics
        metrics['coverage'] = self._compute_coverage(X_real, X_fake)

        return metrics

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy"""
        def rbf_kernel(x, y, sigma=sigma):
            return np.exp(-np.sum((x - y)**2) / (2 * sigma**2))

        n, m = len(X), len(Y)
        mmd_squared = 0

        # X-X term
        for i in range(n):
            for j in range(n):
                if i != j:
                    mmd_squared += rbf_kernel(X[i], X[j])
        mmd_squared /= n * (n - 1)

        # Y-Y term
        for i in range(m):
            for j in range(m):
                if i != j:
                    mmd_squared += rbf_kernel(Y[i], Y[j])
        mmd_squared /= m * (m - 1)

        # X-Y term
        for i in range(n):
            for j in range(m):
                mmd_squared -= 2 * rbf_kernel(X[i], Y[j])
        mmd_squared /= n * m

        return np.sqrt(max(0, mmd_squared))

    def _compute_coverage(self, X_real: np.ndarray, X_fake: np.ndarray, k: int = 5) -> float:
        """Compute coverage metric"""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k).fit(X_real)
        distances, _ = nbrs.kneighbors(X_fake)

        # Count how many fake samples have real neighbors within threshold
        threshold = np.percentile(distances.flatten(), 50)  # Median distance
        covered = np.sum(distances[:, -1] <= threshold)
        return covered / len(X_fake)
```

### 1.2 Fundamental Concepts

**Probability Density Estimation**
```python
class DensityEstimator:
    """Base class for density estimation methods"""

    def __init__(self):
        self.fitted = False

    def histogram_density(self, X: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Histogram-based density estimation

        Args:
            X: Input data (n_samples, n_features)
            bins: Number of histogram bins

        Returns:
            (density, bin_edges)
        """
        if X.ndim > 1:
            raise ValueError("Histogram density estimation only supports 1D data")

        density, bin_edges = np.histogram(X, bins=bins, density=True)
        return density, bin_edges

    def kernel_density_estimation(self, X: np.ndarray, bandwidth: float = 1.0,
                                kernel: str = 'gaussian') -> Callable[[np.ndarray], np.ndarray]:
        """
        Kernel Density Estimation

        Args:
            X: Training data
            bandwidth: Kernel bandwidth
            kernel: Type of kernel ('gaussian', 'epanechnikov', 'uniform')

        Returns:
            Density function
        """
        n_samples, n_features = X.shape

        def kde_fn(x_new: np.ndarray) -> np.ndarray:
            """Compute density at new points"""
            x_new = np.atleast_2d(x_new)
            densities = np.zeros(len(x_new))

            for i, x in enumerate(x_new):
                for x_train in X:
                    if kernel == 'gaussian':
                        k = np.exp(-0.5 * np.sum((x - x_train)**2) / (bandwidth**2))
                    elif kernel == 'epanechnikov':
                        u = np.sum((x - x_train)**2) / (bandwidth**2)
                        k = 0.75 * (1 - u) if u <= 1 else 0
                    elif kernel == 'uniform':
                        k = 1 if np.sum((x - x_train)**2) <= bandwidth**2 else 0

                    densities[i] += k

            densities /= n_samples * (bandwidth ** n_features)
            return densities

        return kde_fn

    def gaussian_mixture_density(self, X: np.ndarray, n_components: int = 3,
                                max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gaussian Mixture Model density estimation

        Args:
            X: Training data
            n_components: Number of Gaussian components
            max_iter: Maximum EM iterations

        Returns:
            (weights, means, covariances)
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        weights = np.ones(n_components) / n_components
        means = X[np.random.choice(n_samples, n_components, replace=False)]
        covariances = [np.eye(n_features) for _ in range(n_components)]

        # EM algorithm
        for _ in range(max_iter):
            # E-step: Compute responsibilities
            responsibilities = np.zeros((n_samples, n_components))

            for k in range(n_components):
                rv = multivariate_normal(means[k], covariances[k])
                responsibilities[:, k] = weights[k] * rv.pdf(X)

            responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
            responsibilities /= responsibilities_sum + 1e-10

            # M-step: Update parameters
            Nk = responsibilities.sum(axis=0)

            weights = Nk / n_samples

            for k in range(n_components):
                means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / Nk[k]

                diff = X - means[k]
                covariances[k] = np.dot((responsibilities[:, k:k+1] * diff).T, diff) / Nk[k]

        def gmm_density(x_new: np.ndarray) -> np.ndarray:
            """Compute GMM density at new points"""
            x_new = np.atleast_2d(x_new)
            densities = np.zeros(len(x_new))

            for k in range(n_components):
                rv = multivariate_normal(means[k], covariances[k])
                densities += weights[k] * rv.pdf(x_new)

            return densities

        return gmm_density, weights, means, covariances
```

**Information Theory Foundations**
```python
class InformationTheory:
    """Information theory fundamentals for generative modeling"""

    @staticmethod
    def entropy(p: np.ndarray) -> float:
        """
        Shannon entropy: H(p) = -∑p(x)log p(x)

        Args:
            p: Probability distribution

        Returns:
            Entropy value
        """
        p = p[p > 0]  # Remove zero probabilities
        return -np.sum(p * np.log2(p))

    @staticmethod
    def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """
        Cross entropy: H(p,q) = -∑p(x)log q(x)

        Args:
            p: True distribution
            q: Predicted distribution

        Returns:
            Cross entropy value
        """
        return -np.sum(p * np.log2(q + 1e-10))

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Kullback-Leibler divergence: D_KL(p||q) = ∑p(x)log(p(x)/q(x))

        Args:
            p: True distribution
            q: Approximate distribution

        Returns:
            KL divergence value
        """
        return np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon divergence: JSD(p,q) = 0.5 * D_KL(p||m) + 0.5 * D_KL(q||m)

        Args:
            p: First distribution
            q: Second distribution

        Returns:
            JS divergence value
        """
        m = 0.5 * (p + q)
        return 0.5 * (InformationTheory.kl_divergence(p, m) +
                     InformationTheory.kl_divergence(q, m))

    @staticmethod
    def mutual_information(X: np.ndarray, Y: np.ndarray, bins: int = 10) -> float:
        """
        Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)

        Args:
            X: First random variable
            Y: Second random variable
            bins: Number of bins for discretization

        Returns:
            Mutual information value
        """
        # Discretize continuous variables
        X_bins = np.digitize(X, np.linspace(X.min(), X.max(), bins + 1))
        Y_bins = np.digitize(Y, np.linspace(Y.min(), Y.max(), bins + 1))

        # Compute joint and marginal distributions
        joint_dist, _, _ = np.histogram2d(X_bins, Y_bins, bins=(bins, bins))
        joint_dist = joint_dist / joint_dist.sum()

        X_marginal = joint_dist.sum(axis=1)
        Y_marginal = joint_dist.sum(axis=0)

        # Compute mutual information
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if joint_dist[i, j] > 0:
                    mi += joint_dist[i, j] * np.log2(
                        joint_dist[i, j] / (X_marginal[i] * Y_marginal[j] + 1e-10)
                    )

        return mi

    @staticmethod
    def wasserstein_distance_1d(p_dist: tuple, q_dist: tuple) -> float:
        """
        1D Wasserstein distance between two distributions

        Args:
            p_dist: (mean, std) for first distribution
            q_dist: (mean, std) for second distribution

        Returns:
            Wasserstein distance
        """
        mu_p, sigma_p = p_dist
        mu_q, sigma_q = q_dist

        # For 1D Gaussian distributions
        return np.sqrt((mu_p - mu_q)**2 + (sigma_p - sigma_q)**2)
```

## 2. Maximum Likelihood Estimation

### 2.1 Theory and Algorithms

**Maximum Likelihood Principle**
```
θ_MLE = argmax_θ ∏_{i=1}^n P(x_i|θ) = argmax_θ ∑_{i=1}^n log P(x_i|θ)
```

**Gradient Ascent for MLE**
```python
class MaximumLikelihoodEstimator:
    """Maximum likelihood estimation framework"""

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def gaussian_mle(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Analytical MLE for Gaussian distribution

        Args:
            X: Data samples

        Returns:
            (mean, variance)
        """
        mean = np.mean(X)
        variance = np.var(X, ddof=0)  # MLE uses ddof=0
        return mean, variance

    def gaussian_gradient_ascent(self, X: np.ndarray, initial_params: Tuple[float, float] = None) -> Tuple[float, float]:
        """
        Gradient ascent for Gaussian MLE (for demonstration)

        Args:
            X: Data samples
            initial_params: Initial (mean, variance)

        Returns:
            Estimated (mean, variance)
        """
        if initial_params is None:
            mu, sigma_sq = 0.0, 1.0
        else:
            mu, sigma_sq = initial_params

        for _ in range(self.max_iter):
            # Compute gradients
            n = len(X)

            # Gradient for mean
            grad_mu = np.sum(X - mu) / (sigma_sq + 1e-10)

            # Gradient for variance
            grad_sigma_sq = -n / (2 * sigma_sq) + np.sum((X - mu)**2) / (2 * sigma_sq**2)

            # Update parameters
            mu += self.learning_rate * grad_mu
            sigma_sq += self.learning_rate * grad_sigma_sq

            # Ensure variance is positive
            sigma_sq = max(sigma_sq, 1e-6)

        return mu, sigma_sq

    def multivariate_gaussian_mle(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        MLE for multivariate Gaussian distribution

        Args:
            X: Data samples (n_samples, n_features)

        Returns:
            (mean vector, covariance matrix)
        """
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False, bias=True)  # bias=True for MLE
        return mean, cov

    def categorical_mle(self, X: np.ndarray, n_categories: int) -> np.ndarray:
        """
        MLE for categorical distribution

        Args:
            X: Category indices
            n_categories: Number of categories

        Returns:
            Probability vector
        """
        counts = np.bincount(X, minlength=n_categories)
        probabilities = counts / len(X)
        return probabilities

    def expectation_maximization(self, X: np.ndarray, n_components: int,
                                max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        EM algorithm for Gaussian Mixture Model

        Args:
            X: Data samples
            n_components: Number of Gaussian components
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            (weights, means, covariances)
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        weights = np.ones(n_components) / n_components
        means = X[np.random.choice(n_samples, n_components, replace=False)]
        covariances = np.array([np.eye(n_features) for _ in range(n_components)])

        prev_log_likelihood = -np.inf

        for iteration in range(max_iter):
            # E-step: Compute responsibilities
            responsibilities = np.zeros((n_samples, n_components))

            for k in range(n_components):
                try:
                    rv = multivariate_normal(means[k], covariances[k])
                    responsibilities[:, k] = weights[k] * rv.pdf(X)
                except:
                    # Handle numerical issues
                    responsibilities[:, k] = 1e-10

            # Normalize responsibilities
            responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
            responsibilities = responsibilities / (responsibilities_sum + 1e-10)

            # M-step: Update parameters
            Nk = responsibilities.sum(axis=0)

            # Update weights
            weights = Nk / n_samples

            # Update means
            for k in range(n_components):
                means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / Nk[k]

            # Update covariances
            for k in range(n_components):
                diff = X - means[k]
                covariances[k] = np.dot((responsibilities[:, k:k+1] * diff).T, diff) / Nk[k]

                # Ensure positive definiteness
                covariances[k] += 1e-6 * np.eye(n_features)

            # Check convergence
            log_likelihood = self._gmm_log_likelihood(X, weights, means, covariances)
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break

            prev_log_likelihood = log_likelihood

        return weights, means, covariances

    def _gmm_log_likelihood(self, X: np.ndarray, weights: np.ndarray,
                           means: np.ndarray, covariances: np.ndarray) -> float:
        """Compute GMM log likelihood"""
        n_samples = X.shape[0]
        log_likelihood = 0

        for i in range(n_samples):
            sample_likelihood = 0
            for k in range(len(weights)):
                try:
                    rv = multivariate_normal(means[k], covariances[k])
                    sample_likelihood += weights[k] * rv.pdf(X[i])
                except:
                    sample_likelihood += 1e-10

            log_likelihood += np.log(sample_likelihood + 1e-10)

        return log_likelihood
```

### 2.2 Advanced MLE Techniques

**Regularized Maximum Likelihood**
```python
class RegularizedMLE:
    """Regularized maximum likelihood estimation"""

    def __init__(self, regularization_type: str = 'l2', lambda_param: float = 0.01):
        self.regularization_type = regularization_type
        self.lambda_param = lambda_param

    def regularized_gaussian_mle(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Regularized MLE for Gaussian distribution

        Args:
            X: Data samples

        Returns:
            (mean, variance)
        """
        n = len(X)

        # Standard MLE
        mean_mle = np.mean(X)
        var_mle = np.var(X, ddof=0)

        # Apply regularization
        if self.regularization_type == 'l2':
            # L2 regularization (ridge regression for mean)
            mean_reg = mean_mle  # No regularization for mean in simple case

            # Regularize variance towards a prior (e.g., 1.0)
            prior_variance = 1.0
            var_reg = (n * var_mle + self.lambda_param * prior_variance) / (n + self.lambda_param)

        elif self.regularization_type == 'l1':
            # L1 regularization (lasso)
            mean_reg = mean_mle
            var_reg = max(var_mle - self.lambda_param, 1e-6)  # Soft thresholding

        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")

        return mean_reg, var_reg

    def bayesian_mle_gaussian(self, X: np.ndarray, prior_mean: float = 0.0,
                             prior_precision: float = 1.0) -> Tuple[float, float]:
        """
        Bayesian MLE for Gaussian with conjugate prior

        Args:
            X: Data samples
            prior_mean: Prior mean
            prior_precision: Prior precision (1/variance)

        Returns:
            Posterior (mean, variance)
        """
        n = len(X)
        sample_mean = np.mean(X)

        # Posterior parameters for mean (known variance case)
        posterior_precision = prior_precision + n
        posterior_mean = (prior_precision * prior_mean + n * sample_mean) / posterior_precision

        return posterior_mean, 1.0 / posterior_precision

    def map_estimation_categorical(self, X: np.ndarray, n_categories: int,
                                 alpha: np.ndarray = None) -> np.ndarray:
        """
        MAP estimation for categorical distribution with Dirichlet prior

        Args:
            X: Category indices
            n_categories: Number of categories
            alpha: Dirichlet prior parameters

        Returns:
            MAP probability estimates
        """
        if alpha is None:
            alpha = np.ones(n_categories)  # Uniform prior

        counts = np.bincount(X, minlength=n_categories)
        map_estimates = (counts + alpha - 1) / (len(X) + np.sum(alpha) - n_categories)

        return map_estimates
```

## 3. Variational Inference

### 3.1 Theory and Framework

**Variational Inference Principle**
```
log P(X) = ELBO(q) + KL(q(θ) || P(θ|X))
```

Where ELBO (Evidence Lower Bound) is:
```
ELBO(q) = E_q[log P(X,θ)] - E_q[log q(θ)]
```

**Variational Autoencoder Theory**
```python
class VariationalInference:
    """Variational inference framework"""

    def __init__(self):
        self.elbo_history = []

    def mean_field_vi_gaussian(self, X: np.ndarray, n_components: int = 1,
                             max_iter: int = 1000, learning_rate: float = 0.01) -> Dict:
        """
        Mean-field variational inference for Gaussian mixture model

        Args:
            X: Data samples
            n_components: Number of Gaussian components
            max_iter: Maximum iterations
            learning_rate: Learning rate for coordinate ascent

        Returns:
            Dictionary with variational parameters
        """
        n_samples, n_features = X.shape

        # Initialize variational parameters
        # Responsibilities (r_nk)
        responsibilities = np.random.dirichlet(np.ones(n_components), size=n_samples)

        # Component parameters (means, covariances)
        means = X[np.random.choice(n_samples, n_components, replace=False)]
        covariances = [np.eye(n_features) for _ in range(n_components)]

        # Mixing weights (π_k)
        mixing_weights = np.ones(n_components) / n_components

        for iteration in range(max_iter):
            # E-step: Update responsibilities
            for n in range(n_samples):
                log_resps = []

                for k in range(n_components):
                    try:
                        rv = multivariate_normal(means[k], covariances[k])
                        log_resp = np.log(mixing_weights[k] + 1e-10) + np.log(rv.pdf(X[n]) + 1e-10)
                        log_resps.append(log_resp)
                    except:
                        log_resps.append(-np.inf)

                # Normalize responsibilities
                log_resps = np.array(log_resps)
                log_resps -= np.max(log_resps)  # For numerical stability
                responsibilities[n] = np.exp(log_resps)
                responsibilities[n] /= np.sum(responsibilities[n])

            # M-step: Update component parameters
            Nk = responsibilities.sum(axis=0)

            # Update mixing weights
            mixing_weights = Nk / n_samples

            # Update means
            for k in range(n_components):
                means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / Nk[k]

            # Update covariances
            for k in range(n_components):
                diff = X - means[k]
                covariances[k] = np.dot((responsibilities[:, k:k+1] * diff).T, diff) / Nk[k]

                # Ensure positive definiteness
                covariances[k] += 1e-6 * np.eye(n_features)

            # Calculate ELBO
            elbo = self._calculate_elbo_gmm(X, responsibilities, mixing_weights, means, covariances)
            self.elbo_history.append(elbo)

        return {
            'responsibilities': responsibilities,
            'mixing_weights': mixing_weights,
            'means': means,
            'covariances': covariances,
            'elbo_history': self.elbo_history
        }

    def _calculate_elbo_gmm(self, X: np.ndarray, responsibilities: np.ndarray,
                           mixing_weights: np.ndarray, means: np.ndarray,
                           covariances: np.ndarray) -> float:
        """Calculate ELBO for Gaussian mixture model"""
        n_samples = X.shape[0]
        elbo = 0

        for n in range(n_samples):
            for k in range(len(mixing_weights)):
                if responsibilities[n, k] > 1e-10:
                    # Log likelihood term
                    try:
                        rv = multivariate_normal(means[k], covariances[k])
                        log_likelihood = np.log(rv.pdf(X[n]) + 1e-10)
                    except:
                        log_likelihood = -np.inf

                    # Entropy term
                    entropy = -responsibilities[n, k] * np.log(responsibilities[n, k] + 1e-10)

                    elbo += responsibilities[n, k] * (log_likelihood + np.log(mixing_weights[k] + 1e-10)) + entropy

        return elbo

    def coordinate_ascent_vi(self, X: np.ndarray, initial_params: Dict,
                           max_iter: int = 100, tol: float = 1e-6) -> Dict:
        """
        Coordinate ascent variational inference

        Args:
            X: Data samples
            initial_params: Initial variational parameters
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Optimized variational parameters
        """
        params = initial_params.copy()
        prev_elbo = -np.inf

        for iteration in range(max_iter):
            # Update each variational parameter in turn
            for param_name in params:
                new_params = params.copy()

                # Simple coordinate update (simplified)
                if param_name == 'mean':
                    # Update mean parameter
                    gradient = self._compute_mean_gradient(X, params)
                    new_params['mean'] += 0.01 * gradient

                elif param_name == 'variance':
                    # Update variance parameter
                    gradient = self._compute_variance_gradient(X, params)
                    new_params['variance'] = np.maximum(new_params['variance'] + 0.01 * gradient, 1e-6)

                params = new_params

            # Check convergence
            current_elbo = self._compute_elbo(X, params)
            if abs(current_elbo - prev_elbo) < tol:
                break

            prev_elbo = current_elbo

        return params

    def _compute_mean_gradient(self, X: np.ndarray, params: Dict) -> float:
        """Compute gradient for mean parameter (simplified)"""
        # This is a simplified version
        return np.mean(X - params['mean'])

    def _compute_variance_gradient(self, X: np.ndarray, params: Dict) -> float:
        """Compute gradient for variance parameter (simplified)"""
        # This is a simplified version
        squared_errors = (X - params['mean'])**2
        return np.mean(squared_errors / params['variance']**2 - 1 / params['variance'])

    def _compute_elbo(self, X: np.ndarray, params: Dict) -> float:
        """Compute ELBO (simplified)"""
        mu = params['mean']
        sigma2 = params['variance']

        # Expected log likelihood
        expected_log_likelihood = -0.5 * np.mean((X - mu)**2) / sigma2 - 0.5 * np.log(2 * np.pi * sigma2)

        # Entropy of variational distribution
        entropy = 0.5 * np.log(2 * np.pi * np.e * sigma2)

        return expected_log_likelihood + entropy
```

### 3.2 Stochastic Variational Inference

**SVI for Large Datasets**
```python
class StochasticVariationalInference:
    """Stochastic variational inference for large datasets"""

    def __init__(self, batch_size: int = 32, learning_rate: float = 0.01):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.elbo_history = []

    def svi_gaussian_mixture(self, X: np.ndarray, n_components: int,
                            n_epochs: int = 100) -> Dict:
        """
        Stochastic VI for Gaussian mixture model

        Args:
            X: Data samples
            n_components: Number of components
            n_epochs: Number of epochs

        Returns:
            Variational parameters
        """
        n_samples, n_features = X.shape

        # Initialize variational parameters
        means = X[np.random.choice(n_samples, n_components, replace=False)]
        covariances = [np.eye(n_features) for _ in range(n_components)]
        mixing_weights = np.ones(n_components) / n_components

        # Natural parameters (for efficiency)
        natural_means = [np.zeros(n_features) for _ in range(n_components)]
        natural_precisions = [np.eye(n_features) for _ in range(n_components)]

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]

                # Compute local variational parameters for batch
                batch_responsibilities = np.zeros((len(X_batch), n_components))

                for i, x in enumerate(X_batch):
                    log_resps = []

                    for k in range(n_components):
                        try:
                            rv = multivariate_normal(means[k], covariances[k])
                            log_resp = np.log(mixing_weights[k] + 1e-10) + np.log(rv.pdf(x) + 1e-10)
                            log_resps.append(log_resp)
                        except:
                            log_resps.append(-np.inf)

                    log_resps = np.array(log_resps)
                    log_resps -= np.max(log_resps)
                    batch_responsibilities[i] = np.exp(log_resps)
                    batch_responsibilities[i] /= np.sum(batch_responsibilities[i])

                # Update global parameters
                for k in range(n_components):
                    # Natural gradient update
                    Nk_batch = np.sum(batch_responsibilities[:, k])

                    # Update mean
                    mean_update = np.sum(batch_responsibilities[:, k:k+1] * X_batch, axis=0) / (Nk_batch + 1e-10)
                    means[k] = (1 - self.learning_rate) * means[k] + self.learning_rate * mean_update

                    # Update covariance
                    diff = X_batch - means[k]
                    cov_update = np.dot((batch_responsibilities[:, k:k+1] * diff).T, diff) / (Nk_batch + 1e-10)
                    covariances[k] = (1 - self.learning_rate) * covariances[k] + self.learning_rate * cov_update

                # Update mixing weights
                mixing_weights_update = np.sum(batch_responsibilities, axis=0) / len(X_batch)
                mixing_weights = (1 - self.learning_rate) * mixing_weights + self.learning_rate * mixing_weights_update

                # Normalize mixing weights
                mixing_weights /= np.sum(mixing_weights)

            # Calculate and store ELBO
            elbo = self._calculate_elbo_gmm(X, mixing_weights, means, covariances, 100)  # Subsample for speed
            self.elbo_history.append(elbo)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, ELBO: {elbo:.4f}")

        return {
            'mixing_weights': mixing_weights,
            'means': means,
            'covariances': covariances,
            'elbo_history': self.elbo_history
        }

    def _calculate_elbo_gmm(self, X: np.ndarray, mixing_weights: np.ndarray,
                           means: np.ndarray, covariances: np.ndarray,
                           n_samples: int = 1000) -> float:
        """Calculate ELBO using subsampling"""
        # Subsample for efficiency
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sub = X[indices]
        else:
            X_sub = X

        elbo = 0
        for x in X_sub:
            responsibilities = []

            for k in range(len(mixing_weights)):
                try:
                    rv = multivariate_normal(means[k], covariances[k])
                    resp = mixing_weights[k] * rv.pdf(x)
                    responsibilities.append(resp)
                except:
                    responsibilities.append(1e-10)

            total_resp = sum(responsibilities)
            if total_resp > 0:
                elbo += np.log(total_resp + 1e-10)

        return elbo / len(X_sub)
```

## 4. Monte Carlo Methods

### 4.1 Sampling Techniques

**Importance Sampling**
```python
class MonteCarloMethods:
    """Monte Carlo methods for generative modeling"""

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)

    def importance_sampling(self, target_density: callable, proposal_density: callable,
                          proposal_sampler: callable, n_samples: int = 10000) -> Dict:
        """
        Importance sampling for expectation estimation

        Args:
            target_density: Target density function p(x)
            proposal_density: Proposal density function q(x)
            proposal_sampler: Function to sample from proposal
            n_samples: Number of samples

        Returns:
            Dictionary with estimates and diagnostics
        """
        # Sample from proposal
        samples = proposal_sampler(n_samples)

        # Calculate importance weights
        target_vals = target_density(samples)
        proposal_vals = proposal_density(samples)

        # Handle numerical issues
        proposal_vals = np.maximum(proposal_vals, 1e-10)
        importance_weights = target_vals / proposal_vals

        # Normalize weights
        normalized_weights = importance_weights / np.sum(importance_weights)

        # Estimate expectation (assuming we want E[f(X)] = X)
        expectation = np.sum(normalized_weights * samples)
        variance = np.sum(normalized_weights * (samples - expectation)**2)

        # Effective sample size
        ess = 1 / np.sum(normalized_weights**2)

        return {
            'expectation': expectation,
            'variance': variance,
            'effective_sample_size': ess,
            'samples': samples,
            'weights': normalized_weights
        }

    def adaptive_importance_sampling(self, target_density: callable,
                                    initial_proposal: callable, n_samples: int = 10000,
                                    n_iterations: int = 5) -> Dict:
        """
        Adaptive importance sampling

        Args:
            target_density: Target density function
            initial_proposal: Initial proposal distribution
            n_samples: Samples per iteration
            n_iterations: Number of adaptation iterations

        Returns:
            Adaptive sampling results
        """
        results = []

        # Start with initial proposal
        proposal_sampler = lambda n: initial_proposal.rvs(size=n)
        proposal_density = initial_proposal.pdf

        for iteration in range(n_iterations):
            # Perform importance sampling
            iter_result = self.importance_sampling(
                target_density, proposal_density, proposal_sampler, n_samples
            )
            results.append(iter_result)

            # Adapt proposal based on weighted samples
            if iteration < n_iterations - 1:
                # Fit new proposal to weighted samples
                from scipy.stats import norm
                weighted_mean = np.sum(iter_result['weights'] * iter_result['samples'])
                weighted_var = np.sum(iter_result['weights'] * (iter_result['samples'] - weighted_mean)**2)

                # Update proposal
                new_proposal = norm(weighted_mean, np.sqrt(weighted_var))
                proposal_sampler = lambda n: new_proposal.rvs(size=n)
                proposal_density = new_proposal.pdf

        return {
            'iterations': results,
            'final_expectation': results[-1]['expectation'],
            'convergence_history': [r['expectation'] for r in results]
        }

    def rejection_sampling(self, target_density: callable, proposal_density: callable,
                          proposal_sampler: callable, M: float, n_samples: int = 10000) -> np.ndarray:
        """
        Rejection sampling

        Args:
            target_density: Target density p(x)
            proposal_density: Proposal density q(x)
            proposal_sampler: Function to sample from proposal
            M: Upper bound on p(x)/q(x)
            n_samples: Desired number of samples

        Returns:
            Accepted samples
        """
        accepted_samples = []
        total_attempts = 0

        while len(accepted_samples) < n_samples:
            # Sample from proposal
            x = proposal_sampler(1)[0]
            total_attempts += 1

            # Calculate acceptance probability
            p_x = target_density(x)
            q_x = proposal_density(x)

            if q_x > 0:
                acceptance_prob = p_x / (M * q_x)

                # Accept or reject
                if np.random.random() < acceptance_prob:
                    accepted_samples.append(x)

        acceptance_rate = n_samples / total_attempts

        print(f"Acceptance rate: {acceptance_rate:.4f}")

        return np.array(accepted_samples)

    def metropolis_hastings(self, target_density: callable, proposal_sampler: callable,
                           initial_state: float, n_samples: int = 10000,
                           burn_in: int = 1000) -> Dict:
        """
        Metropolis-Hastings algorithm

        Args:
            target_density: Target density (unnormalized)
            proposal_sampler: Function that samples proposal given current state
            initial_state: Initial chain state
            n_samples: Total samples to generate
            burn_in: Number of burn-in samples

        Returns:
            MCMC results
        """
        chain = [initial_state]
        current_state = initial_state
        accepted = 0

        for i in range(n_samples + burn_in):
            # Generate proposal
            proposed_state = proposal_sampler(current_state)

            # Calculate acceptance probability
            current_target = target_density(current_state)
            proposed_target = target_density(proposed_state)

            # For symmetric proposals, q ratio = 1
            acceptance_ratio = proposed_target / (current_target + 1e-10)

            # Accept or reject
            if np.random.random() < min(1, acceptance_ratio):
                current_state = proposed_state
                accepted += 1

            chain.append(current_state)

        # Remove burn-in
        samples = np.array(chain[burn_in+1:])
        acceptance_rate = accepted / (n_samples + burn_in)

        return {
            'samples': samples,
            'acceptance_rate': acceptance_rate,
            'chain': chain
        }

    def gibbs_sampling(self, joint_density: callable, conditional_samplers: List[callable],
                      initial_state: np.ndarray, n_samples: int = 10000,
                      burn_in: int = 1000) -> np.ndarray:
        """
        Gibbs sampling for multivariate distributions

        Args:
            joint_density: Joint density function (for diagnostics)
            conditional_samplers: List of conditional sampling functions
            initial_state: Initial state vector
            n_samples: Number of samples
            burn_in: Burn-in period

        Returns:
            Sampled chain
        """
        n_dimensions = len(initial_state)
        chain = [initial_state.copy()]
        current_state = initial_state.copy()

        for _ in range(n_samples + burn_in):
            for dim in range(n_dimensions):
                # Sample from conditional distribution
                current_state[dim] = conditional_samplers[dim](current_state)

            chain.append(current_state.copy())

        return np.array(chain[burn_in+1:])
```

### 4.2 Markov Chain Monte Carlo

**Advanced MCMC Techniques**
```python
class AdvancedMCMC:
    """Advanced Markov Chain Monte Carlo methods"""

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)

    def hamiltonian_monte_carlo(self, log_target_density: callable,
                               gradient_function: callable,
                               initial_state: np.ndarray,
                               n_samples: int = 10000,
                               step_size: float = 0.1,
                               n_leapfrog: int = 10) -> Dict:
        """
        Hamiltonian Monte Carlo

        Args:
            log_target_density: Log target density
            gradient_function: Gradient of log target density
            initial_state: Initial state
            n_samples: Number of samples
            step_size: Integration step size
            n_leapfrog: Number of leapfrog steps

        Returns:
            HMC samples and diagnostics
        """
        n_dim = len(initial_state)
        chain = [initial_state.copy()]
        current_state = initial_state.copy()
        accepted = 0

        for _ in range(n_samples):
            # Sample momentum
            current_momentum = np.random.randn(n_dim)

            # Hamiltonian at current state
            current_hamiltonian = -log_target_density(current_state) + 0.5 * np.sum(current_momentum**2)

            # Leapfrog integration
            proposed_state = current_state.copy()
            proposed_momentum = current_momentum.copy()

            # Half step for momentum
            proposed_momentum += 0.5 * step_size * gradient_function(proposed_state)

            # Full steps for position and momentum
            for _ in range(n_leapfrog):
                proposed_state += step_size * proposed_momentum
                proposed_momentum += step_size * gradient_function(proposed_state)

            # Half step for momentum
            proposed_momentum += 0.5 * step_size * gradient_function(proposed_state)

            # Hamiltonian at proposed state
            proposed_hamiltonian = -log_target_density(proposed_state) + 0.5 * np.sum(proposed_momentum**2)

            # Accept or reject
            delta_hamiltonian = proposed_hamiltonian - current_hamiltonian

            if np.random.random() < np.exp(-delta_hamiltonian):
                current_state = proposed_state.copy()
                accepted += 1

            chain.append(current_state.copy())

        acceptance_rate = accepted / n_samples

        return {
            'samples': np.array(chain),
            'acceptance_rate': acceptance_rate,
            'step_size': step_size,
            'n_leapfrog': n_leapfrog
        }

    def nuts_sampler(self, log_target_density: callable,
                    gradient_function: callable,
                    initial_state: np.ndarray,
                    n_samples: int = 10000) -> Dict:
        """
        Simplified No-U-Turn Sampler

        Args:
            log_target_density: Log target density
            gradient_function: Gradient function
            initial_state: Initial state
            n_samples: Number of samples

        Returns:
            NUTS samples
        """
        n_dim = len(initial_state)
        chain = [initial_state.copy()]
        current_state = initial_state.copy()
        accepted = 0

        # Initial step size (adaptive)
        step_size = 0.1

        for i in range(n_samples):
            # Build binary tree
            tree_state = self._build_nuts_tree(
                current_state, log_target_density, gradient_function,
                step_size, max_depth=10
            )

            if tree_state['accept']:
                current_state = tree_state['sample']
                accepted += 1

            chain.append(current_state.copy())

            # Adapt step size (simplified)
            if i < 1000:  # Adaptation period
                acceptance_rate = accepted / (i + 1)
                if acceptance_rate < 0.6:
                    step_size *= 0.95
                elif acceptance_rate > 0.8:
                    step_size *= 1.05

        return {
            'samples': np.array(chain),
            'acceptance_rate': accepted / n_samples,
            'final_step_size': step_size
        }

    def _build_nuts_tree(self, current_state: np.ndarray,
                        log_target_density: callable,
                        gradient_function: callable,
                        step_size: float,
                        max_depth: int = 10) -> Dict:
        """Build NUTS binary tree (simplified implementation)"""
        # This is a simplified version of NUTS
        # Full implementation would involve building a proper binary tree

        # Sample momentum
        n_dim = len(current_state)
        momentum = np.random.randn(n_dim)

        # Take a single leapfrog step
        new_state = current_state.copy()
        new_momentum = momentum.copy()

        new_momentum += 0.5 * step_size * gradient_function(new_state)
        new_state += step_size * new_momentum
        new_momentum += 0.5 * step_size * gradient_function(new_state)

        # Calculate acceptance probability
        current_h = -log_target_density(current_state) + 0.5 * np.sum(momentum**2)
        new_h = -log_target_density(new_state) + 0.5 * np.sum(new_momentum**2)

        accept_prob = np.exp(min(0, current_h - new_h))

        return {
            'sample': new_state if np.random.random() < accept_prob else current_state,
            'accept': np.random.random() < accept_prob
        }

    def parallel_tempering(self, target_densities: List[callable],
                          initial_states: List[np.ndarray],
                          n_samples: int = 10000,
                          swap_interval: int = 100) -> Dict:
        """
        Parallel tempering for multimodal distributions

        Args:
            target_densities: List of target densities (temperature levels)
            initial_states: Initial states for each chain
            n_samples: Number of samples
            swap_interval: Interval for chain swaps

        Returns:
            Results from all chains
        """
        n_chains = len(target_densities)
        chains = [state.copy() for state in initial_states]
        all_chains = [[] for _ in range(n_chains)]
        swap_attempts = 0
        swap_acceptances = 0

        for iteration in range(n_samples):
            # Run MCMC for each chain
            for i in range(n_chains):
                # Simple Metropolis step for each chain
                proposal = chains[i] + np.random.randn(len(chains[i])) * 0.1

                current_log_prob = np.log(target_densities[i](chains[i]) + 1e-10)
                proposal_log_prob = np.log(target_densities[i](proposal) + 1e-10)

                if np.random.random() < np.exp(min(0, proposal_log_prob - current_log_prob)):
                    chains[i] = proposal

                all_chains[i].append(chains[i].copy())

            # Attempt chain swaps
            if iteration % swap_interval == 0 and iteration > 0:
                swap_attempts += 1

                # Randomly select two adjacent chains
                i = np.random.randint(0, n_chains - 1)
                j = i + 1

                # Calculate swap probability
                log_p_i = np.log(target_densities[i](chains[i]) + 1e-10)
                log_p_j = np.log(target_densities[j](chains[j]) + 1e-10)
                log_p_i_swap = np.log(target_densities[i](chains[j]) + 1e-10)
                log_p_j_swap = np.log(target_densities[j](chains[i]) + 1e-10)

                swap_prob = min(1, np.exp((log_p_i_swap + log_p_j_swap) - (log_p_i + log_p_j)))

                if np.random.random() < swap_prob:
                    chains[i], chains[j] = chains[j], chains[i]
                    swap_acceptances += 1

        swap_rate = swap_acceptances / max(1, swap_attempts)

        return {
            'chains': [np.array(chain) for chain in all_chains],
            'swap_rate': swap_rate,
            'final_states': chains
        }
```

## 5. Advanced Generative Modeling Theory

### 5.1 Normalizing Flows

**Normalizing Flow Theory**
```
log p_X(x) = log p_Z(f(x)) + log|det J_f(x)|

Where:
- f: Bijective transformation
- J_f: Jacobian matrix of f
- p_Z: Base distribution (usually Gaussian)
```

```python
class NormalizingFlow:
    """Normalizing flow implementations"""

    def __init__(self, base_dist: callable = None):
        if base_dist is None:
            # Standard normal base distribution
            self.base_dist = lambda z: multivariate_normal.pdf(z, mean=np.zeros(len(z)), cov=np.eye(len(z)))
            self.base_log_prob = lambda z: multivariate_normal.logpdf(z, mean=np.zeros(len(z)), cov=np.eye(len(z)))
        else:
            self.base_dist = base_dist

    def planar_flow(self, z: np.ndarray, w: np.ndarray, u: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
        """
        Planar flow transformation

        Args:
            z: Input
            w, u, b: Flow parameters

        Returns:
            (transformed_z, log_det_jacobian)
        """
        # Ensure invertibility condition
        w_dot_u = np.dot(w, u)
        if w_dot_u < -1:
            # Make u such that w·u > -1
            u = u + (-1 - w_dot_u) * w / (np.dot(w, w) + 1e-10)

        # Transformation
        h = np.tanh(np.dot(w, z) + b)
        f_z = z + u * h

        # Log determinant of Jacobian
        psi = (1 - h**2) * w
        log_det = np.log(np.abs(1 + np.dot(u, psi)) + 1e-10)

        return f_z, log_det

    def radial_flow(self, z: np.ndarray, z0: np.ndarray, alpha: float, beta: float) -> Tuple[np.ndarray, float]:
        """
        Radial flow transformation

        Args:
            z: Input
            z0: Reference point
            alpha, beta: Flow parameters

        Returns:
            (transformed_z, log_det_jacobian)
        """
        r = np.linalg.norm(z - z0)
        h = 1 / (alpha + r)
        f_z = z + beta * h * (z - z0)

        # Log determinant
        d = len(z)
        log_det = np.log(1 + beta / (alpha + r)**(d+1) + 1e-10)

        return f_z, log_det

    def affine_coupling(self, z: np.ndarray, scale_net: callable, translate_net: callable) -> Tuple[np.ndarray, float]:
        """
        Affine coupling layer

        Args:
            z: Input
            scale_net: Neural network for scale
            translate_net: Neural network for translation

        Returns:
            (transformed_z, log_det_jacobian)
        """
        d = len(z)
        d1 = d // 2

        z1, z2 = z[:d1], z[d1:]

        # Compute scale and translation
        s = scale_net(z1)
        t = translate_net(z1)

        # Transform
        y1 = z1
        y2 = z2 * np.exp(s) + t
        y = np.concatenate([y1, y2])

        # Log determinant
        log_det = np.sum(s)

        return y, log_det

    def real_nvp(self, z: np.ndarray, masks: List[np.ndarray],
                 scale_networks: List[callable], translate_networks: List[callable]) -> Tuple[np.ndarray, float]:
        """
        Real NVP (Real-valued Non-Volume Preserving) flow

        Args:
            z: Input
            masks: Binary masks for partitioning
            scale_networks: Scale networks
            translate_networks: Translation networks

        Returns:
            (transformed_z, total_log_det)
        """
        x = z.copy()
        total_log_det = 0

        for mask, scale_net, translate_net in zip(masks, scale_networks, translate_networks):
            # Apply affine coupling
            x, log_det = self.affine_coupling(x, scale_net, translate_net)
            total_log_det += log_det

            # Permute (for expressiveness)
            x = x[mask]

        return x, total_log_det

    def flow_density(self, x: np.ndarray, flow_layers: List[callable]) -> float:
        """
        Compute density of flow model

        Args:
            x: Input point
            flow_layers: List of flow transformations

        Returns:
            log probability density
        """
        # Transform through flow
        z = x.copy()
        total_log_det = 0

        for flow_layer in flow_layers:
            z, log_det = flow_layer(z)
            total_log_det += log_det

        # Evaluate base distribution
        log_p_z = self.base_log_prob(z)
        log_p_x = log_p_z + total_log_det

        return log_p_x

    def train_flow(self, X: np.ndarray, flow_layers: List[callable],
                  n_epochs: int = 1000, learning_rate: float = 0.001) -> List[float]:
        """
        Train normalizing flow model

        Args:
            X: Training data
            flow_layers: Flow transformations
            n_epochs: Training epochs
            learning_rate: Learning rate

        Returns:
            Training loss history
        """
        # This is a simplified training procedure
        # In practice, you'd use proper optimization libraries

        loss_history = []

        for epoch in range(n_epochs):
            total_loss = 0

            for x in X:
                # Forward pass
                log_p_x = self.flow_density(x, flow_layers)
                loss = -log_p_x  # Negative log likelihood

                # Backward pass (simplified)
                # In practice, you'd compute gradients and update parameters

                total_loss += loss

            avg_loss = total_loss / len(X)
            loss_history.append(avg_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return loss_history
```

### 5.2 Energy-Based Models

**Energy-Based Model Theory**
```
p(x) = exp(-E(x)) / Z

Where:
- E(x): Energy function
- Z: Partition function (normalization constant)
```

```python
class EnergyBasedModel:
    """Energy-based model implementations"""

    def __init__(self, energy_function: callable):
        self.energy_function = energy_function

    def boltzmann_distribution(self, x: np.ndarray, temperature: float = 1.0) -> float:
        """
        Boltzmann distribution

        Args:
            x: Input point
            temperature: Temperature parameter

        Returns:
            Probability density (unnormalized)
        """
        energy = self.energy_function(x)
        return np.exp(-energy / temperature)

    def langevin_dynamics(self, initial_x: np.ndarray, n_steps: int = 1000,
                         step_size: float = 0.01, temperature: float = 1.0) -> np.ndarray:
        """
        Langevin dynamics for sampling

        Args:
            initial_x: Initial state
            n_steps: Number of steps
            step_size: Step size
            temperature: Temperature

        Returns:
            Sampled points
        """
        x = initial_x.copy()
        samples = [x.copy()]

        for _ in range(n_steps):
            # Compute gradient of energy
            gradient = self._compute_energy_gradient(x)

            # Langevin update
            noise = np.random.randn(len(x)) * np.sqrt(2 * step_size * temperature)
            x = x - step_size * gradient + noise

            samples.append(x.copy())

        return np.array(samples)

    def contrastive_divergence(self, visible_data: np.ndarray, hidden_dim: int,
                             n_iterations: int = 1000, learning_rate: float = 0.01,
                             k_cd: int = 1) -> Dict:
        """
        Contrastive Divergence for training (simplified RBM)

        Args:
            visible_data: Training data
            hidden_dim: Number of hidden units
            n_iterations: Training iterations
            learning_rate: Learning rate
            k_cd: CD-k parameter

        Returns:
            Trained model parameters
        """
        n_samples, visible_dim = visible_data.shape

        # Initialize parameters
        W = np.random.randn(visible_dim, hidden_dim) * 0.01
        b_visible = np.zeros(visible_dim)
        b_hidden = np.zeros(hidden_dim)

        loss_history = []

        for iteration in range(n_iterations):
            total_loss = 0

            for v_data in visible_data:
                # Positive phase
                h_prob = self._sigmoid(np.dot(v_data, W) + b_hidden)
                h_data = (np.random.rand(hidden_dim) < h_prob).astype(float)

                # Negative phase (CD-k)
                v_model = v_data.copy()
                for _ in range(k_cd):
                    h_prob = self._sigmoid(np.dot(v_model, W) + b_hidden)
                    h_model = (np.random.rand(hidden_dim) < h_prob).astype(float)

                    v_prob = self._sigmoid(np.dot(h_model, W.T) + b_visible)
                    v_model = (np.random.rand(visible_dim) < v_prob).astype(float)

                # Update parameters
                positive_grad = np.outer(v_data, h_data)
                negative_grad = np.outer(v_model, h_model)

                W += learning_rate * (positive_grad - negative_grad)
                b_visible += learning_rate * (v_data - v_model)
                b_hidden += learning_rate * (h_data - h_model)

                # Compute reconstruction error
                reconstruction = self._sigmoid(np.dot(h_data, W.T) + b_visible)
                loss = np.mean((v_data - reconstruction)**2)
                total_loss += loss

            avg_loss = total_loss / len(visible_data)
            loss_history.append(avg_loss)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {avg_loss:.4f}")

        return {
            'weights': W,
            'visible_bias': b_visible,
            'hidden_bias': b_hidden,
            'loss_history': loss_history
        }

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def _compute_energy_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of energy function (finite differences)"""
        epsilon = 1e-6
        gradient = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i] += epsilon
            x_minus[i] -= epsilon

            energy_plus = self.energy_function(x_plus)
            energy_minus = self.energy_function(x_minus)

            gradient[i] = (energy_plus - energy_minus) / (2 * epsilon)

        return gradient
```

## Conclusion

This comprehensive guide to probabilistic generative models provides the mathematical foundations and practical implementations for:

1. **Fundamental Concepts**: Density estimation, information theory, and MLE
2. **Variational Inference**: ELBO optimization and stochastic methods
3. **Monte Carlo Methods**: Importance sampling, MCMC, and advanced techniques
4. **Advanced Models**: Normalizing flows and energy-based models

These mathematical foundations are essential for understanding modern generative AI techniques including VAEs, GANs, diffusion models, and transformer-based generative models. The theoretical framework provided here serves as the basis for more complex generative systems and enables rigorous analysis and improvement of generative AI algorithms.