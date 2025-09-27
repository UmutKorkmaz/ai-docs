# Numerical Methods Applications in Machine Learning

## Overview

This practical guide demonstrates how numerical methods are applied in real machine learning scenarios. We'll implement complete workflows showing the practical importance of numerical stability, efficient algorithms, and computational considerations.

## 1. Robust Linear Regression with Numerical Stability

### 1.1 Problem: Solving the Normal Equations

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve, lstsq, svd, pinv, qr
from scipy.optimize import minimize, least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def robust_linear_regression_demo():
    """Demonstrate numerical issues in linear regression and robust solutions"""
    print("=== Robust Linear Regression with Numerical Stability ===")

    # Create a dataset with potential numerical issues
    np.random.seed(42)
    n_samples = 100
    n_features = 50  # High dimensional, potential multicollinearity

    # Generate correlated features
    base_features = np.random.randn(n_samples, 5)
    X = np.column_stack([base_features + 0.01 * np.random.randn(n_samples, 5) for _ in range(10)])
    true_beta = np.random.randn(50)
    true_beta[5:] = 0  # Sparse true coefficients

    y = X @ true_beta + 0.5 * np.random.randn(n_samples)

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Condition number of X: {np.linalg.cond(X):.2e}")
    print(f"Rank of X: {np.linalg.matrix_rank(X)}/{n_features}")

    # Method 1: Normal equations (naive approach)
    print("\n=== Method 1: Normal Equations ===")

    try:
        beta_normal = np.linalg.solve(X.T @ X, X.T @ y)
        error_normal = np.linalg.norm(X @ beta_normal - y)
        print(f"Success: Error = {error_normal:.4f}")
    except np.linalg.LinAlgError:
        print("Failed: Matrix is singular!")
        beta_normal = None

    # Method 2: QR decomposition
    print("\n=== Method 2: QR Decomposition ===")

    Q, R = qr(X, mode='economic')
    beta_qr = np.linalg.solve(R, Q.T @ y)
    error_qr = np.linalg.norm(X @ beta_qr - y)
    print(f"QR solution: Error = {error_qr:.4f}")

    # Method 3: SVD (pseudoinverse)
    print("\n=== Method 3: SVD/Pseudoinverse ===")

    U, s, Vt = svd(X, full_matrices=False)
    rank = np.sum(s > 1e-10)
    print(f"Numerical rank: {rank}")

    # Truncated SVD for regularization
    k = min(rank, 40)  # Use top k singular values
    S_inv = np.zeros(len(s))
    S_inv[:k] = 1.0 / s[:k]
    X_pinv = Vt.T @ np.diag(S_inv) @ U.T

    beta_svd = X_pinv @ y
    error_svd = np.linalg.norm(X @ beta_svd - y)
    print(f"SVD solution (rank {k}): Error = {error_svd:.4f}")

    # Method 4: Ridge regression (regularized)
    print("\n=== Method 4: Ridge Regression ===")

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    for alpha in alphas:
        X_reg = X.T @ X + alpha * np.eye(n_features)
        try:
            beta_ridge = np.linalg.solve(X_reg, X.T @ y)
            error_ridge = np.linalg.norm(X @ beta_ridge - y)
            cond_reg = np.linalg.cond(X_reg)
            print(f"Ridge (α={alpha}): Error = {error_ridge:.4f}, Cond = {cond_reg:.2e}")
        except np.linalg.LinAlgError:
            print(f"Ridge (α={alpha}): Failed to solve!")

    # Method 5: Iterative methods (Conjugate Gradient)
    print("\n=== Method 5: Conjugate Gradient ===")

    from scipy.sparse.linalg import cg

    def cg_callback(xk):
        """Callback to track convergence"""
        if hasattr(cg_callback, 'residuals'):
            cg_callback.residuals.append(np.linalg.norm(X.T @ (X @ xk - y)))
        else:
            cg_callback.residuals = [np.linalg.norm(X.T @ (X @ xk - y))]

    cg_callback.residuals = []

    beta_cg, info = cg(X.T @ X, X.T @ y, callback=cg_callback, tol=1e-10)
    error_cg = np.linalg.norm(X @ beta_cg - y)
    print(f"CG solution: Error = {error_cg:.4f}, Iterations = {info}")
    print(f"CG convergence: {len(cg_callback.residuals)} iterations")

    # Comparison
    print("\n=== Method Comparison ===")

    methods = {
        'QR': beta_qr,
        'SVD': beta_svd,
        'CG': beta_cg
    }

    if beta_normal is not None:
        methods['Normal'] = beta_normal

    print(f"{'Method':<8} {'Error':>10} {'||β||':>10} {'Non-zero':>9}")
    print("-" * 45)

    for name, beta in methods.items():
        error = np.linalg.norm(X @ beta - y)
        norm_beta = np.linalg.norm(beta)
        non_zero = np.sum(np.abs(beta) > 1e-10)
        print(f"{name:<8} {error:>10.4f} {norm_beta:>10.4f} {non_zero:>9}")

    # Cross-validation for optimal regularization
    print("\n=== Cross-validation for Regularization ===")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def ridge_cv(X, y, alphas, cv=5):
        """Cross-validation for ridge regression"""
        n = len(y)
        fold_size = n // cv
        cv_errors = []

        for alpha in alphas:
            fold_errors = []

            for i in range(cv):
                # Split data
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < cv - 1 else n

                val_idx = np.arange(start_idx, end_idx)
                train_idx = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, n)])

                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Fit ridge
                XTX = X_train_fold.T @ X_train_fold + alpha * np.eye(X.shape[1])
                try:
                    beta_fold = np.linalg.solve(XTX, X_train_fold.T @ y_train_fold)
                    pred_val = X_val_fold @ beta_fold
                    fold_errors.append(np.mean((pred_val - y_val_fold)**2))
                except:
                    fold_errors.append(np.inf)

            cv_errors.append(np.mean(fold_errors))

        return np.array(cv_errors)

    alpha_range = np.logspace(-4, 2, 20)
    cv_errors = ridge_cv(X_train, y_train, alpha_range)

    best_alpha = alpha_range[np.argmin(cv_errors)]
    print(f"Best alpha: {best_alpha:.4f}")

    # Final model with best regularization
    X_final = X_train.T @ X_train + best_alpha * np.eye(n_features)
    beta_final = np.linalg.solve(X_final, X_train.T @ y_train)

    train_pred = X_train @ beta_final
    test_pred = X_test @ beta_final

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nFinal model performance:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Cross-validation curve
    ax1.semilogx(alpha_range, cv_errors, 'bo-')
    ax1.axvline(best_alpha, color='r', linestyle='--', label=f'Best α = {best_alpha:.4f}')
    ax1.set_xlabel('Regularization parameter α')
    ax1.set_ylabel('CV Error')
    ax1.set_title('Ridge Regression Cross-validation')
    ax1.legend()
    ax1.grid(True)

    # Coefficient comparison
    if beta_normal is not None:
        ax2.plot(range(n_features), beta_normal, 'b-', alpha=0.7, label='Normal')
    ax2.plot(range(n_features), beta_qr, 'r-', alpha=0.7, label='QR')
    ax2.plot(range(n_features), beta_svd, 'g-', alpha=0.7, label='SVD')
    ax2.plot(range(n_features), beta_final, 'k-', linewidth=2, label='Ridge (optimal)')
    ax2.set_xlabel('Feature index')
    ax2.set_ylabel('Coefficient value')
    ax2.set_title('Coefficient Comparison')
    ax2.legend()
    ax2.grid(True)

    # CG convergence
    ax3.semilogy(range(len(cg_callback.residuals)), cg_callback.residuals, 'bo-')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Residual norm')
    ax3.set_title('Conjugate Gradient Convergence')
    ax3.grid(True)

    # Prediction vs actual
    ax4.scatter(y_test, test_pred, alpha=0.6)
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax4.set_xlabel('Actual values')
    ax4.set_ylabel('Predicted values')
    ax4.set_title(f'Test Predictions (R² = {test_r2:.3f})')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'beta_qr': beta_qr,
        'beta_svd': beta_svd,
        'beta_cg': beta_cg,
        'beta_final': beta_final,
        'best_alpha': best_alpha,
        'test_r2': test_r2
    }

robust_linear_regression_demo()
```

## 2. Efficient PCA Implementation

### 2.1 Comparing PCA Implementation Methods

```python
def efficient_pca_implementation():
    """Compare different PCA implementations for efficiency and accuracy"""
    print("\n=== Efficient PCA Implementation ===")

    # Generate large dataset
    np.random.seed(42)
    n_samples = 5000
    n_features = 100

    # Create data with known structure
    latent_factors = np.random.randn(n_samples, 10)
    noise = 0.1 * np.random.randn(n_samples, n_features)

    # Create loading matrix
    loading_matrix = np.random.randn(10, n_features)
    loading_matrix = loading_matrix / np.linalg.norm(loading_matrix, axis=0)

    X = latent_factors @ loading_matrix + noise

    # Standardize
    X_centered = X - X.mean(axis=0)
    X_std = X_centered / X_centered.std(axis=0)

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Memory usage: {X.nbytes / 1024**2:.1f} MB")

    # Method 1: Full SVD
    print("\n=== Method 1: Full SVD ===")

    import time

    start_time = time.time()
    U_full, s_full, Vt_full = np.linalg.svd(X_std, full_matrices=False)
    full_time = time.time() - start_time

    # Variance explained
    var_explained = s_full**2 / np.sum(s_full**2)
    cum_var = np.cumsum(var_explained)

    print(f"Full SVD time: {full_time:.3f} seconds")
    print(f"Variance explained by top 10 PCs: {cum_var[9]:.3f}")

    # Method 2: Truncated SVD
    print("\n=== Method 2: Truncated SVD ===")

    from scipy.sparse.linalg import svds

    n_components = 10

    start_time = time.time()
    U_trunc, s_trunc, Vt_trunc = svds(X_std, k=n_components)
    trunc_time = time.time() - start_time

    # Sort singular values (svds doesn't guarantee order)
    sort_idx = np.argsort(s_trunc)[::-1]
    s_trunc = s_trunc[sort_idx]
    U_trunc = U_trunc[:, sort_idx]
    Vt_trunc = Vt_trunc[sort_idx, :]

    var_explained_trunc = s_trunc**2 / np.sum(s_full**2)

    print(f"Truncated SVD time: {trunc_time:.3f} seconds")
    print(f"Variance explained: {np.sum(var_explained_trunc):.3f}")

    # Method 3: Power iteration
    print("\n=== Method 3: Power Iteration ===")

    def power_iteration(X, n_components, max_iter=100, tol=1e-10):
        """Power iteration for top eigenvectors"""
        n_samples, n_features = X.shape
        V = np.random.randn(n_features, n_components)
        V, _ = np.linalg.qr(V)

        eigenvalues = []

        for i in range(max_iter):
            V_old = V.copy()

            # Power iteration
            AV = X.T @ (X @ V)
            V, R = np.linalg.qr(AV)

            # Eigenvalue estimates
            eigenvalues = np.diag(R)

            # Check convergence
            if np.linalg.norm(V - V_old) < tol:
                break

        return V, np.array(eigenvalues)

    start_time = time.time()
    V_power, eig_power = power_iteration(X_std, n_components)
    power_time = time.time() - start_time

    var_explained_power = eig_power**2 / np.sum(s_full**2)

    print(f"Power iteration time: {power_time:.3f} seconds")
    print(f"Iterations: 100")
    print(f"Variance explained: {np.sum(var_explained_power):.3f}")

    # Method 4: Randomized SVD
    print("\n=== Method 4: Randomized SVD ===")

    def randomized_svd(X, n_components, n_oversamples=10, n_iter=2):
        """Randomized SVD algorithm"""
        n_samples, n_features = X.shape

        # Stage A: Find range of X
        l = n_components + n_oversamples

        # Generate random matrix
        Omega = np.random.randn(n_features, l)

        # Form Y = X * Omega
        Y = X @ Omega

        # Power iterations
        for i in range(n_iter):
            Y = X @ (X.T @ Y)
            Y, _ = np.linalg.qr(Y)

        # Form Q
        Q, _ = np.linalg.qr(Y)

        # Stage B: SVD of Q'X
        B = Q.T @ X
        U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)

        # Recover U
        U = Q @ U_tilde

        return U[:, :n_components], s[:n_components], Vt[:n_components, :]

    start_time = time.time()
    U_rand, s_rand, Vt_rand = randomized_svd(X_std, n_components)
    rand_time = time.time() - start_time

    var_explained_rand = s_rand**2 / np.sum(s_full**2)

    print(f"Randomized SVD time: {rand_time:.3f} seconds")
    print(f"Variance explained: {np.sum(var_explained_rand):.3f}")

    # Method 5: Incremental PCA
    print("\n=== Method 5: Incremental PCA ===")

    def incremental_pca(X, n_components, batch_size=100, max_iter=10):
        """Incremental PCA implementation"""
        n_samples, n_features = X.shape

        # Initialize
        components = np.random.randn(n_components, n_features)
        components = components / np.linalg.norm(components, axis=1, keepdims=True)

        explained_variance = np.zeros(n_components)

        for iteration in range(max_iter):
            # Shuffle data
            idx = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_idx = idx[i:batch_end]
                X_batch = X[batch_idx]

                # Project batch
                projected = X_batch @ components.T

                # Reconstruct
                reconstructed = projected @ components

                # Compute residual
                residual = X_batch - reconstructed

                # Update components (simplified)
                if i == 0:
                    explained_variance = np.var(projected, axis=0)
                else:
                    explained_variance = 0.9 * explained_variance + 0.1 * np.var(projected, axis=0)

        return components, explained_variance

    start_time = time.time()
    comp_inc, var_inc = incremental_pca(X_std, n_components, batch_size=500)
    inc_time = time.time() - start_time

    print(f"Incremental PCA time: {inc_time:.3f} seconds")
    print(f"Variance explained: {np.sum(var_inc) / np.sum(s_full**2):.3f}")

    # Comparison
    print("\n=== Method Comparison ===")

    results = {
        'Full SVD': {'time': full_time, 'var_explained': cum_var[9], 'accuracy': 1.0},
        'Truncated SVD': {'time': trunc_time, 'var_explained': np.sum(var_explained_trunc), 'accuracy': 1.0},
        'Power Iteration': {'time': power_time, 'var_explained': np.sum(var_explained_power), 'accuracy': 0.95},
        'Randomized SVD': {'time': rand_time, 'var_explained': np.sum(var_explained_rand), 'accuracy': 0.98},
        'Incremental PCA': {'time': inc_time, 'var_explained': np.sum(var_inc) / np.sum(s_full**2), 'accuracy': 0.90}
    }

    print(f"{'Method':<15} {'Time (s)':>10} {'Var Explained':>12} {'Speedup':>10}")
    print("-" * 55)

    baseline_time = full_time
    for method, result in results.items():
        speedup = baseline_time / result['time']
        print(f"{method:<15} {result['time']:>10.3f} {result['var_explained']:>12.3f} {speedup:>10.1f}x")

    # Memory usage comparison
    print("\n=== Memory Usage Analysis ===")

    def memory_usage_pca(X, method, n_components):
        """Estimate memory usage for different PCA methods"""
        n_samples, n_features = X.shape

        if method == 'full_svd':
            # Need to store U, S, Vt
            return (n_samples * n_components + n_components + n_features * n_components) * 8 / 1024**2

        elif method == 'truncated_svd':
            # Only store truncated components
            return (n_samples * n_components + n_components + n_features * n_components) * 8 / 1024**2

        elif method == 'incremental':
            # Store components and current batch
            return (n_features * n_components + 500 * n_features) * 8 / 1024**2

        elif method == 'randomized':
            # Store random matrix and components
            return (n_features * (n_components + 10) + n_features * n_components) * 8 / 1024**2

    print(f"{'Method':<15} {'Memory (MB)':>12}")
    print("-" * 30)

    for method in ['full_svd', 'truncated_svd', 'incremental', 'randomized']:
        mem = memory_usage_pca(X_std, method, n_components)
        print(f"{method:<15} {mem:>12.1f}")

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Time comparison
    methods = list(results.keys())
    times = [results[m]['time'] for m in methods]
    ax1.bar(methods, times)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # Variance explained
    var_explained = [results[m]['var_explained'] for m in methods]
    ax2.bar(methods, var_explained)
    ax2.set_ylabel('Variance Explained')
    ax2.set_title('Variance Explained Comparison')
    ax2.tick_params(axis='x', rotation=45)

    # Scree plot
    ax3.plot(range(1, min(20, len(s_full)) + 1), cum_var[:20], 'bo-')
    ax3.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax3.set_xlabel('Number of components')
    ax3.set_ylabel('Cumulative variance explained')
    ax3.set_title('Scree Plot')
    ax3.legend()
    ax3.grid(True)

    # 2D projection
    X_pca_full = X_std @ Vt_full[:2, :].T
    X_pca_rand = X_std @ Vt_rand[:2, :].T

    ax4.scatter(X_pca_full[:, 0], X_pca_full[:, 1], alpha=0.6, label='Full SVD')
    ax4.scatter(X_pca_rand[:, 0], X_pca_rand[:, 1], alpha=0.6, label='Randomized SVD')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('2D Projection Comparison')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'full_svd': (U_full, s_full, Vt_full),
        'truncated': (U_trunc, s_trunc, Vt_trunc),
        'randomized': (U_rand, s_rand, Vt_rand),
        'incremental': (comp_inc, var_inc)
    }

efficient_pca_implementation()
```

## 3. Optimization in Neural Networks

### 3.1 Implementing Custom Optimizers

```python
def neural_network_optimization():
    """Implement and compare neural network optimizers from scratch"""
    print("\n=== Neural Network Optimization Methods ===")

    # Simple neural network implementation
    class SimpleNeuralNetwork:
        def __init__(self, layer_sizes, activation='relu'):
            self.layer_sizes = layer_sizes
            self.activation = activation
            self.weights = []
            self.biases = []

            # Initialize weights
            for i in range(len(layer_sizes) - 1):
                # He initialization
                scale = np.sqrt(2.0 / layer_sizes[i])
                W = scale * np.random.randn(layer_sizes[i], layer_sizes[i+1])
                b = np.zeros(layer_sizes[i+1])
                self.weights.append(W)
                self.biases.append(b)

        def forward(self, X):
            """Forward propagation"""
            self.activations = [X]
            self.z_values = []

            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                z = self.activations[-1] @ W + b
                self.z_values.append(z)

                if i == len(self.weights) - 1:  # Output layer
                    a = z  # Linear activation for regression
                else:
                    if self.activation == 'relu':
                        a = np.maximum(0, z)
                    elif self.activation == 'sigmoid':
                        a = 1 / (1 + np.exp(-z))
                    elif self.activation == 'tanh':
                        a = np.tanh(z)

                self.activations.append(a)

            return self.activations[-1]

        def backward(self, X, y):
            """Backward propagation"""
            n_samples = X.shape[0]
            self.gradients_W = []
            self.gradients_b = []

            # Output layer gradient
            delta = self.activations[-1] - y

            for i in reversed(range(len(self.weights))):
                # Gradient for weights and biases
                grad_W = self.activations[i].T @ delta / n_samples
                grad_b = np.mean(delta, axis=0)

                self.gradients_W.insert(0, grad_W)
                self.gradients_b.insert(0, grad_b)

                # Backpropagate delta
                if i > 0:
                    if self.activation == 'relu':
                        grad_activation = (self.z_values[i-1] > 0).astype(float)
                    elif self.activation == 'sigmoid':
                        a = self.activations[i]
                        grad_activation = a * (1 - a)
                    elif self.activation == 'tanh':
                        a = self.activations[i]
                        grad_activation = 1 - a**2

                    delta = (delta @ self.weights[i].T) * grad_activation

            return self.gradients_W, self.gradients_b

        def get_parameters(self):
            """Get all parameters as a single vector"""
            params = []
            for W, b in zip(self.weights, self.biases):
                params.extend([W.flatten(), b])
            return np.concatenate(params)

        def set_parameters(self, params):
            """Set parameters from a single vector"""
            idx = 0
            for i in range(len(self.weights)):
                W_size = self.weights[i].size
                b_size = self.biases[i].size

                self.weights[i] = params[idx:idx+W_size].reshape(self.weights[i].shape)
                idx += W_size

                self.biases[i] = params[idx:idx+b_size]
                idx += b_size

        def get_gradients(self):
            """Get all gradients as a single vector"""
            grads = []
            for gW, gb in zip(self.gradients_W, self.gradients_b):
                grads.extend([gW.flatten(), gb])
            return np.concatenate(grads)

    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Non-linear function: sin(x1) + x2^2 + noise
    y = np.sin(X[:, 0]) + X[:, 1]**2 + 0.1 * np.random.randn(n_samples)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Define optimizers
    class SGDOptimizer:
        def __init__(self, learning_rate=0.01):
            self.lr = learning_rate

        def update(self, params, grads):
            return params - self.lr * grads

    class MomentumOptimizer:
        def __init__(self, learning_rate=0.01, momentum=0.9):
            self.lr = learning_rate
            self.momentum = momentum
            self.velocity = None

        def update(self, params, grads):
            if self.velocity is None:
                self.velocity = np.zeros_like(params)

            self.velocity = self.momentum * self.velocity + self.lr * grads
            return params - self.velocity

    class AdamOptimizer:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.lr = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = None
            self.v = None
            self.t = 0

        def update(self, params, grads):
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)

            self.t += 1

            self.m = self.beta1 * self.m + (1 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)

            return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    class RMSpropOptimizer:
        def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
            self.lr = learning_rate
            self.beta = beta
            self.epsilon = epsilon
            self.v = None

        def update(self, params, grads):
            if self.v is None:
                self.v = np.zeros_like(params)

            self.v = self.beta * self.v + (1 - self.beta) * grads**2
            return params - self.lr * grads / (np.sqrt(self.v) + self.epsilon)

    # Training function
    def train_network(nn, optimizer, X_train, y_train, X_test, y_test,
                     n_epochs=100, batch_size=32, print_every=20):
        """Train neural network with given optimizer"""
        train_losses = []
        test_losses = []

        for epoch in range(n_epochs):
            # Shuffle training data
            idx = np.random.permutation(len(X_train))

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward pass
                predictions = nn.forward(X_batch)

                # Backward pass
                nn.backward(X_batch, y_batch)

                # Get parameters and gradients
                params = nn.get_parameters()
                grads = nn.get_gradients()

                # Update parameters
                new_params = optimizer.update(params, grads)
                nn.set_parameters(new_params)

            # Evaluate
            if epoch % print_every == 0:
                train_pred = nn.forward(X_train)
                test_pred = nn.forward(X_test)

                train_loss = np.mean((train_pred - y_train)**2)
                test_loss = np.mean((test_pred - y_test)**2)

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        return train_losses, test_losses

    # Create network
    layer_sizes = [n_features, 20, 10, 1]
    nn_base = SimpleNeuralNetwork(layer_sizes, activation='relu')

    print(f"\nNetwork architecture: {layer_sizes}")
    print(f"Total parameters: {sum(w.size + b.size for w, b in zip(nn_base.weights, nn_base.biases))}")

    # Test different optimizers
    optimizers = {
        'SGD': SGDOptimizer(learning_rate=0.01),
        'Momentum': MomentumOptimizer(learning_rate=0.01, momentum=0.9),
        'Adam': AdamOptimizer(learning_rate=0.001),
        'RMSprop': RMSpropOptimizer(learning_rate=0.001)
    }

    results = {}

    for name, optimizer in optimizers.items():
        print(f"\n=== Training with {name} ===")

        # Create fresh network
        nn = SimpleNeuralNetwork(layer_sizes, activation='relu')

        # Train
        train_losses, test_losses = train_network(
            nn, optimizer, X_train, y_train, X_test, y_test,
            n_epochs=200, batch_size=32, print_every=40
        )

        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_test_loss': test_losses[-1],
            'network': nn
        }

        final_pred = nn.forward(X_test)
        final_mse = np.mean((final_pred - y_test)**2)
        final_r2 = r2_score(y_test, final_pred)

        print(f"Final Test MSE: {final_mse:.4f}")
        print(f"Final Test R²: {final_r2:.4f}")

    # Compare results
    print("\n=== Optimizer Comparison ===")

    print(f"{'Optimizer':<10} {'Final Test Loss':>15} {'Best Epoch':>12}")
    print("-" * 45)

    for name, result in results.items():
        best_epoch = np.argmin(result['test_losses']) * 40
        print(f"{name:<10} {result['final_test_loss']:>15.4f} {best_epoch:>12}")

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Training curves
    epochs = np.arange(0, 200, 40)
    for name, result in results.items():
        ax1.plot(epochs, result['train_losses'], 'o-', label=f'{name} (Train)')
        ax2.plot(epochs, result['test_losses'], 'o-', label=f'{name} (Test)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)

    # Final predictions
    best_optimizer = min(results.keys(), key=lambda x: results[x]['final_test_loss'])
    best_nn = results[best_optimizer]['network']
    predictions = best_nn.forward(X_test)

    ax3.scatter(y_test, predictions, alpha=0.6)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax3.set_xlabel('True values')
    ax3.set_ylabel('Predicted values')
    ax3.set_title(f'Best Model ({best_optimizer}) Predictions')
    ax3.grid(True)

    # Learning rate sensitivity analysis
    print("\n=== Learning Rate Sensitivity Analysis ===")

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    lr_results = []

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")

        nn = SimpleNeuralNetwork(layer_sizes, activation='relu')
        optimizer = AdamOptimizer(learning_rate=lr)

        train_losses, test_losses = train_network(
            nn, optimizer, X_train, y_train, X_test, y_test,
            n_epochs=100, batch_size=32, print_every=100
        )

        lr_results.append({
            'lr': lr,
            'final_loss': test_losses[-1]
        })

        print(f"Final test loss: {test_losses[-1]:.4f}")

    # Plot learning rate sensitivity
    lrs = [r['lr'] for r in lr_results]
    losses = [r['final_loss'] for r in lr_results]

    ax4.semilogx(lrs, losses, 'bo-')
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Final Test Loss')
    ax4.set_title('Learning Rate Sensitivity')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return results

neural_network_optimization()
```

## 4. Numerical Methods for Recommender Systems

### 4.1 Matrix Factorization with Alternating Least Squares

```python
def recommender_system_als():
    """Implement ALS for collaborative filtering with numerical considerations"""
    print("\n=== Recommender System with ALS ===")

    # Generate synthetic rating data
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_factors = 5

    # True latent factors
    U_true = np.random.randn(n_users, n_factors)
    V_true = np.random.randn(n_items, n_factors)

    # Bias terms
    user_bias = np.random.randn(n_users) * 0.5
    item_bias = np.random.randn(n_items) * 0.5
    global_bias = 3.0

    # Generate ratings
    R_true = global_bias + user_bias[:, np.newaxis] + item_bias[np.newaxis, :] + U_true @ V_true.T

    # Add noise and clip ratings
    R = R_true + 0.5 * np.random.randn(n_users, n_items)
    R = np.clip(R, 1, 5)

    # Create mask (simulate sparse observations)
    sparsity = 0.9  # 90% missing
    mask = np.random.rand(n_users, n_items) > sparsity

    R_observed = R.copy()
    R_observed[~mask] = np.nan

    print(f"Dataset: {n_users} users, {n_items} items")
    print(f"Sparsity: {100*sparsity:.1f}% missing")
    print(f"Observed ratings: {np.sum(mask)}/{n_users*n_items}")

    # ALS implementation with numerical stability
    def als_matrix_factorization(R, mask, n_factors, lambda_reg=0.1, max_iter=50, tol=1e-6):
        """Alternating Least Squares with numerical stability"""
        n_users, n_items = R.shape

        # Initialize with SVD on filled matrix
        R_filled = R.copy()
        R_filled[~mask] = np.nanmean(R[mask])
        U_init, s_init, Vt_init = np.linalg.svd(R_filled, full_matrices=False)
        U = U_init[:, :n_factors] * np.sqrt(s_init[:n_factors])
        V = Vt_init[:n_factors, :].T * np.sqrt(s_init[:n_factors])

        # Initialize biases
        user_bias = np.zeros(n_users)
        item_bias = np.zeros(n_items)
        global_bias = np.nanmean(R[mask])

        errors = []

        for iteration in range(max_iter):
            # Store previous values for convergence check
            U_old = U.copy()
            V_old = V.copy()

            # Update user factors
            for u in range(n_users):
                # Get observed items for this user
                observed_items = mask[u]
                if np.sum(observed_items) > 0:
                    V_obs = V[observed_items]
                    R_obs = R[u, observed_items]

                    # Remove biases
                    R_adj = R_obs - global_bias - user_bias[u] - item_bias[observed_items]

                    # Solve regularized least squares
                    A = V_obs.T @ V_obs + lambda_reg * np.eye(n_factors)
                    b = V_obs.T @ R_adj

                    # Check condition number
                    cond_A = np.linalg.cond(A)
                    if cond_A > 1e10:
                        # Use regularization
                        A = V_obs.T @ V_obs + 10 * lambda_reg * np.eye(n_factors)

                    try:
                        U[u] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        # Use pseudo-inverse if singular
                        U[u] = np.linalg.pinv(A) @ b

            # Update item factors
            for i in range(n_items):
                # Get observed users for this item
                observed_users = mask[:, i]
                if np.sum(observed_users) > 0:
                    U_obs = U[observed_users]
                    R_obs = R[observed_users, i]

                    # Remove biases
                    R_adj = R_obs - global_bias - user_bias[observed_users] - item_bias[i]

                    # Solve regularized least squares
                    A = U_obs.T @ U_obs + lambda_reg * np.eye(n_factors)
                    b = U_obs.T @ R_adj

                    try:
                        V[i] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        V[i] = np.linalg.pinv(A) @ b

            # Update biases
            for u in range(n_users):
                observed_items = mask[u]
                if np.sum(observed_items) > 0:
                    residuals = R[u, observed_items] - global_bias - user_bias[u] - item_bias[observed_items]
                    residuals -= U[u] @ V[observed_items].T
                    user_bias[u] = np.mean(residuals)

            for i in range(n_items):
                observed_users = mask[:, i]
                if np.sum(observed_users) > 0:
                    residuals = R[observed_users, i] - global_bias - user_bias[observed_users] - item_bias[i]
                    residuals -= U[observed_users] @ V[i]
                    item_bias[i] = np.mean(residuals)

            # Compute error
            R_pred = global_bias + user_bias[:, np.newaxis] + item_bias[np.newaxis, :] + U @ V.T
            error = np.sqrt(np.sum(mask * (R - R_pred)**2) / np.sum(mask))
            errors.append(error)

            # Check convergence
            if iteration > 0 and abs(errors[-2] - errors[-1]) < tol:
                break

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: RMSE = {error:.4f}")

        return U, V, user_bias, item_bias, global_bias, errors

    # Run ALS
    print("\n=== Running ALS ===")
    U, V, user_bias, item_bias, global_bias, errors = als_matrix_factorization(
        R_observed, mask, n_factors, lambda_reg=0.1, max_iter=100
    )

    # Evaluate
    R_pred = global_bias + user_bias[:, np.newaxis] + item_bias[np.newaxis, :] + U @ V.T

    # Training error
    train_rmse = np.sqrt(np.sum(mask * (R - R_pred)**2) / np.sum(mask))

    # Test error (on missing entries)
    test_mask = ~mask
    if np.sum(test_mask) > 0:
        test_rmse = np.sqrt(np.sum(test_mask * (R - R_pred)**2) / np.sum(test_mask))
    else:
        test_rmse = 0

    print(f"\nResults:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Global bias: {global_bias:.4f}")

    # Compare with direct matrix completion
    print("\n=== Comparison with Direct Methods ===")

    def svd_matrix_completion(R, mask, n_factors):
        """Matrix completion using truncated SVD"""
        R_filled = R.copy()
        R_filled[~mask] = np.nanmean(R[mask])

        U_full, s_full, Vt_full = np.linalg.svd(R_filled, full_matrices=False)

        # Truncate
        U_trunc = U_full[:, :n_factors]
        s_trunc = s_full[:n_factors]
        Vt_trunc = Vt_full[:n_factors, :]

        R_pred = U_trunc @ np.diag(s_trunc) @ Vt_trunc

        return R_pred

    R_pred_svd = svd_matrix_completion(R_observed, mask, n_factors)

    svd_train_rmse = np.sqrt(np.sum(mask * (R - R_pred_svd)**2) / np.sum(mask))
    if np.sum(test_mask) > 0:
        svd_test_rmse = np.sqrt(np.sum(test_mask * (R - R_pred_svd)**2) / np.sum(test_mask))

    print(f"SVD Training RMSE: {svd_train_rmse:.4f}")
    print(f"SVD Test RMSE: {svd_test_rmse:.4f}")

    # Recommendation generation
    print("\n=== Generating Recommendations ===")

    def generate_recommendations(U, V, user_idx, top_k=10):
        """Generate recommendations for a user"""
        user_vector = U[user_idx]
        item_scores = user_vector @ V.T

        # Exclude already rated items
        rated_items = np.where(mask[user_idx])[0]

        # Get top recommendations
        unrated_items = np.setdiff1d(np.arange(n_items), rated_items)
        recommendations = unrated_items[np.argsort(item_scores[unrated_items])[-top_k:]]

        return recommendations, item_scores[recommendations]

    # Example recommendations
    user_idx = 0
    recommendations, scores = generate_recommendations(U, V, user_idx, top_k=5)

    print(f"\nRecommendations for user {user_idx}:")
    for i, (item, score) in enumerate(zip(recommendations, scores)):
        print(f"  {i+1}. Item {item}: predicted score = {score:.3f}")

    # Cold start problem demonstration
    print("\n=== Cold Start Problem ===")

    def cold_start_user_prediction(V, item_ratings, n_factors, lambda_reg=0.1):
        """Predict for new user using only item ratings"""
        observed_items = np.array(list(item_ratings.keys()))
        ratings = np.array(list(item_ratings.values()))

        if len(observed_items) >= n_factors:
            # Solve for user factors
            V_obs = V[observed_items]

            A = V_obs.T @ V_obs + lambda_reg * np.eye(n_factors)
            b = V_obs.T @ ratings

            try:
                user_factors = np.linalg.solve(A, b)
                # Predict all items
                all_predictions = user_factors @ V.T
                return all_predictions, user_factors
            except np.linalg.LinAlgError:
                return None, None
        else:
            return None, None

    # Simulate new user with few ratings
    new_user_ratings = {5: 4.0, 12: 5.0, 23: 3.0}  # Only 3 ratings

    predictions, user_factors = cold_start_user_prediction(V, new_user_ratings, n_factors)

    if predictions is not None:
        print(f"\nNew user with {len(new_user_ratings)} ratings:")
        print("Top recommendations:")
        top_items = np.argsort(predictions)[-5:][::-1]
        for item in top_items:
            if item not in new_user_ratings:
                print(f"  Item {item}: predicted rating = {predictions[item]:.3f}")

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Convergence
    ax1.plot(range(len(errors)), errors, 'b-o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.set_title('ALS Convergence')
    ax1.grid(True)

    # True vs predicted
    mask_flat = mask.flatten()
    true_ratings = R[mask_flat]
    pred_ratings = R_pred[mask_flat]

    ax2.scatter(true_ratings, pred_ratings, alpha=0.6)
    ax2.plot([1, 5], [1, 5], 'r--')
    ax2.set_xlabel('True Rating')
    ax2.set_ylabel('Predicted Rating')
    ax2.set_title('True vs Predicted Ratings')
    ax2.grid(True)

    # User factors visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    U_2d = pca.fit_transform(U)

    ax3.scatter(U_2d[:, 0], U_2d[:, 1], alpha=0.7)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('User Latent Factors (2D Projection)')
    ax3.grid(True)

    # Item factors visualization
    V_2d = pca.fit_transform(V)

    ax4.scatter(V_2d[:, 0], V_2d[:, 1], alpha=0.7)
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('Item Latent Factors (2D Projection)')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'U': U,
        'V': V,
        'user_bias': user_bias,
        'item_bias': item_bias,
        'global_bias': global_bias,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }

recommender_system_als()
```

## 5. Numerical Stability in Deep Learning

### 5.1 Vanishing/Exploding Gradients and Solutions

```python
def deep_learning_numerical_stability():
    """Demonstrate and solve numerical stability issues in deep learning"""
    print("\n=== Numerical Stability in Deep Learning ===")

    # Deep network with different initializations
    class DeepNetwork:
        def __init__(self, layer_sizes, init_method='xavier'):
            self.layer_sizes = layer_sizes
            self.init_method = init_method
            self.weights = []
            self.biases = []

            for i in range(len(layer_sizes) - 1):
                if init_method == 'xavier':
                    scale = np.sqrt(1.0 / layer_sizes[i])
                elif init_method == 'he':
                    scale = np.sqrt(2.0 / layer_sizes[i])
                elif init_method == 'random':
                    scale = 1.0
                elif init_method == 'orthogonal':
                    # Generate random matrix then orthogonalize
                    W = np.random.randn(layer_sizes[i], layer_sizes[i+1])
                    U, _, Vt = np.linalg.svd(W, full_matrices=False)
                    W = U @ Vt
                    scale = 1.0
                else:
                    scale = 0.01

                if init_method != 'orthogonal':
                    W = scale * np.random.randn(layer_sizes[i], layer_sizes[i+1])

                b = np.zeros(layer_sizes[i+1])

                self.weights.append(W)
                self.biases.append(b)

        def forward_with_activations(self, X):
            """Forward pass tracking all activations"""
            activations = [X]
            pre_activations = []

            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                z = activations[-1] @ W + b
                pre_activations.append(z)

                if i == len(self.weights) - 1:  # Output layer
                    a = z  # Linear
                else:
                    a = np.tanh(z)  # tanh activation

                activations.append(a)

            return activations, pre_activations

        def analyze_gradient_flow(self, X):
            """Analyze gradient flow through the network"""
            activations, pre_activations = self.forward_with_activations(X)

            # Compute gradient norms for each layer
            grad_norms = []

            # Start from output (gradient = 1 for simplicity)
            grad = 1.0

            for i in reversed(range(len(self.weights))):
                # Backprop through activation
                if i < len(self.weights) - 1:
                    # tanh derivative: 1 - tanh²(x)
                    tanh_grad = 1 - activations[i+1]**2
                    grad = grad * tanh_grad

                # Backprop through weights
                W = self.weights[i]
                grad_norm = np.linalg.norm(grad * W, 'fro')
                grad_norms.append(grad_norm)

                # Update gradient for next layer
                grad = grad @ W.T

            return list(reversed(grad_norms)), activations

    # Test different network depths and initializations
    depths = [5, 10, 20, 50]
    init_methods = ['xavier', 'he', 'random', 'orthogonal']

    X_test = np.random.randn(1, 10)  # Single sample

    print("Analyzing gradient flow for different configurations...")

    results = {}

    for depth in depths:
        print(f"\n=== Depth: {depth} layers ===")

        layer_sizes = [10] + [100] * (depth - 1) + [1]

        depth_results = {}

        for init_method in init_methods:
            try:
                nn = DeepNetwork(layer_sizes, init_method=init_method)
                grad_norms, activations = nn.analyze_gradient_flow(X_test)

                # Compute statistics
                activation_stats = []
                for i, act in enumerate(activations[1:-1]):  # Skip input and output
                    activation_stats.append({
                        'mean': np.mean(act),
                        'std': np.std(act),
                        'min': np.min(act),
                        'max': np.max(act)
                    })

                depth_results[init_method] = {
                    'grad_norms': grad_norms,
                    'activation_stats': activation_stats,
                    'final_activation_std': activation_stats[-1]['std'] if activation_stats else 0
                }

                print(f"  {init_method}: Final activation std = {depth_results[init_method]['final_activation_std']:.4f}")

            except Exception as e:
                print(f"  {init_method}: Failed - {str(e)}")

        results[depth] = depth_results

    # Visualization of gradient flow
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, depth in enumerate(depths):
        ax = axes[i]

        for init_method in init_methods:
            if init_method in results[depth]:
                grad_norms = results[depth][init_method]['grad_norms']
                layers = range(1, len(grad_norms) + 1)

                ax.semilogy(layers, grad_norms, 'o-', label=init_method)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'Gradient Flow (Depth = {depth})')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Batch Normalization demonstration
    print("\n=== Batch Normalization Demonstration ===")

    class BatchNormLayer:
        def __init__(self, n_features, momentum=0.9, epsilon=1e-5):
            self.gamma = np.ones(n_features)
            self.beta = np.zeros(n_features)
            self.momentum = momentum
            self.epsilon = epsilon
            self.running_mean = np.zeros(n_features)
            self.running_var = np.ones(n_features)
            self.training = True

        def forward(self, x):
            if self.training:
                batch_mean = np.mean(x, axis=0)
                batch_var = np.var(x, axis=0)

                # Update running statistics
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

                # Normalize
                x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            else:
                x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

            # Scale and shift
            out = self.gamma * x_normalized + self.beta

            return out

    # Compare networks with and without batch normalization
    class NetworkWithBatchNorm:
        def __init__(self, layer_sizes, use_batch_norm=True):
            self.use_batch_norm = use_batch_norm
            self.weights = []
            self.biases = []
            self.batch_norms = []

            for i in range(len(layer_sizes) - 1):
                # Xavier initialization
                scale = np.sqrt(1.0 / layer_sizes[i])
                W = scale * np.random.randn(layer_sizes[i], layer_sizes[i+1])
                b = np.zeros(layer_sizes[i+1])

                self.weights.append(W)
                self.biases.append(b)

                if use_batch_norm and i < len(layer_sizes) - 2:  # No BN on output
                    self.batch_norms.append(BatchNormLayer(layer_sizes[i+1]))
                else:
                    self.batch_norms.append(None)

        def forward(self, X):
            activations = [X]

            for i, (W, b, bn) in enumerate(zip(self.weights, self.biases, self.batch_norms)):
                z = activations[-1] @ W + b

                if i == len(self.weights) - 1:  # Output layer
                    a = z
                else:
                    a = np.tanh(z)
                    if bn is not None:
                        a = bn.forward(a)

                activations.append(a)

            return activations

    # Test batch normalization impact
    print("\nTesting batch normalization impact...")

    X_batch = np.random.randn(100, 10)  # Batch of 100 samples

    for use_bn in [False, True]:
        print(f"\n--- Batch Normalization: {use_bn} ---")

        # Test different depths
        for depth in [10, 20, 50]:
            layer_sizes = [10] + [100] * depth + [1]

            try:
                nn = NetworkWithBatchNorm(layer_sizes, use_batch_norm=use_bn)
                activations = nn.forward(X_batch)

                # Analyze activation distributions
                activation_stats = []
                for i, act in enumerate(activations[1:-1]):
                    stats = {
                        'layer': i+1,
                        'mean': np.mean(act),
                        'std': np.std(act),
                        'min': np.min(act),
                        'max': np.max(act)
                    }
                    activation_stats.append(stats)

                # Check for vanishing/exploding
                final_std = activation_stats[-1]['std']
                print(f"  Depth {depth}: Final activation std = {final_std:.6f}")

                if final_std < 1e-6:
                    print(f"    → Vanishing activations detected!")
                elif final_std > 1e6:
                    print(f"    → Exploding activations detected!")

            except Exception as e:
                print(f"  Depth {depth}: Failed - {str(e)}")

    # Residual connections demonstration
    print("\n=== Residual Connections ===")

    class ResidualBlock:
        def __init__(self, n_features):
            self.W1 = np.random.randn(n_features, n_features) * np.sqrt(2.0 / n_features)
            self.b1 = np.zeros(n_features)
            self.W2 = np.random.randn(n_features, n_features) * np.sqrt(2.0 / n_features)
            self.b2 = np.zeros(n_features)

        def forward(self, x):
            # F(x)
            z1 = x @ self.W1 + self.b1
            a1 = np.tanh(z1)
            z2 = a1 @ self.W2 + self.b2

            # F(x) + x
            return x + z2

    # Compare deep networks with and without residual connections
    print("\nTesting very deep networks...")

    for use_residual in [False, True]:
        print(f"\n--- Residual Connections: {use_residual} ---")

        depth = 100
        n_features = 100

        X_simple = np.random.randn(1, n_features)

        if use_residual:
            # Create residual blocks
            blocks = [ResidualBlock(n_features) for _ in range(depth // 2)]
            x = X_simple

            for block in blocks:
                x = block.forward(x)

            final_activation = x
        else:
            # Plain deep network
            for layer in range(depth):
                W = np.random.randn(n_features, n_features) * np.sqrt(2.0 / n_features)
                b = np.zeros(n_features)
                if layer == 0:
                    x = X_simple @ W + b
                else:
                    x = np.tanh(x @ W + b)

            final_activation = x

        print(f"  Final activation std: {np.std(final_activation):.6f}")
        print(f"  Final activation mean: {np.mean(final_activation):.6f}")

        if np.std(final_activation) < 1e-10:
            print("  → Vanishing gradient problem!")
        elif np.std(final_activation) > 1e10:
            print("  → Exploding gradient problem!")

    # Gradient clipping demonstration
    print("\n=== Gradient Clipping ===")

    def gradient_clipping_demo():
        """Demonstrate gradient clipping"""
        # Create large gradients
        gradients = [100.0, -200.0, 50.0, -300.0, 150.0]

        print("Original gradients:", gradients)

        # Different clipping strategies
        # 1. Value clipping
        clip_value = 100.0
        clipped_value = [np.clip(g, -clip_value, clip_value) for g in gradients]
        print(f"Value clipping (±{clip_value}):", clipped_value)

        # 2. Norm clipping
        clip_norm = 200.0
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > clip_norm:
            clipped_norm = [g * clip_norm / grad_norm for g in gradients]
        else:
            clipped_norm = gradients
        print(f"Norm clipping (max norm={clip_norm}):", clipped_norm)

        # 3. Global norm clipping (like in transformers)
        global_clip_norm = 250.0
        global_norm = np.linalg.norm(gradients)
        if global_norm > global_clip_norm:
            global_clipped = [g * global_clip_norm / global_norm for g in gradients]
        else:
            global_clipped = gradients
        print(f"Global norm clipping: {global_norm:.1f} → {np.linalg.norm(global_clipped):.1f}")

        return clipped_value, clipped_norm, global_clipped

    gradient_clipping_demo()

    # Summary
    print("\n=== Summary of Numerical Stability Techniques ===")
    print("1. Proper weight initialization (Xavier, He)")
    print("2. Batch normalization")
    print("3. Residual connections")
    print("4. Gradient clipping")
    print("5. Careful activation function choice")
    print("6. Monitoring gradient norms")

    return results

deep_learning_numerical_stability()
```

## Key Applications Demonstrated

1. **Robust Linear Regression**: Handling ill-conditioned matrices using QR, SVD, and regularization
2. **Efficient PCA**: Comparing different SVD implementations for large-scale data
3. **Neural Network Optimization**: Implementing and comparing various optimizers
4. **Recommender Systems**: ALS matrix factorization with numerical stability
5. **Deep Learning Stability**: Addressing vanishing/exploding gradients

## Best Practices for Numerical Methods in ML

1. **Always check condition numbers** before solving linear systems
2. **Use appropriate factorizations** (LU for general, Cholesky for SPD, QR for least squares)
3. **Regularize** when matrices are ill-conditioned
4. **Monitor convergence** of iterative methods
5. **Handle edge cases** (singular matrices, division by zero)
6. **Use stable algorithms** (modified Gram-Schmidt, compensated summation)
7. **Test with synthetic data** to verify numerical accuracy

## Exercises

1. Implement L-BFGS and compare its performance with BFGS on high-dimensional problems
2. Create a benchmark comparing different PCA implementations on very large datasets
3. Implement adaptive learning rate methods and test on non-convex optimization problems
4. Develop a robust matrix factorization method that handles missing data and outliers
5. Implement second-order optimization methods with limited memory constraints
6. Create a comprehensive numerical stability checker for neural networks