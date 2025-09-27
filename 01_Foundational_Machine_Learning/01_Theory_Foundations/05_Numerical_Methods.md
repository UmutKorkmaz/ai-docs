# Numerical Methods in Machine Learning

## Overview

Numerical methods form the backbone of computational machine learning, providing the mathematical tools to solve problems that don't have closed-form solutions. This section covers essential numerical techniques used in machine learning algorithms, optimization, and data processing.

## 1. Floating Point Arithmetic and Numerical Stability

### 1.1 Floating Point Representation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve, svd, qr, lu
from scipy.sparse import csr_matrix, diags
from scipy.optimize import newton, minimize_scalar, least_squares
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def floating_point_basics():
    """Understanding floating point representation and limitations"""
    print("=== Floating Point Arithmetic Basics ===")

    # Machine epsilon
    epsilon = np.finfo(float).eps
    print(f"Machine epsilon: {epsilon:.16e}")

    # Demonstrate floating point precision issues
    print("\n=== Precision Issues ===")
    a = 1e20
    b = 1
    c = -1e20

    # Associative property doesn't hold
    result1 = (a + b) + c
    result2 = a + (b + c)
    print(f"(1e20 + 1) - 1e20 = {result1}")
    print(f"1e20 + (1 - 1e20) = {result2}")
    print(f"Difference: {abs(result1 - result2)}")

    # Catastrophic cancellation
    print("\n=== Catastrophic Cancellation ===")
    x = 1.23456789012345
    y = 1.23456789012340
    print(f"x = {x:.16f}")
    print(f"y = {y:.16f}")
    print(f"x - y = {x - y:.16f}")

    # Floating point range
    print(f"\n=== Floating Point Range ===")
    print(f"Smallest positive: {np.finfo(float).tiny}")
    print(f"Largest finite: {np.finfo(float).max}")
    print(f"Smallest normal: {np.finfo(float).smallest_normal}")

    return epsilon

def numerical_stability_analysis():
    """Analyze numerical stability of common operations"""
    print("\n=== Numerical Stability Analysis ===")

    # Matrix inversion stability
    print("Matrix Inversion Stability:")
    n = 10
    A = np.random.randn(n, n)
    A[0, 0] = 1e-16  # Nearly singular

    # Condition number
    cond_A = np.linalg.cond(A)
    print(f"Condition number: {cond_A:.2e}")

    try:
        A_inv = np.linalg.inv(A)
        reconstruction_error = np.linalg.norm(A @ A_inv - np.eye(n))
        print(f"Reconstruction error: {reconstruction_error:.2e}")
    except np.linalg.LinAlgError:
        print("Matrix is numerically singular!")

    # Summation stability
    print("\nSummation Stability:")
    numbers = np.array([1e16, 1, -1e16])

    # Naive summation
    naive_sum = np.sum(numbers)
    print(f"Naive sum: {naive_sum}")

    # Compensated summation (Kahan algorithm)
    def kahan_sum(arr):
        s = 0.0
        c = 0.0
        for x in arr:
            y = x - c
            t = s + y
            c = (t - s) - y
            s = t
        return s

    kahan_result = kahan_sum(numbers)
    print(f"Kahan sum: {kahan_result}")

    return cond_A

floating_point_basics()
numerical_stability_analysis()
```

### 1.2 Condition Numbers and Stability

```python
def condition_number_analysis():
    """Analyze condition numbers and their impact on numerical stability"""
    print("\n=== Condition Number Analysis ===")

    # Create matrices with different condition numbers
    n = 5

    # Well-conditioned matrix
    A_well = np.random.randn(n, n)
    cond_well = np.linalg.cond(A_well)

    # Ill-conditioned matrix
    A_ill = np.random.randn(n, n)
    A_ill[0, :] = A_ill[1, :] * (1 + 1e-10)  # Nearly linearly dependent
    cond_ill = np.linalg.cond(A_ill)

    print(f"Well-conditioned matrix condition number: {cond_well:.2e}")
    print(f"Ill-conditioned matrix condition number: {cond_ill:.2e}")

    # Solve linear systems
    b = np.random.randn(n)

    # Well-conditioned system
    x_well = np.linalg.solve(A_well, b)
    perturbed_b = b + 1e-10 * np.random.randn(n)
    x_well_perturbed = np.linalg.solve(A_well, perturbed_b)
    relative_error_well = np.linalg.norm(x_well_perturbed - x_well) / np.linalg.norm(x_well)

    # Ill-conditioned system
    x_ill = np.linalg.solve(A_ill, b)
    x_ill_perturbed = np.linalg.solve(A_ill, perturbed_b)
    relative_error_ill = np.linalg.norm(x_ill_perturbed - x_ill) / np.linalg.norm(x_ill)

    print(f"\nRelative error (well-conditioned): {relative_error_well:.2e}")
    print(f"Relative error (ill-conditioned): {relative_error_ill:.2e}")

    # Theoretical bound
    print(f"\nTheoretical error bound (well-conditioned): {cond_well * 1e-10:.2e}")
    print(f"Theoretical error bound (ill-conditioned): {cond_ill * 1e-10:.2e}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Singular values
    U, s_well, Vh = np.linalg.svd(A_well)
    _, s_ill, _ = np.linalg.svd(A_ill)

    ax1.semilogy(s_well, 'bo-', label='Well-conditioned')
    ax1.semilogy(s_ill, 'ro-', label='Ill-conditioned')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Values Comparison')
    ax1.legend()
    ax1.grid(True)

    # Error amplification
    perturbation_sizes = np.logspace(-15, -5, 20)
    errors_well = []
    errors_ill = []

    for eps in perturbation_sizes:
        b_pert = b + eps * np.random.randn(n)
        x_well_pert = np.linalg.solve(A_well, b_pert)
        x_ill_pert = np.linalg.solve(A_ill, b_pert)

        errors_well.append(np.linalg.norm(x_well_pert - x_well) / np.linalg.norm(x_well))
        errors_ill.append(np.linalg.norm(x_ill_pert - x_ill) / np.linalg.norm(x_ill))

    ax2.loglog(perturbation_sizes, errors_well, 'b-o', label='Well-conditioned')
    ax2.loglog(perturbation_sizes, errors_ill, 'r-o', label='Ill-conditioned')
    ax2.loglog(perturbation_sizes, cond_well * perturbation_sizes, 'b--', label='Bound (well)')
    ax2.loglog(perturbation_sizes, cond_ill * perturbation_sizes, 'r--', label='Bound (ill)')
    ax2.set_xlabel('Perturbation Size')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Error Amplification')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return A_well, A_ill, cond_well, cond_ill

condition_number_analysis()
```

## 2. Matrix Factorization Methods

### 2.1 LU Decomposition

```python
def lu_decomposition_demo():
    """LU decomposition and its applications"""
    print("\n=== LU Decomposition ===")

    # Create a square matrix
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    print(f"Original matrix A:\n{A}")

    # Perform LU decomposition
    P, L, U = lu(A)

    print(f"\nPermutation matrix P:\n{P}")
    print(f"\nLower triangular matrix L:\n{L}")
    print(f"\nUpper triangular matrix U:\n{U}")

    # Verify decomposition
    print(f"\nVerification (P.T @ A = L @ U):")
    print(np.allclose(P.T @ A, L @ U))

    # Solve linear system Ax = b
    b = np.array([4, 10, 26])

    # Using LU decomposition
    Pb = P.T @ b

    # Forward substitution: Ly = Pb
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # Back substitution: Ux = y
    x = np.zeros_like(b)
    for i in range(len(b)-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    print(f"\nSolution using LU: {x}")
    print(f"Direct solution: {np.linalg.solve(A, b)}")

    # Efficiency comparison
    print("\n=== Efficiency Comparison ===")
    sizes = [50, 100, 200, 500]

    for n in sizes:
        A_large = np.random.randn(n, n)
        b_large = np.random.randn(n)

        # Direct solve
        %timeit -n 5 -r 3 x_direct = np.linalg.solve(A_large, b_large)

        # LU decomposition
        %timeit -n 5 -r 3 x_lu = solve(A_large, b_large, assume_a='sym')

    return P, L, U

lu_decomposition_demo()
```

### 2.2 QR Decomposition

```python
def qr_decomposition_demo():
    """QR decomposition and its applications"""
    print("\n=== QR Decomposition ===")

    # Create a matrix
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [11, 12, 13]], dtype=float)
    print(f"Original matrix A shape: {A.shape}")

    # Perform QR decomposition
    Q, R = qr(A)

    print(f"\nOrthogonal matrix Q shape: {Q.shape}")
    print(f"Orthogonality check (Q.T @ Q = I):")
    print(np.allclose(Q.T @ Q, np.eye(Q.shape[1])))

    print(f"\nUpper triangular matrix R:\n{R}")

    # Verify decomposition
    print(f"\nVerification (A = Q @ R):")
    print(np.allclose(A, Q @ R))

    # Least squares using QR
    print("\n=== Least Squares using QR ===")
    X = np.random.randn(100, 3)
    true_beta = np.array([1.5, -2.0, 0.5])
    y = X @ true_beta + 0.1 * np.random.randn(100)

    # QR decomposition
    Q_qr, R_qr = qr(X)

    # Solve R @ beta = Q.T @ y
    beta_qr = np.linalg.solve(R_qr[:3], Q_qr.T @ y)[:3]

    print(f"True beta: {true_beta}")
    print(f"QR solution: {beta_qr}")
    print(f"Direct solution: {np.linalg.lstsq(X, y, rcond=None)[0]}")

    # Gram-Schmidt vs Modified Gram-Schmidt
    print("\n=== Gram-Schmidt Stability ===")

    def classical_gram_schmidt(A):
        """Classical Gram-Schmidt orthogonalization"""
        Q = np.zeros_like(A)
        R = np.zeros((A.shape[1], A.shape[1]))

        for j in range(A.shape[1]):
            q = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                q = q - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(q)
            Q[:, j] = q / R[j, j]

        return Q, R

    def modified_gram_schmidt(A):
        """Modified Gram-Schmidt orthogonalization"""
        Q = np.zeros_like(A)
        R = np.zeros((A.shape[1], A.shape[1]))
        V = A.copy()

        for i in range(A.shape[1]):
            R[i, i] = np.linalg.norm(V[:, i])
            Q[:, i] = V[:, i] / R[i, i]

            for j in range(i+1, A.shape[1]):
                R[i, j] = np.dot(Q[:, i], V[:, j])
                V[:, j] = V[:, j] - R[i, j] * Q[:, i]

        return Q, R

    # Test with nearly linearly dependent vectors
    A_test = np.random.randn(10, 5)
    A_test[:, 2] = A_test[:, 0] + 1e-10 * A_test[:, 1]

    Q_classical, R_classical = classical_gram_schmidt(A_test)
    Q_modified, R_modified = modified_gram_schmidt(A_test)

    orthogonality_classical = np.linalg.norm(Q_classical.T @ Q_classical - np.eye(5))
    orthogonality_modified = np.linalg.norm(Q_modified.T @ Q_modified - np.eye(5))

    print(f"Classical Gram-Schmidt orthogonality error: {orthogonality_classical:.2e}")
    print(f"Modified Gram-Schmidt orthogonality error: {orthogonality_modified:.2e}")

    return Q, R

qr_decomposition_demo()
```

### 2.3 Cholesky Decomposition

```python
def cholesky_decomposition_demo():
    """Cholesky decomposition for positive definite matrices"""
    print("\n=== Cholesky Decomposition ===")

    # Create a positive definite matrix
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
    print(f"Original matrix A:\n{A}")

    # Check if positive definite
    eigenvals = np.linalg.eigvals(A)
    print(f"\nEigenvalues: {eigenvals}")
    print(f"Positive definite: {np.all(eigenvals > 0)}")

    # Perform Cholesky decomposition
    try:
        L = np.linalg.cholesky(A)
        print(f"\nLower triangular matrix L:\n{L}")

        # Verify decomposition
        print(f"\nVerification (A = L @ L.T):")
        print(np.allclose(A, L @ L.T))

        # Solve linear system
        b = np.array([4, 1, -2])

        # Forward substitution: Ly = b
        y = np.linalg.solve(L, b)

        # Back substitution: L.T x = y
        x = np.linalg.solve(L.T, y)

        print(f"\nSolution using Cholesky: {x}")
        print(f"Direct solution: {np.linalg.solve(A, b)}")

    except np.linalg.LinAlgError:
        print("Matrix is not positive definite!")

    # Application: Generating correlated random variables
    print("\n=== Correlated Random Variables ===")

    # Correlation matrix
    corr_matrix = np.array([[1.0, 0.7, 0.3],
                          [0.7, 1.0, 0.5],
                          [0.3, 0.5, 1.0]])

    L_corr = np.linalg.cholesky(corr_matrix)

    # Generate correlated samples
    n_samples = 1000
    independent_samples = np.random.randn(n_samples, 3)
    correlated_samples = independent_samples @ L_corr.T

    # Compute sample correlation
    sample_corr = np.corrcoef(correlated_samples.T)

    print(f"Target correlation matrix:\n{corr_matrix}")
    print(f"Sample correlation matrix:\n{sample_corr}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(correlated_samples[:, 0], correlated_samples[:, 1], alpha=0.5)
    axes[0].set_xlabel('Variable 1')
    axes[0].set_ylabel('Variable 2')
    axes[0].set_title(f'Correlation: {sample_corr[0, 1]:.3f}')

    axes[1].scatter(correlated_samples[:, 0], correlated_samples[:, 2], alpha=0.5)
    axes[1].set_xlabel('Variable 1')
    axes[1].set_ylabel('Variable 3')
    axes[1].set_title(f'Correlation: {sample_corr[0, 2]:.3f}')

    axes[2].scatter(correlated_samples[:, 1], correlated_samples[:, 2], alpha=0.5)
    axes[2].set_xlabel('Variable 2')
    axes[2].set_ylabel('Variable 3')
    axes[2].set_title(f'Correlation: {sample_corr[1, 2]:.3f}')

    plt.tight_layout()
    plt.show()

    return L

cholesky_decomposition_demo()
```

## 3. Iterative Methods for Linear Systems

### 3.1 Jacobi Method

```python
def jacobi_method_demo():
    """Jacobi iterative method for solving linear systems"""
    print("\n=== Jacobi Method ===")

    # Create diagonally dominant matrix
    A = np.array([[10, -1, 2, 0],
                  [-1, 11, -1, 3],
                  [2, -1, 10, -1],
                  [0, 3, -1, 8]], dtype=float)

    b = np.array([6, 25, -11, 15])

    print(f"Matrix A:\n{A}")
    print(f"\nVector b: {b}")

    # Check diagonal dominance
    diag_abs = np.abs(np.diag(A))
    row_sums = np.sum(np.abs(A), axis=1) - diag_abs
    is_diagonally_dominant = np.all(diag_abs > row_sums)

    print(f"\nDiagonally dominant: {is_diagonally_dominant}")

    def jacobi_iteration(A, b, max_iter=100, tol=1e-10):
        """Jacobi iteration implementation"""
        n = len(b)
        x = np.zeros(n)
        x_history = [x.copy()]

        D = np.diag(np.diag(A))
        R = A - D

        for k in range(max_iter):
            x_new = np.linalg.solve(D, b - R @ x)

            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new
            x_history.append(x.copy())

        return x, k+1, np.array(x_history)

    # Solve using Jacobi method
    x_jacobi, iterations, x_history = jacobi_iteration(A, b)
    x_direct = np.linalg.solve(A, b)

    print(f"\nSolution converged in {iterations} iterations")
    print(f"Jacobi solution: {x_jacobi}")
    print(f"Direct solution: {x_direct}")
    print(f"Error: {np.linalg.norm(x_jacobi - x_direct):.2e}")

    # Convergence analysis
    print("\n=== Convergence Analysis ===")

    # Spectral radius of iteration matrix
    D_inv = np.diag(1.0 / np.diag(A))
    G_jacobi = np.eye(len(b)) - D_inv @ A
    spectral_radius = np.max(np.abs(np.linalg.eigvals(G_jacobi)))

    print(f"Spectral radius of Jacobi iteration matrix: {spectral_radius:.4f}")
    print(f"Convergence guaranteed: {spectral_radius < 1}")

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error vs iteration
    errors = [np.linalg.norm(x - x_direct) for x in x_history]
    ax1.semilogy(range(len(errors)), errors, 'bo-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error')
    ax1.set_title('Convergence History')
    ax1.grid(True)

    # Theoretical convergence rate
    theoretical_errors = [errors[0] * (spectral_radius ** k) for k in range(len(errors))]
    ax1.semilogy(range(len(theoretical_errors)), theoretical_errors, 'r--',
                label=f'Theoretical (ρ={spectral_radius:.3f})')
    ax1.legend()

    # Component convergence
    for i in range(4):
        component_history = x_history[:, i]
        ax2.plot(range(len(component_history)), component_history, 'o-',
                label=f'x{i+1}')

    ax2.axhline(y=x_direct[0], color='b', linestyle='--', alpha=0.7)
    ax2.axhline(y=x_direct[1], color='g', linestyle='--', alpha=0.7)
    ax2.axhline(y=x_direct[2], color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=x_direct[3], color='c', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.set_title('Component Convergence')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return x_jacobi, iterations

jacobi_method_demo()
```

### 3.2 Gauss-Seidel Method

```python
def gauss_seidel_method_demo():
    """Gauss-Seidel iterative method"""
    print("\n=== Gauss-Seidel Method ===")

    # Use the same system as Jacobi
    A = np.array([[10, -1, 2, 0],
                  [-1, 11, -1, 3],
                  [2, -1, 10, -1],
                  [0, 3, -1, 8]], dtype=float)

    b = np.array([6, 25, -11, 15])

    def gauss_seidel_iteration(A, b, max_iter=100, tol=1e-10):
        """Gauss-Seidel iteration implementation"""
        n = len(b)
        x = np.zeros(n)
        x_history = [x.copy()]

        for k in range(max_iter):
            x_new = x.copy()

            for i in range(n):
                sum_ax = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
                x_new[i] = (b[i] - sum_ax) / A[i, i]

            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new
            x_history.append(x.copy())

        return x, k+1, np.array(x_history)

    # Solve using Gauss-Seidel
    x_gs, gs_iterations, gs_history = gauss_seidel_iteration(A, b)
    x_direct = np.linalg.solve(A, b)

    print(f"Gauss-Seidel converged in {gs_iterations} iterations")
    print(f"Gauss-Seidel solution: {x_gs}")
    print(f"Error: {np.linalg.norm(x_gs - x_direct):.2e}")

    # Compare with Jacobi
    def jacobi_iteration(A, b, max_iter=100, tol=1e-10):
        n = len(b)
        x = np.zeros(n)
        x_history = [x.copy()]

        for k in range(max_iter):
            x_new = np.zeros(n)
            for i in range(n):
                sum_ax = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
                x_new[i] = (b[i] - sum_ax) / A[i, i]

            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new
            x_history.append(x.copy())

        return x, k+1, np.array(x_history)

    x_jacobi, jacobi_iterations, jacobi_history = jacobi_iteration(A, b)

    print(f"\nComparison:")
    print(f"Jacobi iterations: {jacobi_iterations}")
    print(f"Gauss-Seidel iterations: {gs_iterations}")

    # Convergence comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error comparison
    jacobi_errors = [np.linalg.norm(x - x_direct) for x in jacobi_history]
    gs_errors = [np.linalg.norm(x - x_direct) for x in gs_history]

    ax1.semilogy(range(len(jacobi_errors)), jacobi_errors, 'b-o', label='Jacobi')
    ax1.semilogy(range(len(gs_errors)), gs_errors, 'r-o', label='Gauss-Seidel')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True)

    # Spectral radii comparison
    n = len(b)
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    # Jacobi iteration matrix
    G_jacobi = np.linalg.solve(D, -(L + U))
    rho_jacobi = np.max(np.abs(np.linalg.eigvals(G_jacobi)))

    # Gauss-Seidel iteration matrix
    G_gs = np.linalg.solve(D + L, -U)
    rho_gs = np.max(np.abs(np.linalg.eigvals(G_gs)))

    print(f"\nSpectral radii:")
    print(f"Jacobi: {rho_jacobi:.4f}")
    print(f"Gauss-Seidel: {rho_gs:.4f}")

    # Theoretical convergence
    theoretical_jacobi = [jacobi_errors[0] * (rho_jacobi ** k) for k in range(len(jacobi_errors))]
    theoretical_gs = [gs_errors[0] * (rho_gs ** k) for k in range(len(gs_errors))]

    ax1.semilogy(range(len(theoretical_jacobi)), theoretical_jacobi, 'b--', alpha=0.7)
    ax1.semilogy(range(len(theoretical_gs)), theoretical_gs, 'r--', alpha=0.7)

    # Component comparison for first variable
    ax2.plot(range(len(jacobi_history)), jacobi_history[:, 0], 'b-o', label='Jacobi x1')
    ax2.plot(range(len(gs_history)), gs_history[:, 0], 'r-o', label='GS x1')
    ax2.axhline(y=x_direct[0], color='k', linestyle='--', alpha=0.5, label='True value')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('x1 Value')
    ax2.set_title('First Variable Convergence')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return x_gs, gs_iterations

gauss_seidel_method_demo()
```

### 3.3 Conjugate Gradient Method

```python
def conjugate_gradient_demo():
    """Conjugate Gradient method for symmetric positive definite systems"""
    print("\n=== Conjugate Gradient Method ===")

    # Create symmetric positive definite matrix
    n = 50
    np.random.seed(42)
    A = np.random.randn(n, n)
    A = A @ A.T + 0.1 * np.eye(n)  # Make positive definite

    b = np.random.randn(n)

    print(f"Matrix shape: {A.shape}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")

    def conjugate_gradient(A, b, max_iter=None, tol=1e-10):
        """Conjugate Gradient implementation"""
        n = len(b)
        if max_iter is None:
            max_iter = n

        x = np.zeros(n)
        r = b - A @ x
        p = r.copy()

        errors = [np.linalg.norm(r)]

        for k in range(max_iter):
            Ap = A @ p
            alpha = np.dot(r, r) / np.dot(p, Ap)

            x = x + alpha * p
            r_new = r - alpha * Ap

            if np.linalg.norm(r_new) < tol:
                break

            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new

            errors.append(np.linalg.norm(r))

        return x, k+1, np.array(errors)

    # Solve using CG
    x_cg, cg_iterations, cg_errors = conjugate_gradient(A, b)
    x_direct = np.linalg.solve(A, b)

    print(f"\nCG converged in {cg_iterations} iterations")
    print(f"Error: {np.linalg.norm(x_cg - x_direct):.2e}")

    # Theoretical convergence bound
    eigenvals = np.linalg.eigvals(A)
    kappa = np.max(eigenvals) / np.min(eigenvals)

    print(f"Condition number κ: {kappa:.2e}")
    print(f"Theoretical bound: {2 * np.sqrt(kappa):.2f} iterations")

    # Compare with gradient descent
    def gradient_descent(A, b, max_iter=1000, tol=1e-10):
        """Gradient descent for comparison"""
        n = len(b)
        x = np.zeros(n)
        alpha = 2 / (np.max(eigenvals) + np.min(eigenvals))  # Optimal step size

        errors = []

        for k in range(max_iter):
            r = b - A @ x
            error = np.linalg.norm(r)
            errors.append(error)

            if error < tol:
                break

            x = x + alpha * r

        return x, k+1, np.array(errors)

    x_gd, gd_iterations, gd_errors = gradient_descent(A, b)

    print(f"\nGradient descent converged in {gd_iterations} iterations")

    # Convergence comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error comparison
    ax1.semilogy(range(len(cg_errors)), cg_errors, 'b-o', label='Conjugate Gradient')
    ax1.semilogy(range(len(gd_errors)), gd_errors, 'r-o', label='Gradient Descent')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Residual Norm')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True)

    # Theoretical bounds
    cg_theoretical = [cg_errors[0] * (2 / (np.sqrt(kappa) + 1/np.sqrt(kappa)))**k
                     for k in range(min(len(cg_errors), 100))]
    gd_theoretical = [gd_errors[0] * ((kappa - 1) / (kappa + 1))**k
                     for k in range(min(len(gd_errors), 1000))]

    ax1.semilogy(range(len(cg_theoretical)), cg_theoretical, 'b--', alpha=0.7)
    ax1.semilogy(range(len(gd_theoretical)), gd_theoretical, 'r--', alpha=0.7)

    # A-orthogonality check
    print("\n=== A-Orthogonality Check ===")

    def conjugate_gradient_with_directions(A, b, max_iter=10):
        """CG with direction tracking"""
        n = len(b)
        x = np.zeros(n)
        r = b - A @ x
        p = r.copy()

        directions = [p.copy()]

        for k in range(max_iter):
            Ap = A @ p
            alpha = np.dot(r, r) / np.dot(p, Ap)

            x = x + alpha * p
            r_new = r - alpha * Ap

            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new

            directions.append(p.copy())

        return np.array(directions)

    directions = conjugate_gradient_with_directions(A, b, max_iter=5)

    # Check A-orthogonality
    for i in range(directions.shape[0]):
        for j in range(i+1, directions.shape[0]):
            a_inner = np.dot(directions[i], A @ directions[j])
            print(f"<d{i+1}, A d{j+1}> = {a_inner:.2e}")

    plt.tight_layout()
    plt.show()

    return x_cg, cg_iterations

conjugate_gradient_demo()
```

## 4. Numerical Optimization Methods

### 4.1 Newton's Method

```python
def newton_method_optimization():
    """Newton's method for optimization"""
    print("\n=== Newton's Method for Optimization ===")

    # 1D Example
    def f1d(x):
        return x**4 - 3*x**3 + 2*x**2 + x - 5

    def df1d(x):
        return 4*x**3 - 9*x**2 + 4*x + 1

    def d2f1d(x):
        return 12*x**2 - 18*x + 4

    def newton_1d(f, df, d2f, x0, max_iter=50, tol=1e-10):
        """1D Newton's method"""
        x = x0
        history = [x]

        for i in range(max_iter):
            fx = f(x)
            dfx = df(x)
            d2fx = d2f(x)

            if abs(dfx) < tol:
                break

            if abs(d2fx) < 1e-10:
                print("Second derivative too small!")
                break

            x_new = x - dfx / d2fx

            if abs(x_new - x) < tol:
                break

            x = x_new
            history.append(x)

        return x, i+1, np.array(history)

    # Find minimum
    x0 = 0
    x_min, iterations, history = newton_1d(f1d, df1d, d2f1d, x0)

    print(f"1D Newton's method:")
    print(f"Initial guess: {x0}")
    print(f"Found minimum at: {x_min:.6f}")
    print(f"Iterations: {iterations}")
    print(f"Function value: {f1d(x_min):.6f}")

    # Multivariate example
    def rosenbrock(x):
        """Rosenbrock function"""
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_rosenbrock(x):
        """Gradient of Rosenbrock function"""
        return np.array([
            -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
            200*(x[1] - x[0]**2)
        ])

    def hess_rosenbrock(x):
        """Hessian of Rosenbrock function"""
        return np.array([
            [-400*(x[1] - 3*x[0]**2) + 2, -400*x[0]],
            [-400*x[0], 200]
        ])

    def newton_nd(f, grad, hess, x0, max_iter=100, tol=1e-10):
        """Multivariate Newton's method"""
        x = x0.copy()
        history = [x.copy()]

        for i in range(max_iter):
            g = grad(x)
            H = hess(x)

            if np.linalg.norm(g) < tol:
                break

            try:
                delta = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                print("Hessian is singular!")
                break

            x_new = x + delta

            if np.linalg.norm(delta) < tol:
                break

            x = x_new
            history.append(x.copy())

        return x, i+1, np.array(history)

    # Solve Rosenbrock
    x0_nd = np.array([-1.5, 2.0])
    x_min_nd, iterations_nd, history_nd = newton_nd(
        rosenbrock, grad_rosenbrock, hess_rosenbrock, x0_nd
    )

    print(f"\nMultivariate Newton's method:")
    print(f"Initial guess: {x0_nd}")
    print(f"Found minimum at: {x_min_nd}")
    print(f"Iterations: {iterations_nd}")
    print(f"Function value: {rosenbrock(x_min_nd):.6f}")

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1D convergence
    x_range = np.linspace(-2, 3, 100)
    ax1.plot(x_range, f1d(x_range))
    ax1.plot(history, f1d(np.array(history)), 'ro-', label='Newton iterations')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('1D Newton\'s Method')
    ax1.legend()
    ax1.grid(True)

    # 2D Rosenbrock function
    X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))
    Z = 100*(Y - X**2)**2 + (1 - X)**2

    contour = ax2.contour(X, Y, Z, levels=50, alpha=0.6)
    ax2.plot(history_nd[:, 0], history_nd[:, 1], 'ro-', label='Newton path')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Newton\'s Method on Rosenbrock')
    ax2.legend()

    # Convergence rates
    def convergence_rate(history, x_star):
        """Estimate convergence rate"""
        errors = [np.linalg.norm(x - x_star) for x in history]
        if len(errors) < 3:
            return errors

        rates = []
        for i in range(1, len(errors)-1):
            if errors[i] > 1e-12 and errors[i-1] > 1e-12:
                rate = np.log(errors[i+1] / errors[i]) / np.log(errors[i] / errors[i-1])
                rates.append(rate)

        return errors, rates

    # 1D convergence rate
    errors_1d, rates_1d = convergence_rate(history, x_min)
    ax3.semilogy(range(len(errors_1d)), errors_1d, 'bo-')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Error')
    ax3.set_title(f'1D Convergence (Rate: {np.mean(rates_1d):.2f})')
    ax3.grid(True)

    # 2D convergence rate
    errors_2d, rates_2d = convergence_rate(history_nd, np.array([1, 1]))
    ax4.semilogy(range(len(errors_2d)), errors_2d, 'ro-')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Error')
    ax4.set_title(f'2D Convergence (Rate: {np.mean(rates_2d):.2f})')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return x_min, x_min_nd

newton_method_optimization()
```

### 4.2 Quasi-Newton Methods

```python
def quasi_newton_methods():
    """Quasi-Newton methods (BFGS, DFP)"""
    print("\n=== Quasi-Newton Methods ===")

    def rosenbrock(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_rosenbrock(x):
        return np.array([
            -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
            200*(x[1] - x[0]**2)
        ])

    def bfgs(f, grad, x0, max_iter=100, tol=1e-10):
        """BFGS method"""
        n = len(x0)
        x = x0.copy()
        H = np.eye(n)  # Initial Hessian approximation

        history = [x.copy()]

        for k in range(max_iter):
            g = grad(x)

            if np.linalg.norm(g) < tol:
                break

            # Search direction
            d = -H @ g

            # Line search (simplified)
            alpha = 1.0
            c1 = 0.1
            rho = 0.9

            while f(x + alpha * d) > f(x) + c1 * alpha * np.dot(g, d):
                alpha *= rho

            # Update
            x_new = x + alpha * d
            g_new = grad(x_new)
            s = x_new - x
            y = g_new - g

            # BFGS update
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

            x = x_new
            history.append(x.copy())

        return x, k+1, np.array(history)

    def dfp(f, grad, x0, max_iter=100, tol=1e-10):
        """DFP method"""
        n = len(x0)
        x = x0.copy()
        H = np.eye(n)  # Initial Hessian approximation

        history = [x.copy()]

        for k in range(max_iter):
            g = grad(x)

            if np.linalg.norm(g) < tol:
                break

            # Search direction
            d = -H @ g

            # Line search
            alpha = 1.0
            c1 = 0.1
            rho = 0.9

            while f(x + alpha * d) > f(x) + c1 * alpha * np.dot(g, d):
                alpha *= rho

            # Update
            x_new = x + alpha * d
            g_new = grad(x_new)
            s = x_new - x
            y = g_new - g

            # DFP update
            H = H - (H @ np.outer(y, y) @ H) / np.dot(y, H @ y) + np.outer(s, s) / np.dot(s, y)

            x = x_new
            history.append(x.copy())

        return x, k+1, np.array(history)

    # Test methods
    x0 = np.array([-1.5, 2.0])

    print(f"Initial point: {x0}")
    print(f"Function value: {rosenbrock(x0):.6f}")

    # BFGS
    x_bfgs, iter_bfgs, hist_bfgs = bfgs(rosenbrock, grad_rosenbrock, x0)
    print(f"\nBFGS:")
    print(f"Solution: {x_bfgs}")
    print(f"Iterations: {iter_bfgs}")
    print(f"Function value: {rosenbrock(x_bfgs):.6f}")

    # DFP
    x_dfp, iter_dfp, hist_dfp = dfp(rosenbrock, grad_rosenbrock, x0)
    print(f"\nDFP:")
    print(f"Solution: {x_dfp}")
    print(f"Iterations: {iter_dfp}")
    print(f"Function value: {rosenbrock(x_dfp):.6f}")

    # Compare with scipy's BFGS
    from scipy.optimize import minimize

    result_scipy = minimize(rosenbrock, x0, method='BFGS', jac=grad_rosenbrock)
    print(f"\nScipy BFGS:")
    print(f"Solution: {result_scipy.x}")
    print(f"Iterations: {result_scipy.nit}")
    print(f"Function value: {result_scipy.fun:.6f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Contour plot with optimization paths
    X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))
    Z = 100*(Y - X**2)**2 + (1 - X)**2

    ax1.contour(X, Y, Z, levels=50, alpha=0.6)
    ax1.plot(hist_bfgs[:, 0], hist_bfgs[:, 1], 'b-o', label='BFGS', markersize=4)
    ax1.plot(hist_dfp[:, 0], hist_dfp[:, 1], 'r-o', label='DFP', markersize=4)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Quasi-Newton Methods on Rosenbrock')
    ax1.legend()

    # Convergence comparison
    errors_bfgs = [rosenbrock(x) - 1 for x in hist_bfgs]  # True minimum is 1
    errors_dfp = [rosenbrock(x) - 1 for x in hist_dfp]

    ax2.semilogy(range(len(errors_bfgs)), errors_bfgs, 'b-o', label='BFGS')
    ax2.semilogy(range(len(errors_dfp)), errors_dfp, 'r-o', label='DFP')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('f(x) - f*')
    ax2.set_title('Convergence Comparison')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return x_bfgs, x_dfp

quasi_newton_methods()
```

## 5. Numerical Integration and Differentiation

### 5.1 Numerical Integration

```python
def numerical_integration_methods():
    """Numerical integration methods"""
    print("\n=== Numerical Integration Methods ===")

    # Test functions
    def f1(x):
        return np.exp(-x**2)  # Gaussian

    def f2(x):
        return 1 / (1 + x**2)  # Arctangent derivative

    def exact_f1(a, b):
        """Exact integral of exp(-x^2) using error function"""
        return np.sqrt(np.pi) / 2 * (scipy.special.erf(b) - scipy.special.erf(a))

    import scipy.special

    # Trapezoidal rule
    def trapezoidal(f, a, b, n):
        """Trapezoidal rule"""
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    # Simpson's rule
    def simpson(f, a, b, n):
        """Simpson's rule (n must be even)"""
        if n % 2 != 0:
            n += 1

        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n

        integral = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
        return integral

    # Gaussian quadrature
    def gaussian_quadrature(f, a, b, n):
        """Gaussian quadrature using scipy"""
        from scipy.integrate import quadrature
        result, error = quadrature(f, a, b, tol=1e-10, rtol=1e-10, maxiter=n)
        return result

    # Test integration
    a, b = 0, 2
    exact = exact_f1(a, b)

    print(f"Integral of exp(-x^2) from {a} to {b}")
    print(f"Exact value: {exact:.12f}")

    # Compare methods
    ns = [10, 50, 100, 500]

    print(f"\n{'n':>5} {'Trapezoidal':>12} {'Error':>12} {'Simpson':>12} {'Error':>12} {'Gaussian':>12} {'Error':>12}")
    print("-" * 80)

    for n in ns:
        trap_result = trapezoidal(f1, a, b, n)
        simp_result = simpson(f1, a, b, n)
        gauss_result = gaussian_quadrature(f1, a, b, n)

        trap_error = abs(trap_result - exact)
        simp_error = abs(simp_result - exact)
        gauss_error = abs(gauss_result - exact)

        print(f"{n:5d} {trap_result:12.8f} {trap_error:12.2e} {simp_result:12.8f} {simp_error:12.2e} {gauss_result:12.8f} {gauss_error:12.2e}")

    # Error analysis
    print("\n=== Error Analysis ===")

    # Convergence rates
    n_range = np.logspace(1, 3, 20, dtype=int)
    trap_errors = []
    simp_errors = []

    for n in n_range:
        trap_result = trapezoidal(f1, a, b, n)
        simp_result = simpson(f1, a, b, n)

        trap_errors.append(abs(trap_result - exact))
        simp_errors.append(abs(simp_result - exact))

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(n_range, trap_errors, 'b-o', label='Trapezoidal')
    ax1.loglog(n_range, simp_errors, 'r-o', label='Simpson')

    # Theoretical rates
    ax1.loglog(n_range, 1/n_range**2, 'b--', alpha=0.7, label='O(h²)')
    ax1.loglog(n_range, 1/n_range**4, 'r--', alpha=0.7, label='O(h⁴)')

    ax1.set_xlabel('Number of intervals')
    ax1.set_ylabel('Error')
    ax1.set_title('Integration Error Convergence')
    ax1.legend()
    ax1.grid(True)

    # Function visualization
    x_fine = np.linspace(a, b, 1000)
    y_fine = f1(x_fine)

    ax2.plot(x_fine, y_fine, 'b-', linewidth=2, label='exp(-x²)')
    ax2.fill_between(x_fine, 0, y_fine, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Function to Integrate')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return trap_errors, simp_errors

numerical_integration_methods()
```

### 5.2 Numerical Differentiation

```python
def numerical_differentiation_methods():
    """Numerical differentiation methods"""
    print("\n=== Numerical Differentiation Methods ===")

    # Test functions
    def f(x):
        return np.sin(x) + 0.5 * np.cos(2*x)

    def df_exact(x):
        """Exact derivative"""
        return np.cos(x) - np.sin(2*x)

    def d2f_exact(x):
        """Exact second derivative"""
        return -np.sin(x) - 2*np.cos(2*x)

    # Differentiation methods
    def forward_diff(f, x, h):
        """Forward difference"""
        return (f(x + h) - f(x)) / h

    def backward_diff(f, x, h):
        """Backward difference"""
        return (f(x) - f(x - h)) / h

    def central_diff(f, x, h):
        """Central difference"""
        return (f(x + h) - f(x - h)) / (2 * h)

    def second_derivative(f, x, h):
        """Second derivative using central difference"""
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

    # Test at a point
    x0 = np.pi / 4
    exact_deriv = df_exact(x0)
    exact_second_deriv = d2f_exact(x0)

    print(f"Differentiation at x = π/4 = {x0:.4f}")
    print(f"Exact first derivative: {exact_deriv:.12f}")
    print(f"Exact second derivative: {exact_second_deriv:.12f}")

    # Compare methods for different h
    h_values = np.logspace(-1, -15, 15)

    forward_errors = []
    backward_errors = []
    central_errors = []
    second_deriv_errors = []

    print(f"\n{'h':>12} {'Forward':>12} {'Backward':>12} {'Central':>12} {'Second':>12}")
    print("-" * 65)

    for h in h_values:
        fwd = forward_diff(f, x0, h)
        bwd = backward_diff(f, x0, h)
        cent = central_diff(f, x0, h)
        second = second_derivative(f, x0, h)

        fwd_err = abs(fwd - exact_deriv)
        bwd_err = abs(bwd - exact_deriv)
        cent_err = abs(cent - exact_deriv)
        second_err = abs(second - exact_second_deriv)

        forward_errors.append(fwd_err)
        backward_errors.append(bwd_err)
        central_errors.append(cent_err)
        second_deriv_errors.append(second_err)

        print(f"{h:12.2e} {fwd_err:12.2e} {bwd_err:12.2e} {cent_err:12.2e} {second_err:12.2e}")

    # Find optimal h
    print("\n=== Optimal Step Size Analysis ===")

    def theoretical_error(h, order):
        """Theoretical error for differentiation"""
        # Total error = truncation error + rounding error
        truncation = h**order
        rounding = 1e-16 / h
        return truncation + rounding

    # Find optimal h for central difference (order 2)
    h_fine = np.logspace(-1, -15, 1000)
    theoretical_central = [theoretical_error(h, 2) for h in h_fine]
    optimal_h = h_fine[np.argmin(theoretical_central)]

    print(f"Theoretical optimal h for central difference: {optimal_h:.2e}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # First derivative methods
    ax1.loglog(h_values, forward_errors, 'b-o', label='Forward (O(h))', markersize=6)
    ax1.loglog(h_values, backward_errors, 'g-o', label='Backward (O(h))', markersize=6)
    ax1.loglog(h_values, central_errors, 'r-o', label='Central (O(h²))', markersize=6)
    ax1.loglog(h_fine, theoretical_central, 'r--', alpha=0.7, label='Theoretical (O(h²))')

    ax1.set_xlabel('Step size h')
    ax1.set_ylabel('Error')
    ax1.set_title('First Derivative Methods')
    ax1.legend()
    ax1.grid(True)

    # Second derivative
    ax2.loglog(h_values, second_deriv_errors, 'purple', marker='o', label='Second derivative (O(h²))', markersize=6)
    ax2.loglog(h_fine, [theoretical_error(h, 2) for h in h_fine], 'purple', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Step size h')
    ax2.set_ylabel('Error')
    ax2.set_title('Second Derivative Method')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Automatic differentiation example
    print("\n=== Automatic Differentiation Example ===")

    class DualNumber:
        """Simple automatic differentiation using dual numbers"""
        def __init__(self, value, derivative=1.0):
            self.value = value
            self.derivative = derivative

        def __add__(self, other):
            if isinstance(other, DualNumber):
                return DualNumber(self.value + other.value, self.derivative + other.derivative)
            return DualNumber(self.value + other, self.derivative)

        def __mul__(self, other):
            if isinstance(other, DualNumber):
                return DualNumber(self.value * other.value,
                                self.value * other.derivative + self.derivative * other.value)
            return DualNumber(self.value * other, self.derivative * other)

        def sin(self):
            return DualNumber(np.sin(self.value), np.cos(self.value) * self.derivative)

        def cos(self):
            return DualNumber(np.cos(self.value), -np.sin(self.value) * self.derivative)

    def ad_function(x):
        """Function evaluated with automatic differentiation"""
        return x.sin() + DualNumber(0.5, 0) * (DualNumber(2, 0) * x).cos()

    x_ad = DualNumber(x0)
    result_ad = ad_function(x_ad)

    print(f"Automatic differentiation result:")
    print(f"Function value: {result_ad.value:.12f}")
    print(f"Derivative: {result_ad.derivative:.12f}")
    print(f"Exact derivative: {exact_deriv:.12f}")
    print(f"AD error: {abs(result_ad.derivative - exact_deriv):.2e}")

    return forward_errors, central_errors, second_deriv_errors

numerical_differentiation_methods()
```

## 6. Applications in Machine Learning

### 6.1 Matrix Completion and Recommendation Systems

```python
def matrix_completion_application():
    """Matrix completion using alternating least squares"""
    print("\n=== Matrix Completion Application ===")

    # Create synthetic rating matrix
    n_users = 50
    n_items = 30
    n_features = 5

    # True latent factors
    U_true = np.random.randn(n_users, n_features)
    V_true = np.random.randn(n_items, n_features)

    # True rating matrix
    R_true = U_true @ V_true.T

    # Add noise
    R_noisy = R_true + 0.1 * np.random.randn(n_users, n_items)

    # Mask some ratings (simulate missing data)
    mask = np.random.rand(n_users, n_items) < 0.7  # 30% missing
    R_observed = R_noisy.copy()
    R_observed[~mask] = np.nan

    print(f"Original matrix shape: {R_observed.shape}")
    print(f"Observed ratings: {np.sum(mask)}/{n_users * n_items} ({100*np.sum(mask)/(n_users*n_items):.1f}%)")

    def alternating_least_squares(R, mask, n_features, max_iter=100, tol=1e-6):
        """Alternating least squares for matrix completion"""
        n_users, n_items = R.shape

        # Initialize
        U = np.random.randn(n_users, n_features)
        V = np.random.randn(n_items, n_features)

        errors = []

        for iteration in range(max_iter):
            # Update U
            for i in range(n_users):
                # Get observed items for user i
                observed_items = mask[i]
                if np.sum(observed_items) > 0:
                    V_obs = V[observed_items]
                    R_obs = R[i, observed_items]

                    # Solve least squares
                    U[i] = np.linalg.solve(V_obs.T @ V_obs + 1e-6 * np.eye(n_features),
                                          V_obs.T @ R_obs)

            # Update V
            for j in range(n_items):
                # Get observed users for item j
                observed_users = mask[:, j]
                if np.sum(observed_users) > 0:
                    U_obs = U[observed_users]
                    R_obs = R[observed_users, j]

                    # Solve least squares
                    V[j] = np.linalg.solve(U_obs.T @ U_obs + 1e-6 * np.eye(n_features),
                                          U_obs.T @ R_obs)

            # Compute error
            R_pred = U @ V.T
            error = np.sqrt(np.sum(mask * (R - R_pred)**2) / np.sum(mask))
            errors.append(error)

            if iteration > 0 and abs(errors[-2] - errors[-1]) < tol:
                break

        return U, V, errors

    # Perform matrix completion
    U_pred, V_pred, errors = alternating_least_squares(R_noisy, mask, n_features)

    # Evaluate
    R_pred = U_pred @ V_pred.T
    test_mask = ~mask  # Use missing entries as test set

    if np.sum(test_mask) > 0:
        test_rmse = np.sqrt(np.sum(test_mask * (R_noisy - R_pred)**2) / np.sum(test_mask))
        print(f"\nTest RMSE: {test_rmse:.4f}")

    print(f"Final training RMSE: {errors[-1]:.4f}")
    print(f"Iterations: {len(errors)}")

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Original matrix
    im1 = ax1.imshow(R_true, cmap='RdYlBu', aspect='auto')
    ax1.set_title('True Matrix')
    plt.colorbar(im1, ax=ax1)

    # Observed matrix
    R_display = R_observed.copy()
    R_display[~mask] = 0
    im2 = ax2.imshow(R_display, cmap='RdYlBu', aspect='auto')
    ax2.set_title('Observed Matrix')
    plt.colorbar(im2, ax=ax2)

    # Predicted matrix
    im3 = ax3.imshow(R_pred, cmap='RdYlBu', aspect='auto')
    ax3.set_title('Predicted Matrix')
    plt.colorbar(im3, ax=ax3)

    # Error convergence
    ax4.plot(range(len(errors)), errors, 'b-o')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Convergence')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return U_pred, V_pred, errors

matrix_completion_application()
```

### 6.2 Stochastic Optimization in Deep Learning

```python
def stochastic_optimization_demo():
    """Stochastic optimization methods comparison"""
    print("\n=== Stochastic Optimization Methods ===")

    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)

    # Add L2 regularization problem
    def objective(w, lambda_reg=0.01):
        """Ridge regression objective"""
        return 0.5 * np.mean((X @ w - y)**2) + 0.5 * lambda_reg * np.sum(w**2)

    def gradient(w, lambda_reg=0.01):
        """Gradient of ridge regression"""
        return (X.T @ (X @ w - y)) / n_samples + lambda_reg * w

    def stochastic_gradient(w, idx, lambda_reg=0.01, batch_size=1):
        """Stochastic gradient"""
        if batch_size == 1:
            xi, yi = X[idx:idx+1], y[idx:idx+1]
            return xi.T @ (xi @ w - yi) + lambda_reg * w
        else:
            batch_idx = idx[:batch_size]
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            return (X_batch.T @ (X_batch @ w - y_batch)) / batch_size + lambda_reg * w

    # Optimization algorithms
    def gd(f, grad, w0, learning_rate=0.01, max_iter=1000):
        """Gradient descent"""
        w = w0.copy()
        history = [w.copy()]
        losses = [f(w)]

        for i in range(max_iter):
            g = grad(w)
            w = w - learning_rate * g
            history.append(w.copy())
            losses.append(f(w))

        return w, np.array(history), np.array(losses)

    def sgd(f, sgrad, w0, learning_rate=0.01, max_iter=1000, batch_size=1):
        """Stochastic gradient descent"""
        w = w0.copy()
        history = [w.copy()]
        losses = [f(w)]

        n_batches = n_samples // batch_size

        for i in range(max_iter):
            idx = np.random.permutation(n_samples)

            for j in range(n_batches):
                batch_idx = idx[j*batch_size:(j+1)*batch_size]
                g = stochastic_gradient(w, batch_idx, batch_size=batch_size)
                w = w - learning_rate * g

            history.append(w.copy())
            losses.append(f(w))

        return w, np.array(history), np.array(losses)

    def sgd_momentum(f, sgrad, w0, learning_rate=0.01, momentum=0.9, max_iter=1000):
        """SGD with momentum"""
        w = w0.copy()
        v = np.zeros_like(w)
        history = [w.copy()]
        losses = [f(w)]

        for i in range(max_iter):
            idx = np.random.randint(0, n_samples)
            g = stochastic_gradient(w, idx)

            v = momentum * v + learning_rate * g
            w = w - v

            history.append(w.copy())
            losses.append(f(w))

        return w, np.array(history), np.array(losses)

    def adam(f, sgrad, w0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
        """Adam optimizer"""
        w = w0.copy()
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        t = 0

        history = [w.copy()]
        losses = [f(w)]

        for i in range(max_iter):
            t += 1
            idx = np.random.randint(0, n_samples)
            g = stochastic_gradient(w, idx)

            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            w = w - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            history.append(w.copy())
            losses.append(f(w))

        return w, np.array(history), np.array(losses)

    # Initialize
    w0 = np.random.randn(n_features)

    print(f"Problem: Ridge regression with {n_samples} samples, {n_features} features")
    print(f"True weights norm: {np.linalg.norm(true_weights):.4f}")

    # Run algorithms
    print("\nRunning optimization algorithms...")

    w_gd, hist_gd, loss_gd = gd(objective, gradient, w0, learning_rate=0.1, max_iter=500)
    w_sgd, hist_sgd, loss_sgd = sgd(objective, stochastic_gradient, w0, learning_rate=0.01, max_iter=500)
    w_momentum, hist_momentum, loss_momentum = sgd_momentum(objective, stochastic_gradient, w0,
                                                          learning_rate=0.01, max_iter=500)
    w_adam, hist_adam, loss_adam = adam(objective, stochastic_gradient, w0, learning_rate=0.001, max_iter=500)

    # Results
    print(f"\nResults:")
    print(f"GD final loss: {loss_gd[-1]:.6f}, error: {np.linalg.norm(w_gd - true_weights):.4f}")
    print(f"SGD final loss: {loss_sgd[-1]:.6f}, error: {np.linalg.norm(w_sgd - true_weights):.4f}")
    print(f"Momentum final loss: {loss_momentum[-1]:.6f}, error: {np.linalg.norm(w_momentum - true_weights):.4f}")
    print(f"Adam final loss: {loss_adam[-1]:.6f}, error: {np.linalg.norm(w_adam - true_weights):.4f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss convergence
    ax1.semilogy(loss_gd, label='GD', linewidth=2)
    ax1.semilogy(loss_sgd, label='SGD', linewidth=2)
    ax1.semilogy(loss_momentum, label='SGD+Momentum', linewidth=2)
    ax1.semilogy(loss_adam, label='Adam', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Convergence')
    ax1.legend()
    ax1.grid(True)

    # Weight convergence (project to 2D)
    def project_to_2d(hist):
        """Project weights to 2D using PCA"""
        weights_matrix = np.array(hist)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        return pca.fit_transform(weights_matrix)

    proj_gd = project_to_2d(hist_gd)
    proj_sgd = project_to_2d(hist_sgd)
    proj_momentum = project_to_2d(hist_momentum)
    proj_adam = project_to_2d(hist_adam)

    ax2.plot(proj_gd[:, 0], proj_gd[:, 1], 'b-', label='GD', linewidth=2, alpha=0.7)
    ax2.plot(proj_sgd[:, 0], proj_sgd[:, 1], 'r-', label='SGD', linewidth=2, alpha=0.7)
    ax2.plot(proj_momentum[:, 0], proj_momentum[:, 1], 'g-', label='Momentum', linewidth=2, alpha=0.7)
    ax2.plot(proj_adam[:, 0], proj_adam[:, 1], 'm-', label='Adam', linewidth=2, alpha=0.7)

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Optimization Trajectories (2D Projection)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'gd': (w_gd, loss_gd),
        'sgd': (w_sgd, loss_sgd),
        'momentum': (w_momentum, loss_momentum),
        'adam': (w_adam, loss_adam)
    }

stochastic_optimization_demo()
```

## Key Takeaways

1. **Numerical Stability**: Understanding floating-point arithmetic and condition numbers is crucial for implementing stable algorithms.

2. **Matrix Factorizations**: Different decompositions (LU, QR, Cholesky) are suited for different types of problems and matrices.

3. **Iterative Methods**: For large systems, iterative methods like Jacobi, Gauss-Seidel, and Conjugate Gradient can be more efficient than direct methods.

4. **Optimization**: Newton's method provides quadratic convergence but requires Hessians, while quasi-Newton methods offer a good compromise.

5. **Numerical Integration/Differentiation**: Different methods offer different trade-offs between accuracy and computational cost.

6. **Machine Learning Applications**: These numerical methods form the foundation of many ML algorithms, from matrix completion to stochastic optimization in deep learning.

## Exercises

1. **Floating Point Exercise**: Implement compensated summation and compare its accuracy with naive summation for adding many small numbers to a large one.

2. **Matrix Factorization**: Compare the numerical stability of different methods for solving linear systems with various condition numbers.

3. **Iterative Methods**: Implement Successive Over-Relaxation (SOR) and compare its convergence with Jacobi and Gauss-Seidel methods.

4. **Optimization**: Implement the L-BFGS method and compare its memory usage and convergence with BFGS.

5. **Integration**: Implement adaptive quadrature methods and compare their efficiency with fixed-step methods.

6. **Machine Learning**: Use numerical methods to implement a simple neural network from scratch, focusing on numerical stability.