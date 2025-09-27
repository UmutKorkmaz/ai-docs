# Linear Algebra for Machine Learning

## Overview

Linear algebra is the mathematical foundation of machine learning, providing the tools to represent and manipulate data in high-dimensional spaces. This section covers the essential linear algebra concepts needed for understanding machine learning algorithms.

## 1. Vectors and Vector Spaces

### 1.1 Basic Vector Operations

**Definition**: A vector is a mathematical object that has both magnitude and direction. In machine learning, vectors typically represent data points, features, or model parameters.

```python
import numpy as np
import matplotlib.pyplot as plt

# Vector operations
def vector_operations():
    # Create vectors
    v1 = np.array([2, 3, 1])
    v2 = np.array([1, 4, 2])

    print("Vector v1:", v1)
    print("Vector v2:", v2)

    # Vector addition
    v_sum = v1 + v2
    print("Sum:", v_sum)

    # Scalar multiplication
    v_scaled = 2 * v1
    print("Scaled (2×v1):", v_scaled)

    # Dot product
    dot_product = np.dot(v1, v2)
    print("Dot product:", dot_product)

    # Vector magnitude (norm)
    norm_v1 = np.linalg.norm(v1)
    print("Norm of v1:", norm_v1)

    # Vector subtraction
    v_diff = v1 - v2
    print("Difference:", v_diff)

    return v1, v2, v_sum, v_scaled, dot_product, norm_v1, v_diff

# Execute and display
v1, v2, v_sum, v_scaled, dot_product, norm_v1, v_diff = vector_operations()
```

**Key Properties**:
- **Commutativity**: $\vec{v} + \vec{w} = \vec{w} + \vec{v}$
- **Associativity**: $(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})$
- **Distributivity**: $c(\vec{v} + \vec{w}) = c\vec{v} + c\vec{w}$

### 1.2 Vector Norms

Vector norms measure the "length" or "magnitude" of a vector. Different norms are used in different machine learning contexts.

```python
def vector_norms():
    v = np.array([3, 4])

    # L2 norm (Euclidean norm)
    l2_norm = np.linalg.norm(v, 2)
    print(f"L2 norm: {l2_norm:.3f}")

    # L1 norm (Manhattan norm)
    l1_norm = np.linalg.norm(v, 1)
    print(f"L1 norm: {l1_norm:.3f}")

    # L∞ norm (maximum norm)
    inf_norm = np.linalg.norm(v, np.inf)
    print(f"L∞ norm: {inf_norm:.3f}")

    # Lp norm general formula
    def lp_norm(vector, p):
        return np.power(np.sum(np.abs(vector) ** p), 1/p)

    l3_norm = lp_norm(v, 3)
    print(f"L3 norm: {l3_norm:.3f}")

    return l2_norm, l1_norm, inf_norm

l2_norm, l1_norm, inf_norm = vector_norms()
```

**Mathematical Definitions**:
- **L1 Norm**: $||\vec{v}||_1 = \sum_{i=1}^n |v_i|$
- **L2 Norm**: $||\vec{v}||_2 = \sqrt{\sum_{i=1}^n v_i^2}$
- **L∞ Norm**: $||\vec{v}||_\infty = \max_{i} |v_i|$
- **Lp Norm**: $||\vec{v}||_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p}$

### 1.3 Vector Spaces and Subspaces

A vector space is a set of vectors that is closed under addition and scalar multiplication.

```python
def vector_space_example():
    # R^3 - 3-dimensional Euclidean space
    basis_vectors = np.array([[1, 0, 0],  # x-axis
                             [0, 1, 0],  # y-axis
                             [0, 0, 1]]) # z-axis

    print("Standard basis for R^3:")
    print(basis_vectors)

    # Linear combination
    coefficients = [2, -1, 3]
    linear_combination = np.dot(coefficients, basis_vectors)
    print(f"Linear combination: {linear_combination}")

    # Check if vectors form a basis (linear independence)
    def check_linear_independence(vectors):
        det = np.linalg.det(vectors)
        return abs(det) > 1e-10

    is_basis = check_linear_independence(basis_vectors)
    print(f"Standard basis is linearly independent: {is_basis}")

    return basis_vectors, linear_combination

basis_vectors, linear_combination = vector_space_example()
```

## 2. Matrices and Matrix Operations

### 2.1 Basic Matrix Operations

```python
def matrix_operations():
    # Create matrices
    A = np.array([[2, 1, 3],
                  [1, 4, 2],
                  [3, 2, 1]])

    B = np.array([[1, 2, 1],
                  [2, 1, 2],
                  [1, 2, 1]])

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # Matrix addition
    A_plus_B = A + B
    print("\nA + B:")
    print(A_plus_B)

    # Matrix multiplication
    A_times_B = np.dot(A, B)
    print("\nA × B:")
    print(A_times_B)

    # Scalar multiplication
    scaled_A = 3 * A
    print("\n3 × A:")
    print(scaled_A)

    # Matrix transpose
    A_transpose = A.T
    print("\nA^T:")
    print(A_transpose)

    # Trace (sum of diagonal elements)
    trace_A = np.trace(A)
    print(f"\nTrace of A: {trace_A}")

    return A, B, A_plus_B, A_times_B, scaled_A, A_transpose, trace_A

A, B, A_plus_B, A_times_B, scaled_A, A_transpose, trace_A = matrix_operations()
```

**Matrix Properties**:
- **Associativity**: $(AB)C = A(BC)$
- **Distributivity**: $A(B + C) = AB + AC$
- **Non-commutativity**: $AB \neq BA$ (generally)
- **Transpose properties**: $(AB)^T = B^T A^T$

### 2.2 Matrix Types and Special Matrices

```python
def special_matrices():
    n = 3

    # Identity matrix
    I = np.eye(n)
    print("Identity matrix:")
    print(I)

    # Zero matrix
    Z = np.zeros((n, n))
    print("\nZero matrix:")
    print(Z)

    # Diagonal matrix
    D = np.diag([1, 2, 3])
    print("\nDiagonal matrix:")
    print(D)

    # Symmetric matrix
    symmetric = np.array([[1, 2, 3],
                        [2, 5, 6],
                        [3, 6, 9]])
    print("\nSymmetric matrix:")
    print(symmetric)

    # Orthogonal matrix (A^T = A^-1)
    # Create a random orthogonal matrix using QR decomposition
    random_matrix = np.random.randn(n, n)
    Q, R = np.linalg.qr(random_matrix)
    print("\nOrthogonal matrix:")
    print(Q)

    # Verify orthogonality
    identity_check = np.dot(Q.T, Q)
    print("\nQ^T × Q (should be identity):")
    print(np.round(identity_check, 10))

    return I, Z, D, symmetric, Q

I, Z, D, symmetric, Q = special_matrices()
```

## 3. Matrix Decompositions

### 3.1 Eigenvalue Decomposition

Eigenvalue decomposition is fundamental to many machine learning algorithms, including PCA and spectral clustering.

```python
def eigenvalue_decomposition():
    # Create a symmetric matrix (guaranteed real eigenvalues)
    A = np.array([[4, 2, 2],
                  [2, 5, 1],
                  [2, 1, 6]])

    print("Matrix A:")
    print(A)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\nEigenvectors (columns):")
    print(eigenvectors)

    # Verify decomposition: A = PDP^(-1)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)

    reconstructed = P @ D @ P_inv
    print("\nReconstructed matrix A = PDP^(-1):")
    print(np.round(reconstructed, 10))

    # Verify eigenvector equation: Av = λv
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_v = eigenvalues[i]
        Av = A @ v
        lambda_v_calculated = Av / v[0]  # Using first component

        print(f"\nEigenpair {i+1}:")
        print(f"λ = {lambda_v:.6f}")
        print(f"Av = λv check: {np.allclose(Av, lambda_v * v)}")

    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = eigenvalue_decomposition()
```

**Eigenvalue Properties**:
- **Characteristic equation**: $\det(A - \lambda I) = 0$
- **Trace**: $\text{tr}(A) = \sum_{i=1}^n \lambda_i$
- **Determinant**: $\det(A) = \prod_{i=1}^n \lambda_i$
- **Spectral theorem**: For symmetric matrices, eigenvalues are real and eigenvectors are orthogonal

### 3.2 Singular Value Decomposition (SVD)

SVD is one of the most important matrix decompositions in machine learning, used in dimensionality reduction, recommendation systems, and many other applications.

```python
def singular_value_decomposition():
    # Create a matrix
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])

    print("Matrix A:")
    print(A)

    # SVD decomposition
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    print("\nU matrix:")
    print(U)

    print("\nSingular values:")
    print(s)

    print("\nV^T matrix:")
    print(Vt)

    # Reconstruct matrix A = UΣV^T
    Sigma = np.zeros(A.shape)
    Sigma[:len(s), :len(s)] = np.diag(s)

    reconstructed = U @ Sigma @ Vt
    print("\nReconstructed matrix A = UΣV^T:")
    print(np.round(reconstructed, 10))

    # Rank of matrix
    rank = np.linalg.matrix_rank(A)
    print(f"\nRank of A: {rank}")

    # Rank approximation (low-rank approximation)
    k = 2  # Use only 2 singular values
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    Sigma_k = np.zeros((A.shape[0], A.shape[1]))
    Sigma_k[:k, :k] = np.diag(s_k)

    A_k = U_k @ Sigma_k @ Vt_k
    print(f"\nRank-{k} approximation:")
    print(A_k)

    # Calculate approximation error
    error = np.linalg.norm(A - A_k, 'fro')
    print(f"\nFrobenius norm error: {error:.6f}")

    return U, s, Vt, A_k

U, s, Vt, A_k = singular_value_decomposition()
```

**SVD Properties**:
- **Full SVD**: $A = U\Sigma V^T$
- **Reduced SVD**: $A = U_r \Sigma_r V_r^T$ where $r = \text{rank}(A)$
- **Frobenius norm**: $||A||_F = \sqrt{\sum_{i=1}^r \sigma_i^2}$
- **Spectral norm**: $||A||_2 = \sigma_1$ (largest singular value)

### 3.3 QR Decomposition

QR decomposition is used in solving linear systems, least squares problems, and eigenvalue algorithms.

```python
def qr_decomposition():
    # Create a square matrix
    A = np.array([[1, 2, 3],
                  [1, 1, 1],
                  [0, 1, 2]])

    print("Matrix A:")
    print(A)

    # QR decomposition
    Q, R = np.linalg.qr(A)

    print("\nQ matrix (orthogonal):")
    print(Q)

    print("\nR matrix (upper triangular):")
    print(R)

    # Verify Q is orthogonal
    print("\nQ^T × Q (should be identity):")
    print(np.round(Q.T @ Q, 10))

    # Reconstruct A = QR
    reconstructed = Q @ R
    print("\nReconstructed matrix A = QR:")
    print(np.round(reconstructed, 10))

    # Solving linear system Ax = b using QR
    b = np.array([6, 3, 3])
    print(f"\nVector b: {b}")

    # Solve Rx = Q^T b
    x = np.linalg.solve(R, Q.T @ b)
    print(f"Solution x: {x}")

    # Verify solution
    print(f"Verification: A @ x = {A @ x}")
    print(f"Should equal b: {b}")
    print(f"Solution correct: {np.allclose(A @ x, b)}")

    return Q, R, x

Q, R, x = qr_decomposition()
```

## 4. Linear Transformations and Projections

### 4.1 Linear Transformations

```python
def linear_transformations():
    # Define some linear transformations

    # Rotation matrix (45 degrees)
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    # Scaling matrix
    scaling = np.array([[2, 0],
                       [0, 0.5]])

    # Shear matrix
    shear = np.array([[1, 0.5],
                     [0, 1]])

    # Create some vectors to transform
    vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]])

    # Apply transformations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    transformations = [
        ("Identity", np.eye(2)),
        ("Rotation (45°)", rotation),
        ("Scaling (2, 0.5)", scaling),
        ("Shear", shear)
    ]

    for idx, (name, transform) in enumerate(transformations):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        # Original vectors
        orig_x = [0, vectors[i, 0] for i in range(len(vectors))]
        orig_y = [0, vectors[i, 1] for i in range(len(vectors))]

        # Transformed vectors
        transformed = np.array([transform @ v for v in vectors])
        trans_x = [0, transformed[i, 0] for i in range(len(transformed))]
        trans_y = [0, transformed[i, 1] for i in range(len(transformed))]

        # Plot
        for i in range(len(vectors)):
            ax.arrow(0, 0, orig_x[i+1], orig_y[i+1],
                    head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.7)
            ax.arrow(0, 0, trans_x[i+1], trans_y[i+1],
                    head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{name}\nBlue: Original, Red: Transformed')

    plt.tight_layout()
    plt.show()

    return transformations

transformations = linear_transformations()
```

### 4.2 Projection Matrices

Projections are used extensively in machine learning, particularly in dimensionality reduction and optimization.

```python
def projection_matrices():
    # Projection onto a line
    # Define a direction vector
    v = np.array([1, 2])
    v_normalized = v / np.linalg.norm(v)

    # Projection matrix onto span{v}
    P_line = np.outer(v_normalized, v_normalized)
    print("Projection matrix onto line:")
    print(P_line)

    # Test with some vectors
    test_vectors = np.array([[1, 0], [0, 1], [3, 4]])
    print("\nTesting projections:")

    for i, x in enumerate(test_vectors):
        projection = P_line @ x
        print(f"Vector {x} -> projection {projection}")
        print(f"Distance to line: {np.linalg.norm(x - projection):.4f}")

    # Projection onto a subspace (column space)
    A = np.array([[1, 2], [3, 4], [5, 6]])
    print("\n\nProjection onto column space of A:")
    print("Matrix A:")
    print(A)

    # Projection matrix: P = A(A^T A)^(-1) A^T
    P_subspace = A @ np.linalg.inv(A.T @ A) @ A.T
    print("\nProjection matrix:")
    print(P_subspace)

    # Test projection
    b = np.array([1, 2, 3])
    projection = P_subspace @ b
    print(f"\nVector b: {b}")
    print(f"Projection: {projection}")
    print(f"Residual: {b - projection}")
    print(f"Check orthogonality (residual perpendicular to A): {np.allclose(A.T @ (b - projection), 0)}")

    return P_line, P_subspace

P_line, P_subspace = projection_matrices()
```

**Projection Properties**:
- **Idempotent**: $P^2 = P$
- **Symmetric**: $P^T = P$ (for orthogonal projections)
- **Range**: $\text{range}(P) = \text{col}(A)$
- **Null space**: $\text{null}(P) = \text{col}(A)^\perp$

## 5. Applications in Machine Learning

### 5.1 Principal Component Analysis (PCA)

PCA uses eigenvalue decomposition to find the principal components of data.

```python
def pca_example():
    # Generate sample data
    np.random.seed(42)
    n_samples = 100

    # Create correlated data
    x1 = np.random.randn(n_samples)
    x2 = 0.8 * x1 + 0.2 * np.random.randn(n_samples)

    X = np.column_stack([x1, x2])
    X_centered = X - X.mean(axis=0)

    # PCA using eigenvalue decomposition of covariance matrix
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("Covariance matrix:")
    print(cov_matrix)

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\nEigenvectors (principal components):")
    print(eigenvectors)

    # Project data onto principal components
    projected_data = X_centered @ eigenvectors

    # Calculate explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance

    print("\nExplained variance ratio:")
    print(explained_variance)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original data
    ax1.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.7)
    ax1.quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0],
               color='red', scale=3, width=0.01, label='PC1')
    ax1.quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1],
               color='blue', scale=3, width=0.01, label='PC2')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Original Data with Principal Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Transformed data
    ax2.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.7)
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_title('Transformed Data (PCA Space)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return eigenvalues, eigenvectors, projected_data

eigenvalues, eigenvectors, projected_data = pca_example()
```

### 5.2 Linear Regression

Linear regression can be solved using matrix operations.

```python
def linear_regression_matrix():
    # Generate sample data
    np.random.seed(42)
    n_samples = 100

    # True parameters
    true_slope = 2.5
    true_intercept = 1.0

    # Generate data with noise
    x = np.random.randn(n_samples)
    noise = 0.5 * np.random.randn(n_samples)
    y = true_slope * x + true_intercept + noise

    # Create design matrix X (with intercept term)
    X = np.column_stack([np.ones(n_samples), x])

    print("Design matrix X (first 5 rows):")
    print(X[:5])

    # Normal equation: β = (X^T X)^(-1) X^T y
    XTX = X.T @ X
    XTy = X.T @ y

    # Solve for coefficients
    beta = np.linalg.solve(XTX, XTy)

    print("\nTrue parameters:")
    print(f"Intercept: {true_intercept}, Slope: {true_slope}")

    print("\nEstimated parameters:")
    print(f"Intercept: {beta[0]:.4f}, Slope: {beta[1]:.4f}")

    # Alternative: Using pseudo-inverse
    beta_pinv = np.linalg.pinv(X) @ y
    print("\nUsing pseudo-inverse:")
    print(f"Intercept: {beta_pinv[0]:.4f}, Slope: {beta_pinv[1]:.4f}")

    # Calculate predictions and R-squared
    predictions = X @ beta
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    print(f"\nR-squared: {r_squared:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, label='Data points')
    plt.plot(x, predictions, 'r-', linewidth=2, label='Linear regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression using Matrix Operations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return beta, r_squared

beta, r_squared = linear_regression_matrix()
```

## 6. Advanced Topics

### 6.1 Tensor Operations

Tensors are multi-dimensional arrays that generalize vectors and matrices.

```python
def tensor_operations():
    # Create tensors (3D arrays)
    # Think of this as a 2x2x3 tensor
    tensor_3d = np.array([[[1, 2, 3], [4, 5, 6]],
                         [[7, 8, 9], [10, 11, 12]]])

    print("3D tensor:")
    print(tensor_3d)
    print(f"Shape: {tensor_3d.shape}")

    # Tensor operations
    # Element-wise operations
    tensor_squared = tensor_3d ** 2
    print("\nElement-wise square:")
    print(tensor_squared)

    # Sum along different axes
    sum_axis0 = np.sum(tensor_3d, axis=0)  # Sum along first dimension
    sum_axis1 = np.sum(tensor_3d, axis=1)  # Sum along second dimension
    sum_axis2 = np.sum(tensor_3d, axis=2)  # Sum along third dimension

    print("\nSum along axis 0:")
    print(sum_axis0)

    print("\nSum along axis 1:")
    print(sum_axis1)

    print("\nSum along axis 2:")
    print(sum_axis2)

    # Tensor contraction (matrix multiplication generalized)
    # Create two tensors for multiplication
    A = np.random.randn(3, 4, 5)
    B = np.random.randn(5, 6, 7)

    # Contract over the last dimension of A and first of B
    # Result shape: (3, 4, 6, 7)
    result = np.einsum('ijk,klm->ijlm', A, B)

    print(f"\nTensor contraction:")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Result shape: {result.shape}")

    return tensor_3d, sum_axis0, sum_axis1, sum_axis2, result

tensor_3d, sum_axis0, sum_axis1, sum_axis2, result = tensor_operations()
```

### 6.2 Matrix Calculus

Matrix calculus is essential for understanding optimization in machine learning.

```python
def matrix_calculus():
    # Example: Gradient of linear function f(x) = a^T x
    a = np.array([1, 2, 3])
    x = np.array([4, 5, 6])

    f = np.dot(a, x)
    print(f"Linear function f(x) = a^T x")
    print(f"a = {a}")
    print(f"x = {x}")
    print(f"f(x) = {f}")

    # Gradient ∇f = a
    gradient = a
    print(f"\nGradient ∇f = {gradient}")

    # Example: Gradient of quadratic form f(x) = x^T A x
    A = np.array([[2, 1], [1, 3]])
    x = np.array([1, 2])

    f_quadratic = x.T @ A @ x
    print(f"\nQuadratic form f(x) = x^T A x")
    print(f"A = {A}")
    print(f"x = {x}")
    print(f"f(x) = {f_quadratic}")

    # Gradient ∇f = (A + A^T) x
    gradient_quadratic = (A + A.T) @ x
    print(f"\nGradient ∇f = {gradient_quadratic}")

    # Numerical gradient verification
    epsilon = 1e-6
    numerical_gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        f_plus = x_plus.T @ A @ x_plus
        f_minus = x_minus.T @ A @ x_minus

        numerical_gradient[i] = (f_plus - f_minus) / (2 * epsilon)

    print(f"Numerical gradient: {numerical_gradient}")
    print(f"Analytical gradient: {gradient_quadratic}")
    print(f"Match: {np.allclose(numerical_gradient, gradient_quadratic)}")

    return gradient, gradient_quadratic

gradient, gradient_quadratic = matrix_calculus()
```

## 7. Key Concepts Summary

### 7.1 Essential Linear Algebra for ML

1. **Vectors**: Represent data points, features, parameters
2. **Matrices**: Represent transformations, datasets, weight matrices
3. **Decompositions**:
   - SVD for dimensionality reduction
   - Eigenvalue decomposition for PCA
   - QR for solving linear systems
4. **Projections**: Used in dimensionality reduction and optimization
5. **Matrix calculus**: Essential for understanding optimization algorithms

### 7.2 Important Theorems

- **Spectral Theorem**: Real symmetric matrices have real eigenvalues and orthogonal eigenvectors
- **Rank-Nullity Theorem**: $\text{rank}(A) + \text{nullity}(A) = n$ for $n \times n$ matrix
- **Cayley-Hamilton Theorem**: Every matrix satisfies its characteristic equation
- **Fundamental Theorem of Linear Algebra**: Relationship between row space, column space, and null spaces

### 7.3 Computational Considerations

- **Complexity**: Matrix multiplication is $O(n^3)$ for $n \times n$ matrices
- **Numerical Stability**: Condition number affects accuracy of computations
- **Sparsity**: Exploiting sparsity can dramatically improve performance
- **Parallelization**: Many linear algebra operations can be parallelized

## 8. Exercises

### 8.1 Theory Exercises

1. Prove that the eigenvalues of a symmetric matrix are real.
2. Show that for an orthogonal matrix $Q$, $Q^T Q = I$.
3. Derive the normal equation for linear regression.
4. Prove the properties of projection matrices (idempotent, symmetric).
5. Show that the trace of a matrix equals the sum of its eigenvalues.

### 8.2 Programming Exercises

```python
def linear_algebra_exercises():
    """
    Complete these exercises to test your understanding:

    Exercise 1: Implement matrix multiplication without using np.dot
    Exercise 2: Implement eigenvalue decomposition using power iteration
    Exercise 3: Implement SVD and compare with np.linalg.svd
    Exercise 4: Implement PCA from scratch
    Exercise 5: Implement linear regression using gradient descent
    """

    # Exercise 1: Matrix multiplication
    def matrix_multiply(A, B):
        """Implement matrix multiplication without using np.dot"""
        m, n = A.shape
        p, q = B.shape

        if n != p:
            raise ValueError("Matrix dimensions incompatible for multiplication")

        result = np.zeros((m, q))

        for i in range(m):
            for j in range(q):
                for k in range(n):
                    result[i, j] += A[i, k] * B[k, j]

        return result

    # Test Exercise 1
    A_test = np.array([[1, 2], [3, 4]])
    B_test = np.array([[5, 6], [7, 8]])

    manual_result = matrix_multiply(A_test, B_test)
    numpy_result = A_test @ B_test

    print("Exercise 1: Matrix Multiplication")
    print(f"Manual result:\n{manual_result}")
    print(f"NumPy result:\n{numpy_result}")
    print(f"Match: {np.allclose(manual_result, numpy_result)}")

    # Exercise 2: Power iteration for largest eigenvalue
    def power_iteration(A, num_iterations=1000, tolerance=1e-10):
        """Find largest eigenvalue and corresponding eigenvector"""
        n = A.shape[0]

        # Random initial vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)

        for i in range(num_iterations):
            # Power iteration step
            v_new = A @ v
            v_new = v_new / np.linalg.norm(v_new)

            # Check convergence
            if np.linalg.norm(v_new - v) < tolerance:
                break

            v = v_new

        # Rayleigh quotient for eigenvalue
        eigenvalue = (v.T @ A @ v) / (v.T @ v)

        return eigenvalue, v

    # Test Exercise 2
    A_symmetric = np.array([[4, 2], [2, 3]])
    lambda_power, v_power = power_iteration(A_symmetric)
    lambda_numpy, v_numpy = np.linalg.eig(A_symmetric)

    # Get largest eigenvalue from NumPy
    largest_idx = np.argmax(np.abs(lambda_numpy))
    lambda_largest = lambda_numpy[largest_idx]
    v_largest = v_numpy[:, largest_idx]

    print("\nExercise 2: Power Iteration")
    print(f"Power iteration: λ = {lambda_power:.6f}")
    print(f"NumPy largest: λ = {lambda_largest:.6f}")
    print(f"Eigenvalue match: {np.isclose(lambda_power, lambda_largest)}")

    return manual_result, lambda_power, v_power

manual_result, lambda_power, v_power = linear_algebra_exercises()
```

This comprehensive guide covers the essential linear algebra concepts needed for machine learning, from basic vector operations to advanced topics like tensor operations and matrix calculus. Each section includes mathematical explanations, Python implementations, and real-world applications in machine learning.