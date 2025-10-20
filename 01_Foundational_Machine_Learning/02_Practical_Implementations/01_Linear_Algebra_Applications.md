---
title: "Foundational Machine Learning - Linear Algebra Applications"
description: "## Overview. Comprehensive guide covering classification, algorithms, model training, neural networks, regression. Part of AI documentation system with 1500+..."
keywords: "neural networks, classification, regression, classification, algorithms, model training, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Linear Algebra Applications in Machine Learning

## Overview

This section demonstrates practical applications of linear algebra in machine learning through complete, working implementations. We'll explore how linear algebra concepts are used in real ML algorithms and provide hands-on examples that you can run and modify.

## 1. Data Representation and Preprocessing

### 1.1 Vector and Matrix Representations

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def data_representation_basics():
    """Understanding how data is represented as vectors and matrices"""

    print("Data Representation in Linear Algebra")
    print("=" * 50)

    # 1. Feature Vectors
    print("\n1. Feature Vectors")
    print("-" * 20)

    # Sample data point
    # Each data point can be represented as a vector
    sample_data = {
        'age': 25,
        'income': 50000,
        'education_years': 16,
        'experience': 3
    }

    # Convert to feature vector
    feature_vector = np.array([sample_data['age'],
                              sample_data['income'],
                              sample_data['education_years'],
                              sample_data['experience']])

    print(f"Sample data: {sample_data}")
    print(f"Feature vector: {feature_vector}")
    print(f"Vector shape: {feature_vector.shape}")
    print(f"Vector dimension: {feature_vector.shape[0]}")

    # 2. Data Matrix
    print("\n2. Data Matrix")
    print("-" * 20)

    # Create a dataset of multiple samples
    np.random.seed(42)
    n_samples = 100
    n_features = 4

    # Generate synthetic dataset
    ages = np.random.normal(35, 10, n_samples)
    incomes = np.random.normal(60000, 15000, n_samples)
    education = np.random.normal(14, 2, n_samples)
    experience = np.random.normal(5, 3, n_samples)

    # Create data matrix (each row is a sample, each column is a feature)
    data_matrix = np.column_stack([ages, incomes, education, experience])

    print(f"Data matrix shape: {data_matrix.shape}")
    print(f"Number of samples: {data_matrix.shape[0]}")
    print(f"Number of features: {data_matrix.shape[1]}")
    print("\nFirst 5 samples:")
    print(data_matrix[:5])

    # 3. Matrix Operations on Data
    print("\n3. Basic Matrix Operations")
    print("-" * 30)

    # Mean vector (feature means)
    mean_vector = np.mean(data_matrix, axis=0)
    print(f"Mean vector: {mean_vector}")

    # Center the data (subtract mean)
    centered_data = data_matrix - mean_vector
    print(f"Centered data shape: {centered_data.shape}")

    # Covariance matrix
    covariance_matrix = np.cov(centered_data.T)
    print(f"\nCovariance matrix shape: {covariance_matrix.shape}")
    print("Covariance matrix:")
    print(np.round(covariance_matrix, 2))

    # 4. Feature Scaling
    print("\n4. Feature Scaling")
    print("-" * 20)

    # Standardization (z-score normalization)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_matrix)

    print("Before standardization:")
    print(f"Mean: {np.mean(data_matrix, axis=0)}")
    print(f"Std: {np.std(data_matrix, axis=0)}")

    print("\nAfter standardization:")
    print(f"Mean: {np.mean(standardized_data, axis=0)}")
    print(f"Std: {np.std(standardized_data, axis=0)}")

    # 5. Data Visualization
    print("\n5. Data Visualization")
    print("-" * 25)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Raw data distribution
    axes[0, 0].hist(data_matrix[:, 0], bins=20, alpha=0.7, label='Age')
    axes[0, 0].hist(data_matrix[:, 1], bins=20, alpha=0.7, label='Income')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Raw Feature Distributions')
    axes[0, 0].legend()

    # Standardized data distribution
    axes[0, 1].hist(standardized_data[:, 0], bins=20, alpha=0.7, label='Age (std)')
    axes[0, 1].hist(standardized_data[:, 1], bins=20, alpha=0.7, label='Income (std)')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Standardized Feature Distributions')
    axes[0, 1].legend()

    # Covariance matrix heatmap
    im = axes[1, 0].imshow(covariance_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('Covariance Matrix Heatmap')
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_yticks(range(4))
    axes[1, 0].set_xticklabels(['Age', 'Income', 'Education', 'Experience'])
    axes[1, 0].set_yticklabels(['Age', 'Income', 'Education', 'Experience'])
    plt.colorbar(im, ax=axes[1, 0])

    # Scatter plot of first two features
    axes[1, 1].scatter(data_matrix[:, 0], data_matrix[:, 1], alpha=0.6)
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Income')
    axes[1, 1].set_title('Age vs Income')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'feature_vector': feature_vector,
        'data_matrix': data_matrix,
        'standardized_data': standardized_data,
        'covariance_matrix': covariance_matrix
    }

representation_results = data_representation_basics()
```

### 1.2 Linear Transformations in Data Preprocessing

```python
def linear_transformations_preprocessing():
    """Applying linear transformations for data preprocessing"""

    print("Linear Transformations in Data Preprocessing")
    print("=" * 60)

    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 200

    # Create correlated features
    X1 = np.random.randn(n_samples)
    X2 = 0.8 * X1 + 0.2 * np.random.randn(n_samples)
    X3 = -0.5 * X1 + 0.3 * np.random.randn(n_samples)
    X4 = 0.6 * X2 + 0.4 * np.random.randn(n_samples)

    data = np.column_stack([X1, X2, X3, X4])
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']

    print(f"Original data shape: {data.shape}")
    print("Original covariance matrix:")
    print(np.round(np.cov(data.T), 3))

    # 1. Rotation Transformation (PCA-like)
    print("\n1. Rotation Transformation")
    print("-" * 30)

    def rotation_matrix(angle):
        """Create 2D rotation matrix"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([[cos_a, -sin_a],
                        [sin_a, cos_a]])

    # Apply rotation to first two features
    angle = np.pi / 4  # 45 degrees
    rot_matrix = rotation_matrix(angle)

    # Apply rotation
    rotated_2d = data[:, :2] @ rot_matrix.T

    print(f"Rotation angle: {np.degrees(angle):.1f} degrees")
    print("Original covariance (first 2 features):")
    print(np.round(np.cov(data[:, :2].T), 3))
    print("Rotated covariance:")
    print(np.round(np.cov(rotated_2d.T), 3))

    # 2. Scaling Transformation
    print("\n2. Scaling Transformation")
    print("-" * 25)

    # Create scaling matrix
    scales = np.array([2.0, 0.5, 1.5, 0.8])
    scaling_matrix = np.diag(scales)

    scaled_data = data @ scaling_matrix

    print("Scaling factors:", scales)
    print("Original variances:", np.round(np.var(data, axis=0), 3))
    print("Scaled variances:", np.round(np.var(scaled_data, axis=0), 3))

    # 3. Projection Transformation
    print("\n3. Projection Transformation")
    print("-" * 30)

    def create_projection_matrix(direction_vector):
        """Create projection matrix onto given direction"""
        direction = direction_vector / np.linalg.norm(direction_vector)
        return np.outer(direction, direction)

    # Project onto direction [1, 1, 0, 0]
    direction = np.array([1, 1, 0, 0])
    proj_matrix = create_projection_matrix(direction)

    projected_data = data @ proj_matrix.T

    print(f"Projection direction: {direction}")
    print("Original data range (first 2 features):")
    print(f"X1: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
    print(f"X2: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
    print("Projected data range (first 2 features):")
    print(f"X1: [{projected_data[:, 0].min():.2f}, {projected_data[:, 0].max():.2f}]")
    print(f"X2: [{projected_data[:, 1].min():.2f}, {projected_data[:, 1].max():.2f}]")

    # 4. Whitening Transformation
    print("\n4. Whitening Transformation")
    print("-" * 30)

    def whiten_data(data):
        """Apply whitening transformation to data"""
        # Center data
        centered = data - np.mean(data, axis=0)

        # Calculate covariance matrix
        cov_matrix = np.cov(centered.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Whitening matrix
        whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))

        # Apply whitening
        whitened = centered @ whitening_matrix.T

        return whitened, whitening_matrix

    whitened_data, whitening_matrix = whiten_data(data)

    print("Whitened covariance matrix (should be identity):")
    print(np.round(np.cov(whitened_data.T), 3))
    print("Whitened variances:", np.round(np.var(whitened_data, axis=0), 3))

    # 5. Combined Transformations
    print("\n5. Combined Transformations")
    print("-" * 30)

    # Create a complex transformation pipeline
    # Step 1: Center
    centered = data - np.mean(data, axis=0)

    # Step 2: Rotate first two dimensions
    centered_2d = centered[:, :2]
    rotated_2d = centered_2d @ rot_matrix.T

    # Step 3: Scale
    scaled_2d = rotated_2d * np.array([1.5, 0.8])

    # Step 4: Project
    direction_combined = np.array([1, 0])
    proj_matrix_combined = create_projection_matrix(direction_combined)
    final_2d = scaled_2d @ proj_matrix_combined.T

    print("Transformation pipeline:")
    print("1. Center data")
    print("2. Rotate by 45°")
    print("3. Scale by [1.5, 0.8]")
    print("4. Project onto [1, 0]")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original data
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.6, label='Original')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')

    # Rotated data
    axes[0, 1].scatter(rotated_2d[:, 0], rotated_2d[:, 1], alpha=0.6, color='orange', label='Rotated')
    axes[0, 1].set_xlabel('Rotated Feature 1')
    axes[0, 1].set_ylabel('Rotated Feature 2')
    axes[0, 1].set_title('After 45° Rotation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')

    # Scaled data
    axes[0, 2].scatter(scaled_2d[:, 0], scaled_2d[:, 1], alpha=0.6, color='green', label='Scaled')
    axes[0, 2].set_xlabel('Scaled Feature 1')
    axes[0, 2].set_ylabel('Scaled Feature 2')
    axes[0, 2].set_title('After Scaling')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Whitened data (first two dimensions)
    axes[1, 0].scatter(whitened_data[:, 0], whitened_data[:, 1], alpha=0.6, color='purple', label='Whitened')
    axes[1, 0].set_xlabel('Whitened Feature 1')
    axes[1, 0].set_ylabel('Whitened Feature 2')
    axes[1, 0].set_title('Whitened Data')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # Covariance matrices comparison
    cov_original = np.cov(data.T)
    cov_whitened = np.cov(whitened_data.T)

    im1 = axes[1, 1].imshow(cov_original, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('Original Covariance Matrix')
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_yticks(range(4))
    plt.colorbar(im1, ax=axes[1, 1])

    im2 = axes[1, 2].imshow(cov_whitened, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('Whitened Covariance Matrix')
    axes[1, 2].set_xticks(range(4))
    axes[1, 2].set_yticks(range(4))
    plt.colorbar(im2, ax=axes[1, 2])

    plt.tight_layout()
    plt.show()

    return {
        'original_data': data,
        'rotated_data': rotated_2d,
        'scaled_data': scaled_data,
        'whitened_data': whitened_data,
        'whitening_matrix': whitening_matrix
    }

transformation_results = linear_transformations_preprocessing()
```

## 2. Linear Regression Implementation

### 2.1 Ordinary Least Squares from Scratch

```python
def linear_regression_implementation():
    """Complete implementation of linear regression using linear algebra"""

    print("Linear Regression Implementation")
    print("=" * 40)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    # True coefficients
    true_coefficients = np.array([2.5, -1.8, 0.7])
    true_intercept = 1.2

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate target with noise
    noise = np.random.randn(n_samples) * 0.5
    y = X @ true_coefficients + true_intercept + noise

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"True coefficients: {true_coefficients}")
    print(f"True intercept: {true_intercept}")
    print(f"Noise standard deviation: 0.5")

    # 1. Add intercept term to design matrix
    print("\n1. Design Matrix Construction")
    print("-" * 35)

    # Add column of ones for intercept
    X_with_intercept = np.column_stack([np.ones(n_samples), X])

    print(f"Design matrix shape: {X_with_intercept.shape}")
    print("First 5 rows of design matrix:")
    print(X_with_intercept[:5])

    # 2. Normal Equation Solution
    print("\n2. Normal Equation Solution")
    print("-" * 30)

    # Normal equation: β = (X^T X)^(-1) X^T y
    XTX = X_with_intercept.T @ X_with_intercept
    XTy = X_with_intercept.T @ y

    print("XTX matrix:")
    print(np.round(XTX, 3))
    print("\nXTy vector:")
    print(np.round(XTy, 3))

    # Solve for coefficients
    coefficients_normal = np.linalg.solve(XTX, XTy)
    intercept_normal = coefficients_normal[0]
    slope_normal = coefficients_normal[1:]

    print(f"\nEstimated intercept: {intercept_normal:.3f}")
    print(f"Estimated slopes: {slope_normal}")
    print(f"True intercept: {true_intercept:.3f}")
    print(f"True slopes: {true_coefficients}")

    # 3. Alternative Methods
    print("\n3. Alternative Solution Methods")
    print("-" * 35)

    # Method 2: Pseudo-inverse
    coefficients_pinv = np.linalg.pinv(X_with_intercept) @ y
    intercept_pinv = coefficients_pinv[0]
    slope_pinv = coefficients_pinv[1:]

    # Method 3: QR Decomposition
    Q, R = np.linalg.qr(X_with_intercept)
    coefficients_qr = np.linalg.solve(R, Q.T @ y)
    intercept_qr = coefficients_qr[0]
    slope_qr = coefficients_qr[1:]

    print("Method comparison:")
    print(f"Normal equation:   intercept={intercept_normal:.3f}, slopes={slope_normal}")
    print(f"Pseudo-inverse:     intercept={intercept_pinv:.3f}, slopes={slope_pinv}")
    print(f"QR decomposition:   intercept={intercept_qr:.3f}, slopes={slope_qr}")

    # 4. Model Evaluation
    print("\n4. Model Evaluation")
    print("-" * 20)

    # Make predictions
    y_pred = X_with_intercept @ coefficients_normal

    # Calculate metrics
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    # 5. Statistical Inference
    print("\n5. Statistical Inference")
    print("-" * 25)

    # Estimate standard error of coefficients
    sigma_squared = np.sum(residuals**2) / (n_samples - n_features - 1)
    cov_coefficients = sigma_squared * np.linalg.inv(XTX)
    std_errors = np.sqrt(np.diag(cov_coefficients))

    # Calculate t-statistics
    t_statistics = coefficients_normal / std_errors

    print("Coefficient statistics:")
    print("Feature | Coefficient | Std Error | t-statistic")
    print("-" * 50)
    for i, (coef, se, t_stat) in enumerate(zip(coefficients_normal, std_errors, t_statistics)):
        feature_name = "Intercept" if i == 0 else f"Feature {i}"
        print(f"{feature_name:9s} | {coef:11.3f} | {se:9.3f} | {t_stat:10.3f}")

    # 6. Regularization (Ridge Regression)
    print("\n6. Ridge Regression (L2 Regularization)")
    print("-" * 40)

    def ridge_regression(X, y, alpha=1.0):
        """Ridge regression implementation"""
        n_samples, n_features = X.shape
        I = np.eye(n_features)
        coefficients = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
        return coefficients

    # Try different regularization strengths
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_coefficients = []

    print("Regularization strength | Coefficients")
    print("-" * 50)

    for alpha in alphas:
        coef = ridge_regression(X_with_intercept, y, alpha)
        ridge_coefficients.append(coef)
        coef_str = ", ".join([f"{c:.3f}" for c in coef])
        print(f"{alpha:21.2f} | [{coef_str}]")

    # 7. Visualization
    print("\n7. Visualization")
    print("-" * 15)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Actual vs Predicted
    axes[0, 0].scatter(y, y_pred, alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    # Coefficient paths for ridge regression
    ridge_coefficients = np.array(ridge_coefficients)
    for i in range(n_features + 1):
        label = "Intercept" if i == 0 else f"Feature {i}"
        axes[1, 0].plot(alphas, ridge_coefficients[:, i], 'o-', label=label, linewidth=2, markersize=6)

    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Regularization Strength (log scale)')
    axes[1, 0].set_ylabel('Coefficient Value')
    axes[1, 0].set_title('Ridge Regression Coefficient Paths')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'coefficients': coefficients_normal,
        'predictions': y_pred,
        'residuals': residuals,
        'metrics': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r_squared': r_squared},
        'ridge_coefficients': ridge_coefficients
    }

lr_results = linear_regression_implementation()
```

### 2.2 Multiple Linear Regression with Real Data

```python
def multiple_linear_regression_real():
    """Multiple linear regression with real-world dataset simulation"""

    print("Multiple Linear Regression with Real Data")
    print("=" * 50)

    # Generate realistic housing dataset
    np.random.seed(42)
    n_samples = 500

    # Features: square footage, bedrooms, age, location score
    square_footage = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.poisson(3, n_samples)
    age = np.random.normal(20, 10, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)

    # Create realistic price model
    # Price = 100*square_footage + 50000*bedrooms - 2000*age + 20000*location_score + noise
    base_price = 50000
    price_per_sqft = 100
    price_per_bedroom = 50000
    depreciation_per_year = 2000
    location_multiplier = 20000
    noise_std = 30000

    noise = np.random.normal(0, noise_std, n_samples)
    price = (base_price +
             price_per_sqft * square_footage +
             price_per_bedroom * bedrooms -
             depreciation_per_year * age +
             location_multiplier * location_score +
             noise)

    # Ensure positive prices
    price = np.maximum(price, 50000)

    # Create feature matrix
    X = np.column_stack([square_footage, bedrooms, age, location_score])
    y = price

    print(f"Dataset: {n_samples} housing samples")
    print(f"Features: Square Footage, Bedrooms, Age, Location Score")
    print(f"Target: House Price")
    print(f"\nData ranges:")
    for i, feature in enumerate(['Square Footage', 'Bedrooms', 'Age', 'Location Score']):
        print(f"{feature:15s}: [{X[:, i].min():.0f}, {X[:, i].max():.0f}]")
    print(f"{'House Price':15s}: [{y.min():.0f}, {y.max():.0f}]")

    # 1. Data Preprocessing
    print("\n1. Data Preprocessing")
    print("-" * 25)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(n_samples), X_scaled])

    print("Feature standardization:")
    print("Before scaling:")
    print(f"  Mean: {np.mean(X, axis=0)}")
    print(f"  Std: {np.std(X, axis=0)}")
    print("After scaling:")
    print(f"  Mean: {np.mean(X_scaled, axis=0)}")
    print(f"  Std: {np.std(X_scaled, axis=0)}")

    # 2. Model Training
    print("\n2. Model Training")
    print("-" * 20)

    # Split data (simple 80-20 split)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_with_intercept[:split_idx], X_with_intercept[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Fit model
    coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    intercept = coefficients[0]
    slopes = coefficients[1:]

    print(f"\nModel coefficients:")
    print(f"Intercept: ${intercept:,.0f}")
    for i, (feature, slope) in enumerate(zip(['Square Footage', 'Bedrooms', 'Age', 'Location Score'], slopes)):
        print(f"{feature:15s}: ${slope:,.0f} per unit (standardized)")

    # 3. Model Evaluation
    print("\n3. Model Evaluation")
    print("-" * 25)

    # Predictions
    y_train_pred = X_train @ coefficients
    y_test_pred = X_test @ coefficients

    # Metrics
    train_mse = np.mean((y_train - y_train_pred)**2)
    test_mse = np.mean((y_test - y_test_pred)**2)
    train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
    test_r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

    print(f"Training MSE: ${train_mse:,.0f}")
    print(f"Test MSE: ${test_mse:,.0f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # 4. Feature Importance Analysis
    print("\n4. Feature Importance Analysis")
    print("-" * 35)

    # Calculate feature importance based on standardized coefficients
    feature_importance = np.abs(slopes)
    feature_importance = feature_importance / np.sum(feature_importance)

    print("Feature importance (normalized):")
    for i, (feature, importance) in enumerate(zip(['Square Footage', 'Bedrooms', 'Age', 'Location Score'], feature_importance)):
        print(f"{feature:15s}: {importance:.3f}")

    # 5. Residual Analysis
    print("\n5. Residual Analysis")
    print("-" * 20)

    test_residuals = y_test - y_test_pred

    print(f"Residual statistics:")
    print(f"Mean: {np.mean(test_residuals):.0f}")
    print(f"Std: {np.std(test_residuals):.0f}")
    print(f"Min: {np.min(test_residuals):.0f}")
    print(f"Max: {np.max(test_residuals):.0f}")

    # Check for heteroscedasticity
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(y_test_pred, np.abs(test_residuals))

    print(f"\nHeteroscedasticity test:")
    print(f"Correlation between |residuals| and predicted values: {correlation:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant heteroscedasticity: {p_value < 0.05}")

    # 6. Prediction Example
    print("\n6. Prediction Example")
    print("-" * 25)

    # Example houses
    example_houses = [
        {'sqft': 1500, 'bedrooms': 2, 'age': 10, 'location': 7},
        {'sqft': 2500, 'bedrooms': 4, 'age': 5, 'location': 9},
        {'sqft': 1800, 'bedrooms': 3, 'age': 25, 'location': 4}
    ]

    for i, house in enumerate(example_houses):
        # Create feature vector
        features = np.array([house['sqft'], house['bedrooms'], house['age'], house['location']])
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Add intercept and predict
        X_pred = np.column_stack([np.ones(1), features_scaled])
        predicted_price = X_pred @ coefficients

        print(f"House {i+1}:")
        print(f"  {house['sqft']} sqft, {house['bedrooms']} bedrooms, {house['age']} years old, location score {house['location']}")
        print(f"  Predicted price: ${predicted_price[0]:,.0f}")

    # 7. Visualization
    print("\n7. Visualization")
    print("-" * 15)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Price ($)')
    axes[0, 0].set_ylabel('Predicted Price ($)')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals vs Predicted
    axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    # Feature importance
    axes[1, 0].bar(['Square\nFootage', 'Bedrooms', 'Age', 'Location\nScore'], feature_importance)
    axes[1, 0].set_ylabel('Normalized Importance')
    axes[1, 0].set_title('Feature Importance')
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1, 1].hist(test_residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'coefficients': coefficients,
        'feature_importance': feature_importance,
        'metrics': {'train_r2': train_r2, 'test_r2': test_r2, 'test_mse': test_mse},
        'scaler': scaler
    }

real_lr_results = multiple_linear_regression_real()
```

## 3. Principal Component Analysis (PCA)

### 3.1 PCA from Scratch Implementation

```python
def pca_from_scratch():
    """Complete PCA implementation from scratch using linear algebra"""

    print("Principal Component Analysis (PCA) from Scratch")
    print("=" * 55)

    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 200
    n_features = 6

    # Create correlated features
    base_data = np.random.randn(n_samples, 3)
    X = np.column_stack([
        base_data[:, 0],  # Feature 1
        base_data[:, 0] * 0.8 + base_data[:, 1] * 0.2,  # Feature 2 (correlated with 1)
        base_data[:, 1],  # Feature 3
        base_data[:, 1] * 0.6 + base_data[:, 2] * 0.4,  # Feature 4 (correlated with 3)
        base_data[:, 2],  # Feature 5
        base_data[:, 0] * 0.3 + base_data[:, 1] * 0.3 + base_data[:, 2] * 0.4  # Feature 6 (mixed)
    ])

    # Add some noise
    X += 0.1 * np.random.randn(*X.shape)

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print("Feature correlations:")
    correlation_matrix = np.corrcoef(X.T)
    print(np.round(correlation_matrix, 3))

    # 1. Data Preprocessing
    print("\n1. Data Preprocessing")
    print("-" * 25)

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    print("Data centering:")
    print(f"Original mean: {np.mean(X, axis=0)}")
    print(f"Centered mean: {np.mean(X_centered, axis=0)}")

    # 2. Covariance Matrix Calculation
    print("\n2. Covariance Matrix")
    print("-" * 25)

    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)

    print("Covariance matrix:")
    print(np.round(cov_matrix, 3))

    # Check if matrix is symmetric
    is_symmetric = np.allclose(cov_matrix, cov_matrix.T)
    print(f"Matrix is symmetric: {is_symmetric}")

    # 3. Eigenvalue Decomposition
    print("\n3. Eigenvalue Decomposition")
    print("-" * 30)

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("Eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"PC{i+1}: {ev:.4f}")

    print("\nExplained variance ratio:")
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.1f}%)")

    print(f"\nTotal variance explained: {np.sum(explained_variance_ratio):.4f}")

    # 4. Principal Component Selection
    print("\n4. Principal Component Selection")
    print("-" * 35)

    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("Cumulative explained variance:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"First {i+1} PCs: {cum_var:.4f} ({cum_var*100:.1f}%)")

    # Select number of components to keep 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nNumber of components for 95% variance: {n_components_95}")

    # 5. Data Transformation
    print("\n5. Data Transformation")
    print("-" * 25)

    # Project data onto principal components
    # Use first 3 components for visualization
    n_components_vis = 3
    selected_eigenvectors = eigenvectors[:, :n_components_vis]

    # Transform data
    X_pca = X_centered @ selected_eigenvectors

    print(f"Original data shape: {X.shape}")
    print(f"PCA transformed shape: {X_pca.shape}")

    # Check that transformed data is uncorrelated
    pca_correlation = np.corrcoef(X_pca.T)
    print("\nCorrelation matrix of transformed data:")
    print(np.round(pca_correlation, 3))
    print(f"Off-diagonal correlations (should be ~0): {np.abs(pca_correlation - np.eye(3)).max():.6f}")

    # 6. Reconstruction
    print("\n6. Data Reconstruction")
    print("-" * 25)

    # Reconstruct original data from PCA
    X_reconstructed = X_pca @ selected_eigenvectors.T + np.mean(X, axis=0)

    # Calculate reconstruction error
    reconstruction_error = np.mean((X - X_reconstructed)**2)
    print(f"Reconstruction MSE: {reconstruction_error:.6f}")

    # 7. Visualization
    print("\n7. Visualization")
    print("-" * 15)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Scree plot
    axes[0, 0].plot(range(1, len(eigenvalues) + 1), explained_variance_ratio, 'bo-', linewidth=2)
    axes[0, 0].plot(range(1, len(eigenvalues) + 1), cumulative_variance, 'ro-', linewidth=2)
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].legend(['Individual', 'Cumulative'])
    axes[0, 0].grid(True, alpha=0.3)

    # Original data (first two features)
    axes[0, 1].scatter(X[:, 0], X[:, 1], alpha=0.6, label='Original')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].set_title('Original Data (Features 1 & 2)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PCA transformed data
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='red')
    axes[1, 0].set_xlabel('Principal Component 1')
    axes[1, 0].set_ylabel('Principal Component 2')
    axes[1, 0].set_title('PCA Transformed Data')
    axes[1, 0].grid(True, alpha=0.3)

    # Component loadings
    loading_matrix = eigenvectors[:, :2]
    im = axes[1, 1].imshow(loading_matrix.T, cmap='RdBu', aspect='auto')
    axes[1, 1].set_xlabel('Original Features')
    axes[1, 1].set_ylabel('Principal Components')
    axes[1, 1].set_title('Component Loadings')
    axes[1, 1].set_xticks(range(n_features))
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['PC1', 'PC2'])
    plt.colorbar(im, ax=axes[1, 1])

    # Add loading values as text
    for i in range(n_features):
        for j in range(2):
            axes[1, 1].text(i, j, f'{loading_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

    # 8. Comparison with sklearn
    print("\n8. Comparison with sklearn PCA")
    print("-" * 35)

    from sklearn.decomposition import PCA as SklearnPCA

    sklearn_pca = SklearnPCA(n_components=3)
    X_sklearn = sklearn_pca.fit_transform(X)

    print("sklearn PCA explained variance ratio:")
    for i, ratio in enumerate(sklearn_pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f}")

    print("\nOur implementation explained variance ratio:")
    for i, ratio in enumerate(explained_variance_ratio[:3]):
        print(f"PC{i+1}: {ratio:.4f}")

    # Check if results are similar
    similarity = np.allclose(np.abs(X_pca), np.abs(X_sklearn), atol=1e-10)
    print(f"\nResults match sklearn: {similarity}")

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_variance_ratio': explained_variance_ratio,
        'X_pca': X_pca,
        'reconstruction_error': reconstruction_error
    }

pca_results = pca_from_scratch()
```

### 3.2 PCA Applications: Image Compression and Face Recognition

```python
def pca_applications():
    """PCA applications: image compression and face recognition"""

    print("PCA Applications: Image Compression and Face Recognition")
    print("=" * 65)

    # 1. Image Compression with PCA
    print("\n1. Image Compression with PCA")
    print("-" * 35)

    def create_test_image(size=(64, 64)):
        """Create a test image with patterns"""
        image = np.zeros(size)

        # Add some patterns
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

        # Gradient
        image += x / size[0]

        # Circular pattern
        center = size[0] // 2
        radius = min(size) // 4
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        image[mask] += 0.5

        # Noise
        image += 0.1 * np.random.randn(*size)

        return np.clip(image, 0, 1)

    # Create test image
    test_image = create_test_image((100, 100))

    print(f"Test image shape: {test_image.shape}")
    print(f"Image size in memory: {test_image.nbytes} bytes")

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(test_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Reshape image for PCA
    n_pixels = test_image.shape[0] * test_image.shape[1]
    image_vector = test_image.reshape(n_pixels, 1)

    # Apply PCA with different numbers of components
    n_components_list = [5, 20, 50, 100]
    reconstructed_images = []
    compression_ratios = []

    for n_comp in n_components_list:
        # Center data
        image_centered = image_vector - np.mean(image_vector)

        # Perform PCA
        cov_matrix = np.cov(image_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort and select components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        selected_components = eigenvectors[:, :n_comp]

        # Project and reconstruct
        projected = image_centered.T @ selected_components
        reconstructed = projected @ selected_components.T + np.mean(image_vector)

        reconstructed_image = reconstructed.reshape(test_image.shape)
        reconstructed_images.append(reconstructed_image)

        # Calculate compression ratio
        original_size = test_image.nbytes
        compressed_size = n_comp * test_image.shape[0] + test_image.shape[0] + n_comp  # eigenvectors + mean
        compression_ratio = original_size / compressed_size
        compression_ratios.append(compression_ratio)

        # Plot reconstruction
        plt.subplot(1, len(n_components_list) + 1, n_components_list.index(n_comp) + 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f'{n_comp} PCs\nCR: {compression_ratio:.1f}x')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Compression results:")
    for i, (n_comp, cr) in enumerate(zip(n_components_list, compression_ratios)):
        mse = np.mean((test_image - reconstructed_images[i])**2)
        print(f"{n_comp:3d} components: CR={cr:5.1f}x, MSE={mse:.6f}")

    # 2. Face Recognition with PCA (Eigenfaces)
    print("\n2. Face Recognition with PCA (Eigenfaces)")
    print("-" * 45)

    def create_face_dataset(n_faces=20, face_size=(64, 64)):
        """Create a synthetic face dataset"""
        faces = []

        # Base face template
        x, y = np.meshgrid(np.linspace(-1, 1, face_size[0]), np.linspace(-1, 1, face_size[1]))

        for i in range(n_faces):
            face = np.zeros(face_size)

            # Add face-like features
            # Eyes
            eye_y, eye_x = face_size[0] // 3, face_size[1] // 3
            face[eye_y:eye_y+10, eye_x:eye_x+15] += 0.8
            face[eye_y:eye_y+10, face_x+30:eye_x+45] += 0.8

            # Nose
            nose_x = face_size[1] // 2
            face[eye_y+10:eye_y+25, nose_x-3:nose_x+3] += 0.6

            # Mouth
            mouth_y = int(2 * face_size[0] / 3)
            face[mouth_y:mouth_y+8, nose_x-15:nose_x+15] += 0.7

            # Add variations
            # Position offset
            offset_y = int(np.random.randn() * 5)
            offset_x = int(np.random.randn() * 5)

            # Scale
            scale = 0.8 + 0.4 * np.random.rand()

            # Apply transformations
            face_rolled = np.roll(np.roll(face, offset_y, axis=0), offset_x, axis=1)
            face_scaled = np.clip(scale * face_rolled, 0, 1)

            # Add noise
            face_scaled += 0.05 * np.random.randn(*face_size)

            faces.append(np.clip(face_scaled, 0, 1))

        return np.array(faces)

    # Create face dataset
    faces = create_face_dataset(n_faces=20, face_size=(64, 64))

    print(f"Face dataset: {faces.shape[0]} faces, each {faces.shape[1]}x{faces.shape[2]}")

    # Reshape for PCA
    n_faces, h, w = faces.shape
    face_vectors = faces.reshape(n_faces, -1)

    # Apply PCA
    print("\nApplying PCA to face dataset...")

    # Center data
    mean_face = np.mean(face_vectors, axis=0)
    faces_centered = face_vectors - mean_face

    # Calculate covariance matrix
    cov_matrix = np.cov(faces_centered.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Show top eigenfaces
    n_eigenfaces = min(6, len(eigenvalues))
    eigenfaces = eigenvectors[:, :n_eigenfaces]

    print(f"Top {n_eigenfaces} eigenfaces capture {np.sum(eigenvalues[:n_eigenfaces])/np.sum(eigenvalues)*100:.1f}% of variance")

    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Mean face
    axes[0, 0].imshow(mean_face.reshape(h, w), cmap='gray')
    axes[0, 0].set_title('Mean Face')
    axes[0, 0].axis('off')

    # Eigenfaces
    for i in range(min(7, n_eigenfaces)):
        row = (i + 1) // 4
        col = (i + 1) % 4
        eigenface = eigenfaces[:, i].reshape(h, w)
        axes[row, col].imshow(eigenface, cmap='gray')
        axes[row, col].set_title(f'Eigenface {i+1}\n({eigenvalues[i]/np.sum(eigenvalues)*100:.1f}%)')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    # Face reconstruction
    print("\n3. Face Reconstruction with Different Numbers of Components")
    print("-" * 55)

    test_face_idx = 0
    test_face = face_vectors[test_face_idx]

    n_components_face = [2, 5, 10, 15]
    reconstructed_faces = []

    plt.figure(figsize=(15, 4))

    # Original face
    plt.subplot(1, len(n_components_face) + 1, 1)
    plt.imshow(test_face.reshape(h, w), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    for n_comp in n_components_face:
        # Project and reconstruct
        selected_eigenfaces = eigenfaces[:, :n_comp]
        projected = faces_centered[test_face_idx] @ selected_eigenfaces
        reconstructed = projected @ selected_eigenfaces.T + mean_face

        reconstructed_faces.append(reconstructed)

        # Plot
        plt.subplot(1, len(n_components_face) + 1, n_components_face.index(n_comp) + 2)
        plt.imshow(reconstructed.reshape(h, w), cmap='gray')
        plt.title(f'{n_comp} PCs')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate reconstruction errors
    print("Face reconstruction errors:")
    for i, n_comp in enumerate(n_components_face):
        mse = np.mean((test_face - reconstructed_faces[i])**2)
        print(f"{n_comp:2d} components: MSE = {mse:.6f}")

    # 4. Simple Face Recognition
    print("\n4. Simple Face Recognition Demonstration")
    print("-" * 45)

    # Use first 15 faces for training, last 5 for testing
    train_faces = face_vectors[:15]
    test_faces = face_vectors[15:]

    train_centered = train_faces - mean_face
    test_centered = test_faces - mean_face

    # Project onto eigenfaces
    n_components_recog = 8
    selected_eigenfaces = eigenfaces[:, :n_components_recog]

    train_projected = train_centered @ selected_eigenfaces
    test_projected = test_centered @ selected_eigenfaces

    # Simple nearest neighbor classification
    def recognize_face(test_projection, train_projections, k=3):
        """Recognize face using k-nearest neighbors"""
        distances = np.linalg.norm(train_projections - test_projection, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices, distances[nearest_indices]

    # Test recognition
    print(f"Using {n_components_recog} principal components for recognition")
    print("Recognition results:")

    correct = 0
    for i, test_proj in enumerate(test_projected):
        nearest_indices, distances = recognize_face(test_proj, train_projected, k=1)
        predicted_class = nearest_indices[0]
        actual_class = 15 + i  # Test faces are indices 15-19

        print(f"Test face {i+1}:")
        print(f"  Closest training face: {predicted_class+1}")
        print(f"  Distance: {distances[0]:.4f}")

        # Simple success criterion (faces are different, so just report)
        correct += 1

    # Visualize recognition
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Show test faces and their nearest matches
    for i in range(min(5, len(test_faces))):
        # Test face
        axes[0, i].imshow(test_faces[i].reshape(h, w), cmap='gray')
        axes[0, i].set_title(f'Test Face {i+1}')
        axes[0, i].axis('off')

        # Nearest match
        nearest_idx, _ = recognize_face(test_projected[i], train_projected, k=1)
        axes[1, i].imshow(train_faces[nearest_idx].reshape(h, w), cmap='gray')
        axes[1, i].set_title(f'Nearest Match\nFace {nearest_idx+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'image_compression': {
            'test_image': test_image,
            'reconstructed_images': reconstructed_images,
            'compression_ratios': compression_ratios
        },
        'face_recognition': {
            'eigenfaces': eigenfaces,
            'mean_face': mean_face,
            'train_projected': train_projected,
            'test_projected': test_projected
        }
    }

pca_applications_results = pca_applications()
```

## 4. Singular Value Decomposition (SVD)

### 4.1 SVD from Scratch and Applications

```python
def svd_implementation():
    """SVD implementation from scratch and applications"""

    print("Singular Value Decomposition (SVD)")
    print("=" * 45)

    # Generate test matrix
    np.random.seed(42)
    m, n = 8, 6
    A = np.random.randn(m, n)

    print(f"Test matrix shape: {A.shape}")
    print("Test matrix:")
    print(np.round(A, 3))

    # 1. SVD from scratch using power iteration
    print("\n1. SVD Implementation from Scratch")
    print("-" * 40)

    def power_iteration_svd(A, n_iterations=1000, tol=1e-10):
        """Compute SVD using power iteration method"""
        m, n = A.shape

        # Compute A^T A for right singular vectors
        ATA = A.T @ A

        # Initialize random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)

        # Power iteration for largest singular value/vector
        for i in range(n_iterations):
            v_new = ATA @ v
            v_new = v_new / np.linalg.norm(v_new)

            if np.linalg.norm(v_new - v) < tol:
                break
            v = v_new

        # Largest singular value
        sigma = np.sqrt(v.T @ ATA @ v)

        # Corresponding left singular vector
        u = A @ v / sigma

        return u, sigma, v

    def full_svd_power_iteration(A, max_rank=None):
        """Compute full SVD using power iteration"""
        m, n = A.shape
        if max_rank is None:
            max_rank = min(m, n)

        U = np.zeros((m, max_rank))
        S = np.zeros(max_rank)
        V = np.zeros((n, max_rank))

        A_remaining = A.copy()

        for i in range(max_rank):
            # Find largest singular value/vector
            u, sigma, v = power_iteration_svd(A_remaining)

            U[:, i] = u
            S[i] = sigma
            V[:, i] = v

            # Deflate the matrix
            A_remaining = A_remaining - sigma * np.outer(u, v)

        return U, S, V.T

    # Compute SVD
    U_custom, S_custom, Vt_custom = full_svd_power_iteration(A)

    print("Custom SVD results:")
    print(f"U shape: {U_custom.shape}")
    print(f"S shape: {S_custom.shape}")
    print(f"Vt shape: {Vt_custom.shape}")

    print("\nSingular values:")
    for i, s in enumerate(S_custom):
        print(f"σ{i+1}: {s:.4f}")

    # 2. Compare with numpy SVD
    print("\n2. Comparison with NumPy SVD")
    print("-" * 35)

    U_numpy, S_numpy, Vt_numpy = np.linalg.svd(A, full_matrices=False)

    print("NumPy SVD singular values:")
    for i, s in enumerate(S_numpy):
        print(f"σ{i+1}: {s:.4f}")

    # Check accuracy
    reconstruction_custom = U_custom @ np.diag(S_custom) @ Vt_custom
    reconstruction_numpy = U_numpy @ np.diag(S_numpy) @ Vt_numpy

    error_custom = np.linalg.norm(A - reconstruction_custom)
    error_numpy = np.linalg.norm(A - reconstruction_numpy)

    print(f"\nReconstruction errors:")
    print(f"Custom SVD: {error_custom:.6f}")
    print(f"NumPy SVD: {error_numpy:.6f}")

    # 3. SVD Applications
    print("\n3. SVD Applications")
    print("-" * 25)

    # Application 1: Matrix Approximation
    print("\n3.1 Low-rank Matrix Approximation")
    print("-" * 35)

    def low_rank_approximation(A, rank):
        """Create rank-k approximation of matrix A"""
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vt_k = Vt[:rank, :]
        return U_k @ np.diag(S_k) @ Vt_k

    ranks = [1, 2, 3, 4]
    approximations = []
    errors = []

    for rank in ranks:
        approx = low_rank_approximation(A, rank)
        approximations.append(approx)
        error = np.linalg.norm(A - approx, 'fro')
        errors.append(error)

        print(f"Rank {rank} approximation error: {error:.6f}")

    # Application 2: Image Compression
    print("\n3.2 Image Compression with SVD")
    print("-" * 35)

    def compress_image_svd(image, rank):
        """Compress image using SVD"""
        # For grayscale image
        if len(image.shape) == 2:
            U, S, Vt = np.linalg.svd(image, full_matrices=False)
            U_k = U[:, :rank]
            S_k = S[:rank]
            Vt_k = Vt[:rank, :]
            compressed = U_k @ np.diag(S_k) @ Vt_k
            storage = U_k.nbytes + S_k.nbytes + Vt_k.nbytes
            return compressed, storage
        else:
            # For RGB image, compress each channel
            compressed_channels = []
            total_storage = 0
            for channel in range(3):
                U, S, Vt = np.linalg.svd(image[:, :, channel], full_matrices=False)
                U_k = U[:, :rank]
                S_k = S[:rank]
                Vt_k = Vt[:rank, :]
                compressed_channel = U_k @ np.diag(S_k) @ Vt_k
                compressed_channels.append(compressed_channel)
                total_storage += U_k.nbytes + S_k.nbytes + Vt_k.nbytes
            return np.stack(compressed_channels, axis=2), total_storage

    # Create test image
    test_image = np.random.rand(64, 64)  # Grayscale

    original_size = test_image.nbytes
    ranks_svd = [5, 10, 20, 40]

    print(f"Original image size: {original_size} bytes")
    print("SVD compression results:")

    plt.figure(figsize=(12, 3))
    plt.subplot(151)
    plt.imshow(test_image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    for i, rank in enumerate(ranks_svd):
        compressed, storage = compress_image_svd(test_image, rank)
        compression_ratio = original_size / storage
        mse = np.mean((test_image - compressed)**2)

        print(f"Rank {rank:2d}: CR={compression_ratio:5.1f}x, MSE={mse:.6f}")

        plt.subplot(1, len(ranks_svd) + 1, i + 2)
        plt.imshow(compressed, cmap='gray')
        plt.title(f'Rank {rank}\nCR: {compression_ratio:.1f}x')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Application 3: Pseudoinverse for Linear Systems
    print("\n3.3 Pseudoinverse for Solving Linear Systems")
    print("-" * 50)

    # Create overdetermined system
    m_over, n_over = 10, 3
    X_over = np.random.randn(m_over, n_over)
    true_beta = np.array([2.5, -1.8, 0.7])
    y_over = X_over @ true_beta + 0.1 * np.random.randn(m_over)

    # Solve using SVD
    U, S, Vt = np.linalg.svd(X_over, full_matrices=False)
    S_inv = np.diag(1 / S)
    X_pinv = Vt.T @ S_inv @ U.T
    beta_svd = X_pinv @ y_over

    # Compare with normal equation
    beta_normal = np.linalg.inv(X_over.T @ X_over) @ X_over.T @ y_over

    print("Overdetermined system solution:")
    print(f"True coefficients: {true_beta}")
    print(f"SVD solution: {beta_svd}")
    print(f"Normal equation: {beta_normal}")

    # Application 4: Data Denoising
    print("\n3.4 Data Denoising with SVD")
    print("-" * 35)

    # Create noisy data matrix
    n_samples = 50
    n_features = 20

    # Clean signal (low rank)
    U_clean = np.random.randn(n_samples, 3)
    V_clean = np.random.randn(n_features, 3)
    clean_data = U_clean @ V_clean.T

    # Add noise
    noise_level = 0.5
    noisy_data = clean_data + noise_level * np.random.randn(n_samples, n_features)

    print(f"Clean data rank: {np.linalg.matrix_rank(clean_data)}")
    print(f"Noisy data rank: {np.linalg.matrix_rank(noisy_data)}")

    # Denoise using SVD
    U, S, Vt = np.linalg.svd(noisy_data)

    # Keep only significant singular values
    threshold = 0.1 * S[0]  # Keep components above 10% of largest
    significant_components = np.sum(S > threshold)

    U_denoised = U[:, :significant_components]
    S_denoised = S[:significant_components]
    Vt_denoised = Vt[:significant_components, :]

    denoised_data = U_denoised @ np.diag(S_denoised) @ Vt_denoised

    noise_reduction = np.linalg.norm(noisy_data - clean_data, 'fro') - np.linalg.norm(denoised_data - clean_data, 'fro')
    print(f"Noise reduction: {noise_reduction:.4f}")
    print(f"Components kept: {significant_components} out of {len(S)}")

    # 4. Visualization
    print("\n4. Visualization")
    print("-" * 15)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Singular values comparison
    axes[0, 0].plot(S_custom, 'bo-', label='Custom SVD', markersize=6)
    axes[0, 0].plot(S_numpy, 'ro-', label='NumPy SVD', markersize=6)
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Singular Value')
    axes[0, 0].set_title('Singular Values Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Low-rank approximation error
    axes[0, 1].semilogy(ranks, errors, 'go-', markersize=8)
    axes[0, 1].set_xlabel('Rank')
    axes[0, 1].set_ylabel('Frobenius Norm Error')
    axes[0, 1].set_title('Approximation Error vs Rank')
    axes[0, 1].grid(True, alpha=0.3)

    # Matrix visualization
    im = axes[1, 0].imshow(A, cmap='RdBu', aspect='auto')
    axes[1, 0].set_title('Original Matrix')
    axes[1, 0].set_xlabel('Columns')
    axes[1, 0].set_ylabel('Rows')
    plt.colorbar(im, ax=axes[1, 0])

    # Denoising results
    im2 = axes[1, 1].imshow(denoised_data, cmap='RdBu', aspect='auto')
    axes[1, 1].set_title(f'Denoised Matrix (rank {significant_components})')
    axes[1, 1].set_xlabel('Columns')
    axes[1, 1].set_ylabel('Rows')
    plt.colorbar(im2, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    return {
        'custom_svd': {'U': U_custom, 'S': S_custom, 'Vt': Vt_custom},
        'numpy_svd': {'U': U_numpy, 'S': S_numpy, 'Vt': Vt_numpy},
        'applications': {
            'low_rank_approximations': approximations,
            'compression_ratios': compression_ratios,
            'denoising': {'clean': clean_data, 'noisy': noisy_data, 'denoised': denoised_data}
        }
    }

svd_results = svd_implementation()
```

## 5. Linear Algebra in Neural Networks

### 5.1 Forward and Backward Propagation

```python
def neural_network_linear_algebra():
    """Linear algebra operations in neural networks"""

    print("Linear Algebra in Neural Networks")
    print("=" * 40)

    # Generate dataset
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    n_classes = 3

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    # One-hot encode labels
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1

    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")

    # 1. Network Architecture
    print("\n1. Network Architecture")
    print("-" * 30)

    # Define network architecture
    layer_sizes = [n_features, 8, 6, n_classes]  # Input, Hidden1, Hidden2, Output

    print("Network layers:")
    for i, size in enumerate(layer_sizes):
        if i == 0:
            print(f"  Input layer: {size} neurons")
        elif i == len(layer_sizes) - 1:
            print(f"  Output layer: {size} neurons")
        else:
            print(f"  Hidden layer {i}: {size} neurons")

    # 2. Weight Initialization
    print("\n2. Weight Initialization")
    print("-" * 30)

    def initialize_weights(layer_sizes):
        """Initialize weights using Xavier initialization"""
        weights = {}
        biases = {}

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            weights[f'W{i+1}'] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            biases[f'b{i+1}'] = np.zeros(layer_sizes[i + 1])

        return weights, biases

    weights, biases = initialize_weights(layer_sizes)

    print("Weight matrices:")
    for key, matrix in weights.items():
        print(f"  {key}: {matrix.shape}")

    print("Bias vectors:")
    for key, vector in biases.items():
        print(f"  {key}: {vector.shape}")

    # 3. Forward Propagation
    print("\n3. Forward Propagation")
    print("-" * 25)

    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(X, weights, biases):
        """Forward propagation through the network"""
        activations = {}
        pre_activations = {}

        # Input layer
        activations['A0'] = X

        # Hidden layers
        for i in range(1, len(layer_sizes) - 1):
            pre_activations[f'Z{i}'] = activations[f'A{i-1}'] @ weights[f'W{i}'] + biases[f'b{i}']
            activations[f'A{i}'] = sigmoid(pre_activations[f'Z{i}'])

        # Output layer
        i = len(layer_sizes) - 1
        pre_activations[f'Z{i}'] = activations[f'A{i-1}'] @ weights[f'W{i}'] + biases[f'b{i}']
        activations[f'A{i}'] = softmax(pre_activations[f'Z{i}'])

        return activations, pre_activations

    # Perform forward propagation
    activations, pre_activations = forward_propagation(X, weights, biases)

    print("Forward propagation completed")
    print(f"Output shape: {activations[f'A{len(layer_sizes)-1}'].shape}")

    # 4. Loss Function
    print("\n4. Loss Function")
    print("-" * 20)

    def categorical_crossentropy(y_true, y_pred):
        """Categorical cross-entropy loss"""
        # Avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    loss = categorical_crossentropy(y_onehot, activations[f'A{len(layer_sizes)-1}'])
    print(f"Initial loss: {loss:.4f}")

    # 5. Backward Propagation
    print("\n5. Backward Propagation")
    print("-" * 30)

    def sigmoid_derivative(x):
        """Derivative of sigmoid function"""
        s = sigmoid(x)
        return s * (1 - s)

    def backward_propagation(X, y, activations, pre_activations, weights):
        """Backward propagation to compute gradients"""
        gradients = {}
        n_layers = len(layer_sizes)

        # Output layer gradient
        i = n_layers - 1
        dZ = activations[f'A{i}'] - y
        gradients[f'dW{i}'] = (activations[f'A{i-1}'].T @ dZ) / len(X)
        gradients[f'db{i}'] = np.mean(dZ, axis=0)

        # Hidden layers gradients (backpropagate)
        for i in range(n_layers - 2, 0, -1):
            dZ = (dZ @ weights[f'W{i+1}'].T) * sigmoid_derivative(pre_activations[f'Z{i}'])
            gradients[f'dW{i}'] = (activations[f'A{i-1}'].T @ dZ) / len(X)
            gradients[f'db{i}'] = np.mean(dZ, axis=0)

        return gradients

    gradients = backward_propagation(X, y_onehot, activations, pre_activations, weights)

    print("Gradient shapes:")
    for key, grad in gradients.items():
        print(f"  {key}: {grad.shape}")

    # 6. Gradient Descent Update
    print("\n6. Gradient Descent Update")
    print("-" * 30)

    def update_weights(weights, biases, gradients, learning_rate=0.01):
        """Update weights using gradient descent"""
        updated_weights = weights.copy()
        updated_biases = biases.copy()

        for i in range(1, len(layer_sizes)):
            updated_weights[f'W{i}'] -= learning_rate * gradients[f'dW{i}']
            updated_biases[f'b{i}'] -= learning_rate * gradients[f'db{i}']

        return updated_weights, updated_biases

    learning_rate = 0.1
    weights_updated, biases_updated = update_weights(weights, biases, gradients, learning_rate)

    print(f"Weights updated with learning rate: {learning_rate}")

    # 7. Training Loop
    print("\n7. Training Loop")
    print("-" * 20)

    def train_network(X, y, layer_sizes, epochs=100, learning_rate=0.01):
        """Train neural network"""
        # Initialize weights
        weights, biases = initialize_weights(layer_sizes)

        loss_history = []

        for epoch in range(epochs):
            # Forward propagation
            activations, pre_activations = forward_propagation(X, weights, biases)

            # Compute loss
            loss = categorical_crossentropy(y, activations[f'A{len(layer_sizes)-1}'])
            loss_history.append(loss)

            # Backward propagation
            gradients = backward_propagation(X, y, activations, pre_activations, weights)

            # Update weights
            weights, biases = update_weights(weights, biases, gradients, learning_rate)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")

        return weights, biases, loss_history

    # Train the network
    weights_trained, biases_trained, loss_history = train_network(X, y_onehot, layer_sizes, epochs=200, learning_rate=0.1)

    # 8. Linear Algebra Operations Analysis
    print("\n8. Linear Algebra Operations Analysis")
    print("-" * 40)

    # Count matrix multiplications
    print("Matrix operations per forward pass:")
    for i in range(1, len(layer_sizes)):
        input_shape = (len(X), layer_sizes[i-1])
        weight_shape = (layer_sizes[i-1], layer_sizes[i])
        output_shape = (len(X), layer_sizes[i])
        flops = input_shape[0] * input_shape[1] * output_shape[1]
        print(f"  Layer {i}: {input_shape} @ {weight_shape} = {output_shape} ({flops:,} FLOPs)")

    # Parameter count
    total_params = 0
    print("\nNetwork parameters:")
    for i in range(1, len(layer_sizes)):
        w_params = layer_sizes[i-1] * layer_sizes[i]
        b_params = layer_sizes[i]
        total_params += w_params + b_params
        print(f"  Layer {i}: {w_params} weights + {b_params} biases = {w_params + b_params} parameters")
    print(f"Total parameters: {total_params:,}")

    # 9. Visualization
    print("\n9. Visualization")
    print("-" * 15)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curve
    axes[0, 0].plot(loss_history)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Weight distributions
    all_weights = []
    layer_names = []
    for i in range(1, len(layer_sizes)):
        all_weights.extend(weights_trained[f'W{i}'].flatten())
        layer_names.extend([f'Layer {i} Weights'] * len(weights_trained[f'W{i}'].flatten()))

    axes[0, 1].hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Weight Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Network architecture diagram
    axes[1, 0].remove()
    ax_network = fig.add_subplot(223)

    # Draw network architecture
    layer_positions = []
    for i, size in enumerate(layer_sizes):
        x = i * 2
        y_positions = np.linspace(-2, 2, size)
        layer_positions.append((x, y_positions))

        for y in y_positions:
            circle = plt.Circle((x, y), 0.1, color='lightblue', ec='black')
            ax_network.add_patch(circle)

        # Add layer labels
        if i == 0:
            ax_network.text(x, -3, f'Input\n({size})', ha='center', va='center', fontsize=10)
        elif i == len(layer_sizes) - 1:
            ax_network.text(x, -3, f'Output\n({size})', ha='center', va='center', fontsize=10)
        else:
            ax_network.text(x, -3, f'Hidden\n({size})', ha='center', va='center', fontsize=10)

    # Draw connections
    for i in range(len(layer_positions) - 1):
        x1, y1_positions = layer_positions[i]
        x2, y2_positions = layer_positions[i + 1]

        for y1 in y1_positions:
            for y2 in y2_positions:
                ax_network.plot([x1 + 0.1, x2 - 0.1], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)

    ax_network.set_xlim(-1, 2 * len(layer_positions))
    ax_network.set_ylim(-3.5, 3.5)
    ax_network.set_aspect('equal')
    ax_network.set_title('Network Architecture')
    ax_network.axis('off')

    # Parameter analysis
    param_counts = []
    layer_labels = []
    for i in range(1, len(layer_sizes)):
        params = layer_sizes[i-1] * layer_sizes[i] + layer_sizes[i]
        param_counts.append(params)
        layer_labels.append(f'Layer {i}')

    axes[1, 1].bar(layer_labels, param_counts, color='skyblue', edgecolor='black')
    axes[1, 1].set_ylabel('Number of Parameters')
    axes[1, 1].set_title('Parameters per Layer')
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for i, count in enumerate(param_counts):
        axes[1, 1].text(i, count + max(param_counts) * 0.01, f'{count:,}',
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return {
        'trained_weights': weights_trained,
        'trained_biases': biases_trained,
        'loss_history': loss_history,
        'layer_sizes': layer_sizes,
        'total_parameters': total_params
    }

nn_results = neural_network_linear_algebra()
```

## 6. Key Concepts Summary

### 6.1 Essential Linear Algebra for ML

1. **Data Representation**: Vectors for samples, matrices for datasets
2. **Matrix Operations**: Multiplication, decomposition, inversion
3. **Transformations**: Linear transformations for preprocessing
4. **Decompositions**: SVD, PCA for dimensionality reduction
5. **Optimization**: Normal equations, gradient descent

### 6.2 Practical Applications

- **Linear Regression**: Matrix solutions and statistical inference
- **PCA**: Dimensionality reduction and feature extraction
- **SVD**: Matrix approximation, compression, pseudoinverse
- **Neural Networks**: Forward/backward propagation as matrix operations

### 6.3 Implementation Tips

- **Numerical Stability**: Use appropriate decompositions
- **Computational Efficiency**: Vectorize operations
- **Memory Management**: Handle large matrices efficiently
- **Debugging**: Check matrix shapes and properties

## 7. Exercises

### 7.1 Implementation Exercises

1. Implement LASSO regression using coordinate descent
2. Create a neural network with batch normalization
3. Implement t-SNE for dimensionality reduction
4. Build a recommendation system using matrix factorization
5. Implement a convolution operation as matrix multiplication

### 7.2 Conceptual Exercises

1. Explain why PCA uses eigenvectors of covariance matrix
2. Compare different matrix decomposition methods
3. Analyze computational complexity of neural network operations
4. Explain the relationship between SVD and PCA
5. Discuss numerical stability issues in matrix operations

This comprehensive guide demonstrates practical applications of linear algebra in machine learning, with complete implementations and real-world examples that you can run and modify.