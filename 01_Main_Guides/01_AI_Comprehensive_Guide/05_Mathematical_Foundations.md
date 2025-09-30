# Chapter 5: Mathematical Foundations

> **Prerequisites**: [Early AI Approaches](04_Early_AI_Approaches.md) | Basic college mathematics
>
> **Learning Objectives**:
> - Master the linear algebra concepts essential for AI
> - Understand probability and statistics for machine learning
> - Learn calculus and optimization techniques
> - Apply information theory to AI problems
>
> **Related Topics**: [Computational Theory](06_Computational_Theory.md) | [Introduction to Machine Learning](08_Introduction_to_ML.md)

## Linear Algebra for AI

Linear algebra provides the mathematical foundation for representing and manipulating data in AI systems. It's essential for understanding neural networks, machine learning algorithms, and data transformations.

### Vectors and Matrices

**Fundamental Concepts**

**Vectors** represent data points and features in AI:
- **Row vectors**: [x₁, x₂, ..., xₙ]
- **Column vectors**: ⎡x₁⎤
                  ⎢x₂⎥
                  ⎢⋮ ⎥
                  ⎣xₙ⎦
- **Geometric interpretation**: Points in n-dimensional space
- **Applications**: Feature vectors, word embeddings, data points

**Matrices** represent transformations and datasets:
- **Data matrices**: Each row is a sample, each column is a feature
- **Transformation matrices**: Linear transformations of data
- **Weight matrices**: Neural network parameters
- **Applications**: Dataset representation, linear transformations

**Vector Operations**
```python
# Vector addition
v1 + v2 = [v1₁ + v2₁, v1₂ + v2₂, ..., v1ₙ + v2ₙ]

# Scalar multiplication
α · v = [α·v₁, α·v₂, ..., α·vₙ]

# Dot product
v · w = v₁w₁ + v₂w₂ + ... + vₙwₙ = |v||w|cos(θ)

# Vector magnitude (norm)
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

**Matrix Operations**
```python
# Matrix addition
A + B = [aᵢⱼ + bᵢⱼ]

# Matrix multiplication
(AB)ᵢⱼ = Σₖ aᵢₖbₖⱼ

# Matrix-vector multiplication
Av = b, where bᵢ = Σⱼ aᵢⱼvⱼ

# Matrix transpose
Aᵀ where (Aᵀ)ᵢⱼ = Aⱼᵢ
```

### Key Linear Algebra Concepts

#### Vector Spaces

**Definition**: A vector space is a set of vectors with two operations:
1. **Vector addition**: Closed under addition
2. **Scalar multiplication**: Closed under scalar multiplication
3. Satisfies axioms: associativity, commutativity, distributivity

**Important Subspaces**:
- **Column space**: Range of a matrix transformation
- **Null space**: Vectors that map to zero
- **Row space**: Span of row vectors
- **Left null space**: Vectors orthogonal to column space

**Basis and Dimension**
- **Basis**: Linearly independent spanning set
- **Dimension**: Number of basis vectors
- **Orthonormal basis**: Mutually orthogonal unit vectors

#### Eigenvalues and Eigenvectors

**Definition**: For a square matrix A, an eigenvector v satisfies:
Av = λv

where λ is the eigenvalue and v is the eigenvector.

**Applications in AI**:
- **Principal Component Analysis (PCA)**: Dimensionality reduction
- **Spectral clustering**: Graph partitioning
- **PageRank**: Web page importance
- **Vibration analysis**: Natural frequencies
- **Quantum mechanics**: Energy states

**Characteristic Equation**:
det(A - λI) = 0

**Example**: 2×2 Matrix
```
A = [a b]
    [c d]

det(A - λI) = (a-λ)(d-λ) - bc = 0
λ² - (a+d)λ + (ad-bc) = 0
```

#### Matrix Decomposition

**Singular Value Decomposition (SVD)**
A = UΣVᵀ

Where:
- U: Left singular vectors (orthogonal)
- Σ: Singular values (diagonal matrix)
- Vᵀ: Right singular vectors (orthogonal)

**Applications**:
- **Dimensionality reduction**: Keep top k singular values
- **Collaborative filtering**: Recommendation systems
- **Image compression**: Low-rank approximation
- **Noise reduction**: Remove small singular values
- **Latent semantic analysis**: Text mining

**Eigenvalue Decomposition**
A = PDP⁻¹

Where:
- P: Matrix of eigenvectors
- D: Diagonal matrix of eigenvalues

**QR Decomposition**
A = QR

Where:
- Q: Orthogonal matrix
- R: Upper triangular matrix

**Applications**:
- **Solving linear systems**: More stable than Gaussian elimination
- **Least squares problems**: Normal equations
- **Eigenvalue algorithms**: QR algorithm

#### Tensor Operations

**Tensors**: Multi-dimensional arrays (scalars: 0D, vectors: 1D, matrices: 2D, tensors: nD)

**Tensor Operations**:
- **Tensor product**: Outer product generalization
- **Tensor contraction**: Sum over repeated indices
- **Tensor reshaping**: Change dimensionality
- **Tensor slicing**: Extract sub-tensors

**Applications in Deep Learning**:
- **Convolutional neural networks**: 4D tensors (batch, height, width, channels)
- **Recurrent neural networks**: 3D tensors (time steps, batch, features)
- **Transformers**: Multi-head attention tensors
- **Batch processing**: Efficient parallel computation

### Advanced Linear Algebra for AI

#### Singular Value Decomposition (SVD)

**Mathematical Foundation**
For any matrix A ∈ ℝᵐˣⁿ, SVD decomposes it as:
A = UΣVᵀ

**Computational Aspects**:
- **Power iteration**: Find largest singular value
- **QR algorithm**: Full decomposition
- **Randomized SVD**: Approximate for large matrices
- **Incremental SVD**: Update with new data

**AI Applications**:
```python
# Dimensionality reduction with SVD
U, s, Vt = np.linalg.svd(X_reduced)
X_reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Collaborative filtering
# R = UΣVᵀ where R is user-item rating matrix
# User factors: U, Item factors: Vᵀ
```

#### Principal Component Analysis (PCA)

**Algorithm**:
1. Center the data: X_centered = X - μ
2. Compute covariance matrix: C = (1/n)X_centeredᵀX_centered
3. Find eigenvalues and eigenvectors of C
4. Select top k eigenvectors (principal components)
5. Project data: X_pca = X_centered * Wₖ

**Geometric Interpretation**:
- Find directions of maximum variance
- Orthogonal transformation to uncorrelated variables
- Dimensionality reduction while preserving information

**Implementation**:
```python
# PCA implementation
def pca(X, n_components):
    # Center data
    X_centered = X - X.mean(axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    components = eigenvectors[:, :n_components]

    # Project data
    X_pca = X_centered @ components

    return X_pca, components
```

#### Independent Component Analysis (ICA)

**Goal**: Separate mixed signals into independent sources
**Assumption**: Sources are statistically independent
**Application**: Blind source separation, feature extraction

**Mathematical Model**:
X = AS

Where:
- X: Observed mixed signals
- A: Mixing matrix
- S: Independent sources

**Algorithm**: FastICA
1. Center and whiten data
2. Choose non-linearity (e.g., tanh, cubic)
3. Update rule for unmixing matrix
4. Orthogonalize and normalize

#### Matrix Factorization

**Collaborative Filtering**:
R ≈ PQᵀ

Where:
- R: User-item rating matrix
- P: User factors
- Q: Item factors

**Non-negative Matrix Factorization (NMF)**:
W ≥ 0, H ≥ 0, V ≈ WH

Applications:
- **Topic modeling**: Document-term matrices
- **Image processing**: Parts-based representation
- **Bioinformatics**: Gene expression analysis

**Tensor Decomposition**:
- **CANDECOMP/PARAFAC (CP)**: Generalization of SVD
- **Tucker Decomposition**: Higher-order SVD
- **Tensor Train**: Efficient representation for high dimensions

#### Sparse Matrices

**Motivation**: Many real-world matrices are sparse (mostly zeros)
**Storage formats**:
- **COO**: Coordinate format (row, col, value)
- **CSR**: Compressed sparse row
- **CSC**: Compressed sparse column
- **LIL**: List of lists

**Operations**:
- **Sparse matrix-vector multiplication**: O(nnz) complexity
- **Sparse matrix-matrix multiplication**: Efficient algorithms
- **Sparse solvers**: Iterative methods (Conjugate Gradient, GMRES)

**Applications**:
- **Graph algorithms**: Adjacency matrices
- **Natural language processing**: Term-document matrices
- **Recommendation systems**: User-item interactions
- **Computer vision**: Image processing with sparse representations

#### Kernel Methods

**Idea**: Map data to higher-dimensional space where it's linearly separable
**Kernel trick**: Compute inner products in feature space explicitly

**Common Kernels**:
```python
# Linear kernel
K(x, y) = xᵀy

# Polynomial kernel
K(x, y) = (xᵀy + c)ᵈ

# RBF (Gaussian) kernel
K(x, y) = exp(-γ||x-y||²)

# Sigmoid kernel
K(x, y) = tanh(αxᵀy + c)
```

**Applications**:
- **Support Vector Machines (SVM)**: Non-linear classification
- **Kernel PCA**: Non-linear dimensionality reduction
- **Gaussian Processes**: Non-parametric regression
- **Kernel Ridge Regression**: Regularized linear models

#### Random Matrix Theory

**Study**: Properties of large random matrices
**Applications**:
- **Signal processing**: Array processing, spectrum estimation
- **Machine learning**: High-dimensional statistics
- **Network theory**: Random graphs, community detection
- **Quantum physics**: Random Hamiltonians

**Key Results**:
- **Wigner's semicircle law**: Eigenvalue distribution
- **Marchenko-Pastur law**: Sample covariance matrices
- **Circular law**: Non-Hermitian random matrices
- **Tracy-Widom distribution**: Edge eigenvalues

#### Manifold Learning

**Goal**: Discover low-dimensional structure in high-dimensional data
**Assumption**: Data lies on a lower-dimensional manifold

**Algorithms**:
- **ISOMAP**: Geodesic distances, multidimensional scaling
- **LLE (Locally Linear Embedding)**: Local linear reconstructions
- **Laplacian Eigenmaps**: Graph Laplacian eigenvectors
- **t-SNE**: Visualization of high-dimensional data
- **UMAP**: Uniform manifold approximation and projection

**Mathematical Framework**:
```python
# Laplacian Eigenmaps
1. Construct neighborhood graph
2. Compute graph Laplacian L = D - W
3. Solve generalized eigenvalue problem Lv = λDv
4. Use bottom eigenvectors for embedding
```

#### Geometric Algebra

**Unified framework**: Combines vectors, complex numbers, quaternions
**Operations**:
- **Geometric product**: uv = u·v + u∧v
- **Outer product**: Wedge product for subspaces
- **Inner product**: Dot product generalization

**Applications**:
- **Computer graphics**: 3D transformations, physics simulation
- **Robotics**: Kinematics, dynamics
- **Computer vision**: Projective geometry, camera calibration
- **Quantum computing**: Quantum states, operations

### Applications in AI

#### Word Embeddings

**Word2Vec**: Distributed word representations
- **CBOW**: Predict center word from context
- **Skip-gram**: Predict context from center word

**GloVe**: Global matrix factorization and local context
**FastText**: Subword information, character n-grams

**Mathematical Foundation**:
```python
# Word2Vec skip-gram objective
maximize Σ Σ log(p(w₀|wᵢ))
       wᵢ in corpus w₀ in context(wᵢ)

# Context probability
p(w₀|wᵢ) = exp(v'₀·vᵢ) / Σᵥ exp(v'ᵥ·vᵢ)
```

#### Neural Network Weight Matrices

**Weight matrices**: Transformations between layers
**Weight sharing**: Convolutional neural networks
**Weight tying**: Recurrent neural networks, autoencoders

**Weight initialization**:
- **Xavier initialization**: For tanh activations
- **He initialization**: For ReLU activations
- **Orthogonal initialization**: For recurrent networks

#### Image Representations in CNNs

**Convolution operations**: Local feature extraction
**Pooling operations**: Spatial downsampling
**Feature maps**: Hierarchical representations

**Mathematical operations**:
```python
# Convolution
(f * I)(i,j) = Σₘ Σₙ f(m,n) I(i-m, j-n)

# Max pooling
max_pool(I)(i,j) = max_{m,n in neighborhood} I(m,n)

# Strided convolution
stride_conv(f,I,s)(i,j) = Σₘ Σₙ f(m,n) I(i·s-m, j·s-n)
```

#### State Representations in RL

**State vectors**: Environment representations
**Value functions**: State value estimates
**Policy functions**: Action probabilities

**Mathematical formulations**:
```python
# Value function
V(s) = E[Σ γᵗ rₜ₊ₜ | s₀ = s]

# Q-function
Q(s,a) = E[Σ γᵗ rₜ₊ₜ | s₀ = s, a₀ = a]

# Policy
π(a|s) = P(aₜ = a | sₜ = s)
```

## Probability and Statistics

Probability theory provides the mathematical framework for reasoning under uncertainty, which is essential for most AI applications.

### Probability Theory

#### Bayes' Theorem

**Foundation**: P(A|B) = P(B|A)P(A) / P(B)

**AI Applications**:
- **Naive Bayes classification**: Text classification, spam detection
- **Bayesian networks**: Probabilistic reasoning
- **MCMC methods**: Sampling from complex distributions
- **Variational inference**: Approximate Bayesian inference

**Extended Form**:
P(A|B,C) = P(B|A,C)P(A|C) / P(B|C)

**Chain Rule**:
P(A₁,A₂,...,Aₙ) = P(A₁)P(A₂|A₁)P(A₃|A₁,A₂)...P(Aₙ|A₁,...,Aₙ₋₁)

#### Conditional Probability

**Definition**: P(A|B) = P(A∩B) / P(B)

**Independence**: A and B independent if P(A∩B) = P(A)P(B)

**Conditional Independence**: A ⊥ B | C if P(A,B|C) = P(A|C)P(B|C)

**Law of Total Probability**:
P(A) = Σᵢ P(A|Bᵢ)P(Bᵢ)

#### Bayesian Networks

**Graphical Models**: Directed acyclic graphs representing conditional dependencies

**Components**:
- **Nodes**: Random variables
- **Edges**: Direct dependencies
- **Conditional Probability Tables (CPTs)**: Local distributions

**Inference**:
- **Exact inference**: Variable elimination, junction tree
- **Approximate inference**: MCMC, variational methods
- **Loopy belief propagation**: Iterative message passing

**Example Bayesian Network**:
```
Rain → Sprinkler → Wet Grass
  ↓       ↓
Cloudy ← Sunny
```

#### Markov Chains

**Definition**: Sequence of random variables where future depends only on present

**Transition Matrix**: P(Xₜ₊₁|Xₜ)

**Stationary Distribution**: π such that π = πP

**Applications**:
- **PageRank**: Web page importance
- **Text generation**: Language models
- **Reinforcement learning**: State transitions
- **Time series**: Sequential modeling

**Markov Property**:
P(Xₜ₊₁|Xₜ,Xₜ₋₁,...,X₀) = P(Xₜ₊₁|Xₜ)

### Statistical Learning

#### Maximum Likelihood Estimation

**Principle**: Choose parameters that make observed data most probable

**Likelihood Function**: L(θ|X) = P(X|θ)

**Log-Likelihood**: ℓ(θ) = log L(θ)

**Maximum Likelihood Estimate**: θ̂ = argmax_θ L(θ|X)

**Examples**:
```python
# Gaussian MLE
μ̂ = (1/n) Σᵢ xᵢ
σ̂² = (1/n) Σᵢ (xᵢ - μ̂)²

# Bernoulli MLE
p̂ = (1/n) Σᵢ xᵢ

# Multinomial MLE
p̂ⱼ = nⱼ / n
```

#### Bayesian Inference

**Bayes' Rule for Parameters**:
P(θ|X) = P(X|θ)P(θ) / P(X)

**Components**:
- **Prior**: P(θ) - Initial belief about parameters
- **Likelihood**: P(X|θ) - Data generating process
- **Posterior**: P(θ|X) - Updated belief after seeing data
- **Evidence**: P(X) - Normalizing constant

**Conjugate Priors**:
- **Beta-Binomial**: Beta prior, Binomial likelihood
- **Gamma-Poisson**: Gamma prior, Poisson likelihood
- **Normal-Normal**: Normal prior, Normal likelihood

**Approximate Methods**:
- **MCMC**: Markov Chain Monte Carlo sampling
- **Variational Inference**: Optimization-based approximation
- **Laplace Approximation**: Gaussian approximation
- **Expectation Propagation**: Iterative approximation

#### Hypothesis Testing

**Null Hypothesis**: H₀ (default assumption)
**Alternative Hypothesis**: H₁ (claim to test)

**Test Statistics**: T(X) = function of data
**p-value**: P(T(X) ≥ t_observed | H₀)

**Common Tests**:
- **t-test**: Compare means
- **χ² test**: Goodness of fit, independence
- **ANOVA**: Compare multiple groups
- **Kolmogorov-Smirnov**: Distribution comparison

**Type I Error**: Reject H₀ when H₀ is true (false positive)
**Type II Error**: Fail to reject H₀ when H₁ is true (false negative)

#### Confidence Intervals

**Definition**: Range of plausible parameter values
**Confidence Level**: 1 - α (typically 95%)

**Construction Methods**:
- **Wald interval**: θ̂ ± z_(α/2) · SE(θ̂)
- **Bootstrap**: Resampling-based intervals
- **Bayesian credible intervals**: Posterior quantiles

**Interpretation**: 95% CI means that 95% of similarly constructed intervals would contain the true parameter

### Probability Distributions

#### Discrete Distributions

**Bernoulli Distribution**
- **Support**: {0, 1}
- **PMF**: P(X=x) = pˣ(1-p)¹⁻ˣ
- **Mean**: p
- **Variance**: p(1-p)
- **Applications**: Binary classification, coin flips

**Binomial Distribution**
- **Support**: {0, 1, ..., n}
- **PMF**: P(X=k) = C(n,k) pᵏ(1-p)ⁿ⁻ᵏ
- **Mean**: np
- **Variance**: np(1-p)
- **Applications**: Number of successes in n trials

**Poisson Distribution**
- **Support**: {0, 1, 2, ...}
- **PMF**: P(X=k) = (λᵏ e⁻λ)/k!
- **Mean**: λ
- **Variance**: λ
- **Applications**: Count data, rare events

**Geometric Distribution**
- **Support**: {1, 2, 3, ...}
- **PMF**: P(X=k) = (1-p)ᵏ⁻¹p
- **Mean**: 1/p
- **Variance**: (1-p)/p²
- **Applications**: Time to first success

#### Continuous Distributions

**Normal (Gaussian) Distribution**
- **PDF**: f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
- **Mean**: μ
- **Variance**: σ²
- **Properties**: Symmetric, unimodal, central limit theorem
- **Applications**: Measurement errors, natural phenomena

**Exponential Distribution**
- **PDF**: f(x) = λe⁻λˣ for x ≥ 0
- **Mean**: 1/λ
- **Variance**: 1/λ²
- **Memoryless**: P(X > s+t | X > s) = P(X > t)
- **Applications**: Waiting times, reliability

**Uniform Distribution**
- **PDF**: f(x) = 1/(b-a) for a ≤ x ≤ b
- **Mean**: (a+b)/2
- **Variance**: (b-a)²/12
- **Applications**: Random number generation, prior distributions

**Beta Distribution**
- **PDF**: f(x) = [x^(α-1)(1-x)^(β-1)] / B(α,β)
- **Mean**: α/(α+β)
- **Variance**: αβ/[(α+β)²(α+β+1)]
- **Applications**: Proportions, probabilities

### Advanced Probability and Statistics

#### Multivariate Distributions

**Multivariate Normal Distribution**
- **PDF**: f(x) = (1/√((2π)ᵏ|Σ|)) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
- **Mean**: μ (vector)
- **Covariance**: Σ (matrix)
- **Properties**: Linear combinations are normal, marginals are normal

**Multinomial Distribution**
- **PMF**: P(X₁=x₁,...,Xₖ=xₖ) = (n! / (x₁!...xₖ!)) p₁ˣ¹...pₖˣᵏ
- **Mean**: n pᵢ for each i
- **Covariance**: n(pᵢδᵢⱼ - pᵢpⱼ)
- **Applications**: Categorical data, text classification

**Dirichlet Distribution**
- **PDF**: f(x) = [1/B(α)] Πᵢ xᵢ^(αᵢ-1)
- **Conjugate prior**: For multinomial distribution
- **Applications**: Topic modeling, Bayesian inference

#### Copulas

**Definition**: Functions that couple multivariate distributions to their marginals
**Sklar's Theorem**: Any multivariate distribution can be expressed in terms of its marginals and a copula

**Common Copulas**:
- **Gaussian copula**: Based on multivariate normal
- **t-copula**: Based on multivariate t-distribution
- **Archimedean copulas**: Clayton, Frank, Gumbel

**Applications**:
- **Finance**: Dependence modeling
- **Risk management**: Tail dependence
- **Insurance**: Multivariate claims

#### Extreme Value Theory

**Goal**: Model rare events and outliers
**Three types of extreme value distributions**:
- **Gumbel**: Type I, light tails
- **Fréchet**: Type II, heavy tails
- **Weibull**: Type III, bounded tails

**Applications**:
- **Risk management**: Extreme losses
- **Environmental science**: Natural disasters
- **Engineering**: Material failure
- **Finance**: Market crashes

#### Non-parametric Statistics

**Definition**: Methods that don't assume specific parametric forms
**Techniques**:
- **Kernel density estimation**: Smooth density estimation
- **Empirical CDF**: Step function estimate
- **Rank-based tests**: Wilcoxon, Mann-Whitney
- **Bootstrap**: Resampling inference

**Applications**:
- **Exploratory data analysis**: Unknown distributions
- **Robust statistics**: Outlier resistance
- **Model validation**: Goodness-of-fit tests

#### Bayesian Non-parametrics

**Idea**: Infinite-dimensional models that grow with data
**Examples**:
- **Gaussian Processes**: Non-parametric regression
- **Dirichlet Processes**: Bayesian clustering
- **Beta Processes**: Feature allocation
- **Indian Buffet Processes**: Binary features

**Gaussian Process Regression**:
f(x) ~ GP(m(x), k(x,x'))
y = f(x) + ε, ε ~ N(0, σ²I)

**Dirichlet Process Mixture**:
G ~ DP(α, G₀)
θᵢ | G ~ G
yᵢ | θᵢ ~ F(θᵢ)

---

**Next Chapter**: [Computational Theory](06_Computational_Theory.md) - Understanding the theoretical limits of computation

**Related Topics**: [Cognitive Science Foundations](07_Cognitive_Science_Foundations.md) | [Introduction to Machine Learning](08_Introduction_to_ML.md)

**Mathematical Reference**: See [Appendix A](A_Mathematical_Reference.md) for quick reference