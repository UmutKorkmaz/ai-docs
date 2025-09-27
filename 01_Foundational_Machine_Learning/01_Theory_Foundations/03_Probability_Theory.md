# Probability Theory for Machine Learning

## Overview

Probability theory provides the mathematical foundation for reasoning under uncertainty, which is essential for machine learning, statistical inference, and decision making. This section covers the key concepts of probability theory with applications in machine learning.

## 1. Basic Probability Concepts

### 1.1 Probability Spaces and Random Variables

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import comb, factorial
import pandas as pd

# Basic probability concepts
def basic_probability_concepts():
    """Explore fundamental probability concepts"""

    # Sample space and events
    print("Basic Probability Concepts")
    print("-" * 50)

    # Example: Rolling a fair die
    sample_space = {1, 2, 3, 4, 5, 6}
    print(f"Sample space (die roll): {sample_space}")

    # Events
    event_even = {2, 4, 6}
    event_greater_than_4 = {5, 6}
    event_prime = {2, 3, 5}

    print(f"Event A (even numbers): {event_even}")
    print(f"Event B (> 4): {event_greater_than_4}")
    print(f"Event C (prime numbers): {event_prime}")

    # Probability calculations
    P_A = len(event_even) / len(sample_space)
    P_B = len(event_greater_than_4) / len(sample_space)
    P_C = len(event_prime) / len(sample_space)

    print(f"\nProbabilities:")
    print(f"P(A) = {P_A:.3f}")
    print(f"P(B) = {P_B:.3f}")
    print(f"P(C) = {P_C:.3f}")

    # Set operations
    A_intersect_B = event_even.intersection(event_greater_than_4)
    A_union_B = event_even.union(event_greater_than_4)
    A_complement = sample_space - event_even

    print(f"\nSet Operations:")
    print(f"A ∩ B = {A_intersect_B}")
    print(f"A ∪ B = {A_union_B}")
    print(f"A^c = {A_complement}")

    # Probability rules
    P_A_and_B = len(A_intersect_B) / len(sample_space)
    P_A_or_B = len(A_union_B) / len(sample_space)
    P_not_A = len(A_complement) / len(sample_space)

    print(f"\nProbability Rules:")
    print(f"P(A ∩ B) = {P_A_and_B:.3f}")
    print(f"P(A ∪ B) = {P_A_or_B:.3f}")
    print(f"P(A^c) = {P_not_A:.3f}")

    # Verify addition rule: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    addition_rule = P_A + P_B - P_A_and_B
    print(f"\nAddition Rule Verification:")
    print(f"P(A) + P(B) - P(A ∩ B) = {addition_rule:.3f}")
    print(f"P(A ∪ B) = {P_A_or_B:.3f}")
    print(f"Match: {abs(addition_rule - P_A_or_B) < 1e-10}")

    # Conditional probability
    print(f"\nConditional Probability:")
    P_B_given_A = P_A_and_B / P_A if P_A > 0 else 0
    P_A_given_B = P_A_and_B / P_B if P_B > 0 else 0
    print(f"P(B|A) = P(A ∩ B) / P(A) = {P_B_given_A:.3f}")
    print(f"P(A|B) = P(A ∩ B) / P(B) = {P_A_given_B:.3f}")

    # Independence check
    are_independent = abs(P_A_and_B - P_A * P_B) < 1e-10
    print(f"\nIndependence:")
    print(f"P(A ∩ B) = {P_A_and_B:.3f}")
    print(f"P(A) × P(B) = {P_A * P_B:.3f}")
    print(f"Events A and B are independent: {are_independent}")

    return sample_space, event_even, event_greater_than_4

sample_space, event_even, event_greater_than_4 = basic_probability_concepts()
```

**Key Probability Axioms**:
1. **Non-negativity**: $P(A) \geq 0$ for all events A
2. **Normalization**: $P(\Omega) = 1$ where $\Omega$ is the sample space
3. **Additivity**: For mutually exclusive events $A_1, A_2, ...$, $P(\bigcup_i A_i) = \sum_i P(A_i)$

### 1.2 Random Variables and Probability Distributions

```python
def random_variables_distributions():
    """Random variables and their distributions"""

    print("Random Variables and Probability Distributions")
    print("-" * 60)

    # Discrete Random Variables
    print("\n1. Discrete Random Variables")

    # Example: Number of heads in 3 coin flips
    def binomial_pmf(n, p, k):
        """Probability mass function for binomial distribution"""
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    n_flips = 3
    p_head = 0.5
    possible_values = list(range(n_flips + 1))

    print(f"Binomial Distribution: X ~ Binom(n={n_flips}, p={p_head})")
    print("Possible values:", possible_values)

    probabilities = []
    for k in possible_values:
        prob = binomial_pmf(n_flips, p_head, k)
        probabilities.append(prob)
        print(f"P(X = {k}) = {prob:.4f}")

    # Verify probabilities sum to 1
    total_prob = sum(probabilities)
    print(f"\nTotal probability: {total_prob:.6f}")

    # Expected value and variance
    expected_value = sum(k * binomial_pmf(n_flips, p_head, k) for k in possible_values)
    variance = sum(k**2 * binomial_pmf(n_flips, p_head, k) for k in possible_values) - expected_value**2

    print(f"Expected value E[X] = {expected_value:.3f}")
    print(f"Variance Var(X) = {variance:.3f}")
    print(f"Standard deviation σ = {np.sqrt(variance):.3f}")

    # Continuous Random Variables
    print("\n2. Continuous Random Variables")

    # Normal distribution
    def normal_pdf(x, mu, sigma):
        """Probability density function for normal distribution"""
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    mu_normal = 0
    sigma_normal = 1
    x_range = np.linspace(-4, 4, 1000)
    pdf_values = normal_pdf(x_range, mu_normal, sigma_normal)

    print(f"Normal Distribution: X ~ N(μ={mu_normal}, σ²={sigma_normal**2})")

    # Probability in intervals
    interval_probs = []
    intervals = [(-1, 1), (-2, 2), (-3, 3)]

    for a, b in intervals:
        # Numerical integration using Simpson's rule
        from scipy.integrate import simps
        x_interval = np.linspace(a, b, 1000)
        y_interval = normal_pdf(x_interval, mu_normal, sigma_normal)
        prob = simps(y_interval, x_interval)
        interval_probs.append(prob)
        print(f"P({a} ≤ X ≤ {b}) ≈ {prob:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Discrete distribution (Binomial)
    axes[0, 0].bar(possible_values, probabilities, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Heads')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title('Binomial Distribution (n=3, p=0.5)')
    axes[0, 0].grid(True, alpha=0.3)

    # Continuous distribution (Normal)
    axes[0, 1].plot(x_range, pdf_values, 'r-', linewidth=2, label='PDF')
    for i, (a, b) in enumerate(intervals):
        x_shaded = x_range[(x_range >= a) & (x_range <= b)]
        y_shaded = normal_pdf(x_shaded, mu_normal, sigma_normal)
        axes[0, 1].fill_between(x_shaded, 0, y_shaded, alpha=0.3,
                               label=f'P({a}≤X≤{b})={interval_probs[i]:.2f}')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title('Standard Normal Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # CDF comparison
    # Binomial CDF
    binomial_cdf = []
    cumulative = 0
    for k in possible_values:
        cumulative += binomial_pmf(n_flips, p_head, k)
        binomial_cdf.append(cumulative)

    axes[1, 0].step(possible_values, binomial_cdf, where='post', color='blue', linewidth=2)
    axes[1, 0].set_xlabel('Number of Heads')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Binomial CDF')
    axes[1, 0].grid(True, alpha=0.3)

    # Normal CDF
    normal_cdf = stats.norm.cdf(x_range, mu_normal, sigma_normal)
    axes[1, 1].plot(x_range, normal_cdf, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Normal CDF')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return possible_values, probabilities, x_range, pdf_values

possible_values, probabilities, x_range, pdf_values = random_variables_distributions()
```

## 2. Common Probability Distributions

### 2.1 Discrete Distributions

```python
def discrete_distributions():
    """Common discrete probability distributions"""

    print("Common Discrete Probability Distributions")
    print("-" * 50)

    # 1. Bernoulli Distribution
    print("\n1. Bernoulli Distribution")
    print("X ~ Bernoulli(p)")

    def bernoulli_pmf(p, x):
        return p**x * (1-p)**(1-x) if x in [0, 1] else 0

    p_bernoulli = 0.3
    print(f"Parameter: p = {p_bernoulli}")
    for x in [0, 1]:
        print(f"P(X = {x}) = {bernoulli_pmf(p_bernoulli, x):.4f}")

    # 2. Binomial Distribution
    print("\n2. Binomial Distribution")
    print("X ~ Binomial(n, p)")

    n_binomial = 10
    p_binomial = 0.3
    k_values = list(range(n_binomial + 1))

    binomial_probs = [binomial_pmf(n_binomial, p_binomial, k) for k in k_values]
    print(f"Parameters: n = {n_binomial}, p = {p_binomial}")
    print("Probabilities for k = 0 to 10:")
    for k, prob in zip(k_values[:6], binomial_probs[:6]):  # Show first 6
        print(f"P(X = {k}) = {prob:.4f}")

    # 3. Poisson Distribution
    print("\n3. Poisson Distribution")
    print("X ~ Poisson(λ)")

    def poisson_pmf(lam, k):
        return (lam**k * np.exp(-lam)) / factorial(k)

    lambda_poisson = 3.0
    k_poisson = list(range(0, 11))
    poisson_probs = [poisson_pmf(lambda_poisson, k) for k in k_poisson]

    print(f"Parameter: λ = {lambda_poisson}")
    print("Probabilities for k = 0 to 10:")
    for k, prob in zip(k_poisson, poisson_probs):
        print(f"P(X = {k}) = {prob:.4f}")

    # 4. Geometric Distribution
    print("\n4. Geometric Distribution")
    print("X ~ Geometric(p)")

    def geometric_pmf(p, k):
        return p * (1-p)**(k-1) if k >= 1 else 0

    p_geometric = 0.2
    k_geometric = list(range(1, 11))
    geometric_probs = [geometric_pmf(p_geometric, k) for k in k_geometric]

    print(f"Parameter: p = {p_geometric}")
    print("Probabilities for k = 1 to 10:")
    for k, prob in zip(k_geometric, geometric_probs):
        print(f"P(X = {k}) = {prob:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Bernoulli
    axes[0, 0].bar([0, 1], [bernoulli_pmf(p_bernoulli, 0), bernoulli_pmf(p_bernoulli, 1)],
                  color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('P(X = x)')
    axes[0, 0].set_title(f'Bernoulli Distribution (p = {p_bernoulli})')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)

    # Binomial
    axes[0, 1].bar(k_values, binomial_probs, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('P(X = x)')
    axes[0, 1].set_title(f'Binomial Distribution (n = {n_binomial}, p = {p_binomial})')
    axes[0, 1].grid(True, alpha=0.3)

    # Poisson
    axes[1, 0].bar(k_poisson, poisson_probs, color='red', alpha=0.7)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('P(X = x)')
    axes[1, 0].set_title(f'Poisson Distribution (λ = {lambda_poisson})')
    axes[1, 0].grid(True, alpha=0.3)

    # Geometric
    axes[1, 1].bar(k_geometric, geometric_probs, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('P(X = x)')
    axes[1, 1].set_title(f'Geometric Distribution (p = {p_geometric})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Expected values and variances
    print("\nSummary Statistics:")
    print("-" * 30)

    # Bernoulli
    E_bernoulli = p_bernoulli
    Var_bernoulli = p_bernoulli * (1 - p_bernoulli)
    print(f"Bernoulli: E[X] = {E_bernoulli:.3f}, Var(X) = {Var_bernoulli:.3f}")

    # Binomial
    E_binomial = n_binomial * p_binomial
    Var_binomial = n_binomial * p_binomial * (1 - p_binomial)
    print(f"Binomial: E[X] = {E_binomial:.3f}, Var(X) = {Var_binomial:.3f}")

    # Poisson
    E_poisson = lambda_poisson
    Var_poisson = lambda_poisson
    print(f"Poisson: E[X] = {E_poisson:.3f}, Var(X) = {Var_poisson:.3f}")

    # Geometric
    E_geometric = 1 / p_geometric
    Var_geometric = (1 - p_geometric) / (p_geometric ** 2)
    print(f"Geometric: E[X] = {E_geometric:.3f}, Var(X) = {Var_geometric:.3f}")

    return {
        'bernoulli': (p_bernoulli, E_bernoulli, Var_bernoulli),
        'binomial': (n_binomial, p_binomial, E_binomial, Var_binomial),
        'poisson': (lambda_poisson, E_poisson, Var_poisson),
        'geometric': (p_geometric, E_geometric, Var_geometric)
    }

distributions_info = discrete_distributions()
```

### 2.2 Continuous Distributions

```python
def continuous_distributions():
    """Common continuous probability distributions"""

    print("Common Continuous Probability Distributions")
    print("-" * 50)

    # Generate x range for plotting
    x_range = np.linspace(-5, 5, 1000)

    # 1. Normal Distribution
    print("\n1. Normal Distribution")
    print("X ~ N(μ, σ²)")

    mu_normal = 0
    sigma_normal = 1
    normal_pdf = stats.norm.pdf(x_range, mu_normal, sigma_normal)
    normal_cdf = stats.norm.cdf(x_range, mu_normal, sigma_normal)

    print(f"Parameters: μ = {mu_normal}, σ = {sigma_normal}")
    print(f"P(-1 ≤ X ≤ 1) = {stats.norm.cdf(1, mu_normal, sigma_normal) - stats.norm.cdf(-1, mu_normal, sigma_normal):.4f}")

    # 2. Exponential Distribution
    print("\n2. Exponential Distribution")
    print("X ~ Exp(λ)")

    lambda_exp = 1.0
    x_exp = np.linspace(0, 10, 1000)
    exp_pdf = stats.expon.pdf(x_exp, scale=1/lambda_exp)
    exp_cdf = stats.expon.cdf(x_exp, scale=1/lambda_exp)

    print(f"Parameter: λ = {lambda_exp}")
    print(f"P(X > 1) = {1 - stats.expon.cdf(1, scale=1/lambda_exp):.4f}")

    # 3. Uniform Distribution
    print("\n3. Uniform Distribution")
    print("X ~ U(a, b)")

    a_uniform = -2
    b_uniform = 3
    x_uniform = np.linspace(-3, 4, 1000)
    uniform_pdf = stats.uniform.pdf(x_uniform, loc=a_uniform, scale=b_uniform-a_uniform)
    uniform_cdf = stats.uniform.cdf(x_uniform, loc=a_uniform, scale=b_uniform-a_uniform)

    print(f"Parameters: a = {a_uniform}, b = {b_uniform}")
    print(f"P(0 ≤ X ≤ 2) = {stats.uniform.cdf(2, loc=a_uniform, scale=b_uniform-a_uniform) - stats.uniform.cdf(0, loc=a_uniform, scale=b_uniform-a_uniform):.4f}")

    # 4. Beta Distribution
    print("\n4. Beta Distribution")
    print("X ~ Beta(α, β)")

    alpha_beta = 2
    beta_beta = 5
    x_beta = np.linspace(0, 1, 1000)
    beta_pdf = stats.beta.pdf(x_beta, alpha_beta, beta_beta)
    beta_cdf = stats.beta.cdf(x_beta, alpha_beta, beta_beta)

    print(f"Parameters: α = {alpha_beta}, β = {beta_beta}")
    print(f"P(0.2 ≤ X ≤ 0.8) = {stats.beta.cdf(0.8, alpha_beta, beta_beta) - stats.beta.cdf(0.2, alpha_beta, beta_beta):.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Normal Distribution
    axes[0, 0].plot(x_range, normal_pdf, 'b-', linewidth=2, label='PDF')
    axes[0, 0].fill_between(x_range[(x_range >= -1) & (x_range <= 1)], 0,
                           normal_pdf[(x_range >= -1) & (x_range <= 1)],
                           alpha=0.3, label='P(-1 ≤ X ≤ 1)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title(f'Normal Distribution N({mu_normal}, {sigma_normal}²)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Exponential Distribution
    axes[0, 1].plot(x_exp, exp_pdf, 'r-', linewidth=2, label='PDF')
    axes[0, 1].fill_between(x_exp[x_exp >= 1], 0, exp_pdf[x_exp >= 1],
                           alpha=0.3, label='P(X > 1)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title(f'Exponential Distribution Exp({lambda_exp})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Uniform Distribution
    axes[1, 0].plot(x_uniform, uniform_pdf, 'g-', linewidth=2, label='PDF')
    axes[1, 0].fill_between(x_uniform[(x_uniform >= 0) & (x_uniform <= 2)], 0,
                           uniform_pdf[(x_uniform >= 0) & (x_uniform <= 2)],
                           alpha=0.3, label='P(0 ≤ X ≤ 2)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].set_title(f'Uniform Distribution U({a_uniform}, {b_uniform})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Beta Distribution
    axes[1, 1].plot(x_beta, beta_pdf, 'm-', linewidth=2, label='PDF')
    axes[1, 1].fill_between(x_beta[(x_beta >= 0.2) & (x_beta <= 0.8)], 0,
                           beta_pdf[(x_beta >= 0.2) & (x_beta <= 0.8)],
                           alpha=0.3, label='P(0.2 ≤ X ≤ 0.8)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].set_title(f'Beta Distribution Beta({alpha_beta}, {beta_beta})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 30)

    # Normal
    E_normal = mu_normal
    Var_normal = sigma_normal**2
    print(f"Normal: E[X] = {E_normal:.3f}, Var(X) = {Var_normal:.3f}")

    # Exponential
    E_exp = 1/lambda_exp
    Var_exp = 1/(lambda_exp**2)
    print(f"Exponential: E[X] = {E_exp:.3f}, Var(X) = {Var_exp:.3f}")

    # Uniform
    E_uniform = (a_uniform + b_uniform) / 2
    Var_uniform = (b_uniform - a_uniform)**2 / 12
    print(f"Uniform: E[X] = {E_uniform:.3f}, Var(X) = {Var_uniform:.3f}")

    # Beta
    E_beta = alpha_beta / (alpha_beta + beta_beta)
    Var_beta = (alpha_beta * beta_beta) / ((alpha_beta + beta_beta)**2 * (alpha_beta + beta_beta + 1))
    print(f"Beta: E[X] = {E_beta:.3f}, Var(X) = {Var_beta:.3f}")

    return {
        'normal': (mu_normal, sigma_normal, E_normal, Var_normal),
        'exponential': (lambda_exp, E_exp, Var_exp),
        'uniform': (a_uniform, b_uniform, E_uniform, Var_uniform),
        'beta': (alpha_beta, beta_beta, E_beta, Var_beta)
    }

cont_distributions_info = continuous_distributions()
```

## 3. Joint and Conditional Probability

### 3.1 Joint Distributions

```python
def joint_distributions():
    """Joint probability distributions and independence"""

    print("Joint Probability Distributions")
    print("-" * 40)

    # Discrete joint distribution example
    print("\n1. Discrete Joint Distribution")
    print("Example: Two dice rolls")

    # Create joint probability table for two independent dice
    die1_values = list(range(1, 7))
    die2_values = list(range(1, 7))

    # Joint PMF (since independent: P(X=x, Y=y) = P(X=x) × P(Y=y))
    joint_pmf = np.zeros((6, 6))
    for i, x in enumerate(die1_values):
        for j, y in enumerate(die2_values):
            joint_pmf[i, j] = (1/6) * (1/6)

    print("Joint PMF Table (P(X=x, Y=y)):")
    print("    Y=1   Y=2   Y=3   Y=4   Y=5   Y=6")
    for i, x in enumerate(die1_values):
        row_str = f"X={x} "
        for j in range(6):
            row_str += f"{joint_pmf[i, j]:.4f} "
        print(row_str)

    # Marginal distributions
    marginal_x = np.sum(joint_pmf, axis=1)
    marginal_y = np.sum(joint_pmf, axis=0)

    print(f"\nMarginal P(X): {marginal_x}")
    print(f"Marginal P(Y): {marginal_y}")

    # Check independence
    print("\nIndependence Check:")
    for i, x in enumerate([1, 3, 5]):  # Check some values
        for j, y in enumerate([2, 4, 6]):
            joint_prob = joint_pmf[i, j]
            marginal_prob = marginal_x[i] * marginal_y[j]
            print(f"P(X={x}, Y={y}) = {joint_prob:.4f}")
            print(f"P(X={x}) × P(Y={y}) = {marginal_prob:.4f}")
            print(f"Independent: {abs(joint_prob - marginal_prob) < 1e-10}")
            print()

    # Continuous joint distribution
    print("2. Continuous Joint Distribution")
    print("Example: Bivariate Normal Distribution")

    def bivariate_normal_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
        """Bivariate normal PDF"""
        z_x = (x - mu_x) / sigma_x
        z_y = (y - mu_y) / sigma_y
        denom = 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)
        exp_term = -0.5 * (z_x**2 - 2*rho*z_x*z_y + z_y**2) / (1 - rho**2)
        return np.exp(exp_term) / denom

    # Parameters
    mu_x, mu_y = 0, 0
    sigma_x, sigma_y = 1, 1
    rho = 0.7  # Correlation coefficient

    # Create grid for visualization
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = bivariate_normal_pdf(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho)

    print(f"Parameters: μx={mu_x}, μy={mu_y}, σx={sigma_x}, σy={sigma_y}, ρ={rho}")

    # Calculate some probabilities
    print("\nProbability Calculations:")
    # P(X > 0, Y > 0)
    x_pos = X[X > 0]
    y_pos = Y[Y > 0]
    prob_xy_pos = 0.25  # For standard normal with positive correlation, approximately 0.25 + 0.5*rho/4
    print(f"P(X > 0, Y > 0) ≈ {prob_xy_pos:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Joint PMF heatmap
    im1 = axes[0, 0].imshow(joint_pmf, cmap='Blues', aspect='equal')
    axes[0, 0].set_title('Joint PMF: Two Independent Dice')
    axes[0, 0].set_xlabel('Die 2')
    axes[0, 0].set_ylabel('Die 1')
    axes[0, 0].set_xticks(range(6))
    axes[0, 0].set_yticks(range(6))
    axes[0, 0].set_xticklabels(range(1, 7))
    axes[0, 0].set_yticklabels(range(1, 7))
    plt.colorbar(im1, ax=axes[0, 0])

    # Bivariate normal contour plot
    contour = axes[0, 1].contour(X, Y, Z, levels=20, alpha=0.6)
    axes[0, 1].clabel(contour, inline=True, fontsize=8)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title(f'Bivariate Normal (ρ = {rho})')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')

    # Marginal distributions
    axes[1, 0].bar(die1_values, marginal_x, alpha=0.7, color='blue', label='X')
    axes[1, 0].bar(die2_values, marginal_y, alpha=0.7, color='red', label='Y')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title('Marginal Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Covariance and correlation visualization
    axes[1, 1].scatter(np.random.randn(1000), np.random.randn(1000), alpha=0.6, label='ρ = 0')
    correlated_data = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 1000)
    axes[1, 1].scatter(correlated_data[:, 0], correlated_data[:, 1], alpha=0.6, label=f'ρ = {rho}')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('Correlation Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return joint_pmf, marginal_x, marginal_y

joint_pmf, marginal_x, marginal_y = joint_distributions()
```

### 3.2 Conditional Probability and Bayes' Theorem

```python
def conditional_probability():
    """Conditional probability and Bayes' theorem"""

    print("Conditional Probability and Bayes' Theorem")
    print("-" * 50)

    # Medical testing example
    print("\n1. Medical Testing Example")
    print("Disease prevalence, test sensitivity, and specificity")

    # Parameters
    prevalence = 0.01  # 1% of population has disease
    sensitivity = 0.95  # P(positive|disease) = 95%
    specificity = 0.90  # P(negative|no disease) = 90%

    print(f"Disease prevalence (P(D)) = {prevalence:.3f}")
    print(f"Test sensitivity (P(+|D)) = {sensitivity:.3f}")
    print(f"Test specificity (P(-|¬D)) = {specificity:.3f}")

    # Calculate probabilities
    p_disease = prevalence
    p_no_disease = 1 - prevalence
    p_pos_given_disease = sensitivity
    p_neg_given_no_disease = specificity
    p_pos_given_no_disease = 1 - specificity

    print(f"\nP(¬D) = {p_no_disease:.3f}")
    print(f"P(+|¬D) = {p_pos_given_no_disease:.3f}")

    # Total probability of positive test
    p_positive = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease

    # Posterior probability using Bayes' theorem
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive

    print(f"\nBayes' Theorem Application:")
    print(f"P(D|+) = [P(+|D) × P(D)] / P(+)")
    print(f"        = [{sensitivity:.3f} × {prevalence:.3f}] / {p_positive:.4f}")
    print(f"        = {p_disease_given_pos:.4f}")

    print(f"\nInterpretation:")
    print(f"Even with a positive test result, there's only a {p_disease_given_pos*100:.1f}% chance")
    print(f"that the patient actually has the disease!")

    # Multiple scenarios
    prevalences = [0.001, 0.01, 0.05, 0.1, 0.2]
    posterior_probs = []

    print(f"\n2. Effect of Prevalence on Posterior Probability")
    print(f"(Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f})")
    print("-" * 60)

    for prev in prevalences:
        p_pos = sensitivity * prev + (1 - specificity) * (1 - prev)
        p_d_given_pos = (sensitivity * prev) / p_pos if p_pos > 0 else 0
        posterior_probs.append(p_d_given_pos)
        print(f"Prevalence = {prev:.3f}: P(D|+) = {p_d_given_pos:.4f}")

    # Machine learning example: Naive Bayes classifier
    print(f"\n3. Naive Bayes Classifier Example")
    print("Email spam classification")

    # Training data
    # Features: contains 'free', contains 'money', contains 'urgent'
    # Classes: spam, not_spam
    training_data = [
        ([1, 1, 0], 'spam'),    # contains free, money
        ([1, 0, 1], 'spam'),    # contains free, urgent
        ([0, 1, 0], 'spam'),    # contains money
        ([1, 1, 1], 'spam'),    # contains all
        ([0, 0, 1], 'not_spam'), # contains urgent
        ([0, 0, 0], 'not_spam'), # contains none
        ([1, 0, 0], 'not_spam'), # contains free
        ([0, 1, 1], 'not_spam')  # contains money, urgent
    ]

    # Calculate class priors
    total_emails = len(training_data)
    spam_count = sum(1 for _, label in training_data if label == 'spam')
    not_spam_count = total_emails - spam_count

    p_spam = spam_count / total_emails
    p_not_spam = not_spam_count / total_emails

    print(f"Training data: {total_emails} emails")
    print(f"P(spam) = {p_spam:.3f}")
    print(f"P(not_spam) = {p_not_spam:.3f}")

    # Calculate likelihoods (with Laplace smoothing)
    def calculate_likelihoods(feature_idx, class_label, smoothing=1):
        """Calculate P(feature=1|class) with Laplace smoothing"""
        class_data = [features for features, label in training_data if label == class_label]
        feature_count = sum(features[feature_idx] for features in class_data)
        total = len(class_data)
        return (feature_count + smoothing) / (total + 2 * smoothing)

    # Likelihoods for each feature
    features = ['free', 'money', 'urgent']
    likelihoods_spam = []
    likelihoods_not_spam = []

    for i in range(3):
        p_feature_given_spam = calculate_likelihoods(i, 'spam')
        p_feature_given_not_spam = calculate_likelihoods(i, 'not_spam')
        likelihoods_spam.append(p_feature_given_spam)
        likelihoods_not_spam.append(p_feature_given_not_spam)

    print(f"\nLikelihoods:")
    for i, feature in enumerate(features):
        print(f"P({feature}=1|spam) = {likelihoods_spam[i]:.3f}")
        print(f"P({feature}=1|not_spam) = {likelihoods_not_spam[i]:.3f}")

    # Classify a new email
    new_email_features = [1, 1, 0]  # contains 'free' and 'money'

    # Calculate posterior probabilities
    # P(spam|features) ∝ P(features|spam) × P(spam)
    # P(not_spam|features) ∝ P(features|not_spam) × P(not_spam)

    def calculate_posterior(features, class_label, class_prior, likelihoods):
        """Calculate posterior probability for a class"""
        likelihood_product = 1.0
        for i, feature_value in enumerate(features):
            if feature_value == 1:
                likelihood_product *= likelihoods[i]
            else:
                likelihood_product *= (1 - likelihoods[i])
        return likelihood_product * class_prior

    posterior_spam = calculate_posterior(new_email_features, 'spam', p_spam, likelihoods_spam)
    posterior_not_spam = calculate_posterior(new_email_features, 'not_spam', p_not_spam, likelihoods_not_spam)

    # Normalize
    total_posterior = posterior_spam + posterior_not_spam
    p_spam_given_features = posterior_spam / total_posterior
    p_not_spam_given_features = posterior_not_spam / total_posterior

    print(f"\nClassifying email with features: {features[i] for i in range(3) if new_email_features[i] == 1}")
    print(f"P(spam|features) = {p_spam_given_features:.4f}")
    print(f"P(not_spam|features) = {p_not_spam_given_features:.4f}")
    print(f"Prediction: {'spam' if p_spam_given_features > p_not_spam_given_features else 'not_spam'}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Medical testing results
    axes[0, 0].bar(['1%', '10%', '20%'], [posterior_probs[1], posterior_probs[3], posterior_probs[4]],
                  color=['red', 'orange', 'green'], alpha=0.7)
    axes[0, 0].set_ylabel('P(Disease|Positive Test)')
    axes[0, 0].set_title('Effect of Disease Prevalence on Test Accuracy')
    axes[0, 0].grid(True, alpha=0.3)

    # Bayes theorem visualization
    # Show the components of Bayes theorem
    components = ['P(+|D)×P(D)', 'P(+|¬D)×P(¬D)', 'Total P(+)']
    values = [sensitivity * prevalence, (1-specificity) * (1-prevalence), p_positive]

    axes[0, 1].bar(components, values, color=['blue', 'red', 'green'], alpha=0.7)
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_title('Components of Bayes Theorem')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Confusion matrix for medical test
    confusion_matrix = np.array([
        [sensitivity * prevalence * 1000, (1-specificity) * (1-prevalence) * 1000],  # True positive, False positive
        [(1-sensitivity) * prevalence * 1000, specificity * (1-prevalence) * 1000]   # False negative, True negative
    ])

    im = axes[1, 0].imshow(confusion_matrix, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Confusion Matrix (per 1000 people)')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Positive', 'Negative'])
    axes[1, 0].set_yticklabels(['Disease', 'No Disease'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{confusion_matrix[i, j]:.0f}',
                           ha='center', va='center', fontsize=12)

    plt.colorbar(im, ax=axes[1, 0])

    # Naive Bayes feature importance
    axes[1, 1].bar(features, likelihoods_spam, alpha=0.7, color='red', label='Spam')
    axes[1, 1].bar(features, likelihoods_not_spam, alpha=0.7, color='blue', label='Not Spam')
    axes[1, 1].set_ylabel('P(feature=1|class)')
    axes[1, 1].set_title('Feature Likelihoods in Naive Bayes')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return p_disease_given_pos, p_spam_given_features

p_disease_given_pos, p_spam_given_features = conditional_probability()
```

## 4. Expectation and Variance

### 4.1 Expected Value and Moments

```python
def expectation_variance():
    """Expected value, variance, and moments"""

    print("Expected Value, Variance, and Higher Moments")
    print("-" * 50)

    # Discrete random variable example
    print("\n1. Discrete Random Variable: Roll of a fair die")

    # Die roll: X = {1, 2, 3, 4, 5, 6} with P(X=x) = 1/6
    die_values = np.array([1, 2, 3, 4, 5, 6])
    die_probs = np.array([1/6] * 6)

    # Expected value
    E_X = np.sum(die_values * die_probs)
    print(f"E[X] = ∑ x × P(X=x) = {E_X:.3f}")

    # E[X²]
    E_X2 = np.sum(die_values**2 * die_probs)
    print(f"E[X²] = ∑ x² × P(X=x) = {E_X2:.3f}")

    # Variance
    Var_X = E_X2 - E_X**2
    print(f"Var(X) = E[X²] - (E[X])² = {Var_X:.3f}")

    # Standard deviation
    sigma_X = np.sqrt(Var_X)
    print(f"σ(X) = √Var(X) = {sigma_X:.3f}")

    # Higher moments
    # Skewness: E[(X-μ)³]/σ³
    central_moments = []
    for n in range(1, 5):
        moment = np.sum((die_values - E_X)**n * die_probs)
        central_moments.append(moment)
        print(f"Central moment μ{n} = E[(X-μ)ⁿ] = {moment:.3f}")

    skewness = central_moments[2] / sigma_X**3
    kurtosis = central_moments[3] / sigma_X**4
    print(f"Skewness = μ³/σ³ = {skewness:.3f}")
    print(f"Kurtosis = μ⁴/σ⁴ = {kurtosis:.3f}")

    # Continuous random variable example
    print("\n2. Continuous Random Variable: Normal Distribution")

    # Normal distribution parameters
    mu_normal = 5
    sigma_normal = 2

    print(f"X ~ N(μ={mu_normal}, σ={sigma_normal})")

    # Theoretical moments
    E_X_normal = mu_normal
    Var_X_normal = sigma_normal**2
    sigma_X_normal = sigma_normal

    print(f"Theoretical:")
    print(f"E[X] = {E_X_normal:.3f}")
    print(f"Var(X) = {Var_X_normal:.3f}")
    print(f"σ(X) = {sigma_X_normal:.3f}")

    # Sample moments from simulation
    np.random.seed(42)
    samples = np.random.normal(mu_normal, sigma_normal, 10000)

    E_X_sample = np.mean(samples)
    Var_X_sample = np.var(samples, ddof=1)  # Sample variance
    sigma_X_sample = np.std(samples, ddof=1)

    skewness_sample = stats.skew(samples)
    kurtosis_sample = stats.kurtosis(samples) + 3  # Pearson's kurtosis

    print(f"\nSample (n=10,000):")
    print(f"E[X] = {E_X_sample:.3f}")
    print(f"Var(X) = {Var_X_sample:.3f}")
    print(f"σ(X) = {sigma_X_sample:.3f}")
    print(f"Skewness = {skewness_sample:.3f}")
    print(f"Kurtosis = {kurtosis_sample:.3f}")

    # Linearity of expectation
    print("\n3. Linearity of Expectation")
    print("Example: Sum of two dice rolls")

    # Y = X1 + X2 where X1, X2 are independent die rolls
    E_X1 = E_X
    E_X2 = E_X
    E_Y = E_X1 + E_X2

    print(f"E[X1] = {E_X1:.3f}")
    print(f"E[X2] = {E_X2:.3f}")
    print(f"E[X1 + X2] = E[X1] + E[X2] = {E_Y:.3f}")

    # For independent variables: Var(X1 + X2) = Var(X1) + Var(X2)
    Var_X1 = Var_X
    Var_X2 = Var_X
    Var_Y = Var_X1 + Var_X2

    print(f"Var(X1) = {Var_X1:.3f}")
    print(f"Var(X2) = {Var_X2:.3f}")
    print(f"Var(X1 + X2) = Var(X1) + Var(X2) = {Var_Y:.3f}")

    # Covariance and correlation
    print("\n4. Covariance and Correlation")

    # Generate correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]  # Correlation = 0.7
    data = np.random.multivariate_normal(mean, cov, 1000)

    X = data[:, 0]
    Y = data[:, 1]

    # Sample covariance
    cov_sample = np.cov(X, Y, ddof=1)[0, 1]
    corr_sample = np.corrcoef(X, Y)[0, 1]

    print(f"Sample covariance: Cov(X,Y) = {cov_sample:.4f}")
    print(f"Sample correlation: ρ(X,Y) = {corr_sample:.4f}")

    # Theoretical values
    cov_theoretical = 0.7
    corr_theoretical = 0.7

    print(f"Theoretical covariance: {cov_theoretical:.4f}")
    print(f"Theoretical correlation: {corr_theoretical:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Die roll distribution
    axes[0, 0].bar(die_values, die_probs, color='blue', alpha=0.7)
    axes[0, 0].axvline(E_X, color='red', linestyle='--', linewidth=2, label=f'E[X] = {E_X}')
    axes[0, 0].axvline(E_X - sigma_X, color='green', linestyle=':', linewidth=1, label=f'μ-σ')
    axes[0, 0].axvline(E_X + sigma_X, color='green', linestyle=':', linewidth=1, label=f'μ+σ')
    axes[0, 0].set_xlabel('Die Value')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title('Die Roll Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Normal distribution with moments
    x_normal = np.linspace(mu_normal - 4*sigma_normal, mu_normal + 4*sigma_normal, 100)
    pdf_normal = stats.norm.pdf(x_normal, mu_normal, sigma_normal)

    axes[0, 1].plot(x_normal, pdf_normal, 'b-', linewidth=2, label='PDF')
    axes[0, 1].axvline(mu_normal, color='red', linestyle='--', linewidth=2, label=f'μ = {mu_normal}')
    axes[0, 1].axvline(mu_normal - sigma_normal, color='green', linestyle=':', linewidth=1, label='μ±σ')
    axes[0, 1].axvline(mu_normal + sigma_normal, color='green', linestyle=':', linewidth=1)
    axes[0, 1].axvline(mu_normal - 2*sigma_normal, color='orange', linestyle=':', linewidth=1, label='μ±2σ')
    axes[0, 1].axvline(mu_normal + 2*sigma_normal, color='orange', linestyle=':', linewidth=1)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title(f'Normal Distribution N({mu_normal}, {sigma_normal}²)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram of samples vs theoretical
    axes[1, 0].hist(samples, bins=50, density=True, alpha=0.7, color='blue', label='Sample histogram')
    axes[1, 0].plot(x_normal, pdf_normal, 'r-', linewidth=2, label='Theoretical PDF')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Sample vs Theoretical Normal Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Correlated variables
    axes[1, 1].scatter(X, Y, alpha=0.6, s=10)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title(f'Correlated Variables (ρ = {corr_sample:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return {
        'die': {'E_X': E_X, 'Var_X': Var_X, 'skewness': skewness},
        'normal': {'E_X': E_X_normal, 'Var_X': Var_X_normal},
        'correlation': {'cov_sample': cov_sample, 'corr_sample': corr_sample}
    }

moments_info = expectation_variance()
```

## 5. Limit Theorems and Convergence

### 5.1 Law of Large Numbers and Central Limit Theorem

```python
def limit_theorems():
    """Law of Large Numbers and Central Limit Theorem"""

    print("Limit Theorems in Probability")
    print("-" * 40)

    # Law of Large Numbers
    print("\n1. Law of Large Numbers (LLN)")
    print("Sample mean converges to population mean as n → ∞")

    # Generate data from a known distribution
    np.random.seed(42)
    true_mean = 5
    true_var = 4
    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]

    print(f"True distribution: N(μ={true_mean}, σ²={true_var})")

    sample_means = []
    sample_vars = []

    for n in sample_sizes:
        samples = np.random.normal(true_mean, np.sqrt(true_var), n)
        sample_mean = np.mean(samples)
        sample_var = np.var(samples, ddof=1)

        sample_means.append(sample_mean)
        sample_vars.append(sample_var)

        print(f"n = {n:5d}: Sample mean = {sample_mean:.4f}, Sample var = {sample_var:.4f}")

    # Central Limit Theorem
    print("\n2. Central Limit Theorem (CLT)")
    print("Sum/mean of independent random variables → Normal distribution")

    # CLT demonstration with different parent distributions
    def clt_demo(distribution_name, sampler, sample_size=1000, n_experiments=10000):
        """Demonstrate CLT with a given distribution"""
        print(f"\nCLT with {distribution_name} distribution:")

        # Sample means from the distribution
        sample_means = []
        for _ in range(n_experiments):
            sample = sampler(sample_size)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)

        # Calculate statistics of sample means
        mean_of_means = np.mean(sample_means)
        var_of_means = np.var(sample_means, ddof=1)
        std_of_means = np.std(sample_means, ddof=1)

        print(f"Mean of sample means: {mean_of_means:.4f}")
        print(f"Variance of sample means: {var_of_means:.4f}")
        print(f"Standard deviation of sample means: {std_of_means:.4f}")

        # Theoretical values (if known)
        if hasattr(sampler, 'true_mean'):
            theoretical_mean = sampler.true_mean
            theoretical_std = sampler.true_std / np.sqrt(sample_size)
            print(f"Theoretical mean: {theoretical_mean:.4f}")
            print(f"Theoretical std: {theoretical_std:.4f}")

        # Normality test
        from scipy.stats import normaltest
        _, p_value = normaltest(sample_means)
        print(f"Normality test p-value: {p_value:.4f}")
        print(f"Is normal? {'Yes' if p_value > 0.05 else 'No'}")

        return sample_means

    # Different distributions to test CLT
    class UniformSampler:
        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.true_mean = (a + b) / 2
            self.true_var = (b - a)**2 / 12
            self.true_std = np.sqrt(self.true_var)

        def __call__(self, n):
            return np.random.uniform(self.a, self.b, n)

    class ExponentialSampler:
        def __init__(self, lam):
            self.lam = lam
            self.true_mean = 1 / lam
            self.true_var = 1 / (lam**2)
            self.true_std = 1 / lam

        def __call__(self, n):
            return np.random.exponential(1/self.lam, n)

    class BernoulliSampler:
        def __init__(self, p):
            self.p = p
            self.true_mean = p
            self.true_var = p * (1 - p)
            self.true_std = np.sqrt(self.true_var)

        def __call__(self, n):
            return np.random.binomial(1, self.p, n)

    # Run CLT demonstrations
    uniform_sampler = UniformSampler(0, 6)  # U(0,6)
    exp_sampler = ExponentialSampler(1)      # Exp(1)
    bernoulli_sampler = BernoulliSampler(0.3)  # Bernoulli(0.3)

    uniform_means = clt_demo("Uniform(0,6)", uniform_sampler)
    exp_means = clt_demo("Exponential(1)", exp_sampler)
    bernoulli_means = clt_demo("Bernoulli(0.3)", bernoulli_sampler)

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Law of Large Numbers
    axes[0, 0].plot(sample_sizes, sample_means, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=true_mean, color='red', linestyle='--', linewidth=2, label=f'True mean = {true_mean}')
    axes[0, 0].set_xlabel('Sample Size')
    axes[0, 0].set_ylabel('Sample Mean')
    axes[0, 0].set_title('Law of Large Numbers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')

    axes[0, 1].plot(sample_sizes, sample_vars, 'go-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=true_var, color='red', linestyle='--', linewidth=2, label=f'True variance = {true_var}')
    axes[0, 1].set_xlabel('Sample Size')
    axes[0, 1].set_ylabel('Sample Variance')
    axes[0, 1].set_title('Sample Variance Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')

    # CLT distributions
    # Uniform distribution CLT
    axes[1, 0].hist(uniform_means, bins=50, density=True, alpha=0.7, color='blue', label='Sample means')

    # Theoretical normal approximation
    x_theory = np.linspace(np.mean(uniform_means) - 4*np.std(uniform_means),
                          np.mean(uniform_means) + 4*np.std(uniform_means), 100)
    theory_pdf = stats.norm.pdf(x_theory, uniform_sampler.true_mean,
                               uniform_sampler.true_std / np.sqrt(1000))
    axes[1, 0].plot(x_theory, theory_pdf, 'r-', linewidth=2, label='Normal approximation')
    axes[1, 0].set_xlabel('Sample Mean')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('CLT: Uniform Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Exponential distribution CLT
    axes[1, 1].hist(exp_means, bins=50, density=True, alpha=0.7, color='green', label='Sample means')

    x_theory = np.linspace(np.mean(exp_means) - 4*np.std(exp_means),
                          np.mean(exp_means) + 4*np.std(exp_means), 100)
    theory_pdf = stats.norm.pdf(x_theory, exp_sampler.true_mean,
                               exp_sampler.true_std / np.sqrt(1000))
    axes[1, 1].plot(x_theory, theory_pdf, 'r-', linewidth=2, label='Normal approximation')
    axes[1, 1].set_xlabel('Sample Mean')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('CLT: Exponential Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Bernoulli distribution CLT
    axes[2, 0].hist(bernoulli_means, bins=50, density=True, alpha=0.7, color='orange', label='Sample means')

    x_theory = np.linspace(np.mean(bernoulli_means) - 4*np.std(bernoulli_means),
                          np.mean(bernoulli_means) + 4*np.std(bernoulli_means), 100)
    theory_pdf = stats.norm.pdf(x_theory, bernoulli_sampler.true_mean,
                               bernoulli_sampler.true_std / np.sqrt(1000))
    axes[2, 0].plot(x_theory, theory_pdf, 'r-', linewidth=2, label='Normal approximation')
    axes[2, 0].set_xlabel('Sample Mean')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].set_title('CLT: Bernoulli Distribution')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Q-Q plots for normality check
    from scipy.stats import probplot

    axes[2, 1] = plt.subplot(3, 2, 6)
    probplot(exp_means, plot=axes[2, 1])
    axes[2, 1].set_title('Q-Q Plot: Exponential Sample Means')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary of limit theorems
    print("\n3. Summary of Limit Theorems")
    print("-" * 40)

    print("Law of Large Numbers:")
    print("- Sample mean converges to population mean")
    print("- Variance of sample mean decreases as 1/n")
    print("- Holds for i.i.d. random variables with finite mean")

    print("\nCentral Limit Theorem:")
    print("- Sample mean distribution approaches normal distribution")
    print("- Holds regardless of parent distribution (with finite variance)")
    print("- Convergence rate depends on the shape of parent distribution")

    print("\nPractical Implications:")
    print("- Confidence intervals for population parameters")
    print("- Hypothesis testing")
    print("- Quality control and process monitoring")

    return sample_means, uniform_means, exp_means, bernoulli_means

limit_theorems_results = limit_theorems()
```

## 6. Applications in Machine Learning

### 6.1 Maximum Likelihood Estimation

```python
def maximum_likelihood_estimation():
    """Maximum Likelihood Estimation in Machine Learning"""

    print("Maximum Likelihood Estimation (MLE)")
    print("-" * 40)

    # Generate synthetic data
    np.random.seed(42)
    true_mean = 3.5
    true_std = 1.2
    n_samples = 100

    data = np.random.normal(true_mean, true_std, n_samples)

    print(f"Generated data: {n_samples} samples from N({true_mean}, {true_std}²)")
    print(f"Sample mean: {np.mean(data):.4f}")
    print(f"Sample std: {np.std(data, ddof=1):.4f}")

    # MLE for normal distribution
    def normal_log_likelihood(data, mu, sigma):
        """Log likelihood for normal distribution"""
        n = len(data)
        return -n/2 * np.log(2*np.pi) - n * np.log(sigma) - \
               np.sum((data - mu)**2) / (2 * sigma**2)

    # Grid search for MLE
    mu_range = np.linspace(2, 5, 100)
    sigma_range = np.linspace(0.5, 2.5, 100)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)

    log_likelihood_grid = np.zeros_like(MU)
    for i in range(len(mu_range)):
        for j in range(len(sigma_range)):
            log_likelihood_grid[j, i] = normal_log_likelihood(data, MU[j, i], SIGMA[j, i])

    # Find MLE
    max_idx = np.unravel_index(np.argmax(log_likelihood_grid), log_likelihood_grid.shape)
    mu_mle = mu_range[max_idx[1]]
    sigma_mle = sigma_range[max_idx[0]]

    print(f"\nMLE Results:")
    print(f"μ_MLE = {mu_mle:.4f} (true = {true_mean:.4f})")
    print(f"σ_MLE = {sigma_mle:.4f} (true = {true_std:.4f})")

    # Analytical MLE for normal distribution
    mu_analytical = np.mean(data)
    sigma_analytical = np.std(data, ddof=0)  # MLE uses ddof=0

    print(f"\nAnalytical MLE:")
    print(f"μ_MLE = {mu_analytical:.4f}")
    print(f"σ_MLE = {sigma_analytical:.4f}")

    # MLE for linear regression
    print("\n2. MLE for Linear Regression")

    # Generate linear regression data
    n_reg = 50
    X_reg = np.linspace(0, 10, n_reg)
    true_slope = 2.5
    true_intercept = 1.0
    noise_std = 0.8
    y_reg = true_intercept + true_slope * X_reg + np.random.normal(0, noise_std, n_reg)

    # MLE for linear regression (equivalent to least squares)
    def linear_regression_log_likelihood(y, X, slope, intercept, sigma):
        """Log likelihood for linear regression with Gaussian noise"""
        predictions = intercept + slope * X
        residuals = y - predictions
        n = len(y)
        return -n/2 * np.log(2*np.pi) - n * np.log(sigma) - \
               np.sum(residuals**2) / (2 * sigma**2)

    # Analytical solution for MLE (normal equations)
    X_design = np.column_stack([np.ones(n_reg), X_reg])
    beta_mle = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_reg
    intercept_mle, slope_mle = beta_mle

    # Estimate noise standard deviation
    residuals = y_reg - (intercept_mle + slope_mle * X_reg)
    sigma_mle_reg = np.std(residuals, ddof=0)

    print(f"Linear Regression MLE:")
    print(f"Intercept: {intercept_mle:.4f} (true = {true_intercept:.4f})")
    print(f"Slope: {slope_mle:.4f} (true = {true_slope:.4f})")
    print(f"Noise std: {sigma_mle_reg:.4f} (true = {noise_std:.4f})")

    # MLE for logistic regression
    print("\n3. MLE for Logistic Regression")

    # Generate binary classification data
    n_class = 100
    X_class = np.random.randn(n_class, 2)
    true_weights = np.array([1.5, -2.0])
    logits = X_class @ true_weights
    probs = 1 / (1 + np.exp(-logits))
    y_class = (probs > 0.5).astype(int)

    # Logistic regression log likelihood
    def logistic_log_likelihood(y, X, weights):
        """Log likelihood for logistic regression"""
        logits = X @ weights
        probs = 1 / (1 + np.exp(-logits))
        # Avoid log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    # Gradient of log likelihood
    def logistic_gradient(y, X, weights):
        """Gradient of logistic log likelihood"""
        logits = X @ weights
        probs = 1 / (1 + np.exp(-logits))
        return X.T @ (y - probs)

    # Gradient descent for MLE
    weights_init = np.array([0.0, 0.0])
    learning_rate = 0.01
    max_iter = 1000

    weights = weights_init.copy()
    log_likelihood_history = []

    for i in range(max_iter):
        grad = logistic_gradient(y_class, X_class, weights)
        weights = weights + learning_rate * grad  # Maximizing, so we add

        if i % 10 == 0:
            ll = logistic_log_likelihood(y_class, X_class, weights)
            log_likelihood_history.append(ll)

    print(f"Logistic Regression MLE:")
    print(f"Weights: {weights} (true = {true_weights})")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Normal distribution MLE
    contour = axes[0, 0].contour(MU, SIGMA, log_likelihood_grid, levels=20)
    axes[0, 0].clabel(contour, inline=True, fontsize=8)
    axes[0, 0].plot(mu_mle, sigma_mle, 'r*', markersize=15, label='MLE')
    axes[0, 0].plot(true_mean, true_std, 'g*', markersize=15, label='True')
    axes[0, 0].set_xlabel('μ')
    axes[0, 0].set_ylabel('σ')
    axes[0, 0].set_title('Log Likelihood Contours (Normal Distribution)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Linear regression MLE
    axes[0, 1].scatter(X_reg, y_reg, alpha=0.7, label='Data')
    x_plot = np.linspace(0, 10, 100)
    y_true = true_intercept + true_slope * x_plot
    y_mle = intercept_mle + slope_mle * x_plot
    axes[0, 1].plot(x_plot, y_true, 'g-', linewidth=2, label='True')
    axes[0, 1].plot(x_plot, y_mle, 'r--', linewidth=2, label='MLE')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('Linear Regression MLE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Logistic regression decision boundary
    x1_range = np.linspace(-3, 3, 100)
    x2_range = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Decision boundary: weights[0]*x1 + weights[1]*x2 = 0
    X2_boundary = -(weights[0] * X1) / weights[1]
    X2_true = -(true_weights[0] * X1) / true_weights[1]

    axes[1, 0].scatter(X_class[y_class == 0, 0], X_class[y_class == 0, 1],
                       c='blue', alpha=0.7, label='Class 0')
    axes[1, 0].scatter(X_class[y_class == 1, 0], X_class[y_class == 1, 1],
                       c='red', alpha=0.7, label='Class 1')
    axes[1, 0].plot(X1, X2_boundary, 'r-', linewidth=2, label='MLE boundary')
    axes[1, 0].plot(X1, X2_true, 'g--', linewidth=2, label='True boundary')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    axes[1, 0].set_title('Logistic Regression Decision Boundary')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # Logistic regression convergence
    axes[1, 1].plot(range(0, len(log_likelihood_history)*10, 10), log_likelihood_history, 'b-')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Log Likelihood')
    axes[1, 1].set_title('Logistic Regression: MLE Convergence')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'normal': (mu_mle, sigma_mle),
        'linear_regression': (intercept_mle, slope_mle, sigma_mle_reg),
        'logistic_regression': weights
    }

mle_results = maximum_likelihood_estimation()
```

### 6.2 Bayesian Inference

```python
def bayesian_inference():
    """Bayesian inference in machine learning"""

    print("Bayesian Inference")
    print("-" * 30)

    # Coin flip example
    print("\n1. Bayesian Coin Flip Example")
    print("Estimating bias of a coin from observed flips")

    # Prior distribution (Beta)
    alpha_prior = 2
    beta_prior = 2
    print(f"Prior: Beta(α={alpha_prior}, β={beta_prior})")
    print(f"Prior mean: {alpha_prior/(alpha_prior + beta_prior):.3f}")

    # Observed data
    n_flips = 20
    n_heads = 15
    print(f"Observed data: {n_heads} heads out of {n_flips} flips")

    # Posterior distribution (Beta)
    alpha_posterior = alpha_prior + n_heads
    beta_posterior = beta_prior + (n_flips - n_heads)
    print(f"Posterior: Beta(α={alpha_posterior}, β={beta_posterior})")
    print(f"Posterior mean: {alpha_posterior/(alpha_posterior + beta_posterior):.3f}")

    # Credible intervals
    from scipy.stats import beta as beta_dist

    ci_95 = beta_dist.interval(0.95, alpha_posterior, beta_posterior)
    print(f"95% credible interval: ({ci_95[0]:.3f}, {ci_95[1]:.3f})")

    # Bayesian linear regression
    print("\n2. Bayesian Linear Regression")

    # Generate data
    np.random.seed(42)
    n_bayes = 50
    X_bayes = np.linspace(0, 10, n_bayes)
    true_slope = 2.0
    true_intercept = 1.0
    true_noise_std = 1.0

    y_bayes = true_intercept + true_slope * X_bayes + np.random.normal(0, true_noise_std, n_bayes)

    # Prior parameters
    # Assume weights ~ N(0, σ²_prior)
    sigma_prior = 10.0  # Large uncertainty
    sigma_likelihood = true_noise_std

    # Design matrix
    X_design = np.column_stack([np.ones(n_bayes), X_bayes])

    # Posterior covariance and mean
    sigma_prior_matrix = sigma_prior**2 * np.eye(2)
    sigma_posterior = np.linalg.inv(np.linalg.inv(sigma_prior_matrix) +
                                  (1/sigma_likelihood**2) * X_design.T @ X_design)
    mu_posterior = sigma_posterior @ ((1/sigma_likelihood**2) * X_design.T @ y_bayes)

    print(f"Posterior mean: {mu_posterior}")
    print(f"True parameters: [{true_intercept:.1f}, {true_slope:.1f}]")

    # Posterior predictive distribution
    X_test = np.array([[1, 12]])  # Predict at x = 12

    # Predictive distribution parameters
    mu_predictive = X_test @ mu_posterior
    sigma_predictive = np.sqrt(sigma_likelihood**2 + X_test @ sigma_posterior @ X_test.T)

    print(f"\nPredictive distribution at x=12:")
    print(f"Mean: {mu_predictive[0]:.3f}")
    print(f"Std: {sigma_predictive[0]:.3f}")

    # Bayesian model comparison
    print("\n3. Bayesian Model Comparison")

    # Two models: linear vs quadratic
    # Model 1: y = β₀ + β₁x + ε
    # Model 2: y = β₀ + β₁x + β₂x² + ε

    def calculate_log_evidence(y, X, sigma_noise, sigma_prior):
        """Calculate log model evidence (marginal likelihood)"""
        n, p = X.shape

        # Prior precision
        Lambda_prior = (1/sigma_prior**2) * np.eye(p)

        # Posterior precision
        Lambda_posterior = Lambda_prior + (1/sigma_noise**2) * X.T @ X

        # Log evidence
        log_evidence = -0.5 * n * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(Lambda_posterior)) + \
                      0.5 * np.log(np.linalg.det(Lambda_prior)) - \
                      0.5 * (1/sigma_noise**2) * (y.T @ y - y.T @ X @ np.linalg.inv(Lambda_posterior) @ X.T @ y)

        return log_evidence

    # Design matrices
    X_linear = np.column_stack([np.ones(n_bayes), X_bayes])
    X_quadratic = np.column_stack([np.ones(n_bayes), X_bayes, X_bayes**2])

    # Calculate evidence for both models
    log_evidence_linear = calculate_log_evidence(y_bayes, X_linear, sigma_likelihood, sigma_prior)
    log_evidence_quadratic = calculate_log_evidence(y_bayes, X_quadratic, sigma_likelihood, sigma_prior)

    print(f"Log evidence (linear): {log_evidence_linear:.3f}")
    print(f"Log evidence (quadratic): {log_evidence_quadratic:.3f}")

    # Bayes factor
    bayes_factor = np.exp(log_evidence_linear - log_evidence_quadratic)
    print(f"Bayes factor (linear/quadratic): {bayes_factor:.3f}")

    # Posterior model probabilities (assuming equal prior probabilities)
    total_evidence = np.exp(log_evidence_linear) + np.exp(log_evidence_quadratic)
    p_linear = np.exp(log_evidence_linear) / total_evidence
    p_quadratic = np.exp(log_evidence_quadratic) / total_evidence

    print(f"Posterior probability (linear): {p_linear:.3f}")
    print(f"Posterior probability (quadratic): {p_quadratic:.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Beta distributions
    x_beta = np.linspace(0, 1, 100)
    prior_pdf = beta_dist.pdf(x_beta, alpha_prior, beta_prior)
    posterior_pdf = beta_dist.pdf(x_beta, alpha_posterior, beta_posterior)

    axes[0, 0].plot(x_beta, prior_pdf, 'b-', linewidth=2, label=f'Prior Beta({alpha_prior},{beta_prior})')
    axes[0, 0].plot(x_beta, posterior_pdf, 'r-', linewidth=2, label=f'Posterior Beta({alpha_posterior},{beta_posterior})')
    axes[0, 0].axvline(x=n_heads/n_flips, color='green', linestyle='--', label=f'MLE = {n_heads/n_flips:.3f}')
    axes[0, 0].set_xlabel('Coin bias (p)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Bayesian Coin Flip Inference')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Bayesian linear regression
    axes[0, 1].scatter(X_bayes, y_bayes, alpha=0.7, label='Data')

    # Plot posterior predictive intervals
    x_plot = np.linspace(-2, 12, 100)
    X_plot_design = np.column_stack([np.ones(len(x_plot)), x_plot])

    # Mean prediction
    y_mean = X_plot_design @ mu_posterior

    # Uncertainty in prediction
    y_std = np.sqrt(sigma_likelihood**2 + np.diag(X_plot_design @ sigma_posterior @ X_plot_design.T))

    axes[0, 1].plot(x_plot, y_mean, 'r-', linewidth=2, label='Posterior mean')
    axes[0, 1].fill_between(x_plot, y_mean - 2*y_std, y_mean + 2*y_std,
                           alpha=0.3, color='red', label='95% predictive interval')

    # True relationship
    y_true = true_intercept + true_slope * x_plot
    axes[0, 1].plot(x_plot, y_true, 'g--', linewidth=2, label='True relationship')

    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('Bayesian Linear Regression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Parameter posterior distributions
    # Sample from posterior
    n_samples = 1000
    weight_samples = np.random.multivariate_normal(mu_posterior, sigma_posterior, n_samples)

    axes[1, 0].scatter(weight_samples[:, 0], weight_samples[:, 1], alpha=0.6, s=10)
    axes[1, 0].plot(true_intercept, true_slope, 'r*', markersize=15, label='True parameters')
    axes[1, 0].set_xlabel('Intercept')
    axes[1, 1].set_ylabel('Slope')
    axes[1, 0].set_title('Posterior Parameter Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Model comparison
    models = ['Linear', 'Quadratic']
    log_evidences = [log_evidence_linear, log_evidence_quadratic]
    posterior_probs = [p_linear, p_quadratic]

    x_pos = np.arange(len(models))
    axes[1, 1].bar(x_pos, posterior_probs, color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Posterior Probability')
    axes[1, 1].set_title('Bayesian Model Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].grid(True, alpha=0.3)

    # Add log evidence values as text
    for i, (log_ev, prob) in enumerate(zip(log_evidences, posterior_probs)):
        axes[1, 1].text(i, prob + 0.05, f'log ev: {log_ev:.1f}',
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return {
        'coin_flip': {'alpha_posterior': alpha_posterior, 'beta_posterior': beta_posterior},
        'linear_regression': {'mu_posterior': mu_posterior, 'sigma_posterior': sigma_posterior},
        'model_comparison': {'p_linear': p_linear, 'p_quadratic': p_quadratic}
    }

bayesian_results = bayesian_inference()
```

## 7. Key Concepts Summary

### 7.1 Essential Probability Theory for ML

1. **Basic Probability**: Sample spaces, events, probability axioms
2. **Random Variables**: Discrete and continuous distributions
3. **Joint and Conditional**: Independence, Bayes' theorem
4. **Expectation and Variance**: Moments, covariance, correlation
5. **Limit Theorems**: Law of large numbers, central limit theorem

### 7.2 Common Distributions in ML

- **Normal/Bernoulli**: Fundamental building blocks
- **Poisson/Exponential**: Count and waiting time data
- **Beta/Dirichlet**: Bayesian inference and priors
- **Multivariate Normal**: High-dimensional data modeling

### 7.3 Inference Methods

- **Maximum Likelihood**: Point estimation of parameters
- **Bayesian Inference**: Full posterior distributions
- **MAP Estimation**: Regularized MLE
- **Model Comparison**: Bayesian model evidence

### 7.4 Important Theorems

- **Bayes' Theorem**: Foundation of Bayesian inference
- **Central Limit Theorem**: Justifies normal approximations
- **Law of Large Numbers**: Convergence of sample statistics
- **Chernoff Bounds**: Concentration inequalities

## 8. Exercises

### 8.1 Theory Exercises

1. Derive the maximum likelihood estimator for the exponential distribution.
2. Show that the sample mean is an unbiased estimator of the population mean.
3. Prove the Cramér-Rao lower bound for unbiased estimators.
4. Derive the posterior distribution for Bayesian linear regression.
5. Show how the central limit theorem applies to the binomial distribution.

### 8.2 Programming Exercises

```python
def probability_exercises():
    """
    Complete these exercises to test your understanding:

    Exercise 1: Implement EM algorithm for Gaussian mixture model
    Exercise 2: Create a Bayesian classifier with different priors
    Exercise 3: Implement Markov Chain Monte Carlo sampling
    Exercise 4: Calculate confidence intervals using bootstrap
    Exercise 5: Implement Bayesian linear regression with different priors
    """

    # Exercise 1: Simple EM algorithm for Gaussian mixture
    def em_algorithm_1d(data, n_components=2, max_iter=100, tol=1e-6):
        """EM algorithm for 1D Gaussian mixture model"""
        n = len(data)

        # Initialize parameters
        np.random.seed(42)
        means = np.random.choice(data, n_components)
        stds = np.random.uniform(0.5, 2.0, n_components)
        weights = np.ones(n_components) / n_components

        log_likelihood_history = []

        for iteration in range(max_iter):
            # E-step: Calculate responsibilities
            responsibilities = np.zeros((n, n_components))
            for k in range(n_components):
                responsibilities[:, k] = weights[k] * stats.norm.pdf(data, means[k], stds[k])

            # Normalize responsibilities
            responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

            # M-step: Update parameters
            for k in range(n_components):
                resp_sum = responsibilities[:, k].sum()
                weights[k] = resp_sum / n
                means[k] = np.sum(responsibilities[:, k] * data) / resp_sum
                stds[k] = np.sqrt(np.sum(responsibilities[:, k] * (data - means[k])**2) / resp_sum)

            # Calculate log likelihood
            log_likelihood = 0
            for i in range(n):
                likelihood_i = 0
                for k in range(n_components):
                    likelihood_i += weights[k] * stats.norm.pdf(data[i], means[k], stds[k])
                log_likelihood += np.log(likelihood_i)

            log_likelihood_history.append(log_likelihood)

            # Check convergence
            if iteration > 0 and abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < tol:
                break

        return means, stds, weights, log_likelihood_history

    # Test EM algorithm
    np.random.seed(42)
    data_component1 = np.random.normal(-2, 1, 300)
    data_component2 = np.random.normal(3, 1.5, 200)
    data = np.concatenate([data_component1, data_component2])

    means_est, stds_est, weights_est, ll_history = em_algorithm_1d(data)

    print("Exercise 1: EM Algorithm for Gaussian Mixture")
    print(f"True parameters: Component 1 (μ=-2, σ=1), Component 2 (μ=3, σ=1.5)")
    print(f"Estimated parameters:")
    for i in range(len(means_est)):
        print(f"Component {i+1}: μ={means_est[i]:.3f}, σ={stds_est[i]:.3f}, weight={weights_est[i]:.3f}")

    return means_est, stds_est, weights_est

means_est, stds_est, weights_est = probability_exercises()
```

This comprehensive guide covers the essential probability theory concepts needed for machine learning, from basic probability rules to advanced Bayesian inference methods. Each section includes mathematical explanations, Python implementations, and practical applications in machine learning.