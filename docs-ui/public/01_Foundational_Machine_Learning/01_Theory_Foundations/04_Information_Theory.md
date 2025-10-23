---
title: "Foundational Machine Learning - Information Theory for"
description: "## Overview. Comprehensive guide covering gradient descent, classification, algorithms, machine learning, model training. Part of AI documentation system wit..."
keywords: "machine learning, neural networks, classification, gradient descent, classification, algorithms, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Information Theory for Machine Learning

## Overview

Information theory provides the mathematical framework for quantifying, storing, and communicating information. It is fundamental to many machine learning concepts including feature selection, model evaluation, compression, and deep learning. This section covers the key information theory concepts with applications in ML.

## 1. Basic Information Measures

### 1.1 Shannon Entropy

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import xlogy
import pandas as pd

def shannon_entropy():
    """Shannon entropy and its properties"""

    print("Shannon Entropy")
    print("-" * 30)

    # Define entropy calculation
    def entropy(probabilities, base=2):
        """Calculate Shannon entropy given probability distribution"""
        # Remove zero probabilities to avoid log(0)
        probs = probabilities[probabilities > 0]
        return -np.sum(probs * np.log(probs) / np.log(base))

    # Example 1: Binary entropy function
    print("1. Binary Entropy Function")
    print("H(p) = -p*log2(p) - (1-p)*log2(1-p)")

    p_values = np.linspace(0.001, 0.999, 100)
    binary_entropy = []

    for p in p_values:
        if p == 0 or p == 1:
            h = 0
        else:
            h = -p * np.log2(p) - (1-p) * np.log2(1-p)
        binary_entropy.append(h)

    # Special values
    special_p = [0, 0.5, 1]
    special_h = [0, 1, 0]

    print(f"\nSpecial values:")
    for p, h in zip(special_p, special_h):
        print(f"H({p}) = {h}")

    # Example 2: Fair vs biased coin
    print(f"\n2. Fair vs Biased Coin")
    fair_coin = np.array([0.5, 0.5])
    biased_coin = np.array([0.9, 0.1])
    very_biased = np.array([0.99, 0.01])

    h_fair = entropy(fair_coin)
    h_biased = entropy(biased_coin)
    h_very_biased = entropy(very_biased)

    print(f"Fair coin P=[0.5, 0.5]: H = {h_fair:.3f} bits")
    print(f"Biased coin P=[0.9, 0.1]: H = {h_biased:.3f} bits")
    print(f"Very biased P=[0.99, 0.01]: H = {h_very_biased:.3f} bits")

    # Example 3: Multiple outcomes
    print(f"\n3. Multiple Outcomes")
    # 6-sided die
    fair_die = np.array([1/6] * 6)
    # Weather distribution
    weather = np.array([0.4, 0.3, 0.2, 0.1])  # Sunny, Cloudy, Rainy, Snowy
    # Zipf's law (word frequencies)
    zipf_probs = np.array([1/np.log(100) * 1/i for i in range(1, 6)])
    zipf_probs = zipf_probs / np.sum(zipf_probs)  # Normalize

    h_die = entropy(fair_die)
    h_weather = entropy(weather)
    h_zipf = entropy(zipf_probs)

    print(f"Fair die (6 outcomes): H = {h_die:.3f} bits")
    print(f"Weather distribution: H = {h_weather:.3f} bits")
    print(f"Zipf distribution: H = {h_zipf:.3f} bits")

    # Properties of entropy
    print(f"\n4. Properties of Entropy")
    print("- Non-negative: H(X) ≥ 0")
    print("- Maximum for uniform distribution")
    print("- Concave function")
    print("- Additive for independent variables")

    # Maximum entropy
    n_outcomes = 5
    uniform_dist = np.ones(n_outcomes) / n_outcomes
    max_entropy = np.log2(n_outcomes)

    print(f"For {n_outcomes} outcomes, maximum entropy = log2({n_outcomes}) = {max_entropy:.3f} bits")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Binary entropy function
    axes[0, 0].plot(p_values, binary_entropy, 'b-', linewidth=2)
    axes[0, 0].scatter(special_p, special_h, color='red', s=100, zorder=5)
    axes[0, 0].set_xlabel('Probability p')
    axes[0, 0].set_ylabel('Entropy H(p) (bits)')
    axes[0, 0].set_title('Binary Entropy Function')
    axes[0, 0].grid(True, alpha=0.3)

    # Add annotations
    axes[0, 0].annotate('Maximum uncertainty\n(p = 0.5)', xy=(0.5, 1), xytext=(0.7, 0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
                        fontsize=10)
    axes[0, 0].annotate('Minimum uncertainty\n(p = 0 or 1)', xy=(0.1, 0.1), xytext=(0.3, 0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
                        fontsize=10)

    # Entropy of different distributions
    distributions = ['Fair Coin', 'Biased Coin', 'Fair Die', 'Weather', 'Zipf']
    entropy_values = [h_fair, h_biased, h_die, h_weather, h_zipf]

    axes[0, 1].bar(distributions, entropy_values, color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    axes[0, 1].set_ylabel('Entropy (bits)')
    axes[0, 1].set_title('Entropy of Different Distributions')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Entropy vs number of outcomes (uniform distribution)
    n_range = range(2, 21)
    uniform_entropies = [np.log2(n) for n in n_range]

    axes[1, 0].plot(n_range, uniform_entropies, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Outcomes')
    axes[1, 0].set_ylabel('Maximum Entropy (bits)')
    axes[1, 0].set_title('Maximum Entropy vs Number of Outcomes')
    axes[1, 0].grid(True, alpha=0.3)

    # Entropy and information content
    # Show how entropy relates to information content
    info_content = -np.log2(p_values)
    axes[1, 1].plot(p_values, info_content, 'r-', linewidth=2, label='Information content')
    axes[1, 1].plot(p_values, binary_entropy, 'b-', linewidth=2, label='Expected information (entropy)')
    axes[1, 1].set_xlabel('Probability p')
    axes[1, 1].set_ylabel('Bits')
    axes[1, 1].set_title('Information Content vs Expected Information')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'binary_entropy': (p_values, binary_entropy),
        'distributions': (distributions, entropy_values),
        'uniform_entropies': (n_range, uniform_entropies)
    }

entropy_results = shannon_entropy()
```

### 1.2 Joint and Conditional Entropy

```python
def joint_conditional_entropy():
    """Joint and conditional entropy"""

    print("Joint and Conditional Entropy")
    print("-" * 40)

    # Define joint entropy calculation
    def joint_entropy(joint_probs):
        """Calculate joint entropy H(X,Y)"""
        return -np.sum(joint_probs * np.log2(joint_probs[joint_probs > 0]))

    def marginal_entropy(joint_probs, axis):
        """Calculate marginal entropy along given axis"""
        marginal = np.sum(joint_probs, axis=axis)
        return -np.sum(marginal * np.log2(marginal[marginal > 0]))

    def conditional_entropy(joint_probs, axis):
        """Calculate conditional entropy H(Y|X) or H(X|Y)"""
        if axis == 0:  # H(Y|X)
            marginal_x = np.sum(joint_probs, axis=1)
            cond_probs = joint_probs / marginal_x[:, np.newaxis]
            entropies = -np.sum(joint_probs * np.log2(cond_probs[joint_probs > 0]), axis=1)
            return np.sum(entropies)
        else:  # H(X|Y)
            marginal_y = np.sum(joint_probs, axis=0)
            cond_probs = joint_probs / marginal_y[np.newaxis, :]
            entropies = -np.sum(joint_probs * np.log2(cond_probs[joint_probs > 0]), axis=0)
            return np.sum(entropies)

    # Example 1: Two binary random variables
    print("\n1. Two Binary Random Variables")
    print("Example: Weather and Umbrella usage")

    # Joint probability table: P(Weather, Umbrella)
    # Weather: Sunny=0, Rainy=1
    # Umbrella: No=0, Yes=1
    joint_table = np.array([
        [0.3, 0.1],  # Sunny: No umbrella, With umbrella
        [0.1, 0.5]   # Rainy: No umbrella, With umbrella
    ])

    print("Joint probability table P(Weather, Umbrella):")
    print("           No Umbrella  With Umbrella")
    print(f"Sunny:     {joint_table[0,0]:.2f}        {joint_table[0,1]:.2f}")
    print(f"Rainy:     {joint_table[1,0]:.2f}        {joint_table[1,1]:.2f}")

    # Calculate entropies
    H_joint = joint_entropy(joint_table)
    H_weather = marginal_entropy(joint_table, axis=1)  # Sum over umbrella
    H_umbrella = marginal_entropy(joint_table, axis=0)  # Sum over weather
    H_umbrella_given_weather = conditional_entropy(joint_table, axis=0)
    H_weather_given_umbrella = conditional_entropy(joint_table, axis=1)

    print(f"\nEntropy calculations:")
    print(f"H(Weather, Umbrella) = {H_joint:.3f} bits")
    print(f"H(Weather) = {H_weather:.3f} bits")
    print(f"H(Umbrella) = {H_umbrella:.3f} bits")
    print(f"H(Umbrella|Weather) = {H_umbrella_given_weather:.3f} bits")
    print(f"H(Weather|Umbrella) = {H_weather_given_umbrella:.3f} bits")

    # Verify chain rule: H(X,Y) = H(X) + H(Y|X)
    chain_rule_weather = H_weather + H_umbrella_given_weather
    chain_rule_umbrella = H_umbrella + H_weather_given_umbrella

    print(f"\nChain rule verification:")
    print(f"H(Weather) + H(Umbrella|Weather) = {chain_rule_weather:.3f} bits")
    print(f"H(Umbrella) + H(Weather|Umbrella) = {chain_rule_umbrella:.3f} bits")
    print(f"H(Weather, Umbrella) = {H_joint:.3f} bits")
    print(f"Chain rule holds: {abs(chain_rule_weather - H_joint) < 1e-10}")

    # Example 2: Independent vs dependent variables
    print("\n2. Independent vs Dependent Variables")

    # Independent variables
    joint_independent = np.array([
        [0.4 * 0.6, 0.4 * 0.4],  # P(Sunny) * P(No Umbrella), P(Sunny) * P(With Umbrella)
        [0.6 * 0.6, 0.6 * 0.4]   # P(Rainy) * P(No Umbrella), P(Rainy) * P(With Umbrella)
    ])

    H_independent = joint_entropy(joint_independent)
    H_weather_ind = marginal_entropy(joint_independent, axis=1)
    H_umbrella_ind = marginal_entropy(joint_independent, axis=0)

    print(f"Independent case:")
    print(f"H(Weather, Umbrella) = {H_independent:.3f} bits")
    print(f"H(Weather) + H(Umbrella) = {H_weather_ind + H_umbrella_ind:.3f} bits")
    print(f"Equal for independent variables: {abs(H_independent - (H_weather_ind + H_umbrella_ind)) < 1e-10}")

    # Example 3: Multiple variables
    print("\n3. Multiple Variables Example")
    print("Three variables: Age, Education, Income")

    # Simplified discrete version
    # Age: Young=0, Old=1
    # Education: Low=0, High=1
    # Income: Low=0, High=1

    # Joint distribution (8 possible combinations)
    # This is a simplified example
    joint_3d = np.array([
        [[0.05, 0.15], [0.10, 0.05]],  # Young: (Low,Low), (Low,High), (High,Low), (High,High)
        [[0.05, 0.05], [0.25, 0.30]]   # Old: (Low,Low), (Low,High), (High,Low), (High,High)
    ])

    # Calculate various entropies
    H_age_edu_income = joint_entropy(joint_3d.flatten())

    # Marginal entropies
    H_age = marginal_entropy(joint_3d.reshape(2, -1), axis=1)
    H_edu = marginal_entropy(joint_3d.reshape(2, -1).T, axis=1)
    H_income = marginal_entropy(joint_3d.reshape(-1, 2), axis=0)

    print(f"Three-variable entropies:")
    print(f"H(Age, Education, Income) = {H_age_edu_income:.3f} bits")
    print(f"H(Age) = {H_age:.3f} bits")
    print(f"H(Education) = {H_edu:.3f} bits")
    print(f"H(Income) = {H_income:.3f} bits")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Joint probability heatmap
    im1 = axes[0, 0].imshow(joint_table, cmap='Blues', aspect='auto')
    axes[0, 0].set_title('Joint Distribution P(Weather, Umbrella)')
    axes[0, 0].set_xlabel('Umbrella (0=No, 1=Yes)')
    axes[0, 0].set_ylabel('Weather (0=Sunny, 1=Rainy)')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(['No', 'Yes'])
    axes[0, 0].set_yticklabels(['Sunny', 'Rainy'])

    # Add probability values as text
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, f'{joint_table[i, j]:.2f}',
                           ha='center', va='center', fontsize=12, color='white' if joint_table[i, j] > 0.3 else 'black')

    plt.colorbar(im1, ax=axes[0, 0])

    # Entropy comparison
    entropy_types = ['H(Weather,Umbrella)', 'H(Weather)', 'H(Umbrella)',
                    'H(Umbrella|Weather)', 'H(Weather|Umbrella)']
    entropy_values = [H_joint, H_weather, H_umbrella, H_umbrella_given_weather, H_weather_given_umbrella]

    bars = axes[0, 1].bar(entropy_types, entropy_values, color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
    axes[0, 1].set_ylabel('Entropy (bits)')
    axes[0, 1].set_title('Entropy Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, entropy_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

    # Chain rule visualization
    chain_rules = ['H(X,Y) = H(X) + H(Y|X)', 'H(X,Y) = H(Y) + H(X|Y)']
    left_sides = [H_joint, H_joint]
    right_sides = [H_weather + H_umbrella_given_weather, H_umbrella + H_weather_given_umbrella]

    x_pos = np.arange(len(chain_rules))
    axes[1, 0].bar(x_pos - 0.2, left_sides, 0.4, label='Left side', color='blue', alpha=0.7)
    axes[1, 0].bar(x_pos + 0.2, right_sides, 0.4, label='Right side', color='red', alpha=0.7)
    axes[1, 0].set_ylabel('Entropy (bits)')
    axes[1, 0].set_title('Chain Rule Verification')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(chain_rules, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Information flow diagram
    axes[1, 1].remove()
    ax_new = fig.add_subplot(224)

    # Create a simple information flow diagram
    from matplotlib.patches import FancyBboxPatch, Circle
    from matplotlib.patches import ConnectionPatch

    # Nodes
    weather_node = Circle((0.2, 0.5), 0.1, color='lightblue', ec='black')
    umbrella_node = Circle((0.8, 0.5), 0.1, color='lightgreen', ec='black')
    joint_node = Circle((0.5, 0.2), 0.08, color='lightcoral', ec='black')

    ax_new.add_patch(weather_node)
    ax_new.add_patch(umbrella_node)
    ax_new.add_patch(joint_node)

    # Labels
    ax_new.text(0.2, 0.65, 'Weather', ha='center', fontsize=12, weight='bold')
    ax_new.text(0.8, 0.65, 'Umbrella', ha='center', fontsize=12, weight='bold')
    ax_new.text(0.5, 0.05, 'Joint', ha='center', fontsize=10, weight='bold')

    # Entropy values
    ax_new.text(0.2, 0.35, f'H={H_weather:.2f}', ha='center', fontsize=10)
    ax_new.text(0.8, 0.35, f'H={H_umbrella:.2f}', ha='center', fontsize=10)
    ax_new.text(0.5, 0.5, f'H={H_joint:.2f}', ha='center', fontsize=10)

    # Arrows
    arrow1 = ConnectionPatch((0.2, 0.4), (0.5, 0.28), "data", "data",
                           arrowstyle="-|>", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
    arrow2 = ConnectionPatch((0.8, 0.4), (0.5, 0.28), "data", "data",
                           arrowstyle="-|>", shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")

    ax_new.add_patch(arrow1)
    ax_new.add_patch(arrow2)

    ax_new.set_xlim(0, 1)
    ax_new.set_ylim(0, 0.8)
    ax_new.set_aspect('equal')
    ax_new.set_title('Information Flow and Entropy')
    ax_new.axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'weather_umbrella': {
            'joint': H_joint,
            'marginal_weather': H_weather,
            'marginal_umbrella': H_umbrella,
            'conditional_umbrella_weather': H_umbrella_given_weather,
            'conditional_weather_umbrella': H_weather_given_umbrella
        }
    }

joint_entropy_results = joint_conditional_entropy()
```

## 2. Mutual Information and Information Gain

### 2.1 Mutual Information

```python
def mutual_information():
    """Mutual information and information gain"""

    print("Mutual Information and Information Gain")
    print("-" * 50)

    def mutual_information_calc(joint_probs):
        """Calculate mutual information I(X;Y)"""
        H_X = marginal_entropy(joint_probs, axis=1)
        H_Y = marginal_entropy(joint_probs, axis=0)
        H_X_given_Y = conditional_entropy(joint_probs, axis=1)
        return H_X - H_X_given_Y

    def mutual_information_alternative(joint_probs):
        """Alternative calculation: I(X;Y) = H(X) + H(Y) - H(X,Y)"""
        H_X = marginal_entropy(joint_probs, axis=1)
        H_Y = marginal_entropy(joint_probs, axis=0)
        H_XY = joint_entropy(joint_probs)
        return H_X + H_Y - H_XY

    def kl_divergence(p, q):
        """Calculate Kullback-Leibler divergence DKL(p||q)"""
        # Ensure probabilities sum to 1 and avoid division by zero
        p_norm = p / np.sum(p)
        q_norm = q / np.sum(q)

        # Remove zero probabilities
        mask = (p_norm > 0) & (q_norm > 0)
        p_norm = p_norm[mask]
        q_norm = q_norm[mask]

        return np.sum(p_norm * np.log2(p_norm / q_norm))

    # Example 1: Weather and Umbrella (continued)
    print("\n1. Weather and Umbrella Example")

    # Use the same joint distribution as before
    joint_table = np.array([
        [0.3, 0.1],  # Sunny: No umbrella, With umbrella
        [0.1, 0.5]   # Rainy: No umbrella, With umbrella
    ])

    MI_weather_umbrella = mutual_information_calc(joint_table)
    MI_alt = mutual_information_alternative(joint_table)

    print(f"Mutual Information I(Weather; Umbrella):")
    print(f"Method 1 (H(X) - H(X|Y)): {MI_weather_umbrella:.3f} bits")
    print(f"Method 2 (H(X) + H(Y) - H(X,Y)): {MI_alt:.3f} bits")

    # Interpretation
    H_weather = marginal_entropy(joint_table, axis=1)
    H_umbrella = marginal_entropy(joint_table, axis=0)
    normalized_MI = MI_weather_umbrella / np.sqrt(H_weather * H_umbrella)

    print(f"\nInterpretation:")
    print(f"Uncertainty in Weather: {H_weather:.3f} bits")
    print(f"Uncertainty in Umbrella: {H_umbrella:.3f} bits")
    print(f"Shared information: {MI_weather_umbrella:.3f} bits")
    print(f"Normalized MI: {normalized_MI:.3f}")

    # Example 2: Feature selection
    print("\n2. Feature Selection Example")
    print("Which feature provides most information about the target?")

    # Dataset: Predict whether someone will purchase a product
    # Features: Age, Income, Education Level

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # True relationship: Purchase depends on income and education
    income = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2])
    education = np.random.choice(['Low', 'High'], n_samples, p=[0.4, 0.6])
    age = np.random.choice(['Young', 'Middle', 'Old'], n_samples, p=[0.3, 0.5, 0.2])

    # Purchase probability based on income and education
    purchase_prob = np.zeros(n_samples)
    for i in range(n_samples):
        if income[i] == 'High' and education[i] == 'High':
            purchase_prob[i] = 0.8
        elif income[i] == 'High' or education[i] == 'High':
            purchase_prob[i] = 0.4
        else:
            purchase_prob[i] = 0.1

    purchase = (np.random.random(n_samples) < purchase_prob).astype(int)

    # Calculate mutual information for each feature
    def calculate_mi_categorical(feature, target):
        """Calculate MI between categorical feature and target"""
        # Create contingency table
        unique_features = np.unique(feature)
        unique_targets = np.unique(target)

        joint_counts = np.zeros((len(unique_features), len(unique_targets)))
        for i, feat in enumerate(unique_features):
            for j, targ in enumerate(unique_targets):
                joint_counts[i, j] = np.sum((feature == feat) & (target == targ))

        # Convert to probabilities
        joint_probs = joint_counts / len(feature)

        return mutual_information_calc(joint_probs)

    MI_income = calculate_mi_categorical(income, purchase)
    MI_education = calculate_mi_categorical(education, purchase)
    MI_age = calculate_mi_categorical(age, purchase)

    print(f"Mutual Information with Purchase:")
    print(f"Income: {MI_income:.3f} bits")
    print(f"Education: {MI_education:.3f} bits")
    print(f"Age: {MI_age:.3f} bits")

    # Rank features by importance
    features = ['Income', 'Education', 'Age']
    MI_values = [MI_income, MI_education, MI_age]
    ranking = sorted(zip(features, MI_values), key=lambda x: x[1], reverse=True)

    print(f"\nFeature ranking:")
    for i, (feature, mi) in enumerate(ranking):
        print(f"{i+1}. {feature}: {mi:.3f} bits")

    # Example 3: KL Divergence
    print("\n3. Kullback-Leibler Divergence")

    # Compare two probability distributions
    P = np.array([0.7, 0.2, 0.1])  # True distribution
    Q = np.array([0.5, 0.3, 0.2])  # Approximate distribution
    R = np.array([0.1, 0.1, 0.8])  # Very different distribution

    DKL_PQ = kl_divergence(P, Q)
    DKL_PR = kl_divergence(P, R)

    print(f"Distributions:")
    print(f"P = {P}")
    print(f"Q = {Q}")
    print(f"R = {R}")

    print(f"\nKL Divergence:")
    print(f"DKL(P||Q) = {DKL_PQ:.3f} bits")
    print(f"DKL(P||R) = {DKL_PR:.3f} bits")

    # Properties of KL divergence
    print(f"\nProperties:")
    print(f"- DKL(P||Q) ≥ 0")
    print(f"- DKL(P||Q) = 0 iff P = Q")
    print(f"- Not symmetric: DKL(P||Q) ≠ DKL(Q||P)")

    # Calculate DKL(Q||P) to show asymmetry
    DKL_QP = kl_divergence(Q, P)
    print(f"DKL(Q||P) = {DKL_QP:.3f} bits")
    print(f"DKL(P||Q) = {DKL_PQ:.3f} bits")
    print(f"Symmetric? {abs(DKL_PQ - DKL_QP) < 1e-10}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Mutual information interpretation
    # Create Venn diagram-style visualization
    from matplotlib.patches import Circle
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    ax1 = axes[0, 0]

    # Draw circles for entropies
    circle1 = Circle((0.3, 0.5), 0.25, color='lightblue', alpha=0.7, label=f'Weather (H={H_weather:.2f})')
    circle2 = Circle((0.7, 0.5), 0.25, color='lightgreen', alpha=0.7, label=f'Umbrella (H={H_umbrella:.2f})')

    ax1.add_patch(circle1)
    ax1.add_patch(circle2)

    # Calculate overlap area (this is approximate)
    overlap_text = f'I(X;Y)\n={MI_weather_umbrella:.2f} bits'
    ax1.text(0.5, 0.5, overlap_text, ha='center', va='center', fontsize=10, weight='bold')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Mutual Information Visualization')
    ax1.axis('off')

    # Feature selection results
    axes[0, 1].bar(features, MI_values, color=['blue', 'green', 'orange'], alpha=0.7)
    axes[0, 1].set_ylabel('Mutual Information (bits)')
    axes[0, 1].set_title('Feature Selection: MI with Target')
    axes[0, 1].grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(MI_values):
        axes[0, 1].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')

    # KL divergence visualization
    x = np.arange(len(P))
    width = 0.25

    axes[1, 0].bar(x - width, P, width, label='P (True)', color='blue', alpha=0.7)
    axes[1, 0].bar(x, Q, width, label='Q (Approx)', color='red', alpha=0.7)
    axes[1, 0].bar(x + width, R, width, label='R (Different)', color='green', alpha=0.7)

    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title('Distribution Comparison for KL Divergence')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Cat 1', 'Cat 2', 'Cat 3'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # KL divergence matrix
    distributions = ['P', 'Q', 'R']
    kl_matrix = np.zeros((3, 3))

    kl_matrix[0, 1] = DKL_PQ
    kl_matrix[0, 2] = DKL_PR
    kl_matrix[1, 0] = DKL_QP
    kl_matrix[1, 2] = kl_divergence(Q, R)
    kl_matrix[2, 0] = kl_divergence(R, P)
    kl_matrix[2, 1] = kl_divergence(R, Q)

    im = axes[1, 1].imshow(kl_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('KL Divergence Matrix')
    axes[1, 1].set_xlabel('To Distribution')
    axes[1, 1].set_ylabel('From Distribution')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(distributions)
    axes[1, 1].set_yticklabels(distributions)

    # Add values as text
    for i in range(3):
        for j in range(3):
            if i != j:  # Diagonal is 0
                axes[1, 1].text(j, i, f'{kl_matrix[i, j]:.3f}',
                               ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    return {
        'weather_umbrella_MI': MI_weather_umbrella,
        'feature_selection': list(zip(features, MI_values)),
        'kl_divergence': {'P_Q': DKL_PQ, 'P_R': DKL_PR, 'Q_P': DKL_QP}
    }

mi_results = mutual_information()
```

### 2.2 Information Gain in Decision Trees

```python
def information_gain_decision_trees():
    """Information gain in decision trees"""

    print("Information Gain in Decision Trees")
    print("-" * 40)

    def information_gain(parent_probs, children_probs_list, children_weights=None):
        """Calculate information gain for a split"""
        if children_weights is None:
            # Equal weights if not specified
            children_weights = [1.0 / len(children_probs_list)] * len(children_probs_list)

        # Parent entropy
        H_parent = -np.sum(parent_probs * np.log2(parent_probs[parent_probs > 0]))

        # Weighted average of children entropies
        H_children = 0
        for child_probs, weight in zip(children_probs_list, children_weights):
            H_child = -np.sum(child_probs * np.log2(child_probs[child_probs > 0]))
            H_children += weight * H_child

        return H_parent - H_children

    def gini_index(probs):
        """Calculate Gini index"""
        return 1 - np.sum(probs**2)

    def gini_gain(parent_probs, children_probs_list, children_weights=None):
        """Calculate Gini gain"""
        if children_weights is None:
            children_weights = [1.0 / len(children_probs_list)] * len(children_probs_list)

        gini_parent = gini_index(parent_probs)

        gini_children = 0
        for child_probs, weight in zip(children_probs_list, children_weights):
            gini_children += weight * gini_index(child_probs)

        return gini_parent - gini_children

    # Example 1: Play tennis dataset
    print("\n1. Play Tennis Dataset Example")

    # Simplified dataset
    # Outlook: Sunny, Overcast, Rainy
    # Temperature: Hot, Mild, Cool
    # Humidity: High, Normal
    # Windy: True, False
    # Play: Yes, No

    # Current node (root) - should we play tennis?
    parent_play = np.array([9/14, 5/14])  # 9 Yes, 5 No out of 14

    print(f"Parent node - Play distribution: {parent_play}")
    print(f"Parent entropy: {-np.sum(parent_play * np.log2(parent_play)):.3f} bits")

    # Split by Outlook
    # Sunny: 2 Yes, 3 No -> [2/5, 3/5]
    # Overcast: 4 Yes, 0 No -> [4/4, 0/4]
    # Rainy: 3 Yes, 2 No -> [3/5, 2/5]

    outlook_splits = [
        np.array([2/5, 3/5]),  # Sunny
        np.array([4/4, 0/4]),  # Overcast
        np.array([3/5, 2/5])   # Rainy
    ]
    outlook_weights = [5/14, 4/14, 5/14]  # Number of samples in each split

    IG_outlook = information_gain(parent_play, outlook_splits, outlook_weights)

    print(f"\nSplit by Outlook:")
    print(f"  Sunny: {outlook_splits[0]} -> H = {-np.sum(outlook_splits[0] * np.log2(outlook_splits[0])):.3f}")
    print(f"  Overcast: {outlook_splits[1]} -> H = {-np.sum(outlook_splits[1] * np.log2(outlook_splits[1])):.3f}")
    print(f"  Rainy: {outlook_splits[2]} -> H = {-np.sum(outlook_splits[2] * np.log2(outlook_splits[2])):.3f}")
    print(f"Information Gain: {IG_outlook:.3f} bits")

    # Split by Humidity
    # High: 3 Yes, 4 No -> [3/7, 4/7]
    # Normal: 6 Yes, 1 No -> [6/7, 1/7]

    humidity_splits = [
        np.array([3/7, 4/7]),  # High
        np.array([6/7, 1/7])   # Normal
    ]
    humidity_weights = [7/14, 7/14]

    IG_humidity = information_gain(parent_play, humidity_splits, humidity_weights)

    print(f"\nSplit by Humidity:")
    print(f"  High: {humidity_splits[0]} -> H = {-np.sum(humidity_splits[0] * np.log2(humidity_splits[0])):.3f}")
    print(f"  Normal: {humidity_splits[1]} -> H = {-np.sum(humidity_splits[1] * np.log2(humidity_splits[1])):.3f}")
    print(f"Information Gain: {IG_humidity:.3f} bits")

    # Compare with Gini index
    print(f"\nComparison with Gini Index:")
    print(f"Parent Gini: {gini_index(parent_play):.3f}")

    GG_outlook = gini_gain(parent_play, outlook_splits, outlook_weights)
    GG_humidity = gini_gain(parent_play, humidity_splits, humidity_weights)

    print(f"Gini Gain - Outlook: {GG_outlook:.3f}")
    print(f"Gini Gain - Humidity: {GG_humidity:.3f}")

    # Example 2: Build small decision tree
    print(f"\n2. Building a Small Decision Tree")

    def find_best_split(parent_probs, splits_dict):
        """Find the split with maximum information gain"""
        best_feature = None
        best_ig = -1
        best_splits = None
        best_weights = None

        for feature, (splits, weights) in splits_dict.items():
            ig = information_gain(parent_probs, splits, weights)
            if ig > best_ig:
                best_ig = ig
                best_feature = feature
                best_splits = splits
                best_weights = weights

        return best_feature, best_ig, best_splits, best_weights

    # Available splits
    available_splits = {
        'Outlook': (outlook_splits, outlook_weights),
        'Humidity': (humidity_splits, humidity_weights)
    }

    # Find best split at root
    best_feature, best_ig, best_splits, best_weights = find_best_split(parent_play, available_splits)

    print(f"Best split at root: {best_feature}")
    print(f"Information Gain: {best_ig:.3f} bits")

    # Continue with one level (for Overcast branch - it's pure!)
    if best_feature == 'Outlook':
        print(f"\nOutlook branches:")
        print(f"  Sunny: Not pure (needs further splitting)")
        print(f"  Overcast: Pure! (all Yes) - leaf node")
        print(f"  Rainy: Not pure (needs further splitting)")

    # Example 3: Real dataset simulation
    print(f"\n3. Simulated Real Dataset")

    # Generate a larger dataset
    np.random.seed(42)
    n_samples = 1000

    # Features
    age = np.random.choice(['Young', 'Middle', 'Old'], n_samples)
    income = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master'], n_samples)

    # Target: Will they purchase premium subscription?
    # More likely if: older, higher income, higher education
    purchase_prob = (0.3 if age == 'Young' else 0.5 if age == 'Middle' else 0.7) * \
                   (0.4 if income == 'Low' else 0.6 if income == 'Medium' else 0.8) * \
                   (0.5 if education == 'High School' else 0.7 if education == 'Bachelor' else 0.9)
    purchase_prob = np.clip(purchase_prob, 0.1, 0.9)

    purchase = (np.random.random(n_samples) < purchase_prob).astype(int)

    # Calculate information gain for each feature
    def calculate_feature_ig(feature, target):
        """Calculate information gain for a categorical feature"""
        unique_values = np.unique(feature)
        parent_probs = np.array([np.mean(target), 1 - np.mean(target)])

        splits = []
        weights = []

        for value in unique_values:
            mask = (feature == value)
            if np.sum(mask) > 0:
                child_probs = np.array([np.mean(target[mask]), 1 - np.mean(target[mask])])
                splits.append(child_probs)
                weights.append(np.sum(mask) / len(target))

        return information_gain(parent_probs, splits, weights)

    IG_age = calculate_feature_ig(age, purchase)
    IG_income = calculate_feature_ig(income, purchase)
    IG_education = calculate_feature_ig(education, purchase)

    print(f"Information Gains:")
    print(f"Age: {IG_age:.3f} bits")
    print(f"Income: {IG_income:.3f} bits")
    print(f"Education: {IG_education:.3f} bits")

    # Feature ranking
    features = ['Age', 'Income', 'Education']
    IG_values = [IG_age, IG_income, IG_education]
    ranking = sorted(zip(features, IG_values), key=lambda x: x[1], reverse=True)

    print(f"\nFeature ranking for decision tree:")
    for i, (feature, ig) in enumerate(ranking):
        print(f"{i+1}. {feature}: {ig:.3f} bits")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Information gain comparison
    axes[0, 0].bar(['Outlook', 'Humidity'], [IG_outlook, IG_humidity],
                   color=['blue', 'green'], alpha=0.7)
    axes[0, 0].set_ylabel('Information Gain (bits)')
    axes[0, 0].set_title('Play Tennis: Information Gain by Feature')
    axes[0, 0].grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate([IG_outlook, IG_humidity]):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # Entropy reduction visualization
    parent_entropy = -np.sum(parent_play * np.log2(parent_play))
    children_entropies = []
    for splits, weights in [(outlook_splits, outlook_weights), (humidity_splits, humidity_weights)]:
        weighted_entropy = 0
        for split, weight in zip(splits, weights):
            if np.any(split > 0):
                weighted_entropy += weight * (-np.sum(split * np.log2(split)))
        children_entropies.append(weighted_entropy)

    x_pos = np.arange(2)
    width = 0.35

    axes[0, 1].bar(x_pos - width/2, [parent_entropy, parent_entropy], width,
                   label='Parent Entropy', color='red', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, children_entropies, width,
                   label='Weighted Child Entropy', color='blue', alpha=0.7)
    axes[0, 1].set_ylabel('Entropy (bits)')
    axes[0, 1].set_title('Entropy Reduction by Split')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(['Outlook', 'Humidity'])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Real dataset results
    axes[1, 0].bar(features, IG_values, color=['orange', 'purple', 'brown'], alpha=0.7)
    axes[1, 0].set_ylabel('Information Gain (bits)')
    axes[1, 0].set_title('Simulated Dataset: Feature Importance')
    axes[1, 0].grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(IG_values):
        axes[1, 0].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')

    # Decision tree structure visualization (simplified)
    axes[1, 1].remove()
    ax_tree = fig.add_subplot(224)

    # Simple tree structure
    from matplotlib.patches import Rectangle, FancyBboxPatch

    # Root node
    root = FancyBboxPatch((0.4, 0.8), 0.2, 0.15, boxstyle="round,pad=0.1",
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax_tree.add_patch(root)
    ax_tree.text(0.5, 0.875, 'Play\nTennis?', ha='center', va='center', fontsize=10, weight='bold')

    # Decision node
    decision = FancyBboxPatch((0.4, 0.5), 0.2, 0.15, boxstyle="round,pad=0.1",
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax_tree.add_patch(decision)
    ax_tree.text(0.5, 0.575, 'Outlook?', ha='center', va='center', fontsize=9, weight='bold')

    # Leaf nodes
    leaf1 = FancyBboxPatch((0.1, 0.2), 0.15, 0.1, boxstyle="round,pad=0.1",
                          facecolor='lightcoral', edgecolor='black', linewidth=1)
    leaf2 = FancyBboxPatch((0.425, 0.2), 0.15, 0.1, boxstyle="round,pad=0.1",
                          facecolor='lightgreen', edgecolor='black', linewidth=1)
    leaf3 = FancyBboxPatch((0.75, 0.2), 0.15, 0.1, boxstyle="round,pad=0.1",
                          facecolor='lightyellow', edgecolor='black', linewidth=1)

    ax_tree.add_patch(leaf1)
    ax_tree.add_patch(leaf2)
    ax_tree.add_patch(leaf3)

    ax_tree.text(0.175, 0.25, 'No\n(IG=0.248)', ha='center', va='center', fontsize=8)
    ax_tree.text(0.5, 0.25, 'Yes\n(Pure)', ha='center', va='center', fontsize=8)
    ax_tree.text(0.825, 0.25, 'Yes\n(IG=0.029)', ha='center', va='center', fontsize=8)

    # Labels
    ax_tree.text(0.175, 0.15, 'Sunny', ha='center', va='center', fontsize=8)
    ax_tree.text(0.5, 0.15, 'Overcast', ha='center', va='center', fontsize=8)
    ax_tree.text(0.825, 0.15, 'Rainy', ha='center', va='center', fontsize=8)

    ax_tree.set_xlim(0, 1)
    ax_tree.set_ylim(0, 1)
    ax_tree.set_aspect('equal')
    ax_tree.set_title('Simplified Decision Tree Structure')
    ax_tree.axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'tennis_dataset': {'outlook_ig': IG_outlook, 'humidity_ig': IG_humidity},
        'simulated_dataset': list(zip(features, IG_values))
    }

dt_results = information_gain_decision_trees()
```

## 3. Advanced Information Theory Concepts

### 3.1 Cross-Entropy and Loss Functions

```python
def cross_entropy_loss():
    """Cross-entropy and its role in machine learning"""

    print("Cross-Entropy and Loss Functions")
    print("-" * 40)

    def cross_entropy(p, q):
        """Calculate cross-entropy H(p,q)"""
        # Ensure probabilities are valid
        p_norm = p / np.sum(p)
        q_norm = q / np.sum(q)

        # Remove zero probabilities
        mask = (p_norm > 0) & (q_norm > 0)
        p_norm = p_norm[mask]
        q_norm = q_norm[mask]

        return -np.sum(p_norm * np.log2(q_norm))

    def binary_cross_entropy(y_true, y_pred):
        """Binary cross-entropy loss"""
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def categorical_cross_entropy(y_true, y_pred):
        """Categorical cross-entropy loss"""
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    # Example 1: Cross-entropy vs KL divergence
    print("\n1. Cross-Entropy vs KL Divergence")

    # True distribution and predicted distribution
    p_true = np.array([0.7, 0.2, 0.1])  # True probabilities
    q_good = np.array([0.65, 0.25, 0.1])  # Good approximation
    q_bad = np.array([0.1, 0.1, 0.8])     # Bad approximation

    # Calculate entropies
    H_p = -np.sum(p_true * np.log2(p_true))
    H_pq_good = cross_entropy(p_true, q_good)
    H_pq_bad = cross_entropy(p_true, q_bad)

    # KL divergences
    KL_pq_good = H_pq_good - H_p
    KL_pq_bad = H_pq_bad - H_p

    print(f"True distribution p: {p_true}")
    print(f"Good approximation q: {q_good}")
    print(f"Bad approximation q: {q_bad}")
    print(f"\nEntropy H(p): {H_p:.3f} bits")
    print(f"Cross-entropy H(p,q_good): {H_pq_good:.3f} bits")
    print(f"Cross-entropy H(p,q_bad): {H_pq_bad:.3f} bits")
    print(f"\nKL divergence DKL(p||q_good): {KL_pq_good:.3f} bits")
    print(f"KL divergence DKL(p||q_bad): {KL_pq_bad:.3f} bits")
    print(f"\nRelationship: H(p,q) = H(p) + DKL(p||q)")
    print(f"Good: {H_pq_good:.3f} = {H_p:.3f} + {KL_pq_good:.3f}")
    print(f"Bad: {H_pq_bad:.3f} = {H_p:.3f} + {KL_pq_bad:.3f}")

    # Example 2: Binary classification
    print("\n2. Binary Classification Loss")

    # Generate binary classification data
    np.random.seed(42)
    n_samples = 100

    # True labels
    y_true = np.random.randint(0, 2, n_samples)

    # Model predictions (probabilities)
    # Good model predictions
    y_pred_good = np.where(y_true == 1,
                           np.random.normal(0.8, 0.1, n_samples),
                           np.random.normal(0.2, 0.1, n_samples))
    y_pred_good = np.clip(y_pred_good, 0, 1)

    # Bad model predictions
    y_pred_bad = np.random.uniform(0.4, 0.6, n_samples)

    # Calculate losses
    bce_good = binary_cross_entropy(y_true, y_pred_good)
    bce_bad = binary_cross_entropy(y_true, y_pred_bad)

    print(f"Binary Cross-Entropy Loss:")
    print(f"Good model: {bce_good:.3f}")
    print(f"Bad model: {bce_bad:.3f}")

    # Calculate accuracy
    acc_good = np.mean((y_pred_good > 0.5) == y_true)
    acc_bad = np.mean((y_pred_bad > 0.5) == y_true)

    print(f"Accuracy:")
    print(f"Good model: {acc_good:.3f}")
    print(f"Bad model: {acc_bad:.3f}")

    # Example 3: Multi-class classification
    print("\n3. Multi-class Classification")

    # 3-class classification
    n_classes = 3
    n_samples_multi = 50

    # True labels (one-hot encoded)
    y_true_multi = np.zeros((n_samples_multi, n_classes))
    y_indices = np.random.randint(0, n_classes, n_samples_multi)
    y_true_multi[np.arange(n_samples_multi), y_indices] = 1

    # Model predictions (softmax outputs)
    # Good model
    y_pred_good_multi = np.random.dirichlet([2, 2, 2], n_samples_multi) * 0.3 + y_true_multi * 0.7
    y_pred_good_multi = y_pred_good_multi / y_pred_good_multi.sum(axis=1, keepdims=True)

    # Bad model (random predictions)
    y_pred_bad_multi = np.random.dirichlet([1, 1, 1], n_samples_multi)

    # Calculate losses
    cce_good = categorical_cross_entropy(y_true_multi, y_pred_good_multi)
    cce_bad = categorical_cross_entropy(y_true_multi, y_pred_bad_multi)

    print(f"Categorical Cross-Entropy Loss:")
    print(f"Good model: {cce_good:.3f}")
    print(f"Bad model: {cce_bad:.3f}")

    # Example 4: Information bottleneck principle
    print("\n4. Information Bottleneck Principle")

    def information_bottleneck_demo():
        """Demonstrate information bottleneck principle"""
        # Simple example: compressing a signal while preserving relevant information

        # Generate correlated data
        np.random.seed(42)
        n_samples = 1000

        # Input X
        X = np.random.randn(n_samples)

        # Relevant information Y
        Y = X + 0.5 * np.random.randn(n_samples)

        # Create compressed representations Z with different compression levels
        compression_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        I_XZ = []  # Mutual information between X and Z
        I_ZY = []  # Mutual information between Z and Y

        for level in compression_levels:
            # Compression: quantize X
            n_bins = int(10 * level)
            if n_bins < 2:
                n_bins = 2

            # Create compressed representation
            bins = np.linspace(-3, 3, n_bins)
            Z = np.digitize(X, bins)

            # Calculate mutual information (simplified)
            # This is an approximation
            I_XZ_approx = level * 2  # More compression → less mutual information
            I_ZY_approx = level * 1.5  # Less compression → more preserved information

            I_XZ.append(I_XZ_approx)
            I_ZY.append(I_ZY_approx)

        return compression_levels, I_XZ, I_ZY

    comp_levels, I_XZ, I_ZY = information_bottleneck_demo()

    print(f"Information Bottleneck Results:")
    print(f"Compression Level: I(X,Z), I(Z,Y)")
    for level, ixz, izy in zip(comp_levels, I_XZ, I_ZY):
        print(f"{level:.1f}: {ixz:.2f}, {izy:.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Cross-entropy vs KL divergence
    dist_names = ['Good Approx.', 'Bad Approx.']
    cross_entropies = [H_pq_good, H_pq_bad]
    kl_divergences = [KL_pq_good, KL_pq_bad]

    x_pos = np.arange(len(dist_names))
    width = 0.35

    axes[0, 0].bar(x_pos - width/2, cross_entropies, width, label='Cross-Entropy', color='blue', alpha=0.7)
    axes[0, 0].bar(x_pos + width/2, kl_divergences, width, label='KL Divergence', color='red', alpha=0.7)
    axes[0, 0].set_ylabel('Bits')
    axes[0, 0].set_title('Cross-Entropy vs KL Divergence')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(dist_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Binary classification loss distribution
    axes[0, 1].hist([bce_good, bce_bad], bins=10, alpha=0.7, label=['Good Model', 'Bad Model'], color=['green', 'red'])
    axes[0, 1].set_xlabel('Binary Cross-Entropy Loss')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Loss Distribution (Single Values)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Information bottleneck curve
    axes[1, 0].plot(I_XZ, I_ZY, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('I(X,Z) - Compression Level')
    axes[1, 0].set_ylabel('I(Z,Y) - Preserved Information')
    axes[1, 0].set_title('Information Bottleneck Curve')
    axes[1, 0].grid(True, alpha=0.3)

    # Add annotations
    for i, (level, ixz, izy) in enumerate(zip(comp_levels, I_XZ, I_ZY)):
        axes[1, 0].annotate(f'Level {level:.1f}', (ixz, izy), xytext=(10, 10),
                           textcoords='offset points', fontsize=8)

    # Loss vs accuracy trade-off
    models = ['Good Model', 'Bad Model']
    losses = [bce_good, cce_good]  # Use categorical for multi-class comparison
    accuracies = [acc_good, acc_bad]

    axes[1, 1].scatter(losses, accuracies, s=100, c=['green', 'red'], alpha=0.7)
    for i, (model, loss, acc) in enumerate(zip(models, losses, accuracies)):
        axes[1, 1].annotate(model, (loss, acc), xytext=(10, 10), textcoords='offset points', fontsize=10)

    axes[1, 1].set_xlabel('Loss (Cross-Entropy)')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Loss vs Accuracy Trade-off')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'cross_entropy': {'good': H_pq_good, 'bad': H_pq_bad},
        'kl_divergence': {'good': KL_pq_good, 'bad': KL_pq_bad},
        'classification_loss': {'binary_good': bce_good, 'binary_bad': bce_bad,
                             'categorical_good': cce_good, 'categorical_bad': cce_bad}
    }

cross_entropy_results = cross_entropy_loss()
```

### 3.2 Rate-Distortion Theory

```python
def rate_distortion_theory():
    """Rate-distortion theory and data compression"""

    print("Rate-Distortion Theory")
    print("-" * 30)

    def mean_squared_error(original, reconstructed):
        """Calculate mean squared error (distortion)"""
        return np.mean((original - reconstructed)**2)

    def calculate_rate(bits_per_sample):
        """Calculate rate in bits per sample"""
        return bits_per_sample

    # Example 1: Quantization of continuous signal
    print("\n1. Signal Quantization Example")

    # Generate a continuous signal
    np.random.seed(42)
    n_samples = 1000
    original_signal = np.random.randn(n_samples)

    # Quantize with different bit depths
    bit_depths = [1, 2, 3, 4, 5, 6, 7, 8]
    rates = []
    distortions = []

    for bits in bit_depths:
        # Quantize signal
        levels = 2**bits
        min_val, max_val = np.min(original_signal), np.max(original_signal)
        quantized = np.digitize(original_signal, np.linspace(min_val, max_val, levels + 1))
        quantized = quantized - 1  # Convert to 0-based indexing
        reconstructed = min_val + (quantized + 0.5) * (max_val - min_val) / levels

        # Calculate rate and distortion
        rate = calculate_rate(bits)
        distortion = mean_squared_error(original_signal, reconstructed)

        rates.append(rate)
        distortions.append(distortion)

        print(f"Bits: {bits}, Rate: {rate:.1f} bits/sample, MSE: {distortion:.4f}")

    # Example 2: Image compression simulation
    print("\n2. Image Compression Simulation")

    def simulate_image_compression(quality_levels):
        """Simulate image compression at different quality levels"""
        # Generate a simple "image" (2D signal)
        np.random.seed(42)
        image_size = (64, 64)

        # Create a simple pattern
        x = np.linspace(0, 4*np.pi, image_size[0])
        y = np.linspace(0, 4*np.pi, image_size[1])
        X, Y = np.meshgrid(x, y)
        original_image = np.sin(X) * np.cos(Y) + 0.2 * np.random.randn(*image_size)

        rates_img = []
        distortions_img = []

        for quality in quality_levels:
            # Simulate compression by adding quantization noise
            # Higher quality = less noise
            noise_level = 0.1 / quality
            compressed_image = original_image + noise_level * np.random.randn(*image_size)

            # Calculate compression rate (simplified)
            # In practice, this would depend on the compression algorithm
            rate = 8 / quality  # Simplified rate calculation

            # Calculate distortion
            distortion = mean_squared_error(original_image, compressed_image)

            rates_img.append(rate)
            distortions_img.append(distortion)

        return rates_img, distortions_img, original_image

    quality_levels = [1, 2, 4, 8, 16]
    rates_img, distortions_img, original_image = simulate_image_compression(quality_levels)

    print(f"Image Compression Results:")
    print(f"Quality -> Rate (bits/pixel), MSE")
    for quality, rate, dist in zip(quality_levels, rates_img, distortions_img):
        print(f"{quality:2d} -> {rate:.2f}, {dist:.4f}")

    # Example 3: Rate-distortion function calculation
    print("\n3. Theoretical Rate-Distortion Function")

    # For a Gaussian source, the rate-distortion function is known
    def gaussian_rate_distortion(sigma_squared, D):
        """Rate-distortion function for Gaussian source"""
        if D >= sigma_squared:
            return 0  # No coding needed if distortion >= variance
        else:
            return 0.5 * np.log2(sigma_squared / D)

    # Calculate for different distortion levels
    sigma_squared = 1.0  # Variance of source
    D_range = np.linspace(0.01, 2.0, 100)
    R_theoretical = [gaussian_rate_distortion(sigma_squared, D) for D in D_range]

    print(f"Gaussian Source (σ² = {sigma_squared}):")
    print(f"At D = 0.1: R = {gaussian_rate_distortion(sigma_squared, 0.1):.3f} bits")
    print(f"At D = 0.5: R = {gaussian_rate_distortion(sigma_squared, 0.5):.3f} bits")
    print(f"At D = 1.0: R = {gaussian_rate_distortion(sigma_squared, 1.0):.3f} bits")

    # Example 4: Practical compression trade-offs
    print("\n4. Practical Compression Trade-offs")

    def compression_efficiency(original_size, compressed_size, quality_metric):
        """Calculate compression efficiency metrics"""
        compression_ratio = original_size / compressed_size
        space_saving = 1 - (compressed_size / original_size)
        bits_per_pixel = np.log2(compressed_size) if compressed_size > 0 else 0

        return {
            'compression_ratio': compression_ratio,
            'space_saving_percent': space_saving * 100,
            'bits_per_pixel': bits_per_pixel,
            'quality': quality_metric
        }

    # Simulate different compression scenarios
    scenarios = [
        {'name': 'Lossless', 'size_ratio': 1.0, 'quality': 1.0},
        {'name': 'High Quality', 'size_ratio': 0.3, 'quality': 0.95},
        {'name': 'Medium Quality', 'size_ratio': 0.15, 'quality': 0.85},
        {'name': 'Low Quality', 'size_ratio': 0.05, 'quality': 0.6}
    ]

    print("Compression Scenarios:")
    for scenario in scenarios:
        metrics = compression_efficiency(100, 100 * scenario['size_ratio'], scenario['quality'])
        print(f"{scenario['name']}:")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  Space Saving: {metrics['space_saving_percent']:.1f}%")
        print(f"  Quality: {metrics['quality']:.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Rate-distortion curve for quantization
    axes[0, 0].plot(rates, distortions, 'bo-', linewidth=2, markersize=8, label='Empirical')
    axes[0, 0].set_xlabel('Rate (bits/sample)')
    axes[0, 0].set_ylabel('Distortion (MSE)')
    axes[0, 0].set_title('Rate-Distortion Curve: Signal Quantization')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Add annotations
    for i, (rate, dist) in enumerate(zip(rates, distortions)):
        if i % 2 == 0:  # Annotate every other point
            axes[0, 0].annotate(f'{bit_depths[i]} bits', (rate, dist),
                               xytext=(10, 10), textcoords='offset points', fontsize=8)

    # Rate-distortion curve for image compression
    axes[0, 1].plot(rates_img, distortions_img, 'ro-', linewidth=2, markersize=8, label='Empirical')
    axes[0, 1].set_xlabel('Rate (bits/pixel)')
    axes[0, 1].set_ylabel('Distortion (MSE)')
    axes[0, 1].set_title('Rate-Distortion Curve: Image Compression')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Theoretical vs empirical
    axes[1, 0].plot(D_range, R_theoretical, 'g-', linewidth=2, label='Theoretical (Gaussian)')
    axes[1, 0].plot(distortions, rates, 'bo-', linewidth=2, markersize=6, label='Empirical (Quantization)')
    axes[1, 0].set_xlabel('Distortion D')
    axes[1, 0].set_ylabel('Rate R (bits)')
    axes[1, 0].set_title('Theoretical vs Empirical Rate-Distortion')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Compression efficiency scatter
    compression_ratios = [s['size_ratio'] for s in scenarios]
    qualities = [s['quality'] for s in scenarios]
    colors = ['green', 'blue', 'orange', 'red']

    axes[1, 1].scatter(compression_ratios, qualities, s=100, c=colors, alpha=0.7)
    for i, scenario in enumerate(scenarios):
        axes[1, 1].annotate(scenario['name'], (compression_ratios[i], qualities[i]),
                           xytext=(10, 10), textcoords='offset points', fontsize=10)

    axes[1, 1].set_xlabel('Size Ratio (Compressed/Original)')
    axes[1, 1].set_ylabel('Quality')
    axes[1, 1].set_title('Compression Efficiency: Quality vs Size')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1.1)
    axes[1, 1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()

    return {
        'quantization': {'rates': rates, 'distortions': distortions, 'bit_depths': bit_depths},
        'image_compression': {'rates': rates_img, 'distortions': distortions_img},
        'theoretical': {'D_range': D_range, 'R_theoretical': R_theoretical}
    }

rate_distortion_results = rate_distortion_theory()
```

## 4. Applications in Machine Learning

### 4.1 Information Theory in Deep Learning

```python
def information_theory_deep_learning():
    """Information theory applications in deep learning"""

    print("Information Theory in Deep Learning")
    print("-" * 40)

    # Example 1: Information Bottleneck in Neural Networks
    print("\n1. Information Bottleneck in Neural Networks")

    def simulate_layer_information_flow(input_dim, hidden_dims, output_dim, n_samples=1000):
        """Simulate information flow through neural network layers"""

        np.random.seed(42)

        # Generate input data
        X = np.random.randn(n_samples, input_dim)

        # Simulate neural network layers
        layers_info = []
        current_input = X

        for i, dim in enumerate(hidden_dims + [output_dim]):
            # Random weight matrix
            W = np.random.randn(current_input.shape[1], dim) * 0.1
            b = np.random.randn(dim) * 0.1

            # Forward pass with ReLU activation (except last layer)
            if i < len(hidden_dims):
                output = np.maximum(0, current_input @ W + b)  # ReLU
            else:
                output = current_input @ W + b  # Linear

            # Calculate mutual information (simplified)
            # I(X;T) where T is layer output
            if i == 0:
                # First layer: I(X;T1)
                I_XT = min(dim, input_dim) * 0.8  # Simplified calculation
            else:
                # Subsequent layers: information may decrease
                I_XT = layers_info[-1]['I_XT'] * 0.9

            # I(T;Y) - mutual information with output
            if i == len(hidden_dims):
                # Output layer: maximum information
                I_TY = min(dim, output_dim) * 0.9
            else:
                # Hidden layers: increasing then decreasing
                I_TY = min(dim, output_dim) * (0.3 + 0.6 * (i / len(hidden_dims)))

            layers_info.append({
                'layer': i + 1,
                'dimension': dim,
                'I_XT': I_XT,
                'I_TY': I_TY,
                'output': output
            })

            current_input = output

        return layers_info

    # Simulate information flow
    input_dim = 10
    hidden_dims = [20, 15, 10]
    output_dim = 2

    layers_info = simulate_layer_information_flow(input_dim, hidden_dims, output_dim)

    print("Information Flow Through Network Layers:")
    print("Layer | Dim | I(X;T) | I(T;Y)")
    print("-" * 35)
    for info in layers_info:
        print(f"{info['layer']:5d} | {info['dimension']:3d} | {info['I_XT']:6.2f} | {info['I_TY']:6.2f}")

    # Example 2: Variational Information Maximization
    print("\n2. Variational Information Maximization")

    def variational_information_maximization(X, y, latent_dim=2, n_epochs=100):
        """Simplified variational information maximization"""

        np.random.seed(42)

        # Encoder parameters (simplified)
        n_samples, input_dim = X.shape
        W_encoder = np.random.randn(input_dim, latent_dim) * 0.1
        b_encoder = np.random.randn(latent_dim) * 0.1

        # Decoder parameters
        W_decoder = np.random.randn(latent_dim, input_dim) * 0.1
        b_decoder = np.random.randn(input_dim) * 0.1

        # Training loop (simplified)
        info_history = []

        for epoch in range(n_epochs):
            # Encode
            z = X @ W_encoder + b_encoder

            # Add noise for variational inference
            z_noisy = z + 0.1 * np.random.randn(*z.shape)

            # Decode
            X_reconstructed = z_noisy @ W_decoder + b_decoder

            # Calculate reconstruction error
            reconstruction_error = np.mean((X - X_reconstructed)**2)

            # Calculate mutual information (simplified)
            # I(X;Z) ≈ H(X) - H(X|Z)
            H_X = np.log2(input_dim)  # Simplified
            H_X_given_Z = reconstruction_error  # Simplified approximation
            I_XZ = max(0, H_X - H_X_given_Z)

            info_history.append(I_XZ)

            # Update parameters (simplified gradient descent)
            learning_rate = 0.01
            # This is a very simplified update
            W_encoder -= learning_rate * np.random.randn(*W_encoder.shape) * 0.01
            W_decoder -= learning_rate * np.random.randn(*W_decoder.shape) * 0.01

        return info_history

    # Generate sample data
    X_sample = np.random.randn(100, 5)
    y_sample = np.random.randint(0, 2, 100)

    info_history = variational_information_maximization(X_sample, y_sample)

    print(f"Variational Information Maximization:")
    print(f"Initial I(X;Z): {info_history[0]:.3f}")
    print(f"Final I(X;Z): {info_history[-1]:.3f}")
    print(f"Maximum achieved: {max(info_history):.3f}")

    # Example 3: Information Plane Analysis
    print("\n3. Information Plane Analysis")

    def information_plane_analysis(layers_info):
        """Analyze the information plane of a neural network"""

        I_XT_values = [info['I_XT'] for info in layers_info]
        I_TY_values = [info['I_TY'] for info in layers_info]

        # Calculate compression and prediction
        compression = [I_XT_values[0] - I_XT for I_XT in I_XT_values]
        prediction = I_TY_values

        return I_XT_values, I_TY_values, compression, prediction

    I_XT_vals, I_TY_vals, compression, prediction = information_plane_analysis(layers_info)

    print("Information Plane Analysis:")
    for i, (ixt, ity, comp, pred) in enumerate(zip(I_XT_vals, I_TY_vals, compression, prediction)):
        print(f"Layer {i+1}: I(X;T)={ixt:.2f}, I(T;Y)={ity:.2f}, Compression={comp:.2f}, Prediction={pred:.2f}")

    # Example 4: Mutual Information Neural Estimation (MINE)
    print("\n4. Mutual Information Neural Estimation (MINE) Concept")

    def simple_mine_estimation(X, Y, hidden_dim=10, n_samples=1000):
        """Simplified MINE estimation concept"""

        np.random.seed(42)

        # Generate correlated data
        X_data = np.random.randn(n_samples, 2)
        # Y is a function of X plus noise
        Y_data = X_data[:, 0] * X_data[:, 1] + 0.5 * np.random.randn(n_samples)
        Y_data = Y_data.reshape(-1, 1)

        # Neural network to estimate mutual information
        # This is a conceptual implementation
        def statistics_network(X, Y):
            """Neural network to compute T-statistic"""
            # Simple concatenation and linear transformation
            XY = np.concatenate([X, Y], axis=1)
            W = np.random.randn(X.shape[1] + Y.shape[1], hidden_dim)
            T = np.tanh(XY @ W)
            return T

        # Estimate mutual information using samples
        T_joint = statistics_network(X_data, Y_data)
        T_marginal = statistics_network(np.random.randn(*X_data.shape),
                                       np.random.randn(*Y_data.shape))

        # MINE estimation (simplified)
        mi_estimate = np.mean(T_joint) - np.log(np.mean(np.exp(T_marginal)))

        return mi_estimate

    # Estimate MI for different correlation levels
    correlation_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    mi_estimates = []

    for corr in correlation_levels:
        # Generate data with specified correlation
        X_corr = np.random.randn(100, 2)
        Y_corr = corr * X_corr[:, 0] + np.sqrt(1 - corr**2) * np.random.randn(100)
        mi_est = simple_mine_estimation(X_corr, Y_corr.reshape(-1, 1))
        mi_estimates.append(mi_est)

        print(f"Correlation {corr:.1f}: Estimated MI = {mi_est:.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Information flow through layers
    layer_numbers = [info['layer'] for info in layers_info]
    axes[0, 0].plot(layer_numbers, I_XT_vals, 'bo-', linewidth=2, markersize=8, label='I(X;T)')
    axes[0, 0].plot(layer_numbers, I_TY_vals, 'ro-', linewidth=2, markersize=8, label='I(T;Y)')
    axes[0, 0].set_xlabel('Layer Number')
    axes[0, 0].set_ylabel('Mutual Information (bits)')
    axes[0, 0].set_title('Information Flow Through Network Layers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Information plane
    axes[0, 1].scatter(I_XT_vals, I_TY_vals, s=100, c=layer_numbers, cmap='viridis', alpha=0.7)
    for i, (ixt, ity) in enumerate(zip(I_XT_vals, I_TY_vals)):
        axes[0, 1].annotate(f'L{i+1}', (ixt, ity), xytext=(5, 5), textcoords='offset points')
    axes[0, 1].set_xlabel('I(X;T) - Information about Input')
    axes[0, 1].set_ylabel('I(T;Y) - Information about Output')
    axes[0, 1].set_title('Information Plane')
    axes[0, 1].grid(True, alpha=0.3)

    # Variational information maximization
    axes[1, 0].plot(info_history, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Estimated I(X;Z) (bits)')
    axes[1, 0].set_title('Variational Information Maximization')
    axes[1, 0].grid(True, alpha=0.3)

    # MINE estimation
    axes[1, 1].plot(correlation_levels, mi_estimates, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('True Correlation')
    axes[1, 1].set_ylabel('Estimated Mutual Information')
    axes[1, 1].set_title('MINE: MI Estimation vs Correlation')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'information_flow': layers_info,
        'variational_maximization': info_history,
        'mine_estimation': list(zip(correlation_levels, mi_estimates))
    }

dl_info_results = information_theory_deep_learning()
```

## 5. Key Concepts Summary

### 5.1 Essential Information Theory for ML

1. **Entropy**: Measures uncertainty and information content
2. **Mutual Information**: Quantifies shared information between variables
3. **Cross-Entropy**: Used as loss function in classification
4. **KL Divergence**: Measures difference between probability distributions
5. **Information Bottleneck**: Principle for optimal representation learning

### 5.2 Applications in Machine Learning

- **Feature Selection**: Using mutual information to rank features
- **Decision Trees**: Information gain for splitting criteria
- **Loss Functions**: Cross-entropy for classification tasks
- **Representation Learning**: Information bottleneck principle
- **Compression**: Rate-distortion theory for efficient coding

### 5.3 Important Theorems

- **Shannon's Source Coding Theorem**: Fundamental limit of lossless compression
- **Noisy Channel Coding Theorem**: Reliable communication over noisy channels
- **Data Processing Inequality**: Information cannot increase through processing
- **Fano's Inequality**: Relationship between entropy and error probability

### 5.4 Practical Considerations

- **Estimation**: Mutual information estimation from finite samples
- **Computation**: Efficient calculation for high-dimensional data
- **Interpretation**: Understanding the meaning of information measures
- **Optimization**: Balancing information preservation and compression

## 6. Exercises

### 6.1 Theory Exercises

1. Derive the relationship between cross-entropy and KL divergence.
2. Prove that mutual information is always non-negative.
3. Show how information gain relates to variance reduction.
4. Derive the rate-distortion function for a uniform source.
5. Explain the connection between information bottleneck and autoencoders.

### 6.2 Programming Exercises

```python
def information_theory_exercises():
    """
    Complete these exercises to test your understanding:

    Exercise 1: Implement mutual information estimation from data
    Exercise 2: Create a decision tree using information gain
    Exercise 3: Implement a simple autoencoder with information bottleneck
    Exercise 4: Calculate KL divergence between two neural network distributions
    Exercise 5: Apply information bottleneck to a real dataset
    """

    # Exercise 1: Mutual information estimation
    def estimate_mutual_information(X, Y, n_bins=10):
        """Estimate mutual information between continuous variables using histogram method"""
        # Create 2D histogram
        hist_xy, x_edges, y_edges = np.histogram2d(X, Y, bins=n_bins)

        # Convert to probabilities
        p_xy = hist_xy / np.sum(hist_xy)

        # Calculate marginal probabilities
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Calculate mutual information
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

        return mi

    # Test with correlated data
    np.random.seed(42)
    n_samples = 1000
    X_test = np.random.randn(n_samples)
    Y_test = 0.7 * X_test + 0.3 * np.random.randn(n_samples)

    mi_estimated = estimate_mutual_information(X_test, Y_test)
    print(f"Exercise 1: Mutual Information Estimation")
    print(f"Estimated MI: {mi_estimated:.3f} bits")

    # Exercise 2: Simple decision tree node
    def find_best_split_threshold(X, y, threshold_candidates=None):
        """Find best split threshold using information gain"""
        if threshold_candidates is None:
            threshold_candidates = np.percentile(X, [25, 50, 75])

        best_threshold = None
        best_ig = -np.inf

        parent_entropy = entropy_calculator(y)

        for threshold in threshold_candidates:
            # Split data
            left_mask = X <= threshold
            right_mask = X > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            # Calculate information gain
            left_entropy = entropy_calculator(y[left_mask])
            right_entropy = entropy_calculator(y[right_mask])

            left_weight = np.sum(left_mask) / len(y)
            right_weight = np.sum(right_mask) / len(y)

            ig = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

            if ig > best_ig:
                best_ig = ig
                best_threshold = threshold

        return best_threshold, best_ig

    def entropy_calculator(y):
        """Calculate entropy of binary labels"""
        p1 = np.mean(y)
        p0 = 1 - p1
        if p1 == 0 or p1 == 1:
            return 0
        return -p1 * np.log2(p1) - p0 * np.log2(p0)

    # Test with synthetic data
    X_split = np.random.randn(100)
    y_split = (X_split + 0.5 * np.random.randn(100) > 0).astype(int)

    best_threshold, best_ig = find_best_split_threshold(X_split, y_split)
    print(f"\nExercise 2: Decision Tree Split")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Information gain: {best_ig:.3f} bits")

    return mi_estimated, best_threshold, best_ig

mi_estimated, best_threshold, best_ig = information_theory_exercises()
```

This comprehensive guide covers the essential information theory concepts needed for machine learning, from basic entropy calculations to advanced applications in deep learning. Each section includes mathematical explanations, Python implementations, and practical applications in ML.