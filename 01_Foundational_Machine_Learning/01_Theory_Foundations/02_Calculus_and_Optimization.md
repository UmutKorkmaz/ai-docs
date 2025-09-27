# Calculus and Optimization for Machine Learning

## Overview

Calculus and optimization form the mathematical backbone of machine learning, enabling us to train models, minimize loss functions, and understand algorithm behavior. This section covers the essential calculus concepts and optimization techniques used throughout machine learning.

## 1. Differential Calculus

### 1.1 Derivatives and Gradients

**Definition**: The derivative measures the rate of change of a function with respect to its input variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

# Basic derivatives
def derivatives_example():
    """Explore basic derivatives and their geometric interpretation"""

    # Define simple functions
    def linear(x):
        return 2 * x + 3

    def quadratic(x):
        return x**2 - 4*x + 5

    def cubic(x):
        return x**3 - 2*x**2 + x - 1

    def exponential(x):
        return np.exp(x)

    # Derivatives
    def derivative_linear(x):
        return 2  # Constant

    def derivative_quadratic(x):
        return 2*x - 4

    def derivative_cubic(x):
        return 3*x**2 - 4*x + 1

    def derivative_exponential(x):
        return np.exp(x)

    # Visualize functions and their derivatives
    x = np.linspace(-3, 3, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    functions = [
        ("Linear: f(x) = 2x + 3", linear, derivative_linear),
        ("Quadratic: f(x) = x² - 4x + 5", quadratic, derivative_quadratic),
        ("Cubic: f(x) = x³ - 2x² + x - 1", cubic, derivative_cubic),
        ("Exponential: f(x) = e^x", exponential, derivative_exponential)
    ]

    for idx, (title, func, deriv) in enumerate(functions):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        y = func(x)
        dy = deriv(x)

        ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
        ax.plot(x, dy, 'r--', linewidth=2, label="f'(x)")
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Numerical derivative verification
    def numerical_derivative(func, x, h=1e-6):
        """Compute numerical derivative using finite differences"""
        return (func(x + h) - func(x - h)) / (2 * h)

    # Test at specific points
    test_points = [0, 1, 2]
    print("Derivative Verification at Test Points:")
    print("-" * 50)

    for point in test_points:
        analytic = derivative_quadratic(point)
        numeric = numerical_derivative(quadratic, point)
        error = abs(analytic - numeric)
        print(f"x = {point}: Analytic = {analytic:.6f}, Numeric = {numeric:.6f}, Error = {error:.2e}")

    return functions

functions = derivatives_example()
```

**Derivative Rules**:
- **Power Rule**: $\frac{d}{dx}x^n = nx^{n-1}$
- **Product Rule**: $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$
- **Chain Rule**: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$
- **Quotient Rule**: $\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$

### 1.2 Partial Derivatives and Gradients

For multivariate functions, we need partial derivatives and gradients.

```python
def multivariate_functions():
    """Explore multivariate functions and their gradients"""

    # 2D function: f(x,y) = x² + 2xy + y²
    def f_2d(x, y):
        return x**2 + 2*x*y + y**2

    # Partial derivatives
    def df_dx(x, y):
        return 2*x + 2*y

    def df_dy(x, y):
        return 2*x + 2*y

    # Gradient
    def gradient_2d(x, y):
        return np.array([df_dx(x, y), df_dy(x, y)])

    # Create mesh for visualization
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f_2d(X, Y)

    # Visualize function and gradient field
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('f(x,y) = x² + 2xy + y²')

    # Contour plot with gradient vectors
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)

    # Add gradient vectors
    step = 5
    for i in range(0, len(x_range), step):
        for j in range(0, len(y_range), step):
            x_pos, y_pos = X[i, j], Y[i, j]
            grad = gradient_2d(x_pos, y_pos)
            # Scale gradient for visualization
            ax2.arrow(x_pos, y_pos, grad[0]*0.1, grad[1]*0.1,
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Contour Plot with Gradient Vectors')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Find minimum using gradient information
    print("Finding minimum of f(x,y) = x² + 2xy + y²")
    print("-" * 50)

    # Analytical solution
    print("Analytical minimum: x = 0, y = 0, f(0,0) = 0")

    # Numerical gradient descent
    def gradient_descent_2d(start_x, start_y, learning_rate=0.1, max_iter=100):
        x, y = start_x, start_y
        history = [(x, y, f_2d(x, y))]

        for i in range(max_iter):
            grad = gradient_2d(x, y)
            x = x - learning_rate * grad[0]
            y = y - learning_rate * grad[1]
            history.append((x, y, f_2d(x, y)))

            # Check convergence
            if np.linalg.norm(grad) < 1e-6:
                break

        return np.array(history)

    # Run gradient descent from different starting points
    start_points = [(2, 2), (-1, 1), (3, -2)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot contours
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot gradient descent paths
    colors = ['red', 'blue', 'green']
    for idx, (start_x, start_y) in enumerate(start_points):
        history = gradient_descent_2d(start_x, start_y, learning_rate=0.1)
        path = history[:, :2]

        ax.plot(path[:, 0], path[:, 1], 'o-', color=colors[idx], linewidth=2,
                markersize=4, label=f'Start: ({start_x}, {start_y})')
        ax.plot(path[-1, 0], path[-1, 1], '*', color=colors[idx], markersize=15)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Gradient Descent Paths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.show()

    return f_2d, gradient_2d

f_2d, gradient_2d = multivariate_functions()
```

**Gradient Properties**:
- **Definition**: $\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$
- **Direction**: Points in the direction of steepest ascent
- **Magnitude**: Represents the rate of change in that direction
- **Orthogonality**: Gradient is perpendicular to level sets (contours)

## 2. Integral Calculus

### 2.1 Definite and Indefinite Integrals

```python
def integral_calculus():
    """Explore integration concepts in machine learning"""

    # Define functions for integration
    def linear_func(x):
        return 2 * x + 1

    def quadratic_func(x):
        return x**2 - 3*x + 2

    def exponential_func(x):
        return np.exp(-x**2)

    # Analytical integrals (antiderivatives)
    def integral_linear(x):
        return x**2 + x  # ∫(2x+1)dx = x² + x + C

    def integral_quadratic(x):
        return (1/3)*x**3 - (3/2)*x**2 + 2*x  # ∫(x²-3x+2)dx = x³/3 - 3x²/2 + 2x + C

    # Numerical integration
    def trapezoidal_rule(func, a, b, n=1000):
        """Trapezoidal rule for numerical integration"""
        x = np.linspace(a, b, n)
        y = func(x)
        h = (b - a) / (n - 1)
        return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])

    def simpsons_rule(func, a, b, n=1000):
        """Simpson's rule for numerical integration"""
        if n % 2 == 1:
            n += 1  # Ensure even number of intervals

        x = np.linspace(a, b, n)
        y = func(x)
        h = (b - a) / (n - 1)

        integral = y[0] + y[-1]
        for i in range(1, n-1):
            if i % 2 == 1:  # Odd indices
                integral += 4 * y[i]
            else:  # Even indices
                integral += 2 * y[i]

        return h * integral / 3

    # Compare integration methods
    a, b = 0, 2

    print("Numerical Integration Comparison")
    print("-" * 50)
    print(f"Integrating from {a} to {b}")

    # Test with quadratic function
    analytical_quad = integral_quadratic(b) - integral_quadratic(a)
    trapezoidal_quad = trapezoidal_rule(quadratic_func, a, b)
    simpsons_quad = simpsons_rule(quadratic_func, a, b)

    print(f"\nQuadratic function: f(x) = x² - 3x + 2")
    print(f"Analytical: {analytical_quad:.6f}")
    print(f"Trapezoidal: {trapezoidal_quad:.6f} (Error: {abs(analytical_quad - trapezoidal_quad):.2e})")
    print(f"Simpson's: {simpsons_quad:.6f} (Error: {abs(analytical_quad - simpsons_quad):.2e})")

    # Visualize integration
    x = np.linspace(a-0.5, b+0.5, 100)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Function visualization
    ax1 = axes[0, 0]
    ax1.plot(x, quadratic_func(x), 'b-', linewidth=2, label='f(x) = x² - 3x + 2')
    ax1.fill_between(x[(x >= a) & (x <= b)], 0, quadratic_func(x[(x >= a) & (x <= b)]),
                     alpha=0.3, color='blue', label='Area under curve')
    ax1.axvline(x=a, color='red', linestyle='--', alpha=0.7, label=f'x = {a}')
    ax1.axvline(x=b, color='red', linestyle='--', alpha=0.7, label=f'x = {b}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Definite Integral Visualization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution (integral as area function)
    ax2 = axes[0, 1]
    x_cdf = np.linspace(a, b, 50)
    cumulative_area = [simpsons_rule(quadratic_func, a, x_i) for x_i in x_cdf]

    ax2.plot(x_cdf, cumulative_area, 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('∫f(t)dt from a to x')
    ax2.set_title('Cumulative Area Function')
    ax2.grid(True, alpha=0.3)

    # Integration accuracy comparison
    ax3 = axes[1, 0]
    n_values = np.logspace(1, 4, 20).astype(int)
    trapezoidal_errors = []
    simpsons_errors = []

    for n in n_values:
        trap_result = trapezoidal_rule(quadratic_func, a, b, n)
        simp_result = simpsons_rule(quadratic_func, a, b, n)

        trapezoidal_errors.append(abs(analytical_quad - trap_result))
        simpsons_errors.append(abs(analytical_quad - simp_result))

    ax3.loglog(n_values, trapezoidal_errors, 'o-', label='Trapezoidal Rule')
    ax3.loglog(n_values, simpsons_errors, 's-', label='Simpson\'s Rule')
    ax3.set_xlabel('Number of intervals (n)')
    ax3.set_ylabel('Absolute error')
    ax3.set_title('Integration Error vs Number of Intervals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Application: Probability density function
    ax4 = axes[1, 1]
    # Normal distribution
    def normal_pdf(x, mu=0, sigma=1):
        return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

    x_norm = np.linspace(-4, 4, 200)
    y_norm = normal_pdf(x_norm)

    ax4.plot(x_norm, y_norm, 'purple', linewidth=2, label='Standard Normal PDF')

    # Shade areas for different probabilities
    # P(-1 < X < 1)
    mask_1std = (x_norm >= -1) & (x_norm <= 1)
    ax4.fill_between(x_norm[mask_1std], 0, y_norm[mask_1std],
                     alpha=0.3, color='green', label='P(-1 < X < 1) ≈ 68%')

    # P(-2 < X < 2)
    mask_2std = (x_norm >= -2) & (x_norm <= 2)
    ax4.fill_between(x_norm[mask_2std], 0, y_norm[mask_2std],
                     alpha=0.2, color='orange', label='P(-2 < X < 2) ≈ 95%')

    ax4.set_xlabel('x')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Normal Distribution PDF')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate probabilities using integration
    prob_1std = simpsons_rule(normal_pdf, -1, 1, 1000)
    prob_2std = simpsons_rule(normal_pdf, -2, 2, 1000)

    print(f"\nProbability Calculations:")
    print(f"P(-1 < X < 1) = {prob_1std:.4f} (Expected: ~0.6827)")
    print(f"P(-2 < X < 2) = {prob_2std:.4f} (Expected: ~0.9545)")

    return trapezoidal_rule, simpsons_rule

trapezoidal_rule, simpsons_rule = integral_calculus()
```

## 3. Optimization Methods

### 3.1 Univariate Optimization

```python
def univariate_optimization():
    """Univariate optimization techniques"""

    # Test functions with different properties
    def simple_quadratic(x):
        return x**2 - 4*x + 5

    def multimodal(x):
        return x**4 - 3*x**3 + 2*x**2 + x + 1

    def non_convex(x):
        return np.sin(x) + 0.5*x

    # Derivatives for gradient-based methods
    def grad_simple_quadratic(x):
        return 2*x - 4

    def grad_multimodal(x):
        return 4*x**3 - 9*x**2 + 4*x + 1

    def grad_non_convex(x):
        return np.cos(x) + 0.5

    # Analytical solutions
    def analytical_minimum_quadratic():
        # f(x) = x² - 4x + 5
        # f'(x) = 2x - 4 = 0 => x = 2
        return 2

    print("Univariate Optimization")
    print("-" * 50)

    # 1. Bisection method (for root finding, then optimization)
    def bisection_method(f, a, b, tol=1e-6, max_iter=100):
        """Find root using bisection method"""
        if f(a) * f(b) >= 0:
            raise ValueError("Function must change sign over interval")

        for i in range(max_iter):
            c = (a + b) / 2
            if abs(f(c)) < tol:
                return c
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c

        return (a + b) / 2

    # 2. Golden section search
    def golden_section_search(f, a, b, tol=1e-6, max_iter=100):
        """Find minimum using golden section search"""
        golden_ratio = (np.sqrt(5) - 1) / 2

        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)

        fc = f(c)
        fd = f(d)

        for i in range(max_iter):
            if abs(b - a) < tol:
                return (a + b) / 2

            if fc < fd:
                b = d
                d = c
                c = b - golden_ratio * (b - a)
                fd = fc
                fc = f(c)
            else:
                a = c
                c = d
                d = a + golden_ratio * (b - a)
                fc = fd
                fd = f(d)

        return (a + b) / 2

    # 3. Gradient descent for univariate
    def gradient_descent_1d(f_grad, x0, learning_rate=0.1, tol=1e-6, max_iter=1000):
        """Gradient descent for univariate optimization"""
        x = x0
        history = [x]

        for i in range(max_iter):
            grad = f_grad(x)
            x_new = x - learning_rate * grad
            history.append(x_new)

            if abs(x_new - x) < tol:
                break

            x = x_new

        return x, np.array(history)

    # Test optimization methods
    test_functions = [
        ("Simple Quadratic", simple_quadratic, grad_simple_quadratic, [-5, 10]),
        ("Multimodal", multimodal, grad_multimodal, [-2, 3]),
        ("Non-convex", non_convex, grad_non_convex, [-4, 4])
    ]

    for name, func, grad, interval in test_functions:
        print(f"\n{name}:")
        print(f"Interval: {interval}")

        # Golden section search
        min_golden = golden_section_search(func, interval[0], interval[1])
        print(f"Golden Section: x = {min_golden:.6f}, f(x) = {func(min_golden):.6f}")

        # Gradient descent
        x0 = interval[1]  # Start from right endpoint
        min_grad, history = gradient_descent_1d(grad, x0, learning_rate=0.1)
        print(f"Gradient Descent: x = {min_grad:.6f}, f(x) = {func(min_grad):.6f}")

        # Visualization
        x_vis = np.linspace(interval[0], interval[1], 200)
        y_vis = func(x_vis)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x_vis, y_vis, 'b-', linewidth=2, label=f'{name}')
        plt.plot(min_golden, func(min_golden), 'ro', markersize=10, label='Golden Section')
        plt.plot(min_grad, func(min_grad), 'gs', markersize=10, label='Gradient Descent')
        plt.plot(history, [func(x) for x in history], 'g--', alpha=0.7, label='GD Path')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'{name} - Optimization Results')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history, [func(x) for x in history], 'g-o', linewidth=2, markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')
        plt.title('Gradient Descent Convergence')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return golden_section_search, gradient_descent_1d

golden_section_search, gradient_descent_1d = univariate_optimization()
```

### 3.2 Multivariate Optimization

```python
def multivariate_optimization():
    """Multivariate optimization techniques"""

    # Test functions
    def quadratic_2d(x):
        """f(x,y) = x² + 2y² + xy"""
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1]

    def rosenbrock(x):
        """Rosenbrock function - classic test function"""
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    def himmelblau(x):
        """Himmelblau's function - has multiple minima"""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    # Gradients
    def grad_quadratic_2d(x):
        return np.array([2*x[0] + x[1], 4*x[1] + x[0]])

    def grad_rosenbrock(x):
        return np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
                        200*(x[1] - x[0]**2)])

    def grad_himmelblau(x):
        return np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
                        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)])

    # Hessian matrices
    def hessian_quadratic_2d(x):
        return np.array([[2, 1], [1, 4]])

    def hessian_rosenbrock(x):
        return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
                        [-400*x[0], 200]])

    # 1. Steepest Descent
    def steepest_descent(f, grad_f, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """Steepest descent optimization"""
        x = x0.copy()
        history = [x.copy()]

        for i in range(max_iter):
            grad = grad_f(x)
            x_new = x - learning_rate * grad

            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new
            history.append(x.copy())

        return x, np.array(history)

    # 2. Newton's Method
    def newtons_method(f, grad_f, hessian_f, x0, max_iter=100, tol=1e-6):
        """Newton's method for optimization"""
        x = x0.copy()
        history = [x.copy()]

        for i in range(max_iter):
            grad = grad_f(x)
            hessian = hessian_f(x)

            # Solve Hessian * delta = -gradient
            delta = np.linalg.solve(hessian, -grad)
            x_new = x + delta

            if np.linalg.norm(delta) < tol:
                break

            x = x_new
            history.append(x.copy())

        return x, np.array(history)

    # 3. Conjugate Gradient
    def conjugate_gradient(f, grad_f, x0, max_iter=1000, tol=1e-6):
        """Nonlinear conjugate gradient method (Fletcher-Reeves)"""
        x = x0.copy()
        history = [x.copy()]

        grad = grad_f(x)
        direction = -grad
        beta = 0

        for i in range(max_iter):
            # Line search (simplified)
            alpha = 0.01  # Fixed step size for simplicity
            x_new = x + alpha * direction
            grad_new = grad_f(x_new)

            if np.linalg.norm(grad_new) < tol:
                break

            # Fletcher-Reeves beta
            beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
            direction = -grad_new + beta * direction

            x = x_new
            grad = grad_new
            history.append(x.copy())

        return x, np.array(history)

    # Test optimization methods
    test_functions = [
        ("Quadratic 2D", quadratic_2d, grad_quadratic_2d, hessian_quadratic_2d, np.array([3, 3])),
        ("Rosenbrock", rosenbrock, grad_rosenbrock, hessian_rosenbrock, np.array([-1, 1])),
        ("Himmelblau", himmelblau, grad_himmelblau, None, np.array([0, 0]))
    ]

    print("Multivariate Optimization Results")
    print("-" * 70)

    for name, func, grad, hessian, x0 in test_functions:
        print(f"\n{name}:")
        print(f"Starting point: {x0}")

        # Steepest descent
        min_steepest, history_steepest = steepest_descent(func, grad, x0, learning_rate=0.01)
        print(f"Steepest Descent: x = {min_steepest}, f(x) = {func(min_steepest):.6f}")

        # Newton's method (if Hessian available)
        if hessian is not None:
            min_newton, history_newton = newtons_method(func, grad, hessian, x0)
            print(f"Newton's Method: x = {min_newton}, f(x) = {func(min_newton):.6f}")

        # Conjugate gradient
        min_cg, history_cg = conjugate_gradient(func, grad, x0)
        print(f"Conjugate Gradient: x = {min_cg}, f(x) = {func(min_cg):.6f}")

        # Visualization
        if len(x0) == 2:  # Only visualize 2D functions
            visualize_optimization_2d(func, grad, x0, name,
                                   history_steepest,
                                   history_newton if hessian is not None else None,
                                   history_cg)

    return steepest_descent, newtons_method, conjugate_gradient

def visualize_optimization_2d(func, grad, x0, name, history_steepest, history_newton, history_cg):
    """Visualize optimization paths for 2D functions"""
    # Create contour plot
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    plt.figure(figsize=(15, 5))

    # Steepest descent
    plt.subplot(1, 3, 1)
    plt.contour(X, Y, Z, levels=30, alpha=0.6)
    plt.plot(history_steepest[:, 0], history_steepest[:, 1], 'r-o', markersize=4, label='Path')
    plt.plot(x0[0], x0[1], 'go', markersize=8, label='Start')
    plt.plot(history_steepest[-1, 0], history_steepest[-1, 1], 'r*', markersize=12, label='End')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'{name} - Steepest Descent')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    # Newton's method
    if history_newton is not None:
        plt.subplot(1, 3, 2)
        plt.contour(X, Y, Z, levels=30, alpha=0.6)
        plt.plot(history_newton[:, 0], history_newton[:, 1], 'b-o', markersize=4, label='Path')
        plt.plot(x0[0], x0[1], 'go', markersize=8, label='Start')
        plt.plot(history_newton[-1, 0], history_newton[-1, 1], 'b*', markersize=12, label='End')
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title(f'{name} - Newton\'s Method')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

    # Conjugate gradient
    plt.subplot(1, 3, 3)
    plt.contour(X, Y, Z, levels=30, alpha=0.6)
    plt.plot(history_cg[:, 0], history_cg[:, 1], 'g-o', markersize=4, label='Path')
    plt.plot(x0[0], x0[1], 'go', markersize=8, label='Start')
    plt.plot(history_cg[-1, 0], history_cg[-1, 1], 'g*', markersize=12, label='End')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'{name} - Conjugate Gradient')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

steepest_descent, newtons_method, conjugate_gradient = multivariate_optimization()
```

## 4. Constrained Optimization

### 4.1 Lagrange Multipliers

```python
def constrained_optimization():
    """Constrained optimization using Lagrange multipliers"""

    # Example: Minimize f(x,y) = x² + y² subject to x + y = 1
    def objective_function(x):
        return x[0]**2 + x[1]**2

    def constraint_function(x):
        return x[0] + x[1] - 1

    # Lagrangian: L(x,y,λ) = x² + y² + λ(1 - x - y)
    def lagrangian(x_lambda):
        x, y, lambda_ = x_lambda
        return x**2 + y**2 + lambda_ * (1 - x - y)

    # Gradient of Lagrangian
    def grad_lagrangian(x_lambda):
        x, y, lambda_ = x_lambda
        return np.array([2*x - lambda_, 2*y - lambda_, 1 - x - y])

    # Solve using Newton's method
    x0 = np.array([0.5, 0.5, 0.0])  # Initial guess [x, y, λ]

    print("Constrained Optimization Example")
    print("-" * 50)
    print("Minimize f(x,y) = x² + y² subject to x + y = 1")

    # Analytical solution
    print("\nAnalytical Solution:")
    print("From ∇f = λ∇g and g = 0:")
    print("2x = λ, 2y = λ, x + y = 1")
    print("Thus x = y = 0.5, λ = 1")
    print("Minimum value: f(0.5, 0.5) = 0.5")

    # Numerical solution
    solution = minimize(lagrangian, x0, method='BFGS', jac=grad_lagrangian)

    print(f"\nNumerical Solution:")
    print(f"x = {solution.x[0]:.6f}")
    print(f"y = {solution.x[1]:.6f}")
    print(f"λ = {solution.x[2]:.6f}")
    print(f"Objective value: {objective_function(solution.x[:2]):.6f}")
    print(f"Constraint satisfaction: {abs(constraint_function(solution.x[:2])):.2e}")

    # Visualization
    fig = plt.figure(figsize=(15, 5))

    # 3D plot of objective and constraint
    ax1 = fig.add_subplot(131, projection='3d')
    x_range = np.linspace(-1, 2, 50)
    y_range = np.linspace(-1, 2, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective_function([X, Y])

    surf = ax1.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')

    # Constraint line in 3D
    x_const = np.linspace(-1, 2, 50)
    y_const = 1 - x_const
    z_const = x_const**2 + y_const**2
    ax1.plot(x_const, y_const, z_const, 'r-', linewidth=3, label='Constraint')

    # Optimal point
    ax1.scatter([0.5], [0.5], [0.5], color='red', s=100, label='Optimum')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('3D View: Objective and Constraint')
    ax1.legend()

    # Contour plot with constraint
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=20, alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)

    # Constraint line
    ax2.plot(x_const, y_const, 'r-', linewidth=2, label='Constraint: x + y = 1')
    ax2.scatter([0.5], [0.5], color='red', s=100, label='Optimum')

    # Gradient vectors at optimum
    grad_f = np.array([2*0.5, 2*0.5])  # ∇f at optimum
    grad_g = np.array([1, 1])  # ∇g (constraint gradient)

    ax2.arrow(0.5, 0.5, grad_f[0]*0.2, grad_f[1]*0.2,
             head_width=0.05, head_length=0.05, fc='blue', ec='blue',
             label='∇f')
    ax2.arrow(0.5, 0.5, grad_g[0]*0.2, grad_g[1]*0.2,
             head_width=0.05, head_length=0.05, fc='green', ec='green',
             label='∇g')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot with Gradients')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Level curves and constraint
    ax3 = fig.add_subplot(133)
    levels = np.linspace(0, 3, 15)
    contour = ax3.contour(X, Y, Z, levels=levels, alpha=0.6)
    ax3.clabel(contour, inline=True, fontsize=8)

    ax3.plot(x_const, y_const, 'r-', linewidth=2, label='Constraint')
    ax3.scatter([0.5], [0.5], color='red', s=100, label='Optimum')

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Level Curves and Constraint')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return solution

solution = constrained_optimization()
```

### 4.2 Karush-Kuhn-Tucker (KKT) Conditions

```python
def kkt_conditions():
    """Karush-Kuhn-Tucker conditions for inequality constraints"""

    # Example: Minimize f(x,y) = x² + y² subject to x + y ≥ 1
    def objective_function(x):
        return x[0]**2 + x[1]**2

    def inequality_constraint(x):
        return x[0] + x[1] - 1  # g(x,y) = x + y - 1 ≥ 0

    print("KKT Conditions Example")
    print("-" * 50)
    print("Minimize f(x,y) = x² + y² subject to x + y ≥ 1")

    # KKT conditions:
    # 1. ∇f - λ∇g = 0 (stationarity)
    # 2. g(x,y) ≥ 0 (primal feasibility)
    # 3. λ ≥ 0 (dual feasibility)
    # 4. λg(x,y) = 0 (complementary slackness)

    print("\nKKT Conditions:")
    print("1. Stationarity: [2x, 2y] - λ[1, 1] = 0")
    print("2. Primal feasibility: x + y ≥ 1")
    print("3. Dual feasibility: λ ≥ 0")
    print("4. Complementary slackness: λ(x + y - 1) = 0")

    # Case 1: λ = 0 (constraint inactive)
    print("\nCase 1: λ = 0 (constraint inactive)")
    print("Then 2x = 0, 2y = 0 => x = 0, y = 0")
    print("Check constraint: 0 + 0 = 0 ≥ 1? NO")
    print("This case is invalid.")

    # Case 2: λ > 0 (constraint active)
    print("\nCase 2: λ > 0 (constraint active)")
    print("Then x + y = 1, and 2x = λ, 2y = λ")
    print("Thus x = y = λ/2, and λ/2 + λ/2 = 1")
    print("Therefore λ = 1, x = y = 0.5")
    print("Check: λ = 1 ≥ 0, x + y = 1 ≥ 1")
    print("This is the solution.")

    # Solve numerically
    from scipy.optimize import minimize

    # Set up constrained optimization
    constraints = [{'type': 'ineq', 'fun': inequality_constraint}]
    x0 = np.array([0.5, 0.5])

    result = minimize(objective_function, x0, method='SLSQP', constraints=constraints)

    print(f"\nNumerical Solution:")
    print(f"x = {result.x[0]:.6f}")
    print(f"y = {result.x[1]:.6f}")
    print(f"Objective value: {result.fun:.6f}")
    print(f"Constraint value: {inequality_constraint(result.x):.6f}")

    # Check if constraint is active
    constraint_active = abs(inequality_constraint(result.x)) < 1e-6
    print(f"Constraint active: {constraint_active}")

    # Lagrange multiplier (from KKT conditions)
    lambda_kkt = 2 * result.x[0]  # From 2x = λ
    print(f"Lagrange multiplier λ = {lambda_kkt:.6f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    x_range = np.linspace(-1, 2, 100)
    y_range = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective_function([X, Y])

    # Plot objective function contours
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot feasible region
    x_feasible = np.linspace(-1, 2, 100)
    y_feasible = 1 - x_feasible
    ax.fill_between(x_feasible, y_feasible, 3, alpha=0.3, color='green', label='Feasible Region')

    # Plot constraint boundary
    ax.plot(x_feasible, y_feasible, 'r-', linewidth=2, label='Constraint: x + y = 1')

    # Plot optimal point
    ax.scatter([result.x[0]], [result.x[1]], color='red', s=100, label='Optimum')

    # Plot gradient vectors
    grad_f = np.array([2*result.x[0], 2*result.x[1]])
    grad_g = np.array([1, 1])

    ax.arrow(result.x[0], result.x[1], grad_f[0]*0.1, grad_f[1]*0.1,
             head_width=0.05, head_length=0.05, fc='blue', ec='blue',
             label='∇f')
    ax.arrow(result.x[0], result.x[1], -grad_g[0]*0.1, -grad_g[1]*0.1,
             head_width=0.05, head_length=0.05, fc='orange', ec='orange',
             label='-∇g')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('KKT Conditions: Constrained Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.show()

    return result

result = kkt_conditions()
```

## 5. Applications in Machine Learning

### 5.1 Loss Function Optimization

```python
def loss_function_optimization():
    """Optimization of common machine learning loss functions"""

    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([1.5, -2.0])
    y = (X @ true_weights + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # Loss functions
    def logistic_loss(weights):
        """Logistic loss for binary classification"""
        scores = X @ weights
        probabilities = 1 / (1 + np.exp(-scores))
        # Avoid log(0)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

    def logistic_gradient(weights):
        """Gradient of logistic loss"""
        scores = X @ weights
        probabilities = 1 / (1 + np.exp(-scores))
        gradient = X.T @ (probabilities - y) / len(y)
        return gradient

    def hinge_loss(weights):
        """Hinge loss for SVM"""
        scores = X @ weights
        margins = y * scores
        return np.mean(np.maximum(0, 1 - margins))

    def hinge_gradient(weights):
        """Subgradient of hinge loss"""
        scores = X @ weights
        margins = y * scores
        mask = margins < 1
        gradient = -X[mask].T @ y[mask] / len(y)
        return gradient

    # Optimization methods
    def gradient_descent_loss(loss_func, grad_func, weights_init, learning_rate=0.1, max_iter=1000):
        """Gradient descent for loss minimization"""
        weights = weights_init.copy()
        history = [loss_func(weights)]

        for i in range(max_iter):
            grad = grad_func(weights)
            weights = weights - learning_rate * grad
            history.append(loss_func(weights))

        return weights, np.array(history)

    print("Loss Function Optimization in Machine Learning")
    print("-" * 60)

    # Initialize weights
    weights_init = np.array([0.0, 0.0])

    # Optimize logistic regression
    print("\nLogistic Regression Optimization:")
    weights_logistic, loss_history_logistic = gradient_descent_loss(
        logistic_loss, logistic_gradient, weights_init, learning_rate=0.1)

    print(f"True weights: {true_weights}")
    print(f"Learned weights: {weights_logistic}")
    print(f"Final loss: {loss_history_logistic[-1]:.6f}")

    # Optimize SVM (hinge loss)
    print("\nSVM Optimization (Hinge Loss):")
    weights_svm, loss_history_svm = gradient_descent_loss(
        hinge_loss, hinge_gradient, weights_init, learning_rate=0.01)

    print(f"True weights: {true_weights}")
    print(f"Learned weights: {weights_svm}")
    print(f"Final loss: {loss_history_svm[-1]:.6f}")

    # Compare convergence
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss convergence
    axes[0, 0].plot(loss_history_logistic, 'b-', linewidth=2, label='Logistic Loss')
    axes[0, 0].plot(loss_history_svm, 'r-', linewidth=2, label='Hinge Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Function Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Decision boundaries
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # Logistic regression decision boundary
    Z_logistic = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            point = np.array([X_grid[i, j], Y_grid[i, j]])
            score = point @ weights_logistic
            Z_logistic[i, j] = 1 / (1 + np.exp(-score))

    axes[0, 1].contour(X_grid, Y_grid, Z_logistic, levels=[0.5], colors='blue', linewidths=2)
    axes[0, 1].scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.7, label='Class 0')
    axes[0, 1].scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.7, label='Class 1')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].set_title('Logistic Regression Decision Boundary')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # SVM decision boundary
    Z_svm = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            point = np.array([X_grid[i, j], Y_grid[i, j]])
            score = point @ weights_svm
            Z_svm[i, j] = score

    axes[1, 0].contour(X_grid, Y_grid, Z_svm, levels=[0], colors='red', linewidths=2)
    axes[1, 0].scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.7, label='Class 0')
    axes[1, 0].scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.7, label='Class 1')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    axes[1, 0].set_title('SVM Decision Boundary')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Weight comparison
    methods = ['True', 'Logistic', 'SVM']
    weights_to_plot = [true_weights, weights_logistic, weights_svm]

    x_pos = np.arange(len(methods))
    width = 0.35

    axes[1, 1].bar(x_pos - width/2, [w[0] for w in weights_to_plot], width, label='Weight 1')
    axes[1, 1].bar(x_pos + width/2, [w[1] for w in weights_to_plot], width, label='Weight 2')
    axes[1, 1].set_xlabel('Method')
    axes[1, 1].set_ylabel('Weight Value')
    axes[1, 1].set_title('Weight Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return weights_logistic, weights_svm

weights_logistic, weights_svm = loss_function_optimization()
```

### 5.2 Regularization and Optimization

```python
def regularization_optimization():
    """Regularization techniques and their optimization"""

    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 50, 10
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features) * 0.5
    true_weights[5:] = 0  # Make some weights zero (sparse)
    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    # Regularized loss functions
    def ridge_loss(weights, lambda_ridge=1.0):
        """Ridge regression loss"""
        predictions = X @ weights
        mse = np.mean((predictions - y)**2)
        regularization = lambda_ridge * np.sum(weights**2)
        return mse + regularization

    def ridge_gradient(weights, lambda_ridge=1.0):
        """Gradient of ridge loss"""
        predictions = X @ weights
        mse_gradient = 2 * X.T @ (predictions - y) / len(y)
        regularization_gradient = 2 * lambda_ridge * weights
        return mse_gradient + regularization_gradient

    def lasso_loss(weights, lambda_lasso=1.0):
        """Lasso regression loss"""
        predictions = X @ weights
        mse = np.mean((predictions - y)**2)
        regularization = lambda_lasso * np.sum(np.abs(weights))
        return mse + regularization

    def lasso_subgradient(weights, lambda_lasso=1.0):
        """Subgradient of lasso loss"""
        predictions = X @ weights
        mse_gradient = 2 * X.T @ (predictions - y) / len(y)
        regularization_subgradient = lambda_lasso * np.sign(weights)
        return mse_gradient + regularization_subgradient

    def elastic_net_loss(weights, lambda_lasso=1.0, lambda_ridge=1.0, alpha=0.5):
        """Elastic net loss"""
        predictions = X @ weights
        mse = np.mean((predictions - y)**2)
        l1_reg = alpha * lambda_lasso * np.sum(np.abs(weights))
        l2_reg = (1 - alpha) * lambda_ridge * np.sum(weights**2)
        return mse + l1_reg + l2_reg

    # Optimization with different regularization strengths
    lambda_values = [0.01, 0.1, 1.0, 10.0]

    print("Regularization Optimization")
    print("-" * 50)
    print(f"True weights (first 5 shown): {true_weights[:5]}")
    print(f"Number of non-zero true weights: {np.sum(np.abs(true_weights) > 1e-10)}")

    # Initialize weights
    weights_init = np.random.randn(n_features) * 0.1

    # Ridge regression with different lambda values
    print("\nRidge Regression:")
    ridge_weights = {}
    for lambda_val in lambda_values:
        weights, _ = gradient_descent_loss(
            ridge_loss, ridge_gradient, weights_init,
            learning_rate=0.01, max_iter=1000)
        ridge_weights[lambda_val] = weights
        print(f"λ = {lambda_val:.2f}: Non-zero weights = {np.sum(np.abs(weights) > 1e-10)}")

    # Lasso regression with different lambda values
    print("\nLasso Regression:")
    lasso_weights = {}
    for lambda_val in lambda_values:
        weights, _ = gradient_descent_loss(
            lasso_loss, lasso_subgradient, weights_init,
            learning_rate=0.001, max_iter=1000)
        lasso_weights[lambda_val] = weights
        print(f"λ = {lambda_val:.2f}: Non-zero weights = {np.sum(np.abs(weights) > 1e-10)}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Ridge coefficients path
    axes[0, 0].plot(lambda_values, [np.linalg.norm(ridge_weights[l], 2) for l in lambda_values],
                    'bo-', linewidth=2, markersize=8, label='L2 norm')
    axes[0, 0].plot(lambda_values, [np.sum(np.abs(ridge_weights[l])) for l in lambda_values],
                    'ro-', linewidth=2, markersize=8, label='L1 norm')
    axes[0, 0].set_xlabel('λ (log scale)')
    axes[0, 0].set_ylabel('Norm of coefficients')
    axes[0, 0].set_title('Ridge Regression: Coefficient Norms vs λ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')

    # Lasso coefficients path
    axes[0, 1].plot(lambda_values, [np.linalg.norm(lasso_weights[l], 2) for l in lambda_values],
                    'bo-', linewidth=2, markersize=8, label='L2 norm')
    axes[0, 1].plot(lambda_values, [np.sum(np.abs(lasso_weights[l])) for l in lambda_values],
                    'ro-', linewidth=2, markersize=8, label='L1 norm')
    axes[0, 1].set_xlabel('λ (log scale)')
    axes[0, 1].set_ylabel('Norm of coefficients')
    axes[0, 1].set_title('Lasso Regression: Coefficient Norms vs λ')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')

    # Weight distribution comparison
    feature_idx = range(min(8, n_features))  # Show first 8 features
    x_pos = np.arange(len(feature_idx))
    width = 0.15

    lambda_idx = 1  # Use lambda = 0.1 for comparison
    axes[1, 0].bar(x_pos - width, true_weights[feature_idx], width, label='True')
    axes[1, 0].bar(x_pos, ridge_weights[lambda_values[lambda_idx]][feature_idx], width, label='Ridge')
    axes[1, 0].bar(x_pos + width, lasso_weights[lambda_values[lambda_idx]][feature_idx], width, label='Lasso')

    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Weight Value')
    axes[1, 0].set_title(f'Weight Comparison (λ = {lambda_values[lambda_idx]})')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(feature_idx)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Sparsity comparison
    ridge_sparsity = [np.sum(np.abs(ridge_weights[l]) > 1e-10) for l in lambda_values]
    lasso_sparsity = [np.sum(np.abs(lasso_weights[l]) > 1e-10) for l in lambda_values]

    axes[1, 1].plot(lambda_values, ridge_sparsity, 'bo-', linewidth=2, markersize=8, label='Ridge')
    axes[1, 1].plot(lambda_values, lasso_sparsity, 'ro-', linewidth=2, markersize=8, label='Lasso')
    axes[1, 1].axhline(y=np.sum(np.abs(true_weights) > 1e-10), color='green',
                       linestyle='--', label='True sparsity')
    axes[1, 1].set_xlabel('λ (log scale)')
    axes[1, 1].set_ylabel('Number of non-zero weights')
    axes[1, 1].set_title('Sparsity vs Regularization Strength')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')

    plt.tight_layout()
    plt.show()

    return ridge_weights, lasso_weights

ridge_weights, lasso_weights = regularization_optimization()
```

## 6. Advanced Optimization Topics

### 6.1 Stochastic Optimization

```python
def stochastic_optimization():
    """Stochastic and mini-batch optimization methods"""

    # Generate larger dataset for stochastic methods
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    # Loss and gradient functions
    def mse_loss(weights, X_batch, y_batch):
        """Mean squared error loss"""
        predictions = X_batch @ weights
        return np.mean((predictions - y_batch)**2)

    def mse_gradient(weights, X_batch, y_batch):
        """Gradient of MSE loss"""
        predictions = X_batch @ weights
        return 2 * X_batch.T @ (predictions - y_batch) / len(y_batch)

    # Optimization methods
    def batch_gradient_descent(X, y, weights_init, learning_rate=0.01, max_iter=100):
        """Full batch gradient descent"""
        weights = weights_init.copy()
        history = []

        for i in range(max_iter):
            grad = mse_gradient(weights, X, y)
            weights = weights - learning_rate * grad
            loss = mse_loss(weights, X, y)
            history.append(loss)

        return weights, np.array(history)

    def stochastic_gradient_descent(X, y, weights_init, learning_rate=0.01, max_iter=1000):
        """Stochastic gradient descent"""
        weights = weights_init.copy()
        history = []

        for i in range(max_iter):
            # Random sample
            idx = np.random.randint(0, len(y))
            X_sample = X[idx:idx+1]
            y_sample = y[idx:idx+1]

            grad = mse_gradient(weights, X_sample, y_sample)
            weights = weights - learning_rate * grad

            # Evaluate on full dataset
            if i % 10 == 0:  # Evaluate every 10 iterations
                loss = mse_loss(weights, X, y)
                history.append(loss)

        return weights, np.array(history)

    def mini_batch_gradient_descent(X, y, weights_init, learning_rate=0.01, batch_size=32, max_iter=1000):
        """Mini-batch gradient descent"""
        weights = weights_init.copy()
        history = []

        for i in range(max_iter):
            # Random mini-batch
            indices = np.random.choice(len(y), batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            grad = mse_gradient(weights, X_batch, y_batch)
            weights = weights - learning_rate * grad

            # Evaluate on full dataset
            if i % 10 == 0:
                loss = mse_loss(weights, X, y)
                history.append(loss)

        return weights, np.array(history)

    print("Stochastic Optimization Methods")
    print("-" * 50)
    print(f"Dataset: {n_samples} samples, {n_features} features")

    # Initialize weights
    weights_init = np.random.randn(n_features) * 0.1

    # Run different optimization methods
    print("\nOptimization Results:")

    # Batch gradient descent
    weights_batch, history_batch = batch_gradient_descent(X, y, weights_init, learning_rate=0.01)
    print(f"Batch GD: Final loss = {history_batch[-1]:.6f}")

    # Stochastic gradient descent
    weights_sgd, history_sgd = stochastic_gradient_descent(X, y, weights_init, learning_rate=0.001)
    print(f"SGD: Final loss = {history_sgd[-1]:.6f}")

    # Mini-batch gradient descent
    weights_mini, history_mini = mini_batch_gradient_descent(X, y, weights_init, learning_rate=0.01, batch_size=32)
    print(f"Mini-batch: Final loss = {history_mini[-1]:.6f}")

    print(f"\nTrue weights: {true_weights}")
    print(f"Batch GD learned: {weights_batch}")
    print(f"SGD learned: {weights_sgd}")
    print(f"Mini-batch learned: {weights_mini}")

    # Compare convergence
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss convergence
    axes[0, 0].plot(history_batch, 'b-', linewidth=2, label='Batch GD')
    axes[0, 0].plot(range(0, len(history_sgd)*10, 10), history_sgd, 'r-', linewidth=2, label='SGD')
    axes[0, 0].plot(range(0, len(history_mini)*10, 10), history_mini, 'g-', linewidth=2, label='Mini-batch')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Convergence Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Weight convergence
    weight_errors_batch = [np.linalg.norm(weights_batch - true_weights)]
    weight_errors_sgd = [np.linalg.norm(weights_sgd - true_weights)]
    weight_errors_mini = [np.linalg.norm(weights_mini - true_weights)]

    axes[0, 1].bar(['Batch GD', 'SGD', 'Mini-batch'],
                   [weight_errors_batch[0], weight_errors_sgd[0], weight_errors_mini[0]],
                   color=['blue', 'red', 'green'])
    axes[0, 1].set_ylabel('Weight Error (L2 norm)')
    axes[0, 1].set_title('Weight Estimation Error')
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate sensitivity
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    final_losses = []

    for lr in learning_rates:
        weights, history = batch_gradient_descent(X, y, weights_init, learning_rate=lr, max_iter=50)
        final_losses.append(history[-1])

    axes[1, 0].semilogx(learning_rates, final_losses, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Learning Rate (log scale)')
    axes[1, 0].set_ylabel('Final Loss')
    axes[1, 0].set_title('Learning Rate Sensitivity')
    axes[1, 0].grid(True, alpha=0.3)

    # Batch size sensitivity
    batch_sizes = [1, 8, 32, 64, 128, 256]
    batch_losses = []

    for bs in batch_sizes:
        if bs == 1:
            weights, history = stochastic_gradient_descent(X, y, weights_init, learning_rate=0.001, max_iter=100)
        else:
            weights, history = mini_batch_gradient_descent(X, y, weights_init, learning_rate=0.01, batch_size=bs, max_iter=100)
        batch_losses.append(history[-1])

    axes[1, 1].semilogx(batch_sizes, batch_losses, 'ro-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Batch Size (log scale)')
    axes[1, 1].set_ylabel('Final Loss')
    axes[1, 1].set_title('Batch Size Sensitivity')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return weights_batch, weights_sgd, weights_mini

weights_batch, weights_sgd, weights_mini = stochastic_optimization()
```

## 7. Key Concepts Summary

### 7.1 Essential Calculus for ML

1. **Derivatives**: Measure rate of change, essential for optimization
2. **Gradients**: Multi-dimensional derivatives, used in gradient descent
3. **Integrals**: Used in probability theory and expected values
4. **Optimization**: Finding minima/maxima of functions
5. **Constraints**: Handling restrictions in optimization problems

### 7.2 Optimization Algorithms

- **Gradient Descent**: First-order method, simple but can be slow
- **Newton's Method**: Second-order method, faster convergence but more expensive
- **Conjugate Gradient**: Efficient for large-scale problems
- **Stochastic Methods**: Essential for large datasets
- **Regularization**: Prevents overfitting, adds constraints

### 7.3 Important Theorems

- **Taylor's Theorem**: Local approximation of functions
- **Mean Value Theorem**: Relates function values to derivatives
- **Fundamental Theorem of Calculus**: Connects differentiation and integration
- **Karush-Kuhn-Tucker**: Necessary conditions for constrained optimization

### 7.4 Practical Considerations

- **Learning Rate**: Critical for convergence stability
- **Convexity**: Guarantees global minimum
- **Numerical Stability**: Important for practical implementations
- **Computational Complexity**: Affects scalability

## 8. Exercises

### 8.1 Theory Exercises

1. Derive the gradient of the cross-entropy loss function.
2. Prove that Newton's method converges quadratically near a minimum.
3. Show that the Hessian matrix of a convex function is positive semi-definite.
4. Derive the KKT conditions for a general constrained optimization problem.
5. Analyze the convergence rate of gradient descent for strongly convex functions.

### 8.2 Programming Exercises

```python
def calculus_optimization_exercises():
    """
    Complete these exercises to test your understanding:

    Exercise 1: Implement Newton's method from scratch
    Exercise 2: Implement logistic regression with L2 regularization
    Exercise 3: Compare different optimization algorithms on Rosenbrock function
    Exercise 4: Implement momentum and Adam optimizers
    Exercise 5: Solve a constrained optimization problem using penalty methods
    """

    # Exercise 1: Newton's method implementation
    def newton_method_implementation(f, grad_f, hessian_f, x0, max_iter=100, tol=1e-6):
        """Implement Newton's method from scratch"""
        x = x0.copy()
        history = [x.copy()]

        for i in range(max_iter):
            grad = grad_f(x)
            hessian = hessian_f(x)

            # Check if Hessian is invertible
            if np.linalg.det(hessian) == 0:
                print("Hessian is singular, stopping iteration")
                break

            # Newton step: x = x - H^(-1) * grad
            delta = np.linalg.solve(hessian, -grad)
            x_new = x + delta

            if np.linalg.norm(delta) < tol:
                break

            x = x_new
            history.append(x.copy())

        return x, np.array(history)

    # Test Newton's method
    def quadratic_test(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1]

    def grad_quadratic_test(x):
        return np.array([2*x[0] + x[1], 4*x[1] + x[0]])

    def hessian_quadratic_test(x):
        return np.array([[2, 1], [1, 4]])

    x0_test = np.array([3.0, 3.0])
    result, history = newton_method_implementation(
        quadratic_test, grad_quadratic_test, hessian_quadratic_test, x0_test)

    print("Exercise 1: Newton's Method Implementation")
    print(f"Starting point: {x0_test}")
    print(f"Result: {result}")
    print(f"Objective value: {quadratic_test(result):.6f}")
    print(f"Iterations: {len(history) - 1}")

    return result, history

result, history = calculus_optimization_exercises()
```

This comprehensive guide covers the essential calculus and optimization concepts needed for machine learning, from basic derivatives to advanced stochastic optimization methods. Each section includes mathematical explanations, Python implementations, and practical applications in machine learning.