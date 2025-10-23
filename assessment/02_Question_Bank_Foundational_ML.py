#!/usr/bin/env python3
"""
Question Bank for Foundational Machine Learning Section
Comprehensive assessment questions for all subsections
"""

import json
from typing import List, Dict, Any
from pathlib import Path

class FoundationalMLQuestionBank:
    """Question bank for Section 1: Foundational Machine Learning"""

    def __init__(self):
        self.questions = []
        self.load_questions()

    def load_questions(self):
        """Load all questions for this section"""

        # Mathematical Foundations Questions
        self.questions.extend(self._linear_algebra_questions())
        self.questions.extend(self._calculus_questions())
        self.questions.extend(self._probability_questions())
        self.questions.extend(self._information_theory_questions())
        self.questions.extend(self._numerical_methods_questions())

        # Core Machine Learning Questions
        self.questions.extend(self._supervised_learning_questions())
        self.questions.extend(self._unsupervised_learning_questions())
        self.questions.extend(self._reinforcement_learning_questions())
        self.questions.extend(self._ensemble_methods_questions())

        # Statistical Learning Questions
        self.questions.extend(self._bayesian_methods_questions())
        self.questions.extend(self._causal_inference_questions())

    def _linear_algebra_questions(self) -> List[Dict]:
        """Linear algebra assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Matrix Multiplication Properties",
                "description": "Which of the following properties is NOT true for matrix multiplication?",
                "difficulty": 1,
                "subsection": "linear_algebra",
                "content": {
                    "options": [
                        "Matrix multiplication is associative",
                        "Matrix multiplication is commutative",
                        "Matrix multiplication is distributive over addition",
                        "The identity matrix serves as the multiplicative identity"
                    ]
                },
                "correct_answer": "Matrix multiplication is commutative",
                "points": 10,
                "explanation": "Matrix multiplication is associative (A(BC) = (AB)C) and distributive (A(B+C) = AB + AC), but it is NOT commutative (AB ≠ BA in general).",
                "tags": ["linear_algebra", "matrix_operations", "properties"]
            },
            {
                "question_type": "fill_blank",
                "title": "Eigenvalue Definition",
                "description": "For a square matrix A, if there exists a non-zero vector v such that Av = λv, then λ is called an ______ and v is the corresponding ______.",
                "difficulty": 2,
                "subsection": "linear_algebra",
                "content": {
                    "blanks": ["eigenvalue", "eigenvector"]
                },
                "correct_answer": ["eigenvalue", "eigenvector"],
                "points": 15,
                "explanation": "In linear algebra, eigenvalues and eigenvectors are fundamental concepts where multiplying a matrix by its eigenvector results in a scalar multiple of that vector.",
                "tags": ["linear_algebra", "eigenvalues", "eigenvectors"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Matrix Operations Implementation",
                "description": "Implement basic matrix operations from scratch",
                "difficulty": 2,
                "subsection": "linear_algebra",
                "content": {
                    "description": "Implement functions for matrix addition, multiplication, and transpose without using NumPy",
                    "requirements": [
                        "Matrix addition: add(matrix1, matrix2)",
                        "Matrix multiplication: multiply(matrix1, matrix2)",
                        "Matrix transpose: transpose(matrix)"
                    ],
                    "test_cases": [
                        {"input": {"matrix1": [[1,2],[3,4]], "matrix2": [[5,6],[7,8]]}, "operation": "add", "expected": [[6,8],[10,12]]},
                        {"input": {"matrix1": [[1,2],[3,4]], "matrix2": [[5,6],[7,8]]}, "operation": "multiply", "expected": [[19,22],[43,50]]},
                        {"input": {"matrix": [[1,2,3],[4,5,6]]}, "operation": "transpose", "expected": [[1,4],[2,5],[3,6]]}
                    ]
                },
                "correct_answer": "# Complete implementation",
                "points": 25,
                "tags": ["linear_algebra", "matrix_operations", "programming"]
            },
            {
                "question_type": "true_false",
                "title": "Singular Value Decomposition",
                "description": "SVD (Singular Value Decomposition) can be applied to any matrix, not just square matrices.",
                "difficulty": 3,
                "subsection": "linear_algebra",
                "content": {},
                "correct_answer": True,
                "points": 10,
                "explanation": "Unlike eigenvalue decomposition which only works for square matrices, SVD can decompose any m×n matrix into UΣV^T.",
                "tags": ["linear_algebra", "svd", "matrix_decomposition"]
            },
            {
                "question_type": "essay",
                "title": "Linear Algebra in Machine Learning",
                "description": "Explain the importance of linear algebra in machine learning, providing specific examples of algorithms and techniques that rely heavily on linear algebra concepts.",
                "difficulty": 2,
                "subsection": "linear_algebra",
                "content": {
                    "min_length": 200,
                    "key_concepts": ["eigenvalues", "matrix multiplication", "vector spaces", "orthogonality", "projections"]
                },
                "correct_answer": "Comprehensive explanation",
                "points": 30,
                "explanation": "Linear algebra is fundamental to ML: PCA uses eigenvalues/eigenvectors, neural networks use matrix multiplications, SVMs use dot products and projections, and many optimization algorithms rely on vector space properties.",
                "tags": ["linear_algebra", "machine_learning", "applications"]
            }
        ]

    def _calculus_questions(self) -> List[Dict]:
        """Calculus and optimization assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Gradient Descent Properties",
                "description": "What happens to the learning rate in gradient descent if it's set too high?",
                "difficulty": 1,
                "subsection": "calculus_optimization",
                "content": {
                    "options": [
                        "The algorithm converges faster",
                        "The algorithm may diverge or oscillate",
                        "The algorithm finds better local minima",
                        "The algorithm becomes more stable"
                    ]
                },
                "correct_answer": "The algorithm may diverge or oscillate",
                "points": 10,
                "explanation": "A learning rate that's too high can cause the algorithm to overshoot the minimum, leading to divergence or oscillation around the minimum.",
                "tags": ["calculus", "optimization", "gradient_descent"]
            },
            {
                "question_type": "fill_blank",
                "title": "Partial Derivatives",
                "description": "The gradient of a function f(x,y) is a vector containing the _______ derivatives with respect to each variable: ∇f = [∂f/∂x, _______].",
                "difficulty": 2,
                "subsection": "calculus_optimization",
                "content": {
                    "blanks": ["partial", "∂f/∂y"]
                },
                "correct_answer": ["partial", "∂f/∂y"],
                "points": 15,
                "explanation": "The gradient vector contains all partial derivatives of the function, pointing in the direction of steepest ascent.",
                "tags": ["calculus", "partial_derivatives", "gradient"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Gradient Descent Implementation",
                "description": "Implement gradient descent for linear regression",
                "difficulty": 3,
                "subsection": "calculus_optimization",
                "content": {
                    "description": "Implement gradient descent to find optimal parameters for linear regression",
                    "requirements": [
                        "Compute gradients analytically",
                        "Implement learning rate scheduling",
                        "Include convergence checking",
                        "Visualize learning process"
                    ],
                    "test_cases": [
                        {"dataset": "synthetic_linear", "target_mse": 0.01, "max_iterations": 1000},
                        {"dataset": "real_world_data", "target_mse": 0.1, "max_iterations": 5000}
                    ]
                },
                "correct_answer": "Complete implementation with proper gradient computation",
                "points": 30,
                "tags": ["calculus", "optimization", "gradient_descent", "machine_learning"]
            },
            {
                "question_type": "true_false",
                "title": "Convex Optimization",
                "description": "All convex optimization problems have a unique global minimum.",
                "difficulty": 3,
                "subsection": "calculus_optimization",
                "content": {},
                "correct_answer": False,
                "points": 10,
                "explanation": "Convex functions have only one global minimum, but it may not be unique (e.g., constant functions are convex and every point is a global minimum).",
                "tags": ["calculus", "optimization", "convexity"]
            },
            {
                "question_type": "proof",
                "title": "Gradient Descent Convergence",
                "description": "Prove that for a convex function with Lipschitz continuous gradient, gradient descent with appropriate learning rate converges to the global minimum.",
                "difficulty": 4,
                "subsection": "calculus_optimization",
                "content": {
                    "given": [
                        "f is convex",
                        "∇f is L-Lipschitz continuous",
                        "Learning rate α ≤ 1/L"
                    ],
                    "prove": "Gradient descent converges to global minimum"
                },
                "correct_answer": "Mathematical proof using convexity and Lipschitz conditions",
                "points": 35,
                "explanation": "The proof involves showing that the function value decreases at each iteration and converges to the minimum using the properties of convex functions and Lipschitz continuity.",
                "tags": ["calculus", "optimization", "proof", "convergence"]
            }
        ]

    def _probability_questions(self) -> List[Dict]:
        """Probability theory assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Bayes' Theorem Application",
                "description": "A disease affects 1% of the population. A test is 95% accurate for positive results and 90% accurate for negative results. If a person tests positive, what is the probability they actually have the disease?",
                "difficulty": 3,
                "subsection": "probability_theory",
                "content": {
                    "options": [
                        "About 8.8%",
                        "About 50%",
                        "About 95%",
                        "About 1%"
                    ]
                },
                "correct_answer": "About 8.8%",
                "points": 15,
                "explanation": "Using Bayes' theorem: P(Disease|Positive) = [0.01 × 0.95] / [0.01 × 0.95 + 0.99 × 0.10] ≈ 0.088 or 8.8%. This demonstrates base rate neglect.",
                "tags": ["probability", "bayes_theorem", "conditional_probability"]
            },
            {
                "question_type": "fill_blank",
                "title": "Probability Axioms",
                "description": "The three axioms of probability are: 1) P(A) ≥ ______, 2) P(S) = ______, 3) For mutually exclusive events, P(A∪B) = ______.",
                "difficulty": 2,
                "subsection": "probability_theory",
                "content": {
                    "blanks": ["0", "1", "P(A) + P(B)"]
                },
                "correct_answer": ["0", "1", "P(A) + P(B)"],
                "points": 15,
                "explanation": "Kolmogorov's axioms form the foundation of probability theory: non-negativity, normalization, and additivity.",
                "tags": ["probability", "axioms", "foundations"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Monte Carlo Simulation",
                "description": "Implement Monte Carlo integration to estimate π",
                "difficulty": 2,
                "subsection": "probability_theory",
                "content": {
                    "description": "Use Monte Carlo methods to estimate π by randomly sampling points in a unit square",
                    "requirements": [
                        "Generate random points in [0,1] × [0,1]",
                        "Count points inside unit circle",
                        "Estimate π using the ratio",
                        "Visualize convergence"
                    ],
                    "test_cases": [
                        {"samples": 1000, "tolerance": 0.1},
                        {"samples": 100000, "tolerance": 0.01}
                    ]
                },
                "correct_answer": "Complete Monte Carlo implementation",
                "points": 25,
                "tags": ["probability", "monte_carlo", "simulation", "numerical_methods"]
            },
            {
                "question_type": "true_false",
                "title": "Independent Events",
                "description": "If two events are independent, then they are also mutually exclusive.",
                "difficulty": 2,
                "subsection": "probability_theory",
                "content": {},
                "correct_answer": False,
                "points": 10,
                "explanation": "Independent events can occur together (P(A∩B) = P(A)P(B) > 0 if both probabilities are positive), while mutually exclusive events cannot occur simultaneously.",
                "tags": ["probability", "independence", "mutual_exclusivity"]
            },
            {
                "question_type": "essay",
                "title": "Probability in Machine Learning",
                "description": "Explain how probability theory is fundamental to machine learning, focusing on Bayesian methods, uncertainty quantification, and probabilistic models.",
                "difficulty": 3,
                "subsection": "probability_theory",
                "content": {
                    "min_length": 300,
                    "key_concepts": ["bayesian_inference", "uncertainty", "probabilistic_models", "maximum_likelihood", "posterior_distribution"]
                },
                "correct_answer": "Comprehensive explanation",
                "points": 30,
                "explanation": "Probability theory underpins ML through Bayesian inference, uncertainty quantification in predictions, probabilistic graphical models, and the theoretical foundation for many learning algorithms.",
                "tags": ["probability", "machine_learning", "bayesian_methods"]
            }
        ]

    def _information_theory_questions(self) -> List[Dict]:
        """Information theory assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Entropy Interpretation",
                "description": "What does entropy measure in information theory?",
                "difficulty": 2,
                "subsection": "information_theory",
                "content": {
                    "options": [
                        "The maximum possible information",
                        "The average uncertainty or surprise",
                        "The minimum number of bits needed",
                        "The compression ratio"
                    ]
                },
                "correct_answer": "The average uncertainty or surprise",
                "points": 10,
                "explanation": "Entropy measures the average uncertainty or unpredictability in a random variable. Higher entropy means more uncertainty.",
                "tags": ["information_theory", "entropy", "uncertainty"]
            },
            {
                "question_type": "fill_blank",
                "title": "KL Divergence Formula",
                "description": "The Kullback-Leibler divergence between distributions P and Q is defined as D_KL(P||Q) = Σ P(x) log(_______).",
                "difficulty": 3,
                "subsection": "information_theory",
                "content": {
                    "blanks": ["P(x)/Q(x)"]
                },
                "correct_answer": ["P(x)/Q(x)"],
                "points": 15,
                "explanation": "KL divergence measures how one probability distribution diverges from another, always non-negative and asymmetric.",
                "tags": ["information_theory", "kl_divergence", "distance_measures"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Entropy Calculation",
                "description": "Implement functions to calculate entropy, mutual information, and KL divergence",
                "difficulty": 3,
                "subsection": "information_theory",
                "content": {
                    "description": "Implement core information theory measures",
                    "requirements": [
                        "Entropy: H(X) = -Σ P(x) log₂(P(x))",
                        "Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)",
                        "KL Divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))"
                    ],
                    "test_cases": [
                        {"distribution": [0.5, 0.5], "entropy_expected": 1.0},
                        {"joint_dist": [[0.25, 0.25], [0.25, 0.25]], "mi_expected": 0.0}
                    ]
                },
                "correct_answer": "Complete implementation with proper error handling",
                "points": 30,
                "tags": ["information_theory", "entropy", "mutual_information", "programming"]
            },
            {
                "question_type": "true_false",
                "title": "Cross-Entropy Properties",
                "description": "Cross-entropy is always greater than or equal to entropy.",
                "difficulty": 3,
                "subsection": "information_theory",
                "content": {},
                "correct_answer": True,
                "points": 10,
                "explanation": "Cross-entropy H(P,Q) = H(P) + D_KL(P||Q), and since KL divergence is always non-negative, cross-entropy ≥ entropy.",
                "tags": ["information_theory", "cross_entropy", "entropy"]
            },
            {
                "question_type": "essay",
                "title": "Information Theory in ML",
                "description": "Discuss how information theory concepts are applied in machine learning, including feature selection, model evaluation, and deep learning.",
                "difficulty": 3,
                "subsection": "information_theory",
                "content": {
                    "min_length": 250,
                    "key_concepts": ["feature_selection", "cross_entropy_loss", "information_bottleneck", "mutual_information", "kl_divergence"]
                },
                "correct_answer": "Comprehensive explanation",
                "points": 30,
                "explanation": "Information theory is used in feature selection (mutual information), loss functions (cross-entropy), model compression (information bottleneck), and many deep learning applications.",
                "tags": ["information_theory", "machine_learning", "applications"]
            }
        ]

    def _numerical_methods_questions(self) -> List[Dict]:
        """Numerical methods assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Numerical Stability",
                "description": "Which of the following can cause numerical instability in floating-point arithmetic?",
                "difficulty": 2,
                "subsection": "numerical_methods",
                "content": {
                    "options": [
                        "Adding numbers of very different magnitudes",
                        "Subtracting nearly equal numbers",
                        "Division by very small numbers",
                        "All of the above"
                    ]
                },
                "correct_answer": "All of the above",
                "points": 10,
                "explanation": "All these operations can cause numerical instability: adding different magnitudes causes loss of precision, subtracting similar numbers causes catastrophic cancellation, and division by small numbers amplifies errors.",
                "tags": ["numerical_methods", "numerical_stability", "floating_point"]
            },
            {
                "question_type": "fill_blank",
                "title": "Matrix Factorization",
                "description": "LU decomposition factors a matrix A into the product of a ______ triangular matrix L and an ______ triangular matrix U.",
                "difficulty": 2,
                "subsection": "numerical_methods",
                "content": {
                    "blanks": ["lower", "upper"]
                },
                "correct_answer": ["lower", "upper"],
                "points": 10,
                "explanation": "LU decomposition is a fundamental matrix factorization used for solving linear systems and computing determinants efficiently.",
                "tags": ["numerical_methods", "matrix_factorization", "lu_decomposition"]
            },
            {
                "question_type": "coding_challenge",
                "title": "SVD Implementation",
                "description": "Implement Singular Value Decomposition using power iteration method",
                "difficulty": 4,
                "subsection": "numerical_methods",
                "content": {
                    "description": "Implement SVD using iterative methods without external libraries",
                    "requirements": [
                        "Power iteration for dominant singular value/vector",
                        "Deflation for subsequent singular values",
                        "Orthogonalization procedures",
                        "Convergence checking"
                    ],
                    "test_cases": [
                        {"matrix": [[3, 0], [0, 4]], "tolerance": 1e-6},
                        {"matrix": [[1, 2], [3, 4]], "tolerance": 1e-4}
                    ]
                },
                "correct_answer": "Complete SVD implementation",
                "points": 35,
                "tags": ["numerical_methods", "svd", "matrix_factorization", "iterative_methods"]
            },
            {
                "question_type": "true_false",
                "title": "Condition Number",
                "description": "A high condition number indicates that a matrix is well-conditioned for numerical computations.",
                "difficulty": 2,
                "subsection": "numerical_methods",
                "content": {},
                "correct_answer": False,
                "points": 10,
                "explanation": "A high condition number indicates that the matrix is ill-conditioned, meaning small changes in input can cause large changes in output.",
                "tags": ["numerical_methods", "condition_number", "matrix_properties"]
            },
            {
                "question_type": "essay",
                "title": "Numerical Methods in ML",
                "description": "Explain the importance of numerical methods in machine learning, focusing on optimization algorithms, matrix computations, and stability considerations.",
                "difficulty": 3,
                "subsection": "numerical_methods",
                "content": {
                    "min_length": 250,
                    "key_concepts": ["optimization", "matrix_computations", "numerical_stability", "convergence", "computational_efficiency"]
                },
                "correct_answer": "Comprehensive explanation",
                "points": 25,
                "explanation": "Numerical methods are essential for implementing ML algorithms efficiently and stably, from gradient descent optimization to large-scale matrix operations.",
                "tags": ["numerical_methods", "machine_learning", "computational_efficiency"]
            }
        ]

    def _supervised_learning_questions(self) -> List[Dict]:
        """Supervised learning assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Bias-Variance Tradeoff",
                "description": "Which of the following best describes the bias-variance tradeoff?",
                "difficulty": 2,
                "subsection": "supervised_learning",
                "content": {
                    "options": [
                        "Increasing model complexity always improves performance",
                        "Decreasing bias typically increases variance",
                        "Variance and bias are independent of each other",
                        "The tradeoff only applies to neural networks"
                    ]
                },
                "correct_answer": "Decreasing bias typically increases variance",
                "points": 10,
                "explanation": "The bias-variance tradeoff states that as we decrease bias (by increasing model complexity), we typically increase variance (sensitivity to training data).",
                "tags": ["supervised_learning", "bias_variance", "model_complexity"]
            },
            {
                "question_type": "fill_blank",
                "title": "Cross-Validation",
                "description": "In k-fold cross-validation, the dataset is divided into ______ folds, and the model is trained ______ times.",
                "difficulty": 2,
                "subsection": "supervised_learning",
                "content": {
                    "blanks": ["k", "k"]
                },
                "correct_answer": ["k", "k"],
                "points": 10,
                "explanation": "K-fold cross-validation divides data into k subsets, using each subset once as validation while training on the remaining k-1 subsets.",
                "tags": ["supervised_learning", "cross_validation", "model_evaluation"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Decision Tree Implementation",
                "description": "Implement a decision tree classifier from scratch",
                "difficulty": 4,
                "subsection": "supervised_learning",
                "content": {
                    "description": "Implement decision tree with entropy-based splitting and pruning",
                    "requirements": [
                        "Entropy and information gain calculation",
                        "Tree construction with stopping criteria",
                        "Prediction functionality",
                        "Optional: Post-pruning implementation"
                    ],
                    "test_cases": [
                        {"dataset": "iris", "accuracy_target": 0.9},
                        {"dataset": "breast_cancer", "accuracy_target": 0.85}
                    ]
                },
                "correct_answer": "Complete decision tree implementation",
                "points": 40,
                "tags": ["supervised_learning", "decision_trees", "classification", "programming"]
            },
            {
                "question_type": "true_false",
                "title": "Overfitting",
                "description": "A model that achieves 100% accuracy on training data is always better than one with 95% accuracy.",
                "difficulty": 1,
                "subsection": "supervised_learning",
                "content": {},
                "correct_answer": False,
                "points": 10,
                "explanation": "Perfect training accuracy often indicates overfitting, where the model has memorized training data rather than learned generalizable patterns.",
                "tags": ["supervised_learning", "overfitting", "generalization"]
            },
            {
                "question_type": "case_study",
                "title": "Real-World Classification",
                "description": "Design a supervised learning solution for predicting customer churn in a telecommunications company",
                "difficulty": 3,
                "subsection": "supervised_learning",
                "content": {
                    "scenario": "Telecom company wants to predict which customers are likely to cancel their service",
                    "data": "Customer usage patterns, demographics, service plan, complaints, payment history",
                    "requirements": [
                        "Feature engineering recommendations",
                        "Model selection and justification",
                        "Evaluation metrics appropriate for business context",
                        "Deployment considerations"
                    ]
                },
                "correct_answer": "Comprehensive solution design",
                "points": 35,
                "explanation": "A complete solution should address data preprocessing, feature selection, model choice (e.g., random forest for interpretability), evaluation metrics (precision/recall balance), and practical deployment.",
                "tags": ["supervised_learning", "classification", "business_application", "case_study"]
            }
        ]

    def _unsupervised_learning_questions(self) -> List[Dict]:
        """Unsupervised learning assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "K-means Properties",
                "description": "What is a key limitation of the k-means algorithm?",
                "difficulty": 2,
                "subsection": "unsupervised_learning",
                "content": {
                    "options": [
                        "It always converges to the global optimum",
                        "It can only handle numerical data",
                        "It requires specifying the number of clusters in advance",
                        "It works equally well with all distance metrics"
                    ]
                },
                "correct_answer": "It requires specifying the number of clusters in advance",
                "points": 10,
                "explanation": "K-means requires the number of clusters k to be specified beforehand, which is a significant limitation when the optimal number is unknown.",
                "tags": ["unsupervised_learning", "clustering", "k_means"]
            },
            {
                "question_type": "fill_blank",
                "title": "PCA Components",
                "description": "Principal Component Analysis finds ______ that capture the ______ variance in the data.",
                "difficulty": 2,
                "subsection": "unsupervised_learning",
                "content": {
                    "blanks": ["orthogonal_components", "maximum"]
                },
                "correct_answer": ["orthogonal_components", "maximum"],
                "points": 10,
                "explanation": "PCA finds orthogonal principal components that capture decreasing amounts of variance in the data.",
                "tags": ["unsupervised_learning", "dimensionality_reduction", "pca"]
            },
            {
                "question_type": "coding_challenge",
                "title": "K-means Implementation",
                "description": "Implement k-means clustering algorithm from scratch",
                "difficulty": 3,
                "subsection": "unsupervised_learning",
                "content": {
                    "description": "Implement k-means with proper initialization and convergence",
                    "requirements": [
                        "K-means++ initialization",
                        "Distance calculation and assignment",
                        "Centroid updates",
                        "Convergence checking",
                        "Visualization of results"
                    ],
                    "test_cases": [
                        {"data": "2d_gaussian_clusters", "k": 3, "iterations": 50},
                        {"data": "iris_dataset", "k": 3, "expected_accuracy": 0.8}
                    ]
                },
                "correct_answer": "Complete k-means implementation",
                "points": 30,
                "tags": ["unsupervised_learning", "clustering", "k_means", "programming"]
            },
            {
                "question_type": "true_false",
                "title": "Hierarchical Clustering",
                "description": "Hierarchical clustering produces a fixed number of clusters.",
                "difficulty": 2,
                "subsection": "unsupervised_learning",
                "content": {},
                "correct_answer": False,
                "points": 10,
                "explanation": "Hierarchical clustering produces a tree-like structure (dendrogram) that can be cut at different levels to obtain different numbers of clusters.",
                "tags": ["unsupervised_learning", "clustering", "hierarchical_clustering"]
            },
            {
                "question_type": "essay",
                "title": "Dimensionality Reduction Applications",
                "description": "Discuss the applications and importance of dimensionality reduction in machine learning, including visualization, computational efficiency, and feature extraction.",
                "difficulty": 3,
                "subsection": "unsupervised_learning",
                "content": {
                    "min_length": 250,
                    "key_concepts": ["visualization", "computational_efficiency", "feature_extraction", "noise_reduction", "curse_of_dimensionality"]
                },
                "correct_answer": "Comprehensive explanation",
                "points": 25,
                "explanation": "Dimensionality reduction is crucial for visualization (2D/3D projection), computational efficiency (reduced feature space), noise reduction, and addressing the curse of dimensionality.",
                "tags": ["unsupervised_learning", "dimensionality_reduction", "applications"]
            }
        ]

    def _reinforcement_learning_questions(self) -> List[Dict]:
        """Reinforcement learning assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Q-learning Convergence",
                "description": "Under what conditions does Q-learning converge to the optimal Q-function?",
                "difficulty": 3,
                "subsection": "reinforcement_learning",
                "content": {
                    "options": [
                        "Always, regardless of exploration strategy",
                        "When using ε-greedy exploration with decreasing ε",
                        "When all state-action pairs are visited infinitely often",
                        "Only in deterministic environments"
                    ]
                },
                "correct_answer": "When all state-action pairs are visited infinitely often",
                "points": 15,
                "explanation": "Q-learning converges to the optimal Q-function under the condition that all state-action pairs are visited infinitely often and the learning rate satisfies certain conditions.",
                "tags": ["reinforcement_learning", "q_learning", "convergence"]
            },
            {
                "question_type": "fill_blank",
                "title": "Bellman Equation",
                "description": "The Bellman equation for the optimal value function is: V*(s) = max_a [R(s,a) + γ ______ V*(s')].",
                "difficulty": 3,
                "subsection": "reinforcement_learning",
                "content": {
                    "blanks": ["E"]
                },
                "correct_answer": ["E"],
                "points": 15,
                "explanation": "The Bellman equation expresses the optimal value of a state as the maximum expected return over all possible actions, considering future rewards.",
                "tags": ["reinforcement_learning", "bellman_equation", "dynamic_programming"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Q-learning Agent",
                "description": "Implement a Q-learning agent for a grid world environment",
                "difficulty": 4,
                "subsection": "reinforcement_learning",
                "content": {
                    "description": "Implement Q-learning with experience replay and target network",
                    "requirements": [
                        "Environment representation (grid world)",
                        "Q-table implementation and updates",
                        "ε-greedy exploration strategy",
                        "Experience replay buffer",
                        "Convergence monitoring"
                    ],
                    "test_cases": [
                        {"environment": "4x4_grid", "goal_reach_rate": 0.95, "max_episodes": 1000},
                        {"environment": "8x8_grid", "goal_reach_rate": 0.85, "max_episodes": 2000}
                    ]
                },
                "correct_answer": "Complete Q-learning implementation",
                "points": 40,
                "tags": ["reinforcement_learning", "q_learning", "programming", "grid_world"]
            },
            {
                "question_type": "true_false",
                "title": "Policy Gradient Methods",
                "description": "Policy gradient methods can handle both discrete and continuous action spaces.",
                "difficulty": 2,
                "subsection": "reinforcement_learning",
                "content": {},
                "correct_answer": True,
                "points": 10,
                "explanation": "Policy gradient methods are flexible and can handle both discrete and continuous action spaces, unlike value-based methods like Q-learning.",
                "tags": ["reinforcement_learning", "policy_gradient", "continuous_actions"]
            },
            {
                "question_type": "essay",
                "title": "RL Challenges and Solutions",
                "description": "Discuss the main challenges in reinforcement learning (exploration, credit assignment, sample efficiency) and modern approaches to address them.",
                "difficulty": 4,
                "subsection": "reinforcement_learning",
                "content": {
                    "min_length": 300,
                    "key_concepts": ["exploration", "credit_assignment", "sample_efficiency", "function_approximation", "reward_shaping"]
                },
                "correct_answer": "Comprehensive analysis",
                "points": 35,
                "explanation": "Key challenges include the exploration-exploitation tradeoff, temporal credit assignment, and poor sample efficiency. Modern solutions include intrinsic motivation, hierarchical RL, and deep reinforcement learning.",
                "tags": ["reinforcement_learning", "challenges", "research_directions"]
            }
        ]

    def _ensemble_methods_questions(self) -> List[Dict]:
        """Ensemble methods assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Random Forest Characteristics",
                "description": "What is the main idea behind random forests?",
                "difficulty": 2,
                "subsection": "ensemble_methods",
                "content": {
                    "options": [
                        "Using the same decision tree multiple times",
                        "Combining different types of models",
                        "Building many decision trees on random subsets of data and features",
                        "Training a single very deep tree"
                    ]
                },
                "correct_answer": "Building many decision trees on random subsets of data and features",
                "points": 10,
                "explanation": "Random forests create an ensemble by building multiple decision trees, each trained on a bootstrap sample of the data and using a random subset of features.",
                "tags": ["ensemble_methods", "random_forest", "bagging"]
            },
            {
                "question_type": "fill_blank",
                "title": "Boosting Algorithm",
                "description": "In AdaBoost, subsequent models focus more on ______ classified samples from previous models.",
                "difficulty": 2,
                "subsection": "ensemble_methods",
                "content": {
                    "blanks": ["incorrectly"]
                },
                "correct_answer": ["incorrectly"],
                "points": 10,
                "explanation": "Boosting algorithms like AdaBoost sequentially train models, with each new model focusing on correcting errors made by previous models.",
                "tags": ["ensemble_methods", "boosting", "adaboost"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Gradient Boosting Implementation",
                "description": "Implement gradient boosting for regression from scratch",
                "difficulty": 5,
                "subsection": "ensemble_methods",
                "content": {
                    "description": "Implement gradient boosting with decision tree weak learners",
                    "requirements": [
                        "Gradient calculation for different loss functions",
                        "Decision tree as weak learner",
                        "Gradient descent in function space",
                        "Regularization (learning rate, tree constraints)",
                        "Early stopping criteria"
                    ],
                    "test_cases": [
                        {"dataset": "boston_housing", "target_rmse": 3.0},
                        {"dataset": "california_housing", "target_rmse": 0.6}
                    ]
                },
                "correct_answer": "Complete gradient boosting implementation",
                "points": 45,
                "tags": ["ensemble_methods", "gradient_boosting", "programming", "advanced"]
            },
            {
                "question_type": "true_false",
                "title": "Bagging vs Boosting",
                "description": "Bagging and boosting both aim to reduce variance in the ensemble.",
                "difficulty": 3,
                "subsection": "ensemble_methods",
                "content": {},
                "correct_answer": False,
                "points": 10,
                "explanation": "Bagging primarily reduces variance, while boosting primarily reduces bias. They have different goals and mechanisms.",
                "tags": ["ensemble_methods", "bagging", "boosting", "bias_variance"]
            },
            {
                "question_type": "essay",
                "title": "Ensemble Methods in Practice",
                "description": "Compare and contrast different ensemble methods (bagging, boosting, stacking) and discuss their strengths, weaknesses, and appropriate use cases.",
                "difficulty": 3,
                "subsection": "ensemble_methods",
                "content": {
                    "min_length": 300,
                    "key_concepts": ["bagging", "boosting", "stacking", "variance_reduction", "bias_reduction", "model_diversity"]
                },
                "correct_answer": "Comprehensive comparison",
                "points": 30,
                "explanation": "Each ensemble method has distinct characteristics: bagging reduces variance through parallel training, boosting reduces bias through sequential correction, and stacking combines diverse models.",
                "tags": ["ensemble_methods", "comparison", "practical_applications"]
            }
        ]

    def _bayesian_methods_questions(self) -> List[Dict]:
        """Bayesian methods assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Bayesian Inference",
                "description": "What is the relationship between prior, likelihood, and posterior in Bayesian inference?",
                "difficulty": 2,
                "subsection": "bayesian_methods",
                "content": {
                    "options": [
                        "Posterior = Prior × Likelihood",
                        "Posterior = Prior + Likelihood",
                        "Posterior ∝ Prior × Likelihood",
                        "Posterior = Likelihood / Prior"
                    ]
                },
                "correct_answer": "Posterior ∝ Prior × Likelihood",
                "points": 10,
                "explanation": "Bayes' theorem states that the posterior is proportional to the product of the prior and likelihood, normalized by the evidence.",
                "tags": ["bayesian_methods", "bayes_theorem", "inference"]
            },
            {
                "question_type": "fill_blank",
                "title": "Markov Chain Monte Carlo",
                "description": "MCMC methods are used to ______ samples from complex probability distributions, particularly for ______ inference.",
                "difficulty": 3,
                "subsection": "bayesian_methods",
                "content": {
                    "blanks": ["generate", "bayesian"]
                },
                "correct_answer": ["generate", "bayesian"],
                "points": 15,
                "explanation": "MCMC methods generate samples from posterior distributions when analytical solutions are intractable, making Bayesian inference practical for complex models.",
                "tags": ["bayesian_methods", "mcmc", "sampling"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Bayesian Linear Regression",
                "description": "Implement Bayesian linear regression with conjugate priors",
                "difficulty": 4,
                "subsection": "bayesian_methods",
                "content": {
                    "description": "Implement Bayesian linear regression with Gaussian priors",
                    "requirements": [
                        "Prior distribution specification",
                        "Posterior distribution computation",
                        "Predictive distribution",
                        "Uncertainty quantification",
                        "Visualization of confidence intervals"
                    ],
                    "test_cases": [
                        {"data": "synthetic_linear", "coverage_target": 0.95},
                        {"data": "real_dataset", "coverage_target": 0.9}
                    ]
                },
                "correct_answer": "Complete Bayesian linear regression implementation",
                "points": 35,
                "tags": ["bayesian_methods", "linear_regression", "uncertainty", "programming"]
            },
            {
                "question_type": "true_false",
                "title": "Conjugate Priors",
                "description": "Using conjugate priors guarantees that the posterior distribution will be in the same family as the prior.",
                "difficulty": 3,
                "subsection": "bayesian_methods",
                "content": {},
                "correct_answer": True,
                "points": 10,
                "explanation": "Conjugate priors are specifically chosen so that the posterior distribution belongs to the same family as the prior, making analytical solutions possible.",
                "tags": ["bayesian_methods", "conjugate_priors", "analytical_solutions"]
            },
            {
                "question_type": "essay",
                "title": "Bayesian vs Frequentist Approaches",
                "description": "Compare Bayesian and frequentist approaches to statistical inference, discussing their philosophical differences, practical applications, and strengths/weaknesses.",
                "difficulty": 4,
                "subsection": "bayesian_methods",
                "content": {
                    "min_length": 350,
                    "key_concepts": ["prior_beliefs", "uncertainty_quantification", "interpretability", "computational_complexity", "small_sample"]
                },
                "correct_answer": "Comprehensive comparison",
                "points": 35,
                "explanation": "Bayesian methods incorporate prior knowledge and provide probabilistic uncertainty estimates, while frequentist methods rely solely on data and use confidence intervals.",
                "tags": ["bayesian_methods", "frequentist", "statistical_inference", "philosophy"]
            }
        ]

    def _causal_inference_questions(self) -> List[Dict]:
        """Causal inference assessment questions"""

        return [
            {
                "question_type": "multiple_choice",
                "title": "Causal vs Correlational",
                "description": "Which of the following best distinguishes causal inference from traditional statistical analysis?",
                "difficulty": 3,
                "subsection": "causal_inference",
                "content": {
                    "options": [
                        "Causal inference uses more complex mathematics",
                        "Causal inference focuses on counterfactual reasoning",
                        "Causal inference requires larger datasets",
                        "Causal inference only works with experimental data"
                    ]
                },
                "correct_answer": "Causal inference focuses on counterfactual reasoning",
                "points": 15,
                "explanation": "Causal inference is fundamentally about understanding what would happen under different interventions (counterfactuals), not just describing correlations.",
                "tags": ["causal_inference", "counterfactuals", "philosophy"]
            },
            {
                "question_type": "fill_blank",
                "title": "Potential Outcomes Framework",
                "description": "In the potential outcomes framework, the causal effect for an individual is defined as the difference between their ______ outcome and ______ outcome.",
                "difficulty": 3,
                "subsection": "causal_inference",
                "content": {
                    "blanks": ["treatment", "control"]
                },
                "correct_answer": ["treatment", "control"],
                "points": 15,
                "explanation": "The potential outcomes framework defines causal effects as the difference between what would happen with treatment versus without treatment for the same individual.",
                "tags": ["causal_inference", "potential_outcomes", "rubin_causal_model"]
            },
            {
                "question_type": "coding_challenge",
                "title": "Propensity Score Matching",
                "description": "Implement propensity score matching for causal effect estimation",
                "difficulty": 4,
                "subsection": "causal_inference",
                "content": {
                    "description": "Implement propensity score matching to estimate treatment effects",
                    "requirements": [
                        "Propensity score estimation (logistic regression)",
                        "Matching algorithm (nearest neighbor, caliper)",
                        "Balance checking and diagnostics",
                        "Treatment effect estimation",
                        "Sensitivity analysis"
                    ],
                    "test_cases": [
                        {"data": "observational_study", "treatment_effect_known": "true", "estimation_error_threshold": 0.1},
                        {"data": "real_world_data", "balance_requirements": "standardized_differences < 0.1"}
                    ]
                },
                "correct_answer": "Complete propensity score matching implementation",
                "points": 40,
                "tags": ["causal_inference", "propensity_scores", "matching", "programming"]
            },
            {
                "question_type": "true_false",
                "title": "Instrumental Variables",
                "description": "An instrumental variable must be correlated with the treatment but uncorrelated with the outcome except through the treatment.",
                "difficulty": 3,
                "subsection": "causal_inference",
                "content": {},
                "correct_answer": True,
                "points": 10,
                "explanation": "An instrumental variable must satisfy two key conditions: relevance (correlated with treatment) and exogeneity (uncorrelated with outcome except through treatment).",
                "tags": ["causal_inference", "instrumental_variables", "endogeneity"]
            },
            {
                "question_type": "essay",
                "title": "Causal Inference Applications",
                "description": "Discuss the applications and challenges of causal inference in machine learning, particularly in areas like healthcare, policy evaluation, and recommendation systems.",
                "difficulty": 4,
                "subsection": "causal_inference",
                "content": {
                    "min_length": 350,
                    "key_concepts": ["healthcare", "policy_evaluation", "recommendation_systems", "observational_data", "confounding"]
                },
                "correct_answer": "Comprehensive discussion",
                "points": 35,
                "explanation": "Causal inference is crucial in healthcare (treatment effects), policy evaluation (program impact), and recommendation systems (understanding intervention effects beyond correlation).",
                "tags": ["causal_inference", "applications", "machine_learning", "real_world"]
            }
        ]

    def save_questions(self, filepath: str):
        """Save questions to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.questions, f, indent=2)

    def load_questions_from_file(self, filepath: str):
        """Load questions from JSON file"""
        with open(filepath, 'r') as f:
            self.questions = json.load(f)

    def get_questions_by_difficulty(self, difficulty: int) -> List[Dict]:
        """Get questions filtered by difficulty level"""
        return [q for q in self.questions if q['difficulty'] == difficulty]

    def get_questions_by_subsection(self, subsection: str) -> List[Dict]:
        """Get questions filtered by subsection"""
        return [q for q in self.questions if q['subsection'] == subsection]

    def get_questions_by_type(self, question_type: str) -> List[Dict]:
        """Get questions filtered by question type"""
        return [q for q in self.questions if q['question_type'] == question_type]

if __name__ == "__main__":
    # Create and test the question bank
    question_bank = FoundationalMLQuestionBank()

    print(f"Loaded {len(question_bank.questions)} questions for Foundational ML section")

    # Save questions to file
    question_bank.save_questions("assessment_data/foundational_ml_questions.json")

    # Print some statistics
    print("\nQuestion Statistics:")
    print(f"Total questions: {len(question_bank.questions)}")

    subsections = set(q['subsection'] for q in question_bank.questions)
    print(f"Subsections covered: {len(subsections)}")

    question_types = set(q['question_type'] for q in question_bank.questions)
    print(f"Question types: {question_types}")

    difficulties = [q['difficulty'] for q in question_bank.questions]
    print(f"Difficulty range: {min(difficulties)} to {max(difficulties)}")

    # Sample question
    sample = question_bank.questions[0]
    print(f"\nSample Question: {sample['title']}")
    print(f"Type: {sample['question_type']}")
    print(f"Difficulty: {sample['difficulty']}")
    print(f"Subsection: {sample['subsection']}")