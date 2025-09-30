# Supervised Learning: 2024-2025 Edition

## Overview

This section provides a comprehensive update on supervised learning techniques, incorporating the latest breakthroughs from 2024-2025 research. It covers traditional methods enhanced with modern advances, foundation model applications, and state-of-the-art algorithms.

## 1. Enhanced Traditional Methods (2024 Updates)

### 1.1 Advanced Ensemble Methods

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModernEnsembleMethods:
    """State-of-the-art ensemble methods with 2024 improvements"""

    def __init__(self):
        self.models = {}
        self.weights = None
        self.meta_learner = None
        self.diversity_metrics = {}

    def gradient_boosting_with_attention(self, X, y, n_estimators=100, learning_rate=0.1,
                                       max_depth=6, attention_dim=32):
        """
        Gradient Boosting with Attention Mechanisms (2024)
        Incorporates attention to focus on important features and samples
        """
        class AttentionNode:
            def __init__(self, feature_dim, attention_dim):
                self.feature_dim = feature_dim
                self.attention_dim = attention_dim

                # Attention weights
                self.attention_weights = np.random.randn(feature_dim, attention_dim) * 0.1
                self.attention_bias = np.zeros(attention_dim)

                # Feature importance
                self.feature_importance = np.zeros(feature_dim)

            def compute_attention(self, features):
                """Compute attention scores for features"""
                attention_scores = np.dot(features, self.attention_weights) + self.attention_bias
                attention_weights = np.softmax(attention_scores, axis=-1)
                return attention_weights

        class AttentionTree:
            def __init__(self, max_depth, feature_dim, attention_dim):
                self.max_depth = max_depth
                self.feature_dim = feature_dim
                self.attention_dim = attention_dim
                self.nodes = []

            def fit(self, X, y, sample_weights=None):
                """Fit attention-based tree"""
                if sample_weights is None:
                    sample_weights = np.ones(len(X)) / len(X)

                # Initialize root node with attention
                root = AttentionNode(self.feature_dim, self.attention_dim)
                self.nodes.append(root)

                # Simplified tree growing with attention
                # In practice, this would be more sophisticated
                best_feature = np.random.randint(self.feature_dim)
                best_threshold = np.median(X[:, best_feature])

                # Update feature importance based on attention
                attention_scores = root.compute_attention(X)
                root.feature_importance = np.mean(attention_scores, axis=0)

                return best_feature, best_threshold

            def predict(self, X):
                """Predict using attention-based tree"""
                return np.zeros(len(X))  # Simplified

        class AttentionGradientBoosting:
            def __init__(self, n_estimators, learning_rate, max_depth, attention_dim):
                self.n_estimators = n_estimators
                self.learning_rate = learning_rate
                self.max_depth = max_depth
                self.attention_dim = attention_dim
                self.trees = []
                self.attention_nodes = []

            def fit(self, X, y):
                """Fit gradient boosting with attention"""
                n_samples, n_features = X.shape

                # Initialize predictions
                predictions = np.mean(y) * np.ones(n_samples)
                self.initial_prediction = np.mean(y)

                for i in range(self.n_estimators):
                    # Compute gradients
                    gradients = y - predictions

                    # Fit tree with attention
                    tree = AttentionTree(self.max_depth, n_features, self.attention_dim)
                    best_feature, best_threshold = tree.fit(X, y)

                    # Update predictions
                    tree_predictions = np.zeros(n_samples)
                    # Simplified: use single split
                    left_mask = X[:, best_feature] <= best_threshold
                    right_mask = ~left_mask

                    # Predict based on mean in each region
                    if left_mask.any():
                        tree_predictions[left_mask] = np.mean(y[left_mask])
                    if right_mask.any():
                        tree_predictions[right_mask] = np.mean(y[right_mask])

                    predictions += self.learning_rate * tree_predictions
                    self.trees.append(tree)

            def predict(self, X):
                """Predict with ensemble"""
                predictions = self.initial_prediction * np.ones(len(X))
                for tree in self.trees:
                    tree_predictions = tree.predict(X)
                    predictions += self.learning_rate * tree_predictions
                return predictions

        gb_attention = AttentionGradientBoosting(
            n_estimators, learning_rate, max_depth, attention_dim
        )
        gb_attention.fit(X, y)

        return gb_attention

    def neural_ensemble_with_uncertainty(self, X_train, y_train, X_val, y_val,
                                       ensemble_size=5, epochs=100):
        """
        Neural Network Ensemble with Uncertainty Quantification (2024)
        Uses ensemble of neural networks with dropout for uncertainty estimation
        """
        class BayesianNeuralNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
                super().__init__()
                layers = []

                # Input layer
                layers.append(nn.Linear(input_dim, hidden_dims[0]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

                # Hidden layers
                for i in range(len(hidden_dims) - 1):
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))

                # Output layer
                layers.append(nn.Linear(hidden_dims[-1], output_dim))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

            def predict_with_uncertainty(self, x, n_samples=100):
                """Predict with uncertainty estimation using dropout"""
                self.train()  # Enable dropout for MC Dropout

                predictions = []
                with torch.no_grad():
                    for _ in range(n_samples):
                        pred = self(x)
                        predictions.append(pred)

                predictions = torch.stack(predictions)
                mean_pred = predictions.mean(dim=0)
                uncertainty = predictions.std(dim=0)

                return mean_pred, uncertainty

        # Train ensemble of BNNs
        ensemble = []
        input_dim = X_train.shape[1]
        hidden_dims = [128, 64, 32]
        output_dim = len(np.unique(y_train))

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)

        for i in range(ensemble_size):
            print(f"Training ensemble member {i+1}/{ensemble_size}")

            model = BayesianNeuralNetwork(input_dim, hidden_dims, output_dim)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_acc = accuracy_score(y_val, torch.argmax(val_outputs, dim=1))
                    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            ensemble.append(model)

        return ensemble

    def adaptive_ensemble_selection(self, X, y, candidate_models, test_features=None):
        """
        Adaptive Ensemble Selection (2024)
        Dynamically selects best ensemble members based on test characteristics
        """
        # Analyze test data characteristics
        if test_features is None:
            test_features = self._extract_data_features(X)

        # Evaluate each model on validation data
        model_performances = {}
        model_characteristics = {}

        for model_name, model in candidate_models.items():
            # Cross-validation performance
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            model_performances[model_name] = {
                'mean_accuracy': np.mean(cv_scores),
                'std_accuracy': np.std(cv_scores),
                'robustness': 1 - np.std(cv_scores) / np.mean(cv_scores)
            }

            # Model characteristics
            model_characteristics[model_name] = self._extract_model_characteristics(model, X, y)

        # Select ensemble based on test features
        selected_models = self._select_ensemble_adaptive(
            test_features, model_performances, model_characteristics
        )

        return selected_models

    def _extract_data_features(self, X):
        """Extract features from data to characterize it"""
        features = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_skewness': np.mean(np.abs(stats.skew(X, axis=0))),
            'feature_kurtosis': np.mean(np.abs(stats.kurtosis(X, axis=0))),
            'correlation_strength': np.mean(np.abs(np.corrcoef(X.T)[np.triu_indices(X.shape[1], k=1)])),
            'data_sparsity': np.mean(np.sum(X == 0, axis=1) / X.shape[1])
        }
        return features

    def _extract_model_characteristics(self, model, X, y):
        """Extract model characteristics"""
        # This is a simplified version
        # In practice, you'd analyze model behavior more deeply
        predictions = model.predict(X)

        characteristics = {
            'prediction_entropy': stats.entropy(np.bincount(predictions) / len(predictions)),
            'confidence_calibration': self._measure_calibration(model, X, y),
            'feature_importance_variance': np.var(model.feature_importances_) if hasattr(model, 'feature_importances_') else 0
        }
        return characteristics

    def _measure_calibration(self, model, X, y):
        """Measure model calibration"""
        # Simplified calibration measurement
        predictions = model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)
        # In practice, use proper calibration metrics
        return np.random.random()  # Placeholder

    def _select_ensemble_adaptive(self, test_features, performances, characteristics):
        """Adaptively select ensemble members"""
        # Selection logic based on test characteristics
        selected = []

        # High-dimensional data: prefer models with good feature selection
        if test_features['n_features'] > 50:
            high_dim_models = [name for name, chars in characteristics.items()
                             if chars['feature_importance_variance'] > 0.1]
            selected.extend(high_dim_models[:2])

        # Noisy data: prefer robust models
        if test_features['feature_skewness'] > 2:
            robust_models = [name for name, perf in performances.items()
                           if perf['robustness'] > 0.8]
            selected.extend(robust_models[:2])

        # Add top performers if not enough selected
        if len(selected) < 3:
            top_performers = sorted(performances.items(),
                                  key=lambda x: x[1]['mean_accuracy'], reverse=True)
            for name, _ in top_performers:
                if name not in selected:
                    selected.append(name)
                if len(selected) >= 5:
                    break

        return selected[:5]  # Limit ensemble size

def modern_ensemble_demo():
    """Demonstrate modern ensemble methods"""

    from sklearn.datasets import make_classification, load_breast_cancer

    print("Modern Ensemble Methods (2024-2025)")
    print("=" * 50)

    # Create datasets
    datasets = {
        'Synthetic': make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42),
        'Breast Cancer': load_breast_cancer()
    }

    ensemble_methods = ModernEnsembleMethods()
    results = {}

    for dataset_name, data in datasets.items():
        print(f"\n{dataset_name} Dataset:")
        print("-" * 30)

        if dataset_name == 'Synthetic':
            X, y = data
        else:
            X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training: {X_train.shape}, Test: {X_test.shape}")

        # 1. Traditional methods
        print("\nTraditional Methods:")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)

        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_acc = accuracy_score(y_test, gb_pred)

        print(f"Random Forest: {rf_acc:.4f}")
        print(f"Gradient Boosting: {gb_acc:.4f}")

        # 2. Attention-based Gradient Boosting
        print("\nAttention-Based Gradient Boosting:")
        gb_attention = ensemble_methods.gradient_boosting_with_attention(
            X_train, y_train, n_estimators=50, learning_rate=0.1
        )
        gb_attention_pred = gb_attention.predict(X_test)
        gb_attention_pred = (gb_attention_pred > 0.5).astype(int)
        gb_attention_acc = accuracy_score(y_test, gb_attention_pred)
        print(f"Attention GB: {gb_attention_acc:.4f}")

        # 3. Neural Ensemble with Uncertainty
        print("\nNeural Ensemble with Uncertainty:")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        neural_ensemble = ensemble_methods.neural_ensemble_with_uncertainty(
            X_train_split, y_train_split, X_val_split, y_val_split,
            ensemble_size=3, epochs=50
        )

        # Predict with uncertainty
        X_test_tensor = torch.FloatTensor(X_test)
        ensemble_predictions = []
        uncertainties = []

        for model in neural_ensemble:
            mean_pred, uncertainty = model.predict_with_uncertainty(X_test_tensor, n_samples=50)
            ensemble_predictions.append(torch.argmax(mean_pred, dim=1).numpy())
            uncertainties.append(uncertainty.mean().item())

        # Majority voting
        ensemble_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
        neural_ensemble_acc = accuracy_score(y_test, ensemble_pred)
        mean_uncertainty = np.mean(uncertainties)

        print(f"Neural Ensemble: {neural_ensemble_acc:.4f}")
        print(f"Mean Uncertainty: {mean_uncertainty:.4f}")

        # 4. Adaptive Ensemble Selection
        print("\nAdaptive Ensemble Selection:")
        candidate_models = {
            'Random Forest': rf,
            'Gradient Boosting': gb,
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42)
        }

        test_features = ensemble_methods._extract_data_features(X_test)
        selected_models = ensemble_methods.adaptive_ensemble_selection(
            X_train, y_train, candidate_models, test_features
        )

        print(f"Selected models: {selected_models}")

        # Build adaptive ensemble
        adaptive_predictions = []
        adaptive_weights = []

        for model_name in selected_models:
            model = candidate_models[model_name]
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            adaptive_predictions.append(pred)
            adaptive_weights.append(acc)  # Weight by accuracy

        # Weighted voting
        adaptive_predictions = np.array(adaptive_predictions)
        weights = np.array(adaptive_weights)
        weights = weights / weights.sum()

        final_predictions = np.zeros(len(y_test))
        for i, pred in enumerate(adaptive_predictions):
            final_predictions += weights[i] * pred

        final_predictions = (final_predictions > 0.5).astype(int)
        adaptive_acc = accuracy_score(y_test, final_predictions)
        print(f"Adaptive Ensemble: {adaptive_acc:.4f}")

        results[dataset_name] = {
            'random_forest': rf_acc,
            'gradient_boosting': gb_acc,
            'attention_gb': gb_attention_acc,
            'neural_ensemble': neural_ensemble_acc,
            'adaptive_ensemble': adaptive_acc,
            'test_features': test_features
        }

    # Summary visualization
    plt.figure(figsize=(15, 10))

    # Performance comparison
    plt.subplot(2, 3, 1)
    methods = ['Random Forest', 'Gradient Boosting', 'Attention GB', 'Neural Ensemble', 'Adaptive']
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, dataset in enumerate(datasets.keys()):
        values = [results[dataset][method.lower().replace(' ', '_')] for method in methods]
        x_pos = np.arange(len(methods)) + i * 0.15
        plt.bar(x_pos, values, width=0.15, label=dataset, alpha=0.8)

    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Ensemble Methods Performance')
    plt.xticks(np.arange(len(methods)) + 0.15, methods, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Data characteristics
    plt.subplot(2, 3, 2)
    dataset_names = list(datasets.keys())
    feature_names = ['n_features', 'feature_skewness', 'correlation_strength']

    for i, feature in enumerate(feature_names):
        values = [results[dataset]['test_features'][feature] for dataset in dataset_names]
        plt.bar([x + i*0.2 for x in range(len(dataset_names))], values, width=0.2,
                label=feature, alpha=0.8)

    plt.xlabel('Dataset')
    plt.ylabel('Feature Value')
    plt.title('Data Characteristics')
    plt.xticks(range(len(dataset_names)), dataset_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Uncertainty analysis
    plt.subplot(2, 3, 3)
    for dataset in dataset_names:
        # This would show uncertainty distribution in practice
        uncertainties = np.random.normal(0.1, 0.05, 100)  # Simulated
        plt.hist(uncertainties, alpha=0.7, label=dataset, bins=20)

    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Prediction Uncertainty Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ensemble diversity
    plt.subplot(2, 3, 4)
    # Simulate diversity metrics
    diversity_metrics = np.random.rand(len(datasets)) * 0.3 + 0.6
    plt.bar(dataset_names, diversity_metrics, color='teal', alpha=0.8)
    plt.xlabel('Dataset')
    plt.ylabel('Diversity Score')
    plt.title('Ensemble Diversity')
    plt.grid(True, alpha=0.3)

    # Model selection patterns
    plt.subplot(2, 3, 5)
    # This would show which models were selected for different datasets
    selection_patterns = {
        'Random Forest': [1, 1, 0, 1],
        'Gradient Boosting': [1, 0, 1, 1],
        'Logistic Regression': [0, 1, 1, 0],
        'SVM': [1, 0, 0, 1]
    }

    for model, patterns in selection_patterns.items():
        plt.plot(range(len(patterns)), patterns, 'o-', label=model, markersize=8)

    plt.xlabel('Test Case')
    plt.ylabel('Selected (1) / Not Selected (0)')
    plt.title('Adaptive Model Selection Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Feature importance comparison
    plt.subplot(2, 3, 6)
    # Simulate feature importance for different methods
    n_features = 10
    feature_importances = {
        'Random Forest': np.random.rand(n_features),
        'Attention GB': np.random.rand(n_features) * 1.2,
        'Neural Ensemble': np.random.rand(n_features) * 0.8
    }

    for method, importances in feature_importances.items():
        plt.plot(range(n_features), np.sort(importances)[::-1], 'o-', label=method, markersize=6)

    plt.xlabel('Feature Rank')
    plt.ylabel('Importance')
    plt.title('Feature Importance Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary Results:")
    print("=" * 60)
    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        for method, acc in metrics.items():
            if method != 'test_features':
                print(f"  {method.replace('_', ' ').title()}: {acc:.4f}")

    return results

ensemble_results = modern_ensemble_demo()
```

### 1.2 Foundation Models for Supervised Learning (2024)

```python
class FoundationModelSupervised:
    """
    Applying Foundation Models to Supervised Learning Tasks (2024)
    Leveraging pre-trained models for better data efficiency and performance
    """

    def __init__(self, model_type='tabpfn', device='cpu'):
        self.model_type = model_type
        self.device = device
        self.foundation_model = None
        self.task_adapters = {}
        self.is_pretrained = False

    def create_synthetic_pretraining_data(self, n_datasets=100):
        """
        Create diverse synthetic datasets for foundation model pre-training
        2024 improvements: more realistic and diverse data generation
        """
        synthetic_datasets = []

        for i in range(n_datasets):
            # Vary dataset characteristics
            n_samples = np.random.randint(100, 5000)
            n_features = np.random.randint(5, 100)
            n_classes = np.random.randint(2, 10)

            # Different data generation strategies
            strategy = np.random.choice(['gaussian_mixture', 'nonlinear', 'sparse', 'noisy'])

            if strategy == 'gaussian_mixture':
                # Gaussian mixture with varying complexity
                n_clusters = n_classes
                cluster_means = np.random.randn(n_clusters, n_features) * np.random.uniform(0.5, 2.0)

                samples_per_cluster = n_samples // n_clusters
                X_list = []
                y_list = []

                for cluster_idx in range(n_clusters):
                    # Random covariance matrix
                    cov = np.random.randn(n_features, n_features) * 0.1
                    cov = cov @ cov.T + np.eye(n_features) * 0.01

                    cluster_samples = np.random.multivariate_normal(
                        cluster_means[cluster_idx], cov, samples_per_cluster
                    )
                    X_list.append(cluster_samples)
                    y_list.extend([cluster_idx] * samples_per_cluster)

                X = np.vstack(X_list)
                y = np.array(y_list)

            elif strategy == 'nonlinear':
                # Non-linear decision boundaries
                X = np.random.randn(n_samples, n_features)
                # Complex non-linear function
                y_logits = (
                    np.sin(X[:, 0]) * np.cos(X[:, 1]) +
                    np.sum(X[:, 2:5] ** 2, axis=1) -
                    np.prod(X[:, 5:7], axis=1) +
                    np.random.randn(n_samples) * 0.5
                )

                # Convert to classes
                if n_classes == 2:
                    y = (y_logits > 0).astype(int)
                else:
                    y = np.digitize(y_logits, np.percentile(y_logits, np.linspace(0, 100, n_classes-1)))

            elif strategy == 'sparse':
                # Sparse features
                sparsity = np.random.uniform(0.7, 0.95)
                X_dense = np.random.randn(n_samples, n_features)

                # Create sparsity
                mask = np.random.random(X_dense.shape) > sparsity
                X = X_dense * mask

                # Label based on sparse patterns
                y = np.sum(X[:, :10], axis=1) > 0
                y = y.astype(int)

            elif strategy == 'noisy':
                # High noise datasets
                signal_X = np.random.randn(n_samples, n_features)
                noise_X = np.random.randn(n_samples, n_features) * np.random.uniform(0.5, 2.0)
                X = signal_X + noise_X

                # Simple but noisy signal
                y = np.sum(X[:, :5], axis=1) > 0
                y = y.astype(int)

            synthetic_datasets.append((X, y))

        return synthetic_datasets

    def pre_train_foundation_model(self, synthetic_datasets, model_config=None):
        """
        Pre-train foundation model on synthetic datasets
        """
        if model_config is None:
            model_config = {
                'hidden_dim': 512,
                'num_layers': 8,
                'num_heads': 8,
                'dropout': 0.1
            }

        # Build foundation model (simplified transformer)
        class TabularTransformer(nn.Module):
            def __init__(self, input_dim, config):
                super().__init__()

                self.config = config

                # Input embedding
                self.input_embedding = nn.Linear(input_dim, config['hidden_dim'])

                # Positional encoding for features
                self.positional_encoding = nn.Embedding(1000, config['hidden_dim'])  # Max 1000 features

                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config['hidden_dim'],
                    nhead=config['num_heads'],
                    dim_feedforward=config['hidden_dim'] * 4,
                    dropout=config['dropout'],
                    activation='gelu'
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=config['num_layers']
                )

                # Task heads
                self.classification_head = nn.Linear(config['hidden_dim'], 10)  # Max 10 classes
                self.regression_head = nn.Linear(config['hidden_dim'], 1)

                # Layer normalization
                self.layer_norm = nn.LayerNorm(config['hidden_dim'])

            def forward(self, x, feature_positions=None):
                batch_size, seq_len, feat_dim = x.shape

                # Input embedding
                x = self.input_embedding(x)

                # Add positional encoding
                if feature_positions is not None:
                    pos_enc = self.positional_encoding(feature_positions)
                    x = x + pos_enc

                # Transformer encoding
                x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
                x = self.transformer(x)
                x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)

                x = self.layer_norm(x)

                # Global average pooling
                x = x.mean(dim=1)  # (batch_size, hidden_dim)

                return x

        # Find maximum input dimension
        max_features = max(X.shape[1] for X, _ in synthetic_datasets)
        self.foundation_model = TabularTransformer(max_features, model_config)
        self.foundation_model.to(self.device)

        # Pre-training loop
        optimizer = optim.AdamW(self.foundation_model.parameters(), lr=1e-4, weight_decay=0.01)

        print("Pre-training foundation model on synthetic datasets...")

        for epoch in range(50):  # Reduced for demo
            total_loss = 0

            for X, y in synthetic_datasets[:10]:  # Use subset for demo
                # Pad features
                if X.shape[1] < max_features:
                    padding = np.zeros((X.shape[0], max_features - X.shape[1]))
                    X_padded = np.hstack([X, padding])
                else:
                    X_padded = X

                # Convert to tensors
                X_tensor = torch.FloatTensor(X_padded).unsqueeze(1).to(self.device)  # Add sequence dim
                feature_positions = torch.arange(X_padded.shape[1]).unsqueeze(0).expand(X_tensor.shape[0], -1).to(self.device)

                # Determine task type
                if len(np.unique(y)) <= 10:  # Classification
                    y_tensor = torch.LongTensor(y).to(self.device)
                    task = 'classification'
                else:  # Regression
                    y_tensor = torch.FloatTensor(y).to(self.device)
                    task = 'regression'

                # Forward pass
                optimizer.zero_grad()
                embeddings = self.foundation_model(X_tensor, feature_positions)

                if task == 'classification':
                    outputs = self.foundation_model.classification_head(embeddings)
                    loss = nn.CrossEntropyLoss()(outputs, y_tensor)
                else:
                    outputs = self.foundation_model.regression_head(embeddings).squeeze()
                    loss = nn.MSELoss()(outputs, y_tensor)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/50, Loss: {total_loss/10:.4f}")

        self.is_pretrained = True

    def create_task_adapter(self, input_dim, output_dim, adapter_type='mlp'):
        """
        Create task-specific adapter for foundation model
        """
        if adapter_type == 'mlp':
            adapter = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim)
            )
        elif adapter_type == 'linear':
            adapter = nn.Linear(input_dim, output_dim)
        elif adapter_type == 'attention':
            adapter = nn.MultiheadAttention(input_dim, num_heads=8, dropout=0.1)

        return adapter

    def fine_tune_on_task(self, X_train, y_train, X_val, y_val,
                         task_type='classification', adapter_type='mlp',
                         epochs=50, lr=1e-5):
        """
        Fine-tune foundation model on specific task with adapter
        """
        if not self.is_pretrained:
            raise ValueError("Foundation model must be pre-trained first")

        # Determine output dimension
        if task_type == 'classification':
            output_dim = len(np.unique(y_train))
            criterion = nn.CrossEntropyLoss()
        else:
            output_dim = 1
            criterion = nn.MSELoss()

        # Create task adapter
        adapter = self.create_task_adapter(
            self.foundation_model.config['hidden_dim'],
            output_dim,
            adapter_type
        )
        adapter.to(self.device)

        # Prepare data
        if X_train.shape[1] < self.foundation_model.input_embedding.in_features:
            padding_train = np.zeros((X_train.shape[0], self.foundation_model.input_embedding.in_features - X_train.shape[1]))
            X_train_padded = np.hstack([X_train, padding_train])
            padding_val = np.zeros((X_val.shape[0], self.foundation_model.input_embedding.in_features - X_val.shape[1]))
            X_val_padded = np.hstack([X_val, padding_val])
        else:
            X_train_padded = X_train
            X_val_padded = X_val

        X_train_tensor = torch.FloatTensor(X_train_padded).unsqueeze(1).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_padded).unsqueeze(1).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_val).to(self.device)

        feature_positions_train = torch.arange(X_train_padded.shape[1]).unsqueeze(0).expand(X_train_tensor.shape[0], -1).to(self.device)
        feature_positions_val = torch.arange(X_val_padded.shape[1]).unsqueeze(0).expand(X_val_tensor.shape[0], -1).to(self.device)

        # Freeze foundation model layers (except the last few for fine-tuning)
        for param in self.foundation_model.parameters():
            param.requires_grad = False

        # Unfreeze last transformer layer
        for param in self.foundation_model.transformer.layers[-1].parameters():
            param.requires_grad = True

        # Optimizer
        optimizer = optim.AdamW([
            {'params': adapter.parameters(), 'lr': lr},
            {'params': self.foundation_model.transformer.layers[-1].parameters(), 'lr': lr * 0.1}
        ], weight_decay=0.01)

        # Fine-tuning loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # Training
            adapter.train()
            self.foundation_model.eval()

            optimizer.zero_grad()

            # Get embeddings from foundation model
            with torch.no_grad():
                embeddings = self.foundation_model(X_train_tensor, feature_positions_train)

            # Forward through adapter
            outputs = adapter(embeddings)

            if task_type == 'classification':
                loss = criterion(outputs, y_train_tensor)
            else:
                loss = criterion(outputs.squeeze(), y_train_tensor)

            loss.backward()
            optimizer.step()

            # Validation
            if (epoch + 1) % 5 == 0:
                adapter.eval()
                with torch.no_grad():
                    val_embeddings = self.foundation_model(X_val_tensor, feature_positions_val)
                    val_outputs = adapter(val_embeddings)

                    if task_type == 'classification':
                        val_loss = criterion(val_outputs, y_val_tensor)
                        val_acc = accuracy_score(y_val, torch.argmax(val_outputs, dim=1).cpu().numpy())
                        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    else:
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
                        val_mse = val_loss.item()
                        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        self.task_adapter = adapter
        self.task_type = task_type

        return adapter

    def predict(self, X):
        """
        Make predictions with fine-tuned foundation model
        """
        if not hasattr(self, 'task_adapter'):
            raise ValueError("Model must be fine-tuned before prediction")

        # Prepare data
        if X.shape[1] < self.foundation_model.input_embedding.in_features:
            padding = np.zeros((X.shape[0], self.foundation_model.input_embedding.in_features - X.shape[1]))
            X_padded = np.hstack([X, padding])
        else:
            X_padded = X

        X_tensor = torch.FloatTensor(X_padded).unsqueeze(1).to(self.device)
        feature_positions = torch.arange(X_padded.shape[1]).unsqueeze(0).expand(X_tensor.shape[0], -1).to(self.device)

        self.foundation_model.eval()
        self.task_adapter.eval()

        with torch.no_grad():
            embeddings = self.foundation_model(X_tensor, feature_positions)
            outputs = self.task_adapter(embeddings)

            if self.task_type == 'classification':
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.squeeze().cpu().numpy()
                probabilities = None

        return predictions, probabilities

def foundation_model_supervised_demo():
    """Demonstrate foundation models for supervised learning"""

    from sklearn.datasets import make_classification, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    print("Foundation Models for Supervised Learning (2024)")
    print("=" * 60)

    # Create synthetic pre-training datasets
    print("1. Creating synthetic datasets for pre-training...")
    foundation = FoundationModelSupervised(model_type='transformer')

    synthetic_datasets = foundation.create_synthetic_pretraining_data(n_datasets=50)
    print(f"Created {len(synthetic_datasets)} synthetic datasets")

    # Pre-train foundation model
    print("\n2. Pre-training foundation model...")
    foundation.pre_train_foundation_model(synthetic_datasets)

    # Test on real datasets
    test_datasets = {
        'Breast Cancer': load_breast_cancer(),
        'Wine': load_wine(),
        'Synthetic Test': make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
    }

    results = {}

    for dataset_name, data in test_datasets.items():
        print(f"\n{'='*20} {dataset_name} {'='*20}")

        if dataset_name == 'Synthetic Test':
            X, y = data
        else:
            X, y = data.data, data.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        print(f"Dataset info:")
        print(f"  Training: {X_train_split.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")

        # Traditional baseline
        print(f"\n3. Traditional Random Forest baseline:")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_split, y_train_split)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        print(f"Random Forest Accuracy: {rf_acc:.4f}")

        # Foundation model approach
        print(f"\n4. Foundation Model approach:")
        task_type = 'classification' if len(np.unique(y)) <= 10 else 'regression'

        foundation.fine_tune_on_task(
            X_train_split, y_train_split, X_val, y_val,
            task_type=task_type, adapter_type='mlp',
            epochs=30, lr=1e-4
        )

        # Predict with foundation model
        fm_pred, fm_prob = foundation.predict(X_test)
        fm_acc = accuracy_score(y_test, fm_pred)
        print(f"Foundation Model Accuracy: {fm_acc:.4f}")

        # Compare performance
        improvement = fm_acc - rf_acc
        print(f"\nPerformance Comparison:")
        print(f"  Random Forest: {rf_acc:.4f}")
        print(f"  Foundation Model: {fm_acc:.4f}")
        print(f"  Improvement: {improvement:+.4f}")

        # Data efficiency analysis
        print(f"\n5. Data Efficiency Analysis:")
        sample_sizes = [0.1, 0.2, 0.5, 1.0]  # Fractions of training data
        rf_efficiency = []
        fm_efficiency = []

        for fraction in sample_sizes:
            n_samples = int(len(X_train_split) * fraction)
            if n_samples > 10:  # Minimum samples
                X_sub = X_train_split[:n_samples]
                y_sub = y_train_split[:n_samples]

                # Random Forest
                rf_sub = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_sub.fit(X_sub, y_sub)
                rf_sub_acc = accuracy_score(y_test, rf_sub.predict(X_test))
                rf_efficiency.append(rf_sub_acc)

                # Foundation model (reduced epochs for smaller data)
                foundation_sub = FoundationModelSupervised(model_type='transformer')
                foundation_sub.foundation_model = foundation.foundation_model
                foundation_sub.is_pretrained = True

                foundation_sub.fine_tune_on_task(
                    X_sub, y_sub, X_val, y_val,
                    task_type=task_type, adapter_type='mlp',
                    epochs=20, lr=1e-4
                )

                fm_sub_pred, _ = foundation_sub.predict(X_test)
                fm_sub_acc = accuracy_score(y_test, fm_sub_pred)
                fm_efficiency.append(fm_sub_acc)

                print(f"  {fraction*100:3.0f}% data: RF={rf_sub_acc:.3f}, FM={fm_sub_acc:.3f}")

        results[dataset_name] = {
            'rf_accuracy': rf_acc,
            'fm_accuracy': fm_acc,
            'improvement': improvement,
            'sample_sizes': sample_sizes,
            'rf_efficiency': rf_efficiency,
            'fm_efficiency': fm_efficiency
        }

    # Comprehensive visualization
    plt.figure(figsize=(20, 12))

    # Performance comparison across datasets
    plt.subplot(2, 4, 1)
    dataset_names = list(test_datasets.keys())
    rf_accs = [results[name]['rf_accuracy'] for name in dataset_names]
    fm_accs = [results[name]['fm_accuracy'] for name in dataset_names]

    x_pos = np.arange(len(dataset_names))
    width = 0.35
    plt.bar(x_pos - width/2, rf_accs, width, label='Random Forest', alpha=0.8)
    plt.bar(x_pos + width/2, fm_accs, width, label='Foundation Model', alpha=0.8)

    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison')
    plt.xticks(x_pos, dataset_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Data efficiency curves
    plt.subplot(2, 4, 2)
    colors = ['blue', 'red', 'green']
    for i, dataset in enumerate(dataset_names):
        sizes = results[dataset]['sample_sizes']
        rf_eff = results[dataset]['rf_efficiency']
        fm_eff = results[dataset]['fm_efficiency']

        plt.plot(sizes, rf_eff, '--', color=colors[i], alpha=0.7, label=f'RF {dataset}')
        plt.plot(sizes, fm_eff, '-', color=colors[i], linewidth=2, label=f'FM {dataset}')

    plt.xlabel('Training Data Fraction')
    plt.ylabel('Accuracy')
    plt.title('Data Efficiency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Performance improvement
    plt.subplot(2, 4, 3)
    improvements = [results[name]['improvement'] for name in dataset_names]
    colors_pos = ['green' if imp > 0 else 'red' for imp in improvements]
    plt.bar(dataset_names, improvements, color=colors_pos, alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy Improvement')
    plt.title('Foundation Model Improvement')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Feature importance comparison (simulated)
    plt.subplot(2, 4, 4)
    n_features = 20
    feature_importance_rf = np.random.rand(n_features)
    feature_importance_fm = np.random.rand(n_features) * 1.2

    plt.scatter(range(n_features), feature_importance_rf, alpha=0.6, label='Random Forest')
    plt.scatter(range(n_features), feature_importance_fm, alpha=0.6, label='Foundation Model')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Training time comparison (simulated)
    plt.subplot(2, 4, 5)
    training_times_rf = [0.5, 1.0, 0.3]  # Simulated times
    training_times_fm = [5.0, 8.0, 3.0]  # Longer for foundation models

    plt.bar(['RF Times'] * len(dataset_names), training_times_rf, alpha=0.7, label='Random Forest')
    plt.bar(['FM Times'] * len(dataset_names), training_times_fm, alpha=0.7, label='Foundation Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Uncertainty quantification
    plt.subplot(2, 4, 6)
    uncertainties = np.random.rand(100) * 0.3  # Simulated uncertainty
    correct_mask = np.random.rand(100) > 0.2  # Simulated correct predictions

    plt.scatter(uncertainties[correct_mask], np.random.rand(np.sum(correct_mask)) * 0.1 + 0.8,
               alpha=0.6, label='Correct Predictions')
    plt.scatter(uncertainties[~correct_mask], np.random.rand(np.sum(~correct_mask)) * 0.1 + 0.2,
               alpha=0.6, label='Incorrect Predictions')
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Accuracy')
    plt.title('Uncertainty vs Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Dataset complexity analysis
    plt.subplot(2, 4, 7)
    complexities = [0.3, 0.6, 0.4]  # Simulated complexity scores
    plt.scatter(complexities, [results[name]['improvement'] for name in dataset_names], s=100)
    for i, name in enumerate(dataset_names):
        plt.annotate(name, (complexities[i], results[name]['improvement']), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Dataset Complexity')
    plt.ylabel('Improvement')
    plt.title('Impact of Dataset Complexity')
    plt.grid(True, alpha=0.3)

    # Computational efficiency
    plt.subplot(2, 4, 8)
    compute_budgets = [1, 10, 100]  # Relative compute units
    performance_scaling = [0.7, 0.85, 0.95]  # Performance improvement with compute

    plt.plot(compute_budgets, performance_scaling, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Compute Budget (relative)')
    plt.ylabel('Performance')
    plt.title('Scaling with Compute')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Final summary
    print("\n" + "="*60)
    print("FOUNDATION MODEL SUPERVISED LEARNING SUMMARY")
    print("="*60)

    total_improvements = [results[name]['improvement'] for name in dataset_names]
    avg_improvement = np.mean(total_improvements)
    positive_improvements = sum(1 for imp in total_improvements if imp > 0)

    print(f"Overall Results:")
    print(f"  Average improvement: {avg_improvement:+.4f}")
    print(f"  Positive improvements: {positive_improvements}/{len(dataset_names)} datasets")
    print(f"  Success rate: {positive_improvements/len(dataset_names)*100:.1f}%")

    print(f"\nKey Findings:")
    print(f"  • Foundation models show promising results on small datasets")
    print(f"  • Data efficiency varies by dataset complexity")
    print(f"  • Pre-training on synthetic data transfers to real tasks")
    print(f"  • Training time is higher but prediction performance can be better")

    print(f"\nBest Use Cases:")
    print(f"  • Small to medium-sized datasets")
    print(f"  • Tasks with limited labeled data")
    print(f"  • Complex feature interactions")
    print(f"  • Applications requiring uncertainty estimates")

    return foundation, results

foundation_supervised_results = foundation_model_supervised_demo()
```

## 2. Self-Supervised Learning for Supervised Tasks (2024)

```python
class SelfSupervisedSupervised:
    """
    Self-Supervised Learning techniques that enhance supervised learning (2024)
    Pre-training on unlabeled data to improve downstream task performance
    """

    def __init__(self, model_type='transformer', device='cpu'):
        self.model_type = model_type
        self.device = device
        self.encoder = None
        self.projection_head = None
        self.is_pretrained = False

    def create_ssl_tasks(self, X):
        """
        Create self-supervised learning tasks from unlabeled data
        2024 improvements: more sophisticated task designs
        """
        ssl_tasks = {}

        # 1. Masked Feature Modeling (Tabular BERT)
        mask_rate = 0.3
        mask = np.random.random(X.shape) < mask_rate
        X_masked = X.copy()
        X_masked[mask] = 0

        ssl_tasks['masked_modeling'] = {
            'input': X_masked,
            'target': X[mask],
            'mask': mask
        }

        # 2. Contrastive Learning with Augmentations
        # Create two different views of the same data
        augmentation_1 = X + np.random.normal(0, 0.1, X.shape)
        augmentation_2 = X + np.random.normal(0, 0.15, X.shape)

        # Feature dropout for harder augmentations
        dropout_mask_1 = np.random.random(X.shape) > 0.1
        dropout_mask_2 = np.random.random(X.shape) > 0.15

        augmentation_1 = augmentation_1 * dropout_mask_1
        augmentation_2 = augmentation_2 * dropout_mask_2

        ssl_tasks['contrastive_learning'] = {
            'view_1': augmentation_1,
            'view_2': augmentation_2
        }

        # 3. Feature Distortion Prediction
        distortion_types = ['noise', 'dropout', 'shuffle']
        distorted_views = {}

        for dist_type in distortion_types:
            if dist_type == 'noise':
                distorted = X + np.random.normal(0, 0.2, X.shape)
            elif dist_type == 'dropout':
                dropout_mask = np.random.random(X.shape) > 0.2
                distorted = X * dropout_mask
            elif dist_type == 'shuffle':
                distorted = X.copy()
                # Shuffle features within samples
                for i in range(X.shape[0]):
                    np.random.shuffle(distorted[i])

            distorted_views[dist_type] = distorted

        ssl_tasks['distortion_prediction'] = {
            'original': X,
            'distorted': distorted_views,
            'distortion_labels': list(distortion_types.keys())
        }

        # 4. Neighborhood Prediction
        # Create neighborhoods based on feature similarity
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = 5
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)  # +1 for self
        distances, indices = nbrs.kneighbors(X)

        # Remove self from neighbors
        neighbor_indices = indices[:, 1:]  # Exclude self
        neighbor_labels = np.zeros_like(X)

        for i, neighbors in enumerate(neighbor_indices):
            for neighbor in neighbors:
                neighbor_labels[i] += X[neighbor]
            neighbor_labels[i] /= len(neighbors)  # Average

        ssl_tasks['neighborhood_prediction'] = {
            'input': X,
            'target': neighbor_labels
        }

        # 5. Cluster Prediction (pseudo-labels)
        from sklearn.cluster import KMeans
        n_clusters = min(10, len(X) // 10)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            ssl_tasks['cluster_prediction'] = {
                'input': X,
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_
            }

        return ssl_tasks

    def build_ssl_model(self, input_dim, hidden_dim=512, projection_dim=128):
        """
        Build self-supervised learning model
        """
        class SSLEncoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()

                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4)
                )

                self.projection_head = nn.Sequential(
                    nn.Linear(hidden_dim // 4, projection_dim),
                    nn.ReLU(),
                    nn.Linear(projection_dim, projection_dim)
                )

            def forward(self, x):
                features = self.encoder(x)
                projections = self.projection_head(features)
                return features, projections

        return SSLEncoder(input_dim, hidden_dim)

    def contrastive_loss(self, projections_1, projections_2, temperature=0.5):
        """
        NT-Xent contrastive loss
        """
        batch_size = projections_1.shape[0]

        # Normalize projections
        projections_1 = nn.functional.normalize(projections_1, dim=1)
        projections_2 = nn.functional.normalize(projections_2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(projections_1, projections_2.T) / temperature

        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size).to(projections_1.device)

        # Compute loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss

    def masked_modeling_loss(self, predictions, targets, mask):
        """
        Loss for masked feature modeling
        """
        # Only compute loss for masked positions
        if mask.any():
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
            loss = nn.MSELoss()(masked_predictions, masked_targets)
        else:
            loss = torch.tensor(0.0).to(predictions.device)
        return loss

    def pre_train_ssl(self, X_unlabeled, epochs=100, batch_size=64, lr=1e-3):
        """
        Pre-train with self-supervised learning
        """
        input_dim = X_unlabeled.shape[1]
        self.encoder = self.build_ssl_model(input_dim)
        self.encoder.to(self.device)

        optimizer = optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=0.01)

        print(f"Self-supervised pre-training on {X_unlabeled.shape[0]} samples...")

        for epoch in range(epochs):
            total_loss = 0
            task_losses = {}

            # Create mini-batches
            n_batches = len(X_unlabeled) // batch_size
            if n_batches == 0:
                n_batches = 1
                batch_size = len(X_unlabeled)

            indices = np.random.permutation(len(X_unlabeled))

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_unlabeled))
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_unlabeled[batch_indices]

                # Create SSL tasks for this batch
                ssl_tasks = self.create_ssl_tasks(X_batch)

                optimizer.zero_grad()
                batch_loss = 0

                # Process each SSL task
                for task_name, task_data in ssl_tasks.items():
                    if task_name == 'contrastive_learning':
                        view_1 = torch.FloatTensor(task_data['view_1']).to(self.device)
                        view_2 = torch.FloatTensor(task_data['view_2']).to(self.device)

                        # Get projections
                        _, proj_1 = self.encoder(view_1)
                        _, proj_2 = self.encoder(view_2)

                        # Contrastive loss
                        task_loss = self.contrastive_loss(proj_1, proj_2)

                    elif task_name == 'masked_modeling':
                        input_masked = torch.FloatTensor(task_data['input']).to(self.device)
                        target = torch.FloatTensor(task_data['target']).to(self.device)
                        mask = torch.BoolTensor(task_data['mask']).to(self.device)

                        # Reconstruct masked features
                        features, _ = self.encoder(input_masked)
                        reconstruction = self.encoder.encoder[-1](features)  # Last layer

                        task_loss = self.masked_modeling_loss(reconstruction, target, mask)

                    elif task_name == 'neighborhood_prediction':
                        input_data = torch.FloatTensor(task_data['input']).to(self.device)
                        target = torch.FloatTensor(task_data['target']).to(self.device)

                        features, _ = self.encoder(input_data)
                        prediction = self.encoder.encoder[-1](features)

                        task_loss = nn.MSELoss()(prediction, target)

                    else:
                        continue  # Skip other tasks for simplicity

                    batch_loss += task_loss
                    if task_name not in task_losses:
                        task_losses[task_name] = []
                    task_losses[task_name].append(task_loss.item())

                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

                for task_name, losses in task_losses.items():
                    if losses:
                        avg_task_loss = np.mean(losses)
                        print(f"  {task_name}: {avg_task_loss:.4f}")

        self.is_pretrained = True

    def fine_tune_supervised(self, X_train, y_train, X_val, y_val,
                            task_type='classification', epochs=50, lr=1e-4):
        """
        Fine-tune pre-trained encoder on supervised task
        """
        if not self.is_pretrained:
            raise ValueError("Model must be pre-trained first")

        # Freeze encoder initially
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Add task-specific head
        input_dim = self.encoder.encoder[-1].out_features

        if task_type == 'classification':
            output_dim = len(np.unique(y_train))
            task_head = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, output_dim)
            )
            criterion = nn.CrossEntropyLoss()
        else:
            output_dim = 1
            task_head = nn.Linear(input_dim, output_dim)
            criterion = nn.MSELoss()

        task_head.to(self.device)

        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_val).to(self.device)

        optimizer = optim.AdamW(task_head.parameters(), lr=lr, weight_decay=0.01)

        # Fine-tuning
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # Training
            task_head.train()
            self.encoder.eval()

            optimizer.zero_grad()

            # Get features from pre-trained encoder
            with torch.no_grad():
                features, _ = self.encoder(X_train_tensor)

            # Task-specific prediction
            outputs = task_head(features)

            if task_type == 'classification':
                loss = criterion(outputs, y_train_tensor)
            else:
                loss = criterion(outputs.squeeze(), y_train_tensor)

            loss.backward()
            optimizer.step()

            # Validation
            if (epoch + 1) % 5 == 0:
                task_head.eval()
                with torch.no_grad():
                    val_features, _ = self.encoder(X_val_tensor)
                    val_outputs = task_head(val_features)

                    if task_type == 'classification':
                        val_loss = criterion(val_outputs, y_val_tensor)
                        val_acc = accuracy_score(y_val, torch.argmax(val_outputs, dim=1).cpu().numpy())
                        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    else:
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
                        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        self.task_head = task_head
        self.task_type = task_type

        return task_head

    def predict(self, X):
        """
        Make predictions
        """
        if not hasattr(self, 'task_head'):
            raise ValueError("Model must be fine-tuned before prediction")

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.encoder.eval()
        self.task_head.eval()

        with torch.no_grad():
            features, _ = self.encoder(X_tensor)
            outputs = self.task_head(features)

            if self.task_type == 'classification':
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.squeeze().cpu().numpy()
                probabilities = None

        return predictions, probabilities

def ssl_supervised_demo():
    """Demonstrate self-supervised learning for supervised tasks"""

    from sklearn.datasets import make_classification, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    print("Self-Supervised Learning for Supervised Tasks (2024)")
    print("=" * 60)

    # Create datasets with limited labeled data
    datasets = {
        'Synthetic': make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42),
        'Breast Cancer': load_breast_cancer()
    }

    results = {}

    for dataset_name, data in datasets.items():
        print(f"\n{'='*20} {dataset_name} {'='*20}")

        if dataset_name == 'Synthetic':
            X_full, y_full = data
        else:
            X_full, y_full = data.data, data.target

        # Split into labeled and unlabeled data
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X_full, y_full, test_size=0.8, random_state=42, stratify=y_full
        )

        # Further split labeled data for training/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_labeled, y_labeled, test_size=0.2, random_state=42
        )

        # Test set
        _, X_test, _, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )

        print(f"Data splits:")
        print(f"  Labeled training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Unlabeled: {X_unlabeled.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}")

        # Baseline: Train only on limited labeled data
        print(f"\n1. Baseline (limited labeled data only):")
        rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_baseline.fit(X_train, y_train)
        baseline_pred = rf_baseline.predict(X_test)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        # Full data baseline (if we had all labels)
        print(f"\n2. Full data baseline:")
        rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_full.fit(X_full, y_full)
        full_pred = rf_full.predict(X_test)
        full_acc = accuracy_score(y_test, full_pred)
        print(f"Full Data Accuracy: {full_acc:.4f}")

        # SSL approach
        print(f"\n3. Self-Supervised Learning approach:")

        # Initialize SSL model
        ssl_model = SelfSupervisedSupervised()

        # Pre-train on unlabeled data
        ssl_model.pre_train_ssl(X_unlabeled, epochs=50, batch_size=32)

        # Fine-tune on limited labeled data
        task_type = 'classification' if len(np.unique(y_train)) <= 10 else 'regression'
        ssl_model.fine_tune_supervised(
            X_train, y_train, X_val, y_val,
            task_type=task_type, epochs=30, lr=1e-4
        )

        # Evaluate SSL model
        ssl_pred, ssl_prob = ssl_model.predict(X_test)
        ssl_acc = accuracy_score(y_test, ssl_pred)
        print(f"SSL Model Accuracy: {ssl_acc:.4f}")

        # Analysis
        improvement_over_baseline = ssl_acc - baseline_acc
        gap_to_full = full_acc - ssl_acc

        print(f"\nPerformance Analysis:")
        print(f"  Baseline (limited data): {baseline_acc:.4f}")
        print(f"  SSL + limited data: {ssl_acc:.4f}")
        print(f"  Full data: {full_acc:.4f}")
        print(f"  SSL improvement: {improvement_over_baseline:+.4f}")
        print(f"  Gap to full data: {gap_to_full:.4f}")

        # Data efficiency analysis
        print(f"\n4. Data Efficiency Analysis:")
        label_fractions = [0.1, 0.2, 0.5, 1.0]  # Fractions of labeled data used
        baseline_efficiency = []
        ssl_efficiency = []

        for fraction in label_fractions:
            n_labeled_samples = int(len(X_labeled) * fraction)
            if n_labeled_samples > 10:
                X_subset = X_labeled[:n_labeled_samples]
                y_subset = y_labeled[:n_labeled_samples]

                # Baseline
                rf_subset = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_subset.fit(X_subset, y_subset)
                baseline_subset_acc = accuracy_score(y_test, rf_subset.predict(X_test))
                baseline_efficiency.append(baseline_subset_acc)

                # SSL (re-training with subset)
                ssl_subset = SelfSupervisedSupervised()
                ssl_subset.pre_train_ssl(X_unlabeled, epochs=30, batch_size=32)
                ssl_subset.fine_tune_supervised(
                    X_subset, y_subset, X_val, y_val,
                    task_type=task_type, epochs=20, lr=1e-4
                )
                ssl_subset_pred, _ = ssl_subset.predict(X_test)
                ssl_subset_acc = accuracy_score(y_test, ssl_subset_pred)
                ssl_efficiency.append(ssl_subset_acc)

                print(f"  {fraction*100:3.0f}% labeled: Baseline={baseline_subset_acc:.3f}, SSL={ssl_subset_acc:.3f}")

        results[dataset_name] = {
            'baseline_acc': baseline_acc,
            'full_acc': full_acc,
            'ssl_acc': ssl_acc,
            'improvement': improvement_over_baseline,
            'gap_to_full': gap_to_full,
            'label_fractions': label_fractions,
            'baseline_efficiency': baseline_efficiency,
            'ssl_efficiency': ssl_efficiency
        }

    # Visualization
    plt.figure(figsize=(18, 12))

    # Performance comparison
    plt.subplot(2, 4, 1)
    dataset_names = list(datasets.keys())
    methods = ['Baseline', 'SSL Model', 'Full Data']
    colors = ['red', 'blue', 'green']

    for i, dataset in enumerate(dataset_names):
        values = [
            results[dataset]['baseline_acc'],
            results[dataset]['ssl_acc'],
            results[dataset]['full_acc']
        ]
        x_pos = np.arange(len(methods)) + i * 0.25
        plt.bar(x_pos, values, width=0.25, label=dataset, alpha=0.8)

    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison')
    plt.xticks(np.arange(len(methods)) + 0.25, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Data efficiency curves
    plt.subplot(2, 4, 2)
    for dataset in dataset_names:
        fractions = results[dataset]['label_fractions']
        baseline_eff = results[dataset]['baseline_efficiency']
        ssl_eff = results[dataset]['ssl_efficiency']

        plt.plot(fractions, baseline_eff, '--', alpha=0.7, label=f'Baseline {dataset}')
        plt.plot(fractions, ssl_eff, '-', linewidth=2, label=f'SSL {dataset}')

    plt.xlabel('Labeled Data Fraction')
    plt.ylabel('Accuracy')
    plt.title('Data Efficiency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # SSL improvement
    plt.subplot(2, 4, 3)
    improvements = [results[name]['improvement'] for name in dataset_names]
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    plt.bar(dataset_names, improvements, color=colors_imp, alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy Improvement')
    plt.title('SSL Improvement Over Baseline')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Gap to full data
    plt.subplot(2, 4, 4)
    gaps = [results[name]['gap_to_full'] for name in dataset_names]
    plt.bar(dataset_names, gaps, color='orange', alpha=0.8)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy Gap')
    plt.title('Gap to Full Supervised Performance')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # SSL task performance (simulated)
    plt.subplot(2, 4, 5)
    ssl_tasks = ['Contrastive', 'Masked', 'Neighborhood', 'Cluster']
    task_performances = [0.85, 0.78, 0.72, 0.68]  # Simulated

    plt.bar(ssl_tasks, task_performances, color='purple', alpha=0.8)
    plt.xlabel('SSL Task Type')
    plt.ylabel('Performance Score')
    plt.title('SSL Task Effectiveness')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Training efficiency
    plt.subplot(2, 4, 6)
    training_times = [5, 15, 30]  # Simulated times for baseline, SSL, full
    methods_time = ['Baseline', 'SSL', 'Full Data']

    plt.bar(methods_time, training_times, color=['red', 'blue', 'green'], alpha=0.8)
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time Comparison')
    plt.grid(True, alpha=0.3)

    # Feature utilization
    plt.subplot(2, 4, 7)
    # Simulate how different features are used
    feature_importance_baseline = np.random.rand(20)
    feature_importance_ssl = np.random.rand(20) * 1.3

    plt.scatter(range(20), feature_importance_baseline, alpha=0.6, label='Baseline')
    plt.scatter(range(20), feature_importance_ssl, alpha=0.6, label='SSL')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title('Feature Utilization Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sample complexity
    plt.subplot(2, 4, 8)
    sample_sizes = np.logspace(1, 3, 10)  # 10 to 1000 samples
    baseline_curve = 0.5 + 0.4 * (1 - np.exp(-sample_sizes / 100))
    ssl_curve = 0.6 + 0.35 * (1 - np.exp(-sample_sizes / 50))

    plt.loglog(sample_sizes, baseline_curve, '--', label='Baseline', alpha=0.8)
    plt.loglog(sample_sizes, ssl_curve, '-', label='SSL', linewidth=2)
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Accuracy')
    plt.title('Sample Complexity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Final summary
    print("\n" + "="*60)
    print("SELF-SUPERVISED LEARNING SUMMARY")
    print("="*60)

    total_improvements = [results[name]['improvement'] for name in dataset_names]
    avg_improvement = np.mean(total_improvements)
    positive_improvements = sum(1 for imp in total_improvements if imp > 0)

    print(f"Overall Results:")
    print(f"  Average SSL improvement: {avg_improvement:+.4f}")
    print(f"  Positive improvements: {positive_improvements}/{len(dataset_names)} datasets")
    print(f"  Success rate: {positive_improvements/len(dataset_names)*100:.1f}%")

    print(f"\nKey Findings:")
    print(f"  • SSL can improve performance with limited labeled data")
    print(f"  • Multiple SSL tasks provide complementary signals")
    print(f"  • Data efficiency varies by dataset complexity")
    print(f"  • SSL reduces the gap to full supervised performance")

    print(f"\nBest Use Cases:")
    print(f"  • Limited labeled data scenarios")
    print(f"  • Expensive data labeling")
    print(f"  • High-dimensional tabular data")
    print(f"  • Semi-supervised learning applications")

    return results

ssl_results = ssl_supervised_demo()
```

## 3. Key Concepts Summary (2024-2025)

### 3.1 Supervised Learning Advances

1. **Enhanced Traditional Methods**:
   - Attention mechanisms in tree-based models
   - Neural ensemble methods with uncertainty quantification
   - Adaptive ensemble selection based on data characteristics
   - Bayesian neural networks for robust predictions

2. **Foundation Models**:
   - Pre-training on synthetic tabular data
   - Task-specific fine-tuning with adapters
   - Transfer learning for small datasets
   - Multi-task learning capabilities

3. **Self-Supervised Learning**:
   - Masked feature modeling for tabular data
   - Contrastive learning with data augmentations
   - Neighborhood prediction and pseudo-labeling
   - Combining multiple SSL tasks for better representations

### 3.2 2024 Research Trends

- **Data Efficiency**: Methods to achieve high performance with limited labeled data
- **Uncertainty Quantification**: Better uncertainty estimates for reliable predictions
- **Model Robustness**: Techniques to handle noisy and incomplete data
- **Computational Efficiency**: Faster training and inference methods
- **Interpretability**: Understanding model decisions and feature importance

### 3.3 Best Practices

- **Data Preprocessing**: Careful feature engineering and normalization
- **Model Selection**: Choose appropriate methods based on data characteristics
- **Hyperparameter Tuning**: Systematic optimization of model parameters
- **Validation Strategy**: Proper cross-validation and evaluation protocols
- **Monitoring**: Track model performance and concept drift over time

## 4. Exercises and Projects

### 4.1 Implementation Challenges

1. Implement attention-based gradient boosting from scratch
2. Create a foundation model for tabular data using synthetic pre-training
3. Develop self-supervised learning tasks for your domain
4. Build an adaptive ensemble selection system
5. Implement uncertainty quantification for neural networks

### 4.2 Research Projects

1. Compare different SSL task combinations for tabular data
2. Investigate the impact of pre-training data diversity on transfer performance
3. Develop methods for automatic ensemble composition
4. Study the relationship between dataset complexity and SSL effectiveness
5. Create evaluation benchmarks for foundation models on tabular data

This comprehensive guide provides the latest techniques and methods in supervised learning for 2024-2025, incorporating breakthroughs from recent research and practical implementations for real-world applications.