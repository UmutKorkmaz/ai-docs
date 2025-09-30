"""
Ensemble model implementation for customer churn prediction.
Supports various ensemble methods including weighted averaging, stacking, and voting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
from pathlib import Path
import yaml

from .base_model import BaseModel
from .model_registry import MODEL_REGISTRY
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """
    Advanced ensemble model with multiple combination strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ensemble model with configuration.

        Args:
            config: Configuration dictionary containing ensemble settings
        """
        super().__init__(config)
        self.models = {}
        self.ensemble_method = config.get("ensemble", {}).get("method", "weighted_average")
        self.weights = config.get("ensemble", {}).get("weights", [])
        self.stacking_meta_model = None
        self.model_configs = config.get("models", {})
        self.feature_importances_ = None
        self.cv_scores_ = {}
        self.is_fitted = False

    def _initialize_models(self) -> None:
        """Initialize individual models from configuration."""
        for model_name, model_config in self.model_configs.items():
            try:
                model_class = MODEL_REGISTRY.get(model_config.get("class_name"))
                if model_class is None:
                    logger.warning(f"Model {model_name} not found in registry, skipping")
                    continue

                # Initialize model with parameters
                if hasattr(model_class, '__module__') and model_class.__module__ == 'sklearn.ensemble':
                    # Scikit-learn models
                    model = model_class(**model_config.get("parameters", {}))
                else:
                    # Custom models or other libraries
                    model = model_class(config=model_config)

                self.models[model_name] = model
                logger.info(f"Initialized model: {model_name}")

            except Exception as e:
                logger.error(f"Error initializing model {model_name}: {str(e)}")

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'EnsembleModel':
        """
        Fit all models in the ensemble.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Optional sample weights

        Returns:
            Self for method chaining
        """
        logger.info("Fitting ensemble model...")

        self._initialize_models()

        if not self.models:
            raise ValueError("No models initialized. Check model configuration.")

        # Fit individual models
        for model_name, model in self.models.items():
            try:
                logger.info(f"Fitting model: {model_name}")
                model.fit(X, y, sample_weight=sample_weight)

                # Store cross-validation scores
                cv_scores = self._cross_validate_model(model, X, y)
                self.cv_scores_[model_name] = cv_scores
                logger.info(f"{model_name} CV AUC: {cv_scores['mean_roc_auc']:.4f} (+/- {cv_scores['std_roc_auc']:.4f})")

            except Exception as e:
                logger.error(f"Error fitting model {model_name}: {str(e)}")
                continue

        # Setup ensemble combination
        self._setup_ensemble(X, y)

        self.is_fitted = True
        logger.info("Ensemble model fitting completed")
        return self

    def _cross_validate_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform cross-validation for a single model.

        Args:
            model: Model to validate
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary of cross-validation scores
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        cv_scores = {}
        for score in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=score)
            cv_scores[f'mean_{score}'] = scores.mean()
            cv_scores[f'std_{score}'] = scores.std()

        return cv_scores

    def _setup_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """Setup ensemble combination method."""
        if self.ensemble_method == "stacking":
            self._setup_stacking(X, y)
        elif self.ensemble_method == "weighted_average":
            self._setup_weights(X, y)
        elif self.ensemble_method == "voting":
            self._setup_voting(X, y)

    def _setup_stacking(self, X: np.ndarray, y: np.ndarray) -> None:
        """Setup stacking ensemble with meta-model."""
        logger.info("Setting up stacking ensemble...")

        # Generate out-of-fold predictions for training
        oof_predictions = self._get_oof_predictions(X, y)

        # Initialize meta-model
        meta_config = self.config.get("ensemble", {}).get("stacking_meta_model", {})
        meta_model_class = MODEL_REGISTRY.get(meta_config.get("class_name"))
        if meta_model_class:
            self.stacking_meta_model = meta_model_class(**meta_config.get("parameters", {}))
            self.stacking_meta_model.fit(oof_predictions, y)
        else:
            # Default to logistic regression
            from sklearn.linear_model import LogisticRegression
            self.stacking_meta_model = LogisticRegression(random_state=42)
            self.stacking_meta_model.fit(oof_predictions, y)

    def _get_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get out-of-fold predictions for stacking.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Out-of-fold predictions
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros((len(X), len(self.models)))

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_predictions = []

            for model_idx, (model_name, model) in enumerate(self.models.items()):
                try:
                    # Fit on training fold
                    model_clone = joblib.load(joblib.dump(model, None)[0])
                    model_clone.fit(X_train, y_train)

                    # Predict on validation fold
                    if hasattr(model_clone, 'predict_proba'):
                        pred = model_clone.predict_proba(X_val)[:, 1]
                    else:
                        pred = model_clone.predict(X_val)

                    oof_predictions[val_idx, model_idx] = pred
                    fold_predictions.append(pred)

                except Exception as e:
                    logger.error(f"Error in model {model_name} for fold {fold_idx}: {str(e)}")
                    continue

            logger.info(f"Completed fold {fold_idx + 1}/5")

        return oof_predictions

    def _setup_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Setup weights for weighted averaging ensemble."""
        if not self.weights:
            # If no weights provided, use cross-validation scores
            weights = []
            for model_name in self.models.keys():
                if model_name in self.cv_scores_:
                    weight = self.cv_scores_[model_name]['mean_roc_auc']
                    weights.append(weight)

            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            self.weights = weights.tolist()

        logger.info(f"Using weights: {self.weights}")

    def _setup_voting(self, X: np.ndarray, y: np.ndarray) -> None:
        """Setup voting ensemble (equal weights)."""
        n_models = len(self.models)
        self.weights = [1.0 / n_models] * n_models
        logger.info(f"Using equal voting weights: {self.weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        probabilities = self.predict_proba(X)[:, 1]
        threshold = self.config.get("evaluation", {}).get("threshold", 0.5)
        return (probabilities >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get predictions from all models
        all_predictions = []
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'decision_function'):
                    pred = model.decision_function(X)
                    # Normalize to [0, 1]
                    pred = 1 / (1 + np.exp(-pred))
                else:
                    pred = model.predict(X)

                all_predictions.append(pred)

            except Exception as e:
                logger.error(f"Error predicting with model {model_name}: {str(e)}")
                continue

        if not all_predictions:
            raise ValueError("No models available for prediction")

        all_predictions = np.array(all_predictions).T

        # Combine predictions based on ensemble method
        if self.ensemble_method == "stacking" and self.stacking_meta_model is not None:
            combined_predictions = self.stacking_meta_model.predict(all_predictions)
        else:
            # Weighted average or voting
            weights = np.array(self.weights)
            if len(weights) != all_predictions.shape[1]:
                # Equal weights if mismatch
                weights = np.ones(all_predictions.shape[1]) / all_predictions.shape[1]

            combined_predictions = np.average(all_predictions, axis=1, weights=weights)

        # Return probabilities for both classes
        probabilities = np.column_stack([1 - combined_predictions, combined_predictions])
        return probabilities

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }

        # Calculate PR AUC
        from sklearn.metrics import average_precision_score
        metrics['pr_auc'] = average_precision_score(y, y_proba)

        # Log loss
        from sklearn.metrics import log_loss
        metrics['log_loss'] = log_loss(y, y_proba)

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from ensemble.

        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importances = []

        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    model_importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients
                    model_importance = np.abs(model.coef_[0])
                else:
                    continue

                importances.append({
                    'model': model_name,
                    'importance': model_importance
                })

            except Exception as e:
                logger.warning(f"Could not get feature importance for {model_name}: {str(e)}")
                continue

        if not importances:
            return pd.DataFrame()

        # Aggregate importances
        all_importances = np.array([imp['importance'] for imp in importances])
        avg_importance = all_importances.mean(axis=0)

        # Create DataFrame
        n_features = len(avg_importance)
        feature_names = [f'feature_{i}' for i in range(n_features)]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance,
            'std_importance': all_importances.std(axis=0)
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        self.feature_importances_ = importance_df
        return importance_df

    def get_model_performance(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Get individual model performance metrics.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            DataFrame with model performance metrics
        """
        results = []

        for model_name, model in self.models.items():
            try:
                y_pred = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]
                else:
                    y_proba = y_pred

                metrics = {
                    'model': model_name,
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred),
                    'recall': recall_score(y, y_pred),
                    'f1_score': f1_score(y, y_pred),
                    'roc_auc': roc_auc_score(y, y_proba)
                }

                # Add cross-validation scores if available
                if model_name in self.cv_scores_:
                    cv_auc = self.cv_scores_[model_name]['mean_roc_auc']
                    metrics['cv_roc_auc'] = cv_auc

                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def save_model(self, path: str) -> None:
        """Save ensemble model to file."""
        model_data = {
            'models': self.models,
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'stacking_meta_model': self.stacking_meta_model,
            'config': self.config,
            'cv_scores': self.cv_scores_,
            'is_fitted': self.is_fitted,
            'feature_importances': self.feature_importances_
        }

        joblib.dump(model_data, path)
        logger.info(f"Ensemble model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load ensemble model from file."""
        model_data = joblib.load(path)

        self.models = model_data['models']
        self.ensemble_method = model_data['ensemble_method']
        self.weights = model_data['weights']
        self.stacking_meta_model = model_data['stacking_meta_model']
        self.config = model_data['config']
        self.cv_scores_ = model_data['cv_scores']
        self.is_fitted = model_data['is_fitted']
        self.feature_importances_ = model_data.get('feature_importances')

        logger.info(f"Ensemble model loaded from {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'ensemble_method': self.ensemble_method,
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'weights': self.weights,
            'is_fitted': self.is_fitted,
            'cv_scores': self.cv_scores_
        }

        if self.feature_importances_ is not None:
            summary['top_features'] = self.feature_importances_.head(10)['feature'].tolist()

        return summary


class AdaptiveEnsemble(EnsembleModel):
    """
    Adaptive ensemble that adjusts weights based on recent performance.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.performance_history = []
        self.adaptation_rate = config.get("adaptation_rate", 0.1)
        self.min_weight = config.get("min_weight", 0.05)

    def update_weights(self, recent_performance: Dict[str, float]) -> None:
        """
        Update ensemble weights based on recent performance.

        Args:
            recent_performance: Dictionary of model performance metrics
        """
        self.performance_history.append(recent_performance)

        # Keep only recent history
        max_history = self.config.get("performance_history_size", 10)
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]

        # Calculate average performance
        avg_performance = {}
        for model_name in self.models.keys():
            performances = [perf.get(model_name, 0) for perf in self.performance_history]
            avg_performance[model_name] = np.mean(performances)

        # Update weights
        performance_values = np.array(list(avg_performance.values()))
        performance_values = np.maximum(performance_values, 0)  # Ensure non-negative

        # Apply softmax with adaptation
        exp_performance = np.exp(performance_values / self.adaptation_rate)
        new_weights = exp_performance / exp_performance.sum()

        # Apply minimum weight constraint
        new_weights = np.maximum(new_weights, self.min_weight)
        new_weights = new_weights / new_weights.sum()  # Renormalize

        self.weights = new_weights.tolist()
        logger.info(f"Updated weights: {self.weights}")


if __name__ == "__main__":
    # Example usage
    config = {
        "ensemble": {
            "method": "weighted_average",
            "weights": [0.3, 0.4, 0.3]
        },
        "models": {
            "random_forest": {
                "class_name": "RandomForestClassifier",
                "parameters": {"n_estimators": 100, "random_state": 42}
            },
            "xgboost": {
                "class_name": "XGBClassifier",
                "parameters": {"n_estimators": 100, "random_state": 42}
            },
            "logistic_regression": {
                "class_name": "LogisticRegression",
                "parameters": {"random_state": 42}
            }
        }
    }

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    # Initialize and train ensemble
    ensemble = EnsembleModel(config)
    ensemble.fit(X, y)

    # Evaluate
    metrics = ensemble.evaluate(X, y)
    print("Ensemble Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Get feature importance
    importance_df = ensemble.get_feature_importance()
    print("\nTop 5 Features:")
    print(importance_df.head())

    # Save model
    ensemble.save_model("ensemble_model.joblib")