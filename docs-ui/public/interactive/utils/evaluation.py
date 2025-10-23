"""
Evaluation Utilities for Interactive Notebooks

This module provides standardized evaluation functions for all interactive notebooks,
ensuring consistent performance assessment across different sections.

Author: AI Documentation Project
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation toolkit"""

    def __init__(self):
        self.results_history = []
        self.custom_metrics = {}

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_scores: Optional[np.ndarray] = None,
                               average: str = 'weighted',
                               save_results: bool = True) -> Dict[str, float]:
        """
        Comprehensive classification evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Predicted probabilities/scores (for AUC)
            average: Averaging method for multi-class
            save_results: Whether to save results to history

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

        # Add AUC if scores are provided
        if y_scores is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_scores)
                else:
                    # Multi-class AUC
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    metrics['auc'] = roc_auc_score(y_true_bin, y_scores, multi_class='ovr')
            except:
                metrics['auc'] = None

        # Add confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # Add classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)

        if save_results:
            self._save_evaluation_result('classification', metrics)

        return metrics

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                           save_results: bool = True) -> Dict[str, float]:
        """
        Comprehensive regression evaluation

        Args:
            y_true: True values
            y_pred: Predicted values
            save_results: Whether to save results to history

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else None
        }

        # Add additional metrics
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['mean_error'] = np.mean(y_true - y_pred)
        metrics['std_error'] = np.std(y_true - y_pred)

        if save_results:
            self._save_evaluation_result('regression', metrics)

        return metrics

    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray,
                           save_results: bool = True) -> Dict[str, float]:
        """
        Comprehensive clustering evaluation

        Args:
            X: Feature data
            labels: Cluster labels
            save_results: Whether to save results to history

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Internal clustering metrics
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None

        # Additional clustering statistics
        unique_labels = np.unique(labels)
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_sizes'] = [np.sum(labels == label) for label in unique_labels]

        if save_results:
            self._save_evaluation_result('clustering', metrics)

        return metrics

    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                            cv: int = 5, scoring: Union[str, List[str]] = None,
                            return_train_score: bool = False,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation on a model

        Args:
            model: Model to evaluate
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds
            scoring: Scoring metric(s) to use
            return_train_score: Whether to return training scores
            save_results: Whether to save results to history

        Returns:
            Dictionary of cross-validation results
        """
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'

        results = {}
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, return_train_score=return_train_score)

        if isinstance(scoring, str):
            results[f'{scoring}_mean'] = np.mean(cv_scores)
            results[f'{scoring}_std'] = np.std(cv_scores)
            results[f'{scoring}_scores'] = cv_scores
        else:
            for i, score_name in enumerate(scoring):
                results[f'{score_name}_mean'] = np.mean(cv_scores[:, i])
                results[f'{score_name}_std'] = np.std(cv_scores[:, i])
                results[f'{score_name}_scores'] = cv_scores[:, i]

        results['cv_folds'] = cv
        results['n_samples'] = len(X)

        if save_results:
            self._save_evaluation_result('cross_validation', results)

        return results

    def learning_curve_analysis(self, model, X: np.ndarray, y: np.ndarray,
                               cv: int = 5, train_sizes: Optional[np.ndarray] = None,
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze learning curves for a model

        Args:
            model: Model to evaluate
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds
            train_sizes: Training set sizes to evaluate
            save_results: Whether to save results to history

        Returns:
            Dictionary of learning curve results
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes
        )

        results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'test_scores_mean': np.mean(test_scores, axis=1),
            'test_scores_std': np.std(test_scores, axis=1)
        }

        # Calculate gap between training and test scores
        results['score_gap'] = results['train_scores_mean'] - results['test_scores_mean']
        results['max_gap'] = np.max(results['score_gap'])
        results['final_gap'] = results['score_gap'][-1]

        if save_results:
            self._save_evaluation_result('learning_curve', results)

        return results

    def validation_curve_analysis(self, model, X: np.ndarray, y: np.ndarray,
                                param_name: str, param_range: List[Any],
                                cv: int = 5, scoring: str = None,
                                save_results: bool = True) -> Dict[str, Any]:
        """
        Analyze validation curves for hyperparameter tuning

        Args:
            model: Model to evaluate
            X: Feature data
            y: Target data
            param_name: Parameter name to vary
            param_range: Parameter values to try
            cv: Number of cross-validation folds
            scoring: Scoring metric to use
            save_results: Whether to save results to history

        Returns:
            Dictionary of validation curve results
        """
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'

        train_scores, test_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring
        )

        results = {
            'param_range': param_range,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'test_scores_mean': np.mean(test_scores, axis=1),
            'test_scores_std': np.std(test_scores, axis=1)
        }

        # Find best parameter value
        best_idx = np.argmax(results['test_scores_mean'])
        results['best_param'] = param_range[best_idx]
        results['best_score'] = results['test_scores_mean'][best_idx]

        if save_results:
            self._save_evaluation_result('validation_curve', results)

        return results

    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                      cv: int = 5, scoring: str = None,
                      save_results: bool = True) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation

        Args:
            models: Dictionary of model names to model objects
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds
            scoring: Scoring metric to use
            save_results: Whether to save results to history

        Returns:
            DataFrame of model comparison results
        """
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'

        results = {}

        for model_name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results[model_name] = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'min_score': np.min(cv_scores),
                'max_score': np.max(cv_scores),
                'scores': cv_scores
            }

        df_results = pd.DataFrame(results).T
        df_results = df_results.sort_values('mean_score', ascending=False)

        if save_results:
            self._save_evaluation_result('model_comparison', df_results.to_dict())

        return df_results

    def evaluate_feature_importance(self, model, feature_names: List[str],
                                   save_results: bool = True) -> Dict[str, float]:
        """
        Evaluate and extract feature importance from model

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            save_results: Whether to save results to history

        Returns:
            Dictionary of feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) == 2 else np.abs(model.coef_)
        else:
            raise ValueError("Model does not have feature importances or coefficients")

        feature_importance = dict(zip(feature_names, importances))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        if save_results:
            self._save_evaluation_result('feature_importance', feature_importance)

        return feature_importance

    def add_custom_metric(self, name: str, metric_func: callable):
        """
        Add a custom evaluation metric

        Args:
            name: Name of the metric
            metric_func: Function that computes the metric
        """
        self.custom_metrics[name] = metric_func

    def _save_evaluation_result(self, evaluation_type: str, results: Dict[str, Any]):
        """Save evaluation result to history"""
        result = {
            'type': evaluation_type,
            'timestamp': pd.Timestamp.now(),
            'results': results
        }
        self.results_history.append(result)

    def get_evaluation_history(self) -> pd.DataFrame:
        """Get evaluation history as DataFrame"""
        return pd.DataFrame(self.results_history)

    def clear_history(self):
        """Clear evaluation history"""
        self.results_history = []

    def generate_evaluation_report(self, results: Dict[str, Any],
                                 model_name: str = "Model",
                                 evaluation_type: str = "Evaluation") -> str:
        """
        Generate a comprehensive evaluation report

        Args:
            results: Evaluation results dictionary
            model_name: Name of the model evaluated
            evaluation_type: Type of evaluation

        Returns:
            Formatted report string
        """
        report = f"\n{'='*60}\n"
        report += f"{evaluation_type} Report for {model_name}\n"
        report += f"{'='*60}\n\n"

        if evaluation_type == "classification":
            report += f"Accuracy: {results['accuracy']:.4f}\n"
            report += f"Precision: {results['precision']:.4f}\n"
            report += f"Recall: {results['recall']:.4f}\n"
            report += f"F1-Score: {results['f1']:.4f}\n"
            if 'auc' in results and results['auc'] is not None:
                report += f"AUC: {results['auc']:.4f}\n"

        elif evaluation_type == "regression":
            report += f"RMSE: {results['rmse']:.4f}\n"
            report += f"MAE: {results['mae']:.4f}\n"
            report += f"R²: {results['r2']:.4f}\n"
            if 'mape' in results and results['mape'] is not None:
                report += f"MAPE: {results['mape']:.2f}%\n"

        elif evaluation_type == "clustering":
            report += f"Number of Clusters: {results['n_clusters']}\n"
            if results['silhouette_score'] is not None:
                report += f"Silhouette Score: {results['silhouette_score']:.4f}\n"
            if results['calinski_harabasz_score'] is not None:
                report += f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}\n"
            if results['davies_bouldin_score'] is not None:
                report += f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}\n"

        report += f"\n{'='*60}\n"

        return report

class BenchmarkSuite:
    """Comprehensive benchmark suite for model evaluation"""

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.benchmark_results = {}

    def run_classification_benchmark(self, models: Dict[str, Any],
                                   X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   save_results: bool = True) -> pd.DataFrame:
        """
        Run comprehensive classification benchmark

        Args:
            models: Dictionary of model names to model objects
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            save_results: Whether to save results

        Returns:
            DataFrame of benchmark results
        """
        results = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_scores = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            # Evaluate
            metrics = self.evaluator.evaluate_classification(y_test, y_pred, y_scores, save_results=False)

            # Add cross-validation results
            cv_results = self.evaluator.cross_validate_model(model, X_train, y_train, save_results=False)
            metrics.update({f'cv_{k}': v for k, v in cv_results.items()})

            results[model_name] = metrics

        df_results = pd.DataFrame(results).T
        df_results = df_results.sort_values('accuracy', ascending=False)

        if save_results:
            self.benchmark_results[f'classification_benchmark_{len(self.benchmark_results)}'] = df_results

        return df_results

    def run_regression_benchmark(self, models: Dict[str, Any],
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                save_results: bool = True) -> pd.DataFrame:
        """
        Run comprehensive regression benchmark

        Args:
            models: Dictionary of model names to model objects
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            save_results: Whether to save results

        Returns:
            DataFrame of benchmark results
        """
        results = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate
            metrics = self.evaluator.evaluate_regression(y_test, y_pred, save_results=False)

            # Add cross-validation results
            cv_results = self.evaluator.cross_validate_model(model, X_train, y_train, scoring='r2', save_results=False)
            metrics.update({f'cv_{k}': v for k, v in cv_results.items()})

            results[model_name] = metrics

        df_results = pd.DataFrame(results).T
        df_results = df_results.sort_values('r2', ascending=False)

        if save_results:
            self.benchmark_results[f'regression_benchmark_{len(self.benchmark_results)}'] = df_results

        return df_results

    def get_benchmark_summary(self) -> pd.DataFrame:
        """Get summary of all benchmark results"""
        if not self.benchmark_results:
            return pd.DataFrame()

        summary_data = []
        for benchmark_name, results in self.benchmark_results.items():
            benchmark_type = benchmark_name.split('_')[0]
            best_model = results.index[0]
            best_score = results.iloc[0]['accuracy'] if 'accuracy' in results.columns else results.iloc[0]['r2']

            summary_data.append({
                'benchmark_name': benchmark_name,
                'benchmark_type': benchmark_type,
                'best_model': best_model,
                'best_score': best_score,
                'n_models': len(results)
            })

        return pd.DataFrame(summary_data)

# Global instances
evaluator = ModelEvaluator()
benchmark_suite = BenchmarkSuite()

# Convenience functions
def evaluate_classification(y_true, y_pred, y_scores=None, **kwargs):
    """Convenience function for classification evaluation"""
    return evaluator.evaluate_classification(y_true, y_pred, y_scores, **kwargs)

def evaluate_regression(y_true, y_pred, **kwargs):
    """Convenience function for regression evaluation"""
    return evaluator.evaluate_regression(y_true, y_pred, **kwargs)

def evaluate_clustering(X, labels, **kwargs):
    """Convenience function for clustering evaluation"""
    return evaluator.evaluate_clustering(X, labels, **kwargs)

def cross_validate_model(model, X, y, **kwargs):
    """Convenience function for cross-validation"""
    return evaluator.cross_validate_model(model, X, y, **kwargs)

def learning_curve_analysis(model, X, y, **kwargs):
    """Convenience function for learning curve analysis"""
    return evaluator.learning_curve_analysis(model, X, y, **kwargs)

def compare_models(models, X, y, **kwargs):
    """Convenience function for model comparison"""
    return evaluator.compare_models(models, X, y, **kwargs)

def run_classification_benchmark(models, X_train, y_train, X_test, y_test, **kwargs):
    """Convenience function for classification benchmark"""
    return benchmark_suite.run_classification_benchmark(models, X_train, y_train, X_test, y_test, **kwargs)

def run_regression_benchmark(models, X_train, y_train, X_test, y_test, **kwargs):
    """Convenience function for regression benchmark"""
    return benchmark_suite.run_regression_benchmark(models, X_train, y_train, X_test, y_test, **kwargs)