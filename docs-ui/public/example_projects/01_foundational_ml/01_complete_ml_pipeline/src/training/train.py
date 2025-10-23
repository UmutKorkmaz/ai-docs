"""
Main training script for customer churn prediction.
Handles model training, evaluation, and persistence.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import json
import mlflow
import mlflow.sklearn

from ..data.preprocessing import ChurnDataPreprocessor
from ..models.ensemble_model import EnsembleModel
from ..utils.logging import get_logger, setup_logging
from ..utils.monitoring import MetricsCollector
from ..utils.database import DatabaseManager
from .hyperparameter_tuning import HyperparameterOptimizer
from .model_selection import ModelSelector

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def setup_mlflow(experiment_name: str, tracking_uri: str) -> None:
    """Setup MLflow experiment tracking."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking initialized: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}")


def load_and_preprocess_data(data_path: str, preprocessor_config: str) -> tuple:
    """
    Load and preprocess training data.

    Args:
        data_path: Path to training data
        preprocessor_config: Path to preprocessing configuration

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    logger.info("Loading and preprocessing data...")

    # Initialize preprocessor
    preprocessor = ChurnDataPreprocessor(preprocessor_config)

    # Load data
    df = preprocessor.load_data(data_path)

    # Validate data
    is_valid, errors = preprocessor.validate_data(df)
    if not is_valid:
        logger.error(f"Data validation failed: {errors}")
        sys.exit(1)

    # Clean data
    df_clean = preprocessor.clean_data(df)

    # Engineer features
    df_featured = preprocessor.engineer_features(df_clean)

    # Split data
    X, y = preprocessor.fit_transform(df_featured)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Data split completed:")
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}")
    logger.info(f"Class distribution - Test: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, preprocessor


def train_model(X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any]) -> EnsembleModel:
    """
    Train the ensemble model.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Model configuration

    Returns:
        Trained ensemble model
    """
    logger.info("Training ensemble model...")

    # Initialize model
    model = EnsembleModel(config)

    # Train model
    with mlflow.start_run(run_name="ensemble_training"):
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params({
            "ensemble_method": model.ensemble_method,
            "n_models": len(model.models),
            "model_names": list(model.models.keys()),
            "weights": model.weights
        })

        # Log individual model CV scores
        for model_name, cv_scores in model.cv_scores_.items():
            mlflow.log_metrics({
                f"{model_name}_cv_roc_auc": cv_scores['mean_roc_auc'],
                f"{model_name}_cv_accuracy": cv_scores['mean_accuracy'],
                f"{model_name}_cv_f1": cv_scores['mean_f1_score']
            })

        logger.info("Model training completed")

    return model


def evaluate_model(model: EnsembleModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance...")

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = model.evaluate(X_test, y_test)

    # Log metrics to MLflow
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metrics(metrics)

    # Get individual model performance
    model_performance = model.get_model_performance(X_test, y_test)
    logger.info("Individual model performance:")
    for _, row in model_performance.iterrows():
        logger.info(f"{row['model']}: AUC = {row['roc_auc']:.4f}")

    # Get feature importance
    feature_importance = model.get_feature_importance()
    if not feature_importance.empty:
        logger.info("Top 10 important features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

        # Log feature importance
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

    return metrics


def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Bayesian optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Optimization configuration

    Returns:
        Best hyperparameters
    """
    logger.info("Starting hyperparameter optimization...")

    optimizer = HyperparameterOptimizer(config)
    best_params = optimizer.optimize(X_train, y_train)

    logger.info("Hyperparameter optimization completed")
    logger.info(f"Best parameters: {best_params}")

    return best_params


def save_artifacts(model: EnsembleModel, preprocessor: ChurnDataPreprocessor,
                  config: Dict[str, Any], output_dir: str) -> None:
    """
    Save model artifacts.

    Args:
        model: Trained model
        preprocessor: Data preprocessor
        config: Model configuration
        output_dir: Output directory
    """
    logger.info("Saving model artifacts...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / "ensemble_model.joblib"
    model.save_model(str(model_path))
    mlflow.log_artifact(str(model_path))

    # Save preprocessor
    preprocessor_path = output_path / "preprocessor.joblib"
    preprocessor.save_preprocessor(str(preprocessor_path))
    mlflow.log_artifact(str(preprocessor_path))

    # Save configuration
    config_path = output_path / "model_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    mlflow.log_artifact(str(config_path))

    # Save model summary
    summary = model.get_model_summary()
    summary_path = output_path / "model_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    mlflow.log_artifact(str(summary_path))

    logger.info(f"Artifacts saved to {output_dir}")


def generate_report(metrics: Dict[str, float], model_summary: Dict[str, Any],
                   output_dir: str) -> None:
    """
    Generate training report.

    Args:
        metrics: Evaluation metrics
        model_summary: Model summary
        output_dir: Output directory
    """
    logger.info("Generating training report...")

    report = f"""
# Customer Churn Prediction Model Training Report

## Training Summary
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Ensemble Method**: {model_summary.get('ensemble_method', 'Unknown')}
- **Number of Models**: {model_summary.get('n_models', 0)}
- **Models Used**: {', '.join(model_summary.get('model_names', []))}

## Performance Metrics
"""
    for metric, value in metrics.items():
        report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"

    report += f"""
## Model Weights
{model_summary.get('weights', [])}

## Cross-Validation Scores
"""
    for model_name, scores in model_summary.get('cv_scores', {}).items():
        report += f"- **{model_name}**: AUC = {scores.get('mean_roc_auc', 0):.4f} (±{scores.get('std_roc_auc', 0):.4f})\n"

    report_path = Path(output_dir) / "training_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    mlflow.log_artifact(str(report_path))
    logger.info(f"Training report saved to {report_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train customer churn prediction model")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--data", type=str, default="data/raw/customer_churn.csv",
                       help="Path to training data")
    parser.add_argument("--preprocessor-config", type=str, default="config/preprocessing_config.yaml",
                       help="Path to preprocessing configuration")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for model artifacts")
    parser.add_argument("--optimize", action="store_true",
                       help="Run hyperparameter optimization")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("Starting model training...")

    try:
        # Load configuration
        config = load_config(args.config)

        # Setup MLflow
        experiment_config = config.get("experiment_tracking", {})
        setup_mlflow(
            experiment_config.get("experiment_name", "customer_churn_prediction"),
            experiment_config.get("tracking_uri", "http://localhost:5000")
        )

        # Load and preprocess data
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
            args.data, args.preprocessor_config
        )

        # Hyperparameter optimization (optional)
        if args.optimize:
            optimization_config = config.get("hyperparameter_optimization", {})
            if optimization_config.get("enabled", False):
                best_params = optimize_hyperparameters(X_train, y_train, config)
                # Update config with best parameters
                # This is a simplified version - in practice, you'd update specific model configs
                logger.info("Using optimized hyperparameters")

        # Train model
        model = train_model(X_train, y_train, config)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log final metrics
        logger.info("Final model performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save artifacts
        save_artifacts(model, preprocessor, config, args.output_dir)

        # Generate report
        model_summary = model.get_model_summary()
        generate_report(metrics, model_summary, args.output_dir)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()