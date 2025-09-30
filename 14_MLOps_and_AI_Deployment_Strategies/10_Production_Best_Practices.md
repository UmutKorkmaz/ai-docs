# Production Best Practices

**Navigation**: [← Module 9: AIOps and Automation](09_AIOps_and_Automation.md) | [Main Index](README.md) | [Module 11: Security and Compliance →](11_Security_and_Compliance.md)

## Overview

Production best practices ensure that machine learning systems are reliable, scalable, maintainable, and secure. This module covers comprehensive guidelines for deploying and managing ML models in production environments.

## Production Readiness Checklist

### Model Deployment Readiness

```python
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import yaml
import os
import hashlib
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import requests
from functools import wraps
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class ReadinessCheck:
    """Structure for readiness check results"""
    check_name: str
    passed: bool
    score: float
    message: str
    recommendations: List[str]
    critical: bool = False

@dataclass
class ProductionReadinessReport:
    """Comprehensive production readiness report"""
    timestamp: datetime
    overall_score: float
    is_ready: bool
    checks: Dict[str, ReadinessCheck]
    critical_issues: List[str]
    recommendations: List[str]
    next_steps: List[str]

class ProductionReadinessChecker:
    """
    Comprehensive production readiness evaluation for ML models.
    """

    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        self.model = self.load_model()
        self.config = self.load_config() if config_path else {}
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Setup logger for readiness checker"""
        logger = logging.getLogger('readiness_checker')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def load_model(self):
        """Load model for evaluation"""
        try:
            if self.model_path.endswith('.pkl'):
                return joblib.load(self.model_path)
            elif self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                return torch.load(self.model_path, map_location='cpu')
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def evaluate_production_readiness(self, test_data_path: str = None) -> ProductionReadinessReport:
        """Run comprehensive production readiness evaluation"""
        self.logger.info("Starting production readiness evaluation")

        all_checks = {}

        # Performance checks
        all_checks['model_performance'] = self.check_model_performance(test_data_path)
        all_checks['model_robustness'] = self.check_model_robustness(test_data_path)
        all_checks['inference_performance'] = self.check_inference_performance()

        # Quality checks
        all_checks['data_quality'] = self.check_data_quality()
        all_checks['model_fairness'] = self.check_model_fairness(test_data_path)
        all_checks['model_interpretability'] = self.check_model_interpretability()

        # Infrastructure checks
        all_checks['scalability'] = self.check_scalability()
        all_checks['monitoring'] = self.check_monitoring_readiness()
        all_checks['deployment'] = self.check_deployment_readiness()

        # Operational checks
        all_checks['documentation'] = self.check_documentation()
        all_checks['testing'] = self.check_testing_coverage()
        all_checks['security'] = self.check_security_readiness()

        # Calculate overall score
        total_score = sum(check.score for check in all_checks.values())
        overall_score = total_score / len(all_checks)

        # Identify critical issues
        critical_issues = [
            f"{check.check_name}: {check.message}"
            for check in all_checks.values()
            if check.critical and not check.passed
        ]

        # Generate recommendations
        recommendations = self.generate_recommendations(all_checks)

        # Generate next steps
        next_steps = self.generate_next_steps(all_checks, overall_score)

        report = ProductionReadinessReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            is_ready=overall_score >= 0.8 and len(critical_issues) == 0,
            checks=all_checks,
            critical_issues=critical_issues,
            recommendations=recommendations,
            next_steps=next_steps
        )

        self.logger.info(f"Production readiness evaluation completed. Overall score: {overall_score:.2f}")
        return report

    def check_model_performance(self, test_data_path: str = None) -> ReadinessCheck:
        """Check model performance metrics"""
        try:
            if test_data_path is None:
                return ReadinessCheck(
                    check_name="model_performance",
                    passed=False,
                    score=0.0,
                    message="No test data provided",
                    recommendations=["Provide test data for performance evaluation"],
                    critical=True
                )

            # Load test data
            test_data = pd.read_csv(test_data_path)
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']

            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Get thresholds from config
                thresholds = self.config.get('performance_thresholds', {
                    'accuracy': 0.85,
                    'precision': 0.80,
                    'recall': 0.80,
                    'f1': 0.80
                })

                # Check each metric
                metric_results = {
                    'accuracy': {'value': accuracy, 'threshold': thresholds['accuracy'], 'passed': accuracy >= thresholds['accuracy']},
                    'precision': {'value': precision, 'threshold': thresholds['precision'], 'passed': precision >= thresholds['precision']},
                    'recall': {'value': recall, 'threshold': thresholds['recall'], 'passed': recall >= thresholds['recall']},
                    'f1': {'value': f1, 'threshold': thresholds['f1'], 'passed': f1 >= thresholds['f1']}
                }

                passed_metrics = sum(1 for result in metric_results.values() if result['passed'])
                total_metrics = len(metric_results)
                score = passed_metrics / total_metrics

                failed_metrics = [name for name, result in metric_results.items() if not result['passed']]

                return ReadinessCheck(
                    check_name="model_performance",
                    passed=score >= 0.8,
                    score=score,
                    message=f"Performance evaluation: {passed_metrics}/{total_metrics} metrics met thresholds",
                    recommendations=[
                        f"Improve {metric} (current: {metric_results[metric]['value']:.3f}, threshold: {metric_results[metric]['threshold']:.3f})"
                        for metric in failed_metrics
                    ] + ["Consider model retraining or feature engineering"] if failed_metrics else [],
                    critical=score < 0.6
                )

            else:
                return ReadinessCheck(
                    check_name="model_performance",
                    passed=False,
                    score=0.0,
                    message="Model does not support predict_proba method",
                    recommendations=["Implement probability prediction capability"],
                    critical=True
                )

        except Exception as e:
            self.logger.error(f"Model performance check failed: {e}")
            return ReadinessCheck(
                check_name="model_performance",
                passed=False,
                score=0.0,
                message=f"Performance evaluation failed: {str(e)}",
                recommendations=["Fix model evaluation issues"],
                critical=True
            )

    def check_model_robustness(self, test_data_path: str = None) -> ReadinessCheck:
        """Check model robustness to input variations"""
        try:
            if test_data_path is None:
                return ReadinessCheck(
                    check_name="model_robustness",
                    passed=False,
                    score=0.0,
                    message="No test data provided",
                    recommendations=["Provide test data for robustness evaluation"],
                    critical=False
                )

            # Load test data
            test_data = pd.read_csv(test_data_path)
            X_test = test_data.drop('target', axis=1).head(100)  # Use subset for efficiency
            y_test = test_data['target'].head(100)

            # Get original predictions
            original_pred = self.model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_pred)

            # Test with different noise levels
            noise_tests = [
                (0.001, "Very low noise"),
                (0.01, "Low noise"),
                (0.1, "Medium noise")
            ]

            robustness_results = []

            for noise_level, description in noise_tests:
                noise = np.random.normal(0, noise_level, X_test.shape)
                X_noisy = X_test + noise

                noisy_pred = self.model.predict(X_noisy)
                noisy_accuracy = accuracy_score(y_test, noisy_pred)

                accuracy_drop = original_accuracy - noisy_accuracy

                robustness_results.append({
                    'noise_level': noise_level,
                    'description': description,
                    'accuracy_drop': accuracy_drop,
                    'robust': accuracy_drop < 0.05  # 5% threshold
                })

            # Test with missing values
            X_missing = X_test.copy()
            X_missing.iloc[0, 0] = np.nan

            try:
                missing_pred = self.model.predict(X_missing)
                missing_robust = True
            except Exception:
                missing_robust = False

            # Test with extreme values
            X_extreme = X_test.copy()
            for col in X_extreme.select_dtypes(include=[np.number]).columns:
                X_extreme[col] = X_extreme[col] * 10

            try:
                extreme_pred = self.model.predict(X_extreme)
                extreme_robust = True
            except Exception:
                extreme_robust = False

            # Calculate robustness score
            robust_tests = [
                result['robust'] for result in robustness_results
            ] + [missing_robust, extreme_robust]

            passed_tests = sum(robust_tests)
            total_tests = len(robust_tests)
            score = passed_tests / total_tests

            recommendations = []
            if not all(result['robust'] for result in robustness_results):
                recommendations.extend([
                    "Add input preprocessing and normalization",
                    "Implement input validation and sanitization",
                    "Consider robust training techniques"
                ])

            if not missing_robust:
                recommendations.append("Handle missing values in preprocessing")

            if not extreme_robust:
                recommendations.append("Add input value range validation")

            return ReadinessCheck(
                check_name="model_robustness",
                passed=score >= 0.8,
                score=score,
                message=f"Robustness evaluation: {passed_tests}/{total_tests} tests passed",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Model robustness check failed: {e}")
            return ReadinessCheck(
                check_name="model_robustness",
                passed=False,
                score=0.0,
                message=f"Robustness evaluation failed: {str(e)}",
                recommendations=["Fix robustness testing issues"],
                critical=False
            )

    def check_inference_performance(self) -> ReadinessCheck:
        """Check model inference performance"""
        try:
            # Create test input
            if hasattr(self.model, 'n_features_in_'):
                n_features = self.model.n_features_in_
            else:
                n_features = 10  # Default

            test_input = np.random.randn(1, n_features)

            # Warm up
            _ = self.model.predict(test_input)

            # Measure inference time
            latencies = []
            for _ in range(100):
                start_time = time.time()
                _ = self.model.predict(test_input)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

            latencies = np.array(latencies)

            # Calculate statistics
            mean_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            # Check against thresholds
            thresholds = self.config.get('latency_thresholds', {
                'mean_latency_ms': 100,
                'p95_latency_ms': 200,
                'p99_latency_ms': 500
            })

            latency_checks = [
                ('mean_latency', mean_latency, thresholds['mean_latency_ms']),
                ('p95_latency', p95_latency, thresholds['p95_latency_ms']),
                ('p99_latency', p99_latency, thresholds['p99_latency_ms'])
            ]

            passed_checks = sum(1 for _, value, threshold in latency_checks if value <= threshold)
            total_checks = len(latency_checks)
            score = passed_checks / total_checks

            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Test with larger batch
            batch_input = np.random.randn(10, n_features)
            _ = self.model.predict(batch_input)

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            memory_threshold = self.config.get('memory_threshold_mb', 100)
            memory_check = memory_increase <= memory_threshold

            if not memory_check:
                score *= 0.8  # Reduce score for memory issues

            recommendations = []
            failed_latency = [name for name, value, threshold in latency_checks if value > threshold]

            if failed_latency:
                recommendations.extend([
                    f"Optimize model for {name} (current: {dict(latency_checks)[name]:.1f}ms, threshold: {threshold}ms)"
                    for name, value, threshold in latency_checks if value > threshold
                ])

            if not memory_check:
                recommendations.append(f"Reduce memory footprint (increase: {memory_increase:.1f}MB, threshold: {memory_threshold}MB)")

            recommendations.extend([
                "Consider model quantization or pruning",
                "Implement batch processing for inference",
                "Use hardware acceleration if available"
            ])

            return ReadinessCheck(
                check_name="inference_performance",
                passed=score >= 0.7,
                score=score,
                message=f"Inference performance: {passed_checks}/{total_checks} latency checks passed, memory: {'OK' if memory_check else 'HIGH'}",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Inference performance check failed: {e}")
            return ReadinessCheck(
                check_name="inference_performance",
                passed=False,
                score=0.0,
                message=f"Informance performance evaluation failed: {str(e)}",
                recommendations=["Fix inference performance testing issues"],
                critical=True
            )

    def check_data_quality(self) -> ReadinessCheck:
        """Check data quality and preprocessing"""
        try:
            recommendations = []
            score = 1.0

            # Check if preprocessing pipeline is defined
            preprocessing_config = self.config.get('preprocessing', {})
            if not preprocessing_config:
                score -= 0.3
                recommendations.append("Define preprocessing pipeline configuration")

            # Check for data validation
            data_validation = self.config.get('data_validation', {})
            if not data_validation:
                score -= 0.2
                recommendations.append("Implement data validation rules")

            # Check for missing value handling
            if 'missing_value_handling' not in preprocessing_config:
                score -= 0.2
                recommendations.append("Define missing value handling strategy")

            # Check for feature scaling
            if 'feature_scaling' not in preprocessing_config:
                score -= 0.1
                recommendations.append("Specify feature scaling approach")

            # Check for outlier detection
            if 'outlier_detection' not in preprocessing_config:
                score -= 0.1
                recommendations.append("Implement outlier detection and handling")

            # Check for data versioning
            if not self.config.get('data_versioning_enabled', False):
                score -= 0.1
                recommendations.append("Enable data versioning and lineage tracking")

            return ReadinessCheck(
                check_name="data_quality",
                passed=score >= 0.7,
                score=max(score, 0.0),
                message=f"Data quality checks: {int(score * 10)}/10 requirements met",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Data quality check failed: {e}")
            return ReadinessCheck(
                check_name="data_quality",
                passed=False,
                score=0.0,
                message=f"Data quality evaluation failed: {str(e)}",
                recommendations=["Fix data quality configuration"],
                critical=False
            )

    def check_model_fairness(self, test_data_path: str = None) -> ReadinessCheck:
        """Check model fairness across different groups"""
        try:
            if test_data_path is None:
                return ReadinessCheck(
                    check_name="model_fairness",
                    passed=False,
                    score=0.0,
                    message="No test data provided for fairness evaluation",
                    recommendations=["Provide test data with protected attributes"],
                    critical=False
                )

            # Load test data
            test_data = pd.read_csv(test_data_path)

            # Look for common protected attributes
            protected_attributes = self.config.get('protected_attributes', [])
            if not protected_attributes:
                # Try to detect common protected attributes
                potential_attributes = ['gender', 'race', 'age', 'income', 'education']
                protected_attributes = [attr for attr in potential_attributes if attr in test_data.columns]

            if not protected_attributes:
                return ReadinessCheck(
                    check_name="model_fairness",
                    passed=False,
                    score=0.0,
                    message="No protected attributes found or specified",
                    recommendations=["Identify and specify protected attributes for fairness testing"],
                    critical=False
                )

            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']
            predictions = self.model.predict(X_test)

            fairness_results = []
            for attr in protected_attributes:
                if attr not in test_data.columns:
                    continue

                # Calculate fairness metrics for each group
                unique_values = test_data[attr].unique()
                group_metrics = {}

                for value in unique_values:
                    group_mask = test_data[attr] == value
                    group_y_true = y_test[group_mask]
                    group_y_pred = predictions[group_mask]

                    if len(group_y_true) > 0:
                        group_metrics[value] = {
                            'accuracy': accuracy_score(group_y_true, group_y_pred),
                            'precision': precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                            'sample_size': len(group_y_true)
                        }

                # Calculate fairness metrics
                if len(group_metrics) >= 2:
                    accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
                    accuracy_disparity = max(accuracies) - min(accuracies)

                    # Statistical parity
                    positive_rates = {}
                    for value, metrics in group_metrics.items():
                        group_mask = test_data[attr] == value
                        positive_rates[value] = predictions[group_mask].mean()

                    parity_difference = max(positive_rates.values()) - min(positive_rates.values())

                    fairness_results.append({
                        'attribute': attr,
                        'accuracy_disparity': accuracy_disparity,
                        'parity_difference': parity_difference,
                        'groups': list(group_metrics.keys())
                    })

            # Evaluate fairness results
            if not fairness_results:
                return ReadinessCheck(
                    check_name="model_fairness",
                    passed=False,
                    score=0.0,
                    message="Could not calculate fairness metrics",
                    recommendations=["Ensure sufficient samples per protected group"],
                    critical=False
                )

            # Check against thresholds
            fairness_threshold = self.config.get('fairness_threshold', 0.1)
            fairness_issues = []

            for result in fairness_results:
                if result['accuracy_disparity'] > fairness_threshold:
                    fairness_issues.append(
                        f"High accuracy disparity for {result['attribute']}: {result['accuracy_disparity']:.3f}"
                    )
                if result['parity_difference'] > fairness_threshold:
                    fairness_issues.append(
                        f"High statistical parity difference for {result['attribute']}: {result['parity_difference']:.3f}"
                    )

            score = 1.0 - (len(fairness_issues) / (len(fairness_results) * 2))

            recommendations = []
            if fairness_issues:
                recommendations.extend(fairness_issues)
                recommendations.extend([
                    "Consider bias mitigation techniques",
                    "Collect more diverse training data",
                    "Implement fairness-aware training",
                    "Add fairness metrics to monitoring"
                ])

            return ReadinessCheck(
                check_name="model_fairness",
                passed=score >= 0.8,
                score=max(score, 0.0),
                message=f"Fairness evaluation: {len(fairness_issues)} issues found across {len(fairness_results)} attributes",
                recommendations=recommendations,
                critical=len(fairness_issues) > 2
            )

        except Exception as e:
            self.logger.error(f"Model fairness check failed: {e}")
            return ReadinessCheck(
                check_name="model_fairness",
                passed=False,
                score=0.0,
                message=f"Fairness evaluation failed: {str(e)}",
                recommendations=["Fix fairness evaluation issues"],
                critical=False
            )

    def check_model_interpretability(self) -> ReadinessCheck:
        """Check model interpretability and explainability"""
        try:
            recommendations = []
            score = 1.0

            # Check if interpretability methods are configured
            interpretability_config = self.config.get('interpretability', {})
            if not interpretability_config:
                score -= 0.4
                recommendations.append("Configure model interpretability methods")

            # Check for feature importance
            if not hasattr(self.model, 'feature_importances_') and 'feature_importance' not in interpretability_config:
                score -= 0.2
                recommendations.append("Implement feature importance calculation")

            # Check for SHAP/LIME integration
            if not interpretability_config.get('shap_enabled', False) and not interpretability_config.get('lime_enabled', False):
                score -= 0.2
                recommendations.append("Enable SHAP or LIME for local explanations")

            # Check for explanation storage
            if not interpretability_config.get('store_explanations', False):
                score -= 0.1
                recommendations.append("Configure explanation storage and retrieval")

            # Check for explanation API
            if not interpretability_config.get('explanation_api', False):
                score -= 0.1
                recommendations.append("Implement explanation API endpoint")

            return ReadinessCheck(
                check_name="model_interpretability",
                passed=score >= 0.7,
                score=max(score, 0.0),
                message=f"Model interpretability: {int(score * 10)}/10 requirements met",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Model interpretability check failed: {e}")
            return ReadinessCheck(
                check_name="model_interpretability",
                passed=False,
                score=0.0,
                message=f"Interpretability evaluation failed: {str(e)}",
                recommendations=["Fix interpretability configuration"],
                critical=False
            )

    def check_scalability(self) -> ReadinessCheck:
        """Check model and system scalability"""
        try:
            recommendations = []
            score = 1.0

            # Check for horizontal scaling support
            if not self.config.get('horizontal_scaling_enabled', False):
                score -= 0.3
                recommendations.append("Enable horizontal scaling capabilities")

            # Check for load balancing
            if not self.config.get('load_balancing_configured', False):
                score -= 0.2
                recommendations.append("Configure load balancing for model endpoints")

            # Check for caching strategy
            caching_config = self.config.get('caching', {})
            if not caching_config:
                score -= 0.2
                recommendations.append("Implement inference caching strategy")

            # Check for batch processing
            if not self.config.get('batch_processing_enabled', False):
                score -= 0.1
                recommendations.append("Enable batch processing for high-throughput scenarios")

            # Check for auto-scaling
            if not self.config.get('auto_scaling_configured', False):
                score -= 0.1
                recommendations.append("Configure auto-scaling based on load")

            # Check for distributed inference
            if not self.config.get('distributed_inference_enabled', False):
                score -= 0.1
                recommendations.append("Consider distributed inference for large models")

            return ReadinessCheck(
                check_name="scalability",
                passed=score >= 0.7,
                score=max(score, 0.0),
                message=f"Scalability checks: {int(score * 10)}/10 requirements met",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Scalability check failed: {e}")
            return ReadinessCheck(
                check_name="scalability",
                passed=False,
                score=0.0,
                message=f"Scalability evaluation failed: {str(e)}",
                recommendations=["Fix scalability configuration"],
                critical=True
            )

    def check_monitoring_readiness(self) -> ReadinessCheck:
        """Check monitoring and observability setup"""
        try:
            recommendations = []
            score = 1.0

            # Check for performance monitoring
            monitoring_config = self.config.get('monitoring', {})
            if not monitoring_config:
                score -= 0.3
                recommendations.append("Configure comprehensive monitoring system")

            # Check for metrics collection
            required_metrics = ['prediction_latency', 'prediction_count', 'error_rate', 'model_accuracy']
            configured_metrics = monitoring_config.get('metrics', [])
            missing_metrics = [metric for metric in required_metrics if metric not in configured_metrics]

            if missing_metrics:
                score -= 0.2
                recommendations.append(f"Add missing metrics: {', '.join(missing_metrics)}")

            # Check for alerting
            if not monitoring_config.get('alerting_enabled', False):
                score -= 0.2
                recommendations.append("Configure alerting rules and notifications")

            # Check for dashboard
            if not monitoring_config.get('dashboard_configured', False):
                score -= 0.1
                recommendations.append("Create monitoring dashboard")

            # Check for logging
            if not monitoring_config.get('structured_logging', False):
                score -= 0.1
                recommendations.append("Implement structured logging")

            # Check for drift detection
            if not monitoring_config.get('drift_detection_enabled', False):
                score -= 0.1
                recommendations.append("Enable data and model drift detection")

            return ReadinessCheck(
                check_name="monitoring",
                passed=score >= 0.7,
                score=max(score, 0.0),
                message=f"Monitoring readiness: {int(score * 10)}/10 requirements met",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Monitoring check failed: {e}")
            return ReadinessCheck(
                check_name="monitoring",
                passed=False,
                score=0.0,
                message=f"Monitoring evaluation failed: {str(e)}",
                recommendations=["Fix monitoring configuration"],
                critical=True
            )

    def check_deployment_readiness(self) -> ReadinessCheck:
        """Check deployment configuration and readiness"""
        try:
            recommendations = []
            score = 1.0

            # Check for deployment strategy
            deployment_config = self.config.get('deployment', {})
            if not deployment_config:
                score -= 0.3
                recommendations.append("Define deployment strategy and configuration")

            # Check for containerization
            if not deployment_config.get('containerized', False):
                score -= 0.2
                recommendations.append("Containerize model for consistent deployment")

            # Check for CI/CD pipeline
            if not deployment_config.get('cicd_configured', False):
                score -= 0.2
                recommendations.append("Set up CI/CD pipeline for automated deployment")

            # Check for rollback mechanism
            if not deployment_config.get('rollback_enabled', False):
                score -= 0.1
                recommendations.append("Implement rollback mechanism for failed deployments")

            # Check for blue-green/canary deployment
            if not deployment_config.get('advanced_deployment', False):
                score -= 0.1
                recommendations.append("Consider blue-green or canary deployment strategies")

            # Check for health checks
            if not deployment_config.get('health_checks', False):
                score -= 0.1
                recommendations.append("Implement comprehensive health checks")

            return ReadinessCheck(
                check_name="deployment",
                passed=score >= 0.7,
                score=max(score, 0.0),
                message=f"Deployment readiness: {int(score * 10)}/10 requirements met",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Deployment check failed: {e}")
            return ReadinessCheck(
                check_name="deployment",
                passed=False,
                score=0.0,
                message=f"Deployment evaluation failed: {str(e)}",
                recommendations=["Fix deployment configuration"],
                critical=True
            )

    def check_documentation(self) -> ReadinessCheck:
        """Check documentation completeness"""
        try:
            recommendations = []
            score = 1.0

            # Check for model documentation
            if not self.config.get('model_documentation', False):
                score -= 0.3
                recommendations.append("Create comprehensive model documentation")

            # Check for API documentation
            if not self.config.get('api_documentation', False):
                score -= 0.2
                recommendations.append("Document API endpoints and usage")

            # Check for deployment guide
            if not self.config.get('deployment_guide', False):
                score -= 0.2
                recommendations.append("Write deployment and operations guide")

            # Check for troubleshooting guide
            if not self.config.get('troubleshooting_guide', False):
                score -= 0.1
                recommendations.append("Create troubleshooting guide")

            # Check for monitoring guide
            if not self.config.get('monitoring_guide', False):
                score -= 0.1
                recommendations.append("Document monitoring and alerting setup")

            # Check for onboarding guide
            if not self.config.get('onboarding_guide', False):
                score -= 0.1
                recommendations.append("Create team onboarding documentation")

            return ReadinessCheck(
                check_name="documentation",
                passed=score >= 0.7,
                score=max(score, 0.0),
                message=f"Documentation: {int(score * 10)}/10 components documented",
                recommendations=recommendations,
                critical=score < 0.4
            )

        except Exception as e:
            self.logger.error(f"Documentation check failed: {e}")
            return ReadinessCheck(
                check_name="documentation",
                passed=False,
                score=0.0,
                message=f"Documentation evaluation failed: {str(e)}",
                recommendations=["Fix documentation issues"],
                critical=False
            )

    def check_testing_coverage(self) -> ReadinessCheck:
        """Check testing coverage and quality"""
        try:
            recommendations = []
            score = 1.0

            # Check for unit tests
            testing_config = self.config.get('testing', {})
            if not testing_config.get('unit_tests', False):
                score -= 0.3
                recommendations.append("Implement comprehensive unit tests")

            # Check for integration tests
            if not testing_config.get('integration_tests', False):
                score -= 0.2
                recommendations.append("Add integration tests for end-to-end workflows")

            # Check for performance tests
            if not testing_config.get('performance_tests', False):
                score -= 0.2
                recommendations.append("Create performance and load tests")

            # Check for model tests
            if not testing_config.get('model_tests', False):
                score -= 0.1
                recommendations.append("Implement model-specific tests (accuracy, robustness)")

            # Check for security tests
            if not testing_config.get('security_tests', False):
                score -= 0.1
                recommendations.append("Add security testing for vulnerabilities")

            # Check for test coverage reporting
            if not testing_config.get('coverage_reporting', False):
                score -= 0.1
                recommendations.append("Enable test coverage reporting")

            return ReadinessCheck(
                check_name="testing",
                passed=score >= 0.7,
                score=max(score, 0.0),
                message=f"Testing coverage: {int(score * 10)}/10 test types implemented",
                recommendations=recommendations,
                critical=score < 0.5
            )

        except Exception as e:
            self.logger.error(f"Testing check failed: {e}")
            return ReadinessCheck(
                check_name="testing",
                passed=False,
                score=0.0,
                message=f"Testing evaluation failed: {str(e)}",
                recommendations=["Fix testing configuration"],
                critical=True
            )

    def check_security_readiness(self) -> ReadinessCheck:
        """Check security measures and compliance"""
        try:
            recommendations = []
            score = 1.0

            # Check for authentication
            security_config = self.config.get('security', {})
            if not security_config.get('authentication_enabled', False):
                score -= 0.3
                recommendations.append("Implement authentication for model endpoints")

            # Check for authorization
            if not security_config.get('authorization_enabled', False):
                score -= 0.2
                recommendations.append("Add role-based access control")

            # Check for input validation
            if not security_config.get('input_validation', False):
                score -= 0.2
                recommendations.append("Implement comprehensive input validation")

            # Check for data encryption
            if not security_config.get('encryption_enabled', False):
                score -= 0.1
                recommendations.append("Enable data encryption at rest and in transit")

            # Check for audit logging
            if not security_config.get('audit_logging', False):
                score -= 0.1
                recommendations.append("Implement audit logging for compliance")

            # Check for vulnerability scanning
            if not security_config.get('vulnerability_scanning', False):
                score -= 0.1
                recommendations.append("Set up regular vulnerability scanning")

            return ReadinessCheck(
                check_name="security",
                passed=score >= 0.8,
                score=max(score, 0.0),
                message=f"Security readiness: {int(score * 10)}/10 measures implemented",
                recommendations=recommendations,
                critical=score < 0.6
            )

        except Exception as e:
            self.logger.error(f"Security check failed: {e}")
            return ReadinessCheck(
                check_name="security",
                passed=False,
                score=0.0,
                message=f"Security evaluation failed: {str(e)}",
                recommendations=["Fix security configuration"],
                critical=True
            )

    def generate_recommendations(self, checks: Dict[str, ReadinessCheck]) -> List[str]:
        """Generate prioritized recommendations"""
        all_recommendations = []

        # Get recommendations from failed checks
        for check_name, check in checks.values():
            if not check.passed:
                all_recommendations.extend(check.recommendations)

        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))

        # Prioritize critical check recommendations first
        critical_recommendations = []
        normal_recommendations = []

        for rec in unique_recommendations:
            is_critical = any(
                check.critical and not check.passed and rec in check.recommendations
                for check in checks.values()
            )

            if is_critical:
                critical_recommendations.append(rec)
            else:
                normal_recommendations.append(rec)

        return critical_recommendations + normal_recommendations

    def generate_next_steps(self, checks: Dict[str, ReadinessCheck], overall_score: float) -> List[str]:
        """Generate prioritized next steps"""
        next_steps = []

        if overall_score < 0.5:
            next_steps.append("🚨 Address critical issues before deployment")
        elif overall_score < 0.8:
            next_steps.append("⚠️ Fix major issues before production deployment")
        else:
            next_steps.append("✅ Ready for production deployment with minor improvements")

        # Get failed critical checks
        failed_critical = [
            check_name for check_name, check in checks.items()
            if check.critical and not check.passed
        ]

        if failed_critical:
            next_steps.append(f"🔧 Fix critical checks: {', '.join(failed_critical)}")

        # Get high-impact improvements
        high_impact_checks = [
            (check_name, check)
            for check_name, check in checks.items()
            if not check.passed and check.score < 0.5
        ]

        if high_impact_checks:
            next_steps.append(f"📈 Focus on high-impact improvements: {', '.join([name for name, _ in high_impact_checks])}")

        # Suggest staging deployment
        if 0.6 <= overall_score < 0.8:
            next_steps.append("🧪 Deploy to staging environment for further testing")

        # Suggest monitoring setup
        monitoring_check = checks.get('monitoring')
        if monitoring_check and not monitoring_check.passed:
            next_steps.append("📊 Set up monitoring before full deployment")

        return next_steps

    def generate_report(self, output_path: str = None):
        """Generate comprehensive readiness report"""
        # Run evaluation
        report = self.evaluate_production_readiness()

        # Convert to dictionary
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'overall_score': report.overall_score,
            'is_ready': report.is_ready,
            'critical_issues': report.critical_issues,
            'recommendations': report.recommendations,
            'next_steps': report.next_steps,
            'checks': {
                name: {
                    'check_name': check.check_name,
                    'passed': check.passed,
                    'score': check.score,
                    'message': check.message,
                    'recommendations': check.recommendations,
                    'critical': check.critical
                }
                for name, check in report.checks.items()
            }
        }

        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)

        return report_dict

# Usage example
if __name__ == "__main__":
    # Example configuration
    config = {
        'performance_thresholds': {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.80,
            'f1': 0.80
        },
        'latency_thresholds': {
            'mean_latency_ms': 100,
            'p95_latency_ms': 200,
            'p99_latency_ms': 500
        },
        'memory_threshold_mb': 100,
        'protected_attributes': ['gender', 'age'],
        'fairness_threshold': 0.1,
        'preprocessing': {
            'missing_value_handling': 'mean',
            'feature_scaling': 'standard'
        },
        'monitoring': {
            'metrics': ['prediction_latency', 'prediction_count', 'error_rate'],
            'alerting_enabled': True,
            'structured_logging': True
        },
        'deployment': {
            'containerized': True,
            'cicd_configured': True,
            'health_checks': True
        },
        'security': {
            'authentication_enabled': True,
            'input_validation': True,
            'audit_logging': True
        }
    }

    # Create readiness checker
    checker = ProductionReadinessChecker(
        model_path="models/churn_model.pkl",
        config_path="config/readiness_config.yaml"
    )

    # Generate report
    report = checker.generate_report("production_readiness_report.json")

    print("Production Readiness Report Generated:")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print(f"Ready for Production: {report['is_ready']}")
    print(f"Critical Issues: {len(report['critical_issues'])}")
    print("\nNext Steps:")
    for step in report['next_steps']:
        print(f"- {step}")
```

## Model Deployment Strategies

```python
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import aiohttp
import docker
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import yaml
import uuid

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    model_version: str
    image_name: str
    replicas: int = 3
    cpu_limit: str = "500m"
    memory_limit: str = "1Gi"
    gpu_limit: Optional[str] = None
    autoscaling_enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    environment: str = "production"

@dataclass
class DeploymentResult:
    """Result of deployment operation"""
    deployment_id: str
    status: str  # success, failed, in_progress
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    endpoint_url: Optional[str] = None
    rollback_available: bool = False

class DeploymentStrategy(ABC):
    """Abstract base class for deployment strategies"""

    def __init__(self, k8s_client: client.CoreV1Api, apps_client: client.AppsV1Api):
        self.k8s_client = k8s_client
        self.apps_client = apps_client
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Setup deployment logger"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    @abstractmethod
    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy model using specific strategy"""
        pass

    @abstractmethod
    async def rollback(self, deployment_id: str) -> bool:
        """Rollback deployment"""
        pass

class BlueGreenDeployment(DeploymentStrategy):
    """Blue-green deployment strategy for zero-downtime updates"""

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy using blue-green strategy"""
        deployment_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting blue-green deployment: {deployment_id}")

            # Create green deployment
            green_deployment_name = f"{config.model_name}-green"
            green_service_name = f"{config.model_name}-green-service"

            # Create deployment manifest
            deployment_manifest = self.create_deployment_manifest(config, green_deployment_name)
            service_manifest = self.create_service_manifest(config, green_service_name)

            # Apply green deployment
            await self.apply_kubernetes_manifest(deployment_manifest)
            await self.apply_kubernetes_manifest(service_manifest)

            # Wait for green deployment to be ready
            await self.wait_for_deployment_ready(green_deployment_name, timeout=300)

            # Run health checks on green deployment
            await self.run_health_checks(green_service_name, config.health_check_path)

            # Switch traffic to green (update main service)
            main_service_name = f"{config.model_name}-service"
            await self.switch_service_traffic(main_service_name, green_service_name)

            # Clean up blue deployment (old version)
            blue_deployment_name = f"{config.model_name}-blue"
            await self.cleanup_deployment(blue_deployment_name)

            # Rename green to blue for next deployment
            await self.rename_deployment(green_deployment_name, blue_deployment_name)

            endpoint_url = await self.get_service_endpoint(main_service_name)

            return DeploymentResult(
                deployment_id=deployment_id,
                status="success",
                start_time=start_time,
                end_time=datetime.now(),
                endpoint_url=endpoint_url,
                rollback_available=True
            )

        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            return DeploymentResult(
                deployment_id=deployment_id,
                status="failed",
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )

    async def rollback(self, deployment_id: str) -> bool:
        """Rollback blue-green deployment"""
        try:
            self.logger.info(f"Rolling back deployment: {deployment_id}")

            # Switch traffic back to blue deployment
            main_service_name = f"{config.model_name}-service"
            blue_service_name = f"{config.model_name}-blue-service"

            await self.switch_service_traffic(main_service_name, blue_service_name)

            # Run health checks
            await self.run_health_checks(blue_service_name, "/health")

            self.logger.info("Rollback completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def create_deployment_manifest(self, config: DeploymentConfig, deployment_name: str) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        container_config = {
            "name": "model-server",
            "image": config.image_name,
            "ports": [{"containerPort": 8000}],
            "resources": {
                "limits": {
                    "cpu": config.cpu_limit,
                    "memory": config.memory_limit
                },
                "requests": {
                    "cpu": "250m",
                    "memory": "512Mi"
                }
            },
            "env": [
                {"name": "MODEL_NAME", "value": config.model_name},
                {"name": "MODEL_VERSION", "value": config.model_version},
                {"name": "ENVIRONMENT", "value": config.environment}
            ],
            "livenessProbe": {
                "httpGet": {"path": config.health_check_path, "port": 8000},
                "initialDelaySeconds": 30,
                "periodSeconds": 10
            },
            "readinessProbe": {
                "httpGet": {"path": config.readiness_check_path, "port": 8000},
                "initialDelaySeconds": 5,
                "periodSeconds": 5
            }
        }

        if config.gpu_limit:
            container_config["resources"]["limits"]["nvidia.com/gpu"] = config.gpu_limit

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "labels": {
                    "app": config.model_name,
                    "version": config.model_version,
                    "deployment": deployment_name
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.model_name,
                        "deployment": deployment_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.model_name,
                            "version": config.model_version,
                            "deployment": deployment_name
                        }
                    },
                    "spec": {
                        "containers": [container_config]
                    }
                }
            }
        }

    def create_service_manifest(self, config: DeploymentConfig, service_name: str) -> Dict[str, Any]:
        """Create Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service_name,
                "labels": {
                    "app": config.model_name,
                    "service": service_name
                }
            },
            "spec": {
                "selector": {
                    "app": config.model_name,
                    "service": service_name
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }

    async def apply_kubernetes_manifest(self, manifest: Dict[str, Any]):
        """Apply Kubernetes manifest"""
        if manifest["kind"] == "Deployment":
            await self.apps_client.create_namespaced_deployment(
                namespace="default",
                body=manifest
            )
        elif manifest["kind"] == "Service":
            await self.k8s_client.create_namespaced_service(
                namespace="default",
                body=manifest
            )

    async def wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300):
        """Wait for deployment to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                deployment = await self.apps_client.read_namespaced_deployment(
                    name=deployment_name,
                    namespace="default"
                )

                if (deployment.status.ready_replicas == deployment.spec.replicas and
                    deployment.status.available_replicas == deployment.spec.replicas):
                    self.logger.info(f"Deployment {deployment_name} is ready")
                    return

            except ApiException:
                pass

            await asyncio.sleep(5)

        raise TimeoutError(f"Deployment {deployment_name} not ready within {timeout} seconds")

    async def run_health_checks(self, service_name: str, health_path: str):
        """Run health checks on deployed service"""
        try:
            # Get service cluster IP
            service = await self.k8s_client.read_namespaced_service(
                name=service_name,
                namespace="default"
            )

            cluster_ip = service.spec.cluster_ip
            if not cluster_ip:
                raise ValueError(f"Service {service_name} has no cluster IP")

            # Health check endpoint
            health_url = f"http://{cluster_ip}:8000{health_path}"

            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        self.logger.info(f"Health check passed for {service_name}")
                    else:
                        raise Exception(f"Health check failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}")
            raise

    async def switch_service_traffic(self, main_service_name: str, target_service_name: str):
        """Switch service traffic to target deployment"""
        try:
            # Update main service selector to point to target deployment
            service = await self.k8s_client.read_namespaced_service(
                name=main_service_name,
                namespace="default"
            )

            service.spec.selector = {
                "app": service.metadata.labels.get("app"),
                "service": target_service_name
            }

            await self.k8s_client.patch_namespaced_service(
                name=main_service_name,
                namespace="default",
                body=service
            )

            self.logger.info(f"Traffic switched to {target_service_name}")

        except Exception as e:
            self.logger.error(f"Failed to switch traffic: {e}")
            raise

    async def cleanup_deployment(self, deployment_name: str):
        """Clean up old deployment"""
        try:
            await self.apps_client.delete_namespaced_deployment(
                name=deployment_name,
                namespace="default"
            )
            self.logger.info(f"Cleaned up deployment: {deployment_name}")
        except ApiException:
            pass  # Deployment might not exist

    async def rename_deployment(self, old_name: str, new_name: str):
        """Rename deployment by creating new one and deleting old"""
        try:
            # Get current deployment
            deployment = await self.apps_client.read_namespaced_deployment(
                name=old_name,
                namespace="default"
            )

            # Create new deployment with new name
            new_deployment = deployment.copy()
            new_deployment.metadata.name = new_name
            new_deployment.metadata.labels["deployment"] = new_name
            new_deployment.spec.selector.match_labels["deployment"] = new_name
            new_deployment.spec.template.metadata.labels["deployment"] = new_name

            await self.apps_client.create_namespaced_deployment(
                namespace="default",
                body=new_deployment
            )

            # Wait for new deployment to be ready
            await self.wait_for_deployment_ready(new_name)

            # Delete old deployment
            await self.apps_client.delete_namespaced_deployment(
                name=old_name,
                namespace="default"
            )

            self.logger.info(f"Renamed deployment from {old_name} to {new_name}")

        except Exception as e:
            self.logger.error(f"Failed to rename deployment: {e}")
            raise

    async def get_service_endpoint(self, service_name: str) -> str:
        """Get service endpoint URL"""
        try:
            service = await self.k8s_client.read_namespaced_service(
                name=service_name,
                namespace="default"
            )

            # In production, this would be the actual load balancer URL
            return f"http://{service_name}.default.svc.cluster.local"

        except Exception as e:
            self.logger.error(f"Failed to get service endpoint: {e}")
            return f"http://{service_name}.default.svc.cluster.local"

class CanaryDeployment(DeploymentStrategy):
    """Canary deployment strategy for gradual rollout"""

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy using canary strategy"""
        deployment_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting canary deployment: {deployment_id}")

            # Create canary deployment (small percentage of traffic)
            canary_deployment_name = f"{config.model_name}-canary"
            initial_traffic_percentage = 10  # Start with 10% traffic

            # Deploy canary with initial configuration
            await self.deploy_canary(config, canary_deployment_name, initial_traffic_percentage)

            # Monitor canary performance
            monitoring_duration = 300  # 5 minutes
            canary_healthy = await self.monitor_canary_performance(
                canary_deployment_name, monitoring_duration
            )

            if canary_healthy:
                # Gradually increase traffic
                traffic_percentages = [25, 50, 75, 100]
                for percentage in traffic_percentages:
                    await self.update_canary_traffic(canary_deployment_name, percentage)
                    await self.monitor_canary_performance(canary_deployment_name, 60)  # 1 minute per step

                # Promote canary to stable
                await self.promote_canary_to_stable(config, canary_deployment_name)
            else:
                # Rollback canary
                await self.rollback_canary(canary_deployment_name)
                raise Exception("Canary deployment failed health checks")

            endpoint_url = await self.get_service_endpoint(f"{config.model_name}-service")

            return DeploymentResult(
                deployment_id=deployment_id,
                status="success",
                start_time=start_time,
                end_time=datetime.now(),
                endpoint_url=endpoint_url,
                rollback_available=True
            )

        except Exception as e:
            self.logger.error(f"Canary deployment failed: {e}")
            return DeploymentResult(
                deployment_id=deployment_id,
                status="failed",
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )

    async def deploy_canary(self, config: DeploymentConfig, deployment_name: str, traffic_percentage: int):
        """Deploy canary with specific traffic percentage"""
        # Create canary deployment
        canary_config = DeploymentConfig(
            **config.__dict__,
            replicas=max(1, config.replicas // 4)  # Start with fewer replicas
        )

        deployment_manifest = self.create_deployment_manifest(canary_config, deployment_name)
        await self.apply_kubernetes_manifest(deployment_manifest)

        # Wait for canary to be ready
        await self.wait_for_deployment_ready(deployment_name)

        # Configure traffic split using Istio or similar
        await self.configure_traffic_split(config.model_name, deployment_name, traffic_percentage)

    async def monitor_canary_performance(self, deployment_name: str, duration: int) -> bool:
        """Monitor canary deployment performance"""
        self.logger.info(f"Monitoring canary {deployment_name} for {duration} seconds")

        start_time = time.time()
        health_check_interval = 30

        while time.time() - start_time < duration:
            try:
                # Check deployment health
                deployment = await self.apps_client.read_namespaced_deployment(
                    name=deployment_name,
                    namespace="default"
                )

                if (deployment.status.ready_replicas == 0 or
                    deployment.status.available_replicas == 0):
                    self.logger.error(f"Canary {deployment_name} has no ready replicas")
                    return False

                # Check error rates, latency, etc.
                metrics = await self.get_deployment_metrics(deployment_name)

                error_rate = metrics.get('error_rate', 0)
                avg_latency = metrics.get('avg_latency_ms', 0)

                if error_rate > 5 or avg_latency > 1000:  # 5% error rate or 1s latency
                    self.logger.error(f"Canary {deployment_name} performance degraded")
                    return False

                await asyncio.sleep(health_check_interval)

            except Exception as e:
                self.logger.error(f"Error monitoring canary {deployment_name}: {e}")
                return False

        self.logger.info(f"Canary {deployment_name} monitoring passed")
        return True

    async def update_canary_traffic(self, deployment_name: str, traffic_percentage: int):
        """Update canary traffic percentage"""
        await self.configure_traffic_split(
            deployment_name.replace('-canary', ''),  # Base model name
            deployment_name,
            traffic_percentage
        )
        self.logger.info(f"Updated canary traffic to {traffic_percentage}%")

    async def promote_canary_to_stable(self, config: DeploymentConfig, canary_name: str):
        """Promote canary to stable deployment"""
        self.logger.info(f"Promoting canary {canary_name} to stable")

        # Update main deployment
        stable_deployment_name = f"{config.model_name}"
        stable_manifest = self.create_deployment_manifest(config, stable_deployment_name)

        # Apply stable deployment
        await self.apps_client.patch_namespaced_deployment(
            name=stable_deployment_name,
            namespace="default",
            body=stable_manifest
        )

        # Wait for stable deployment to be ready
        await self.wait_for_deployment_ready(stable_deployment_name)

        # Remove canary
        await self.cleanup_deployment(canary_name)

        # Reset traffic split to 100% stable
        await self.configure_traffic_split(config.model_name, stable_deployment_name, 100)

    async def rollback_canary(self, deployment_name: str):
        """Rollback canary deployment"""
        self.logger.info(f"Rolling back canary {deployment_name}")

        # Reset traffic to 100% stable
        base_name = deployment_name.replace('-canary', '')
        await self.configure_traffic_split(base_name, f"{base_name}-stable", 100)

        # Remove canary deployment
        await self.cleanup_deployment(deployment_name)

    async def configure_traffic_split(self, base_name: str, canary_name: str, canary_percentage: int):
        """Configure traffic split between stable and canary"""
        # This would typically use Istio VirtualService or similar
        # For simplicity, we'll simulate the configuration
        self.logger.info(f"Configuring traffic split: {canary_percentage}% to {canary_name}")

    async def get_deployment_metrics(self, deployment_name: str) -> Dict[str, float]:
        """Get deployment performance metrics"""
        # This would query Prometheus or similar monitoring system
        # For now, return simulated metrics
        return {
            'error_rate': 1.0,  # 1% error rate
            'avg_latency_ms': 150.0,  # 150ms average latency
            'throughput': 100.0  # 100 requests/second
        }

class DeploymentManager:
    """Manages different deployment strategies"""

    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path) if config_path else {}
        self.logger = self.setup_logger()
        self.k8s_configured = self.setup_kubernetes()

        if self.k8s_configured:
            self.k8s_client = client.CoreV1Api()
            self.apps_client = client.AppsV1Api()
        else:
            self.logger.warning("Kubernetes not configured, deployment strategies limited")

        self.strategies = {
            'blue_green': BlueGreenDeployment(self.k8s_client, self.apps_client),
            'canary': CanaryDeployment(self.k8s_client, self.apps_client)
        }

    def setup_logger(self):
        """Setup deployment manager logger"""
        logger = logging.getLogger('deployment_manager')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def setup_kubernetes(self) -> bool:
        """Setup Kubernetes client"""
        try:
            config.load_kube_config()
            return True
        except Exception:
            try:
                config.load_incluster_config()
                return True
            except Exception:
                self.logger.warning("Kubernetes configuration not found")
                return False

    async def deploy_model(self, deployment_config: DeploymentConfig, strategy: str = 'blue_green') -> DeploymentResult:
        """Deploy model using specified strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown deployment strategy: {strategy}")

        if not self.k8s_configured:
            raise Exception("Kubernetes not configured for deployment")

        deployment_strategy = self.strategies[strategy]
        return await deployment_strategy.deploy(deployment_config)

    async def rollback_deployment(self, deployment_id: str, strategy: str = 'blue_green') -> bool:
        """Rollback deployment"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown deployment strategy: {strategy}")

        deployment_strategy = self.strategies[strategy]
        return await deployment_strategy.rollback(deployment_id)

    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            deployment = self.apps_client.read_namespaced_deployment(
                name=deployment_name,
                namespace="default"
            )

            return {
                'name': deployment.metadata.name,
                'replicas': deployment.spec.replicas,
                'ready_replicas': deployment.status.ready_replicas,
                'available_replicas': deployment.status.available_replicas,
                'updated_replicas': deployment.status.updated_replicas,
                'status': 'ready' if (
                    deployment.status.ready_replicas == deployment.spec.replicas and
                    deployment.status.available_replicas == deployment.spec.replicas
                ) else 'updating'
            }

        except ApiException as e:
            return {
                'name': deployment_name,
                'error': str(e),
                'status': 'not_found'
            }

# Usage example
if __name__ == "__main__":
    # Example deployment configuration
    deployment_config = DeploymentConfig(
        model_name="customer-churn-predictor",
        model_version="v2.1.0",
        image_name="ml-registry/customer-churn:v2.1.0",
        replicas=3,
        cpu_limit="1000m",
        memory_limit="2Gi",
        autoscaling_enabled=True,
        min_replicas=2,
        max_replicas=10,
        target_cpu_utilization=70,
        health_check_path="/health",
        environment="production"
    )

    # Create deployment manager
    deployment_manager = DeploymentManager("config/deployment_config.yaml")

    # Deploy using blue-green strategy
    async def deploy_example():
        try:
            result = await deployment_manager.deploy_model(deployment_config, 'blue_green')
            print(f"Deployment completed: {result.status}")
            if result.endpoint_url:
                print(f"Model endpoint: {result.endpoint_url}")
        except Exception as e:
            print(f"Deployment failed: {e}")

    # Run deployment
    asyncio.run(deploy_example())
```

## Quick Reference

### Production Readiness Checklist

1. **Model Performance**
   - Accuracy, precision, recall meet business requirements
   - Performance validated on representative test data
   - Model robustness tested with various inputs
   - Inference latency within acceptable limits

2. **Infrastructure Readiness**
   - Scalability architecture designed and tested
   - Monitoring and alerting configured
   - Deployment strategy selected (blue-green, canary, rolling)
   - Rollback mechanisms in place

3. **Operational Excellence**
   - Comprehensive documentation available
   - Testing coverage meets requirements
   - Security measures implemented
   - Incident response procedures defined

4. **Monitoring and Observability**
   - Key metrics identified and tracked
   - Alert thresholds configured appropriately
   - Dashboards created for visualization
   - Log aggregation and search configured

### Deployment Strategy Comparison

| Strategy | Downtime | Risk | Rollback Ease | Complexity | Best For |
|----------|----------|------|---------------|------------|----------|
| **Blue-Green** | Zero | Low | Easy | Medium | Critical services |
| **Canary** | Zero | Very Low | Easy | High | High-risk changes |
| **Rolling** | Minimal | Medium | Medium | Low | Stateless services |
| **Recreate** | High | High | Easy | Low | Simple applications |

### Best Practices Summary

1. **Start with readiness assessment** - Use comprehensive checklist
2. **Choose appropriate deployment strategy** based on risk tolerance
3. **Implement robust monitoring** before going to production
4. **Document everything** for team knowledge sharing
5. **Test rollback procedures** regularly
6. **Use infrastructure as code** for consistency
7. **Implement gradual rollouts** to minimize risk
8. **Monitor post-deployment** performance closely

## Summary

This module provides comprehensive production best practices for ML systems, including:

- **Production readiness assessment** with detailed scoring
- **Multiple deployment strategies** (blue-green, canary)
- **Comprehensive checklists** for all aspects of production deployment
- **Real-world implementations** with Kubernetes integration
- **Monitoring and observability** guidelines
- **Security and compliance** considerations

The implementation ensures that ML models are deployed safely, reliably, and with minimal risk to production systems.

**Next**: [Module 11: Security and Compliance](11_Security_and_Compliance.md)