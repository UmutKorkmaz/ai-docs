---
title: "Mlops And Ai Deployment Strategies - Module 5: Monitoring"
description: "## Navigation. Comprehensive guide covering classification, algorithms, model training, regression, data preprocessing. Part of AI documentation system with ..."
keywords: "classification, regression, classification, algorithms, model training, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Module 5: Monitoring and Observability

## Navigation
- **← Previous**: [04_Infrastructure_and_Orchestration.md](04_Infrastructure_and_Orchestration.md)
- **→ Next**: [06_Model_Management_and_Versioning.md](06_Model_Management_and_Versioning.md)
- **↑ Up**: [README.md](README.md)

## Overview

Monitoring and observability are critical for maintaining reliable machine learning systems in production. This module covers comprehensive monitoring strategies, drift detection, logging, and alerting systems for ML operations.

## Comprehensive Monitoring Stack

### Advanced Model Monitoring System

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, Info
import time
import functools
import logging
from typing import Callable, Any, Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from enum import Enum

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    alert_type: str
    severity: AlertSeverity
    model_name: str
    model_version: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

class ModelMonitoring:
    """
    Comprehensive monitoring for ML models in production.
    """

    def __init__(self,
                 model_name: str,
                 model_version: str,
                 prometheus_port: int = 8000,
                 enable_alerts: bool = True):

        self.model_name = model_name
        self.model_version = model_version
        self.prometheus_port = prometheus_port
        self.enable_alerts = enable_alerts

        # Initialize metrics registry
        self.registry = CollectorRegistry()

        # Initialize metrics
        self.setup_metrics()

        # Initialize drift detector
        self.drift_detector = DriftDetector()

        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()

        # Initialize alert manager
        self.alert_manager = AlertManager() if enable_alerts else None

        # Initialize monitoring components
        self.setup_monitoring_components()

        # Start monitoring tasks
        self.start_monitoring_tasks()

        # Start Prometheus metrics server
        self.start_metrics_server()

    def setup_metrics(self):
        """Setup comprehensive Prometheus metrics"""

        # Request metrics
        self.request_count = Counter(
            'model_request_count',
            'Total number of requests',
            ['model_name', 'model_version', 'endpoint', 'method'],
            registry=self.registry
        )

        self.request_latency = Histogram(
            'model_request_latency_seconds',
            'Request latency in seconds',
            ['model_name', 'model_version', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )

        # Prediction metrics
        self.prediction_count = Counter(
            'model_prediction_count',
            'Total number of predictions',
            ['model_name', 'model_version', 'prediction_class', 'confidence_level'],
            registry=self.registry
        )

        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Prediction confidence distribution',
            ['model_name', 'model_version', 'prediction_class'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

        # Error metrics
        self.error_count = Counter(
            'model_error_count',
            'Total number of errors',
            ['model_name', 'model_version', 'error_type', 'endpoint'],
            registry=self.registry
        )

        self.error_rate = Gauge(
            'model_error_rate',
            'Error rate as percentage',
            ['model_name', 'model_version', 'endpoint'],
            registry=self.registry
        )

        # Model performance metrics
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy',
            ['model_name', 'model_version', 'dataset'],
            registry=self.registry
        )

        self.model_precision = Gauge(
            'model_precision',
            'Current model precision',
            ['model_name', 'model_version', 'dataset', 'class'],
            registry=self.registry
        )

        self.model_recall = Gauge(
            'model_recall',
            'Current model recall',
            ['model_name', 'model_version', 'dataset', 'class'],
            registry=self.registry
        )

        self.model_f1_score = Gauge(
            'model_f1_score',
            'Current model F1 score',
            ['model_name', 'model_version', 'dataset', 'class'],
            registry=self.registry
        )

        # Drift metrics
        self.model_drift_score = Gauge(
            'model_drift_score',
            'Current drift score',
            ['model_name', 'model_version', 'drift_type', 'feature'],
            registry=self.registry
        )

        self.drift_threshold = Gauge(
            'model_drift_threshold',
            'Drift detection threshold',
            ['model_name', 'model_version', 'drift_type'],
            registry=self.registry
        )

        # Resource metrics
        self.memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage in bytes',
            ['model_name', 'model_version', 'component'],
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'model_cpu_usage_percent',
            'CPU usage percentage',
            ['model_name', 'model_version', 'component'],
            registry=self.registry
        )

        self.gpu_usage = Gauge(
            'model_gpu_usage_percent',
            'GPU usage percentage',
            ['model_name', 'model_version', 'gpu_id', 'component'],
            registry=self.registry
        )

        self.gpu_memory_usage = Gauge(
            'model_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['model_name', 'model_version', 'gpu_id'],
            registry=self.registry
        )

        # Business metrics
        self.business_value = Gauge(
            'model_business_value',
            'Business value generated by model',
            ['model_name', 'model_version', 'metric_type'],
            registry=self.registry
        )

        self.user_satisfaction = Gauge(
            'model_user_satisfaction_score',
            'User satisfaction score',
            ['model_name', 'model_version'],
            registry=self.registry
        )

        # Model info
        self.model_info = Info(
            'model_info',
            'Model information',
            ['model_name', 'model_version'],
            registry=self.registry
        )

        # Set model info
        self.model_info.info(
            {'model_name': self.model_name, 'model_version': self.model_version,
             'created_at': datetime.now().isoformat(), 'monitoring_enabled': True}
        )

    def setup_monitoring_components(self):
        """Setup monitoring components"""
        self.reference_data = None
        self.prediction_buffer = []
        self.performance_window = []
        self.drift_history = []
        self.alert_history = []

        # Configuration
        self.config = {
            'drift_check_interval': 300,  # 5 minutes
            'performance_check_interval': 600,  # 10 minutes
            'buffer_size': 1000,
            'performance_window_size': 100,
            'drift_threshold': 0.05,
            'performance_threshold': 0.1,
            'alert_cooldown': 300  # 5 minutes
        }

    def start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        self.drift_check_thread = threading.Thread(
            target=self.drift_checker,
            daemon=True
        )
        self.drift_check_thread.start()

        self.performance_check_thread = threading.Thread(
            target=self.performance_checker,
            daemon=True
        )
        self.performance_check_thread.start()

        self.metrics_collector_thread = threading.Thread(
            target=self.metrics_collector,
            daemon=True
        )
        self.metrics_collector_thread.start()

    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            prometheus_client.start_http_server(self.prometheus_port, registry=self.registry)
            logging.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            logging.error(f"Failed to start Prometheus server: {e}")

    def track_request(self, endpoint: str, method: str = 'POST'):
        """Decorator to track request metrics"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Increment request count
                self.request_count.labels(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    endpoint=endpoint,
                    method=method
                ).inc()

                # Track latency
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)

                    # Record successful request
                    self.record_successful_request(endpoint, time.time() - start_time)

                    return result
                except Exception as e:
                    # Record error
                    self.record_error(endpoint, type(e).__name__)
                    raise
                finally:
                    # Record latency
                    self.request_latency.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        endpoint=endpoint
                    ).observe(time.time() - start_time)

            return wrapper
        return decorator

    def track_prediction(self,
                        prediction: int,
                        confidence: float,
                        features: np.ndarray = None,
                        true_label: int = None):
        """Track prediction metrics"""

        # Determine confidence level
        confidence_level = self.get_confidence_level(confidence)

        # Track prediction count
        self.prediction_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_class=str(prediction),
            confidence_level=confidence_level
        ).inc()

        # Track confidence distribution
        self.prediction_confidence.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_class=str(prediction)
        ).observe(confidence)

        # Add to prediction buffer for drift detection
        if features is not None:
            self.add_to_prediction_buffer({
                'prediction': prediction,
                'confidence': confidence,
                'features': features,
                'true_label': true_label,
                'timestamp': datetime.now()
            })

        # Update performance metrics if true label is available
        if true_label is not None:
            self.update_performance_metrics(prediction, true_label, confidence)

    def get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level based on probability"""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        else:
            return "low"

    def add_to_prediction_buffer(self, prediction_data: dict):
        """Add prediction to buffer for drift detection"""
        self.prediction_buffer.append(prediction_data)

        # Maintain buffer size
        if len(self.prediction_buffer) > self.config['buffer_size']:
            self.prediction_buffer.pop(0)

    def record_successful_request(self, endpoint: str, latency: float):
        """Record successful request"""
        # Update error rate (decrement)
        current_errors = self.error_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            error_type='total',
            endpoint=endpoint
        )._value.get()

        total_requests = self.request_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            endpoint=endpoint,
            method='POST'
        )._value.get()

        if total_requests > 0:
            error_rate = current_errors / total_requests
            self.error_rate.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                endpoint=endpoint
            ).set(error_rate)

    def record_error(self, endpoint: str, error_type: str):
        """Record error metrics"""
        # Increment error count
        self.error_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            error_type=error_type,
            endpoint=endpoint
        ).inc()

        # Update error rate
        total_requests = self.request_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            endpoint=endpoint,
            method='POST'
        )._value.get()

        error_count = self.error_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            error_type=error_type,
            endpoint=endpoint
        )._value.get()

        if total_requests > 0:
            error_rate = error_count / total_requests
            self.error_rate.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                endpoint=endpoint
            ).set(error_rate)

        # Send alert for high error rates
        if error_rate > 0.1:  # 10% error rate threshold
            self.send_alert(
                Alert(
                    alert_type='high_error_rate',
                    severity=AlertSeverity.CRITICAL,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    message=f"High error rate detected: {error_rate:.2%}",
                    timestamp=datetime.now(),
                    metadata={'endpoint': endpoint, 'error_rate': error_rate}
                )
            )

    def update_performance_metrics(self, prediction: int, true_label: int, confidence: float):
        """Update model performance metrics"""
        performance_data = {
            'prediction': prediction,
            'true_label': true_label,
            'confidence': confidence,
            'timestamp': datetime.now()
        }

        self.performance_window.append(performance_data)

        # Maintain window size
        if len(self.performance_window) > self.config['performance_window_size']:
            self.performance_window.pop(0)

        # Calculate performance metrics
        if len(self.performance_window) >= 10:  # Minimum samples
            self.calculate_performance_metrics()

    def calculate_performance_metrics(self):
        """Calculate and update performance metrics"""
        if not self.performance_window:
            return

        # Extract data
        predictions = [p['prediction'] for p in self.performance_window]
        true_labels = [p['true_label'] for p in self.performance_window]

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        import sklearn.metrics as metrics

        # Accuracy
        accuracy = metrics.accuracy_score(true_labels, predictions)
        self.model_accuracy.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            dataset='production'
        ).set(accuracy)

        # Classification report
        try:
            report = classification_report(true_labels, predictions, output_dict=True)

            # Update class-specific metrics
            for class_label in report.keys():
                if class_label.isdigit():
                    self.model_precision.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        dataset='production',
                        class=class_label
                    ).set(report[class_label]['precision'])

                    self.model_recall.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        dataset='production',
                        class=class_label
                    ).set(report[class_label]['recall'])

                    self.model_f1_score.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        dataset='production',
                        class=class_label
                    ).set(report[class_label]['f1-score'])
        except Exception as e:
            logging.error(f"Error calculating classification metrics: {e}")

    def check_drift(self, current_features: np.ndarray, reference_features: np.ndarray):
        """Check for data drift"""
        drift_scores = self.drift_detector.detect_drift(
            current_features,
            reference_features
        )

        for drift_type, score in drift_scores.items():
            # Update drift score metric
            self.model_drift_score.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                drift_type=drift_type,
                feature='overall'
            ).set(score)

            # Set drift threshold
            threshold = self.drift_detector.get_threshold(drift_type)
            self.drift_threshold.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                drift_type=drift_type
            ).set(threshold)

            # Check if drift exceeds threshold
            if score > threshold:
                self.trigger_drift_alert(drift_type, score, threshold)

                # Add to drift history
                self.drift_history.append({
                    'drift_type': drift_type,
                    'score': score,
                    'threshold': threshold,
                    'timestamp': datetime.now()
                })

    def trigger_drift_alert(self, drift_type: str, score: float, threshold: float):
        """Trigger alert for drift detection"""
        severity = AlertSeverity.WARNING if score < threshold * 2 else AlertSeverity.CRITICAL

        alert = Alert(
            alert_type='drift_detected',
            severity=severity,
            model_name=self.model_name,
            model_version=self.model_version,
            message=f"{drift_type} drift detected: score={score:.4f}, threshold={threshold:.4f}",
            timestamp=datetime.now(),
            metadata={
                'drift_type': drift_type,
                'drift_score': score,
                'drift_threshold': threshold,
                'severity_factor': score / threshold
            }
        )

        self.send_alert(alert)

    def send_alert(self, alert: Alert):
        """Send alert to monitoring system"""
        if not self.enable_alerts or not self.alert_manager:
            logging.warning(f"Alert: {alert}")
            return

        # Check cooldown
        if self.is_alert_in_cooldown(alert):
            return

        # Send alert
        self.alert_manager.send_alert(alert)

        # Add to alert history
        self.alert_history.append(alert)

    def is_alert_in_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period"""
        if not self.alert_history:
            return False

        # Find recent similar alerts
        recent_time = datetime.now() - timedelta(seconds=self.config['alert_cooldown'])
        recent_alerts = [a for a in self.alert_history if a.timestamp > recent_time]

        similar_alerts = [
            a for a in recent_alerts
            if a.alert_type == alert.alert_type and
               a.model_name == alert.model_name and
               a.model_version == alert.model_version
        ]

        return len(similar_alerts) > 0

    def drift_checker(self):
        """Background thread for drift checking"""
        while True:
            try:
                time.sleep(self.config['drift_check_interval'])

                if len(self.prediction_buffer) < 50:  # Minimum samples
                    continue

                # Extract features from buffer
                current_features = np.array([p['features'] for p in self.prediction_buffer[-50:]])

                if self.reference_data is None:
                    # Initialize reference data
                    self.reference_data = current_features.copy()
                    continue

                # Check drift
                self.check_drift(current_features, self.reference_data)

            except Exception as e:
                logging.error(f"Error in drift checker: {e}")
                time.sleep(60)  # Wait before retrying

    def performance_checker(self):
        """Background thread for performance checking"""
        while True:
            try:
                time.sleep(self.config['performance_check_interval'])

                if not self.performance_window:
                    continue

                # Check performance degradation
                current_accuracy = self.model_accuracy.labels(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    dataset='production'
                )._value.get()

                if current_accuracy < 0.8:  # 80% threshold
                    self.send_alert(
                        Alert(
                            alert_type='low_accuracy',
                            severity=AlertSeverity.WARNING,
                            model_name=self.model_name,
                            model_version=self.model_version,
                            message=f"Low model accuracy detected: {current_accuracy:.2%}",
                            timestamp=datetime.now(),
                            metadata={'current_accuracy': current_accuracy}
                        )
                    )

            except Exception as e:
                logging.error(f"Error in performance checker: {e}")
                time.sleep(60)

    def metrics_collector(self):
        """Background thread for system metrics collection"""
        while True:
            try:
                time.sleep(30)  # Collect every 30 seconds

                # Collect system metrics
                self.collect_system_metrics()

            except Exception as e:
                logging.error(f"Error in metrics collector: {e}")
                time.sleep(30)

    def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            import psutil
            process = psutil.Process()

            # Memory usage
            memory_info = process.memory_info()
            self.memory_usage.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                component='main'
            ).set(memory_info.rss)

            # CPU usage
            cpu_percent = process.cpu_percent(interval=1)
            self.cpu_usage.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                component='main'
            ).set(cpu_percent)

            # GPU metrics (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # GPU utilization
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_usage.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        gpu_id=str(i),
                        component='gpu'
                    ).set(gpu_util.gpu)

                    # GPU memory
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_memory_usage.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        gpu_id=str(i)
                    ).set(memory_info.used)

            except ImportError:
                # pynvml not available
                pass

        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")

    def get_monitoring_summary(self) -> dict:
        """Get monitoring summary"""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'total_requests': self.request_count._value.get(),
            'total_errors': sum(e._value.get() for e in self.error_count.collect()),
            'drift_checks': len(self.drift_history),
            'alerts_sent': len(self.alert_history),
            'buffer_size': len(self.prediction_buffer),
            'performance_window_size': len(self.performance_window)
        }

class DriftDetector:
    """Advanced drift detection for ML models"""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.drift_methods = {
            'kolmogorov_smirnov': self.ks_test,
            'population_stability_index': self.psi,
            'jensen_shannon': self.jensen_shannon_distance,
            'wasserstein': self.wasserstein_distance,
            'kl_divergence': self.kl_divergence,
            'chi_square': self.chi_square_test
        }

        self.reference_stats = {}

    def detect_drift(self, current: np.ndarray, reference: np.ndarray) -> dict:
        """Detect drift using multiple methods"""
        drift_scores = {}

        for method_name, method_func in self.drift_methods.items():
            try:
                score = method_func(current, reference)
                drift_scores[method_name] = score
            except Exception as e:
                logging.error(f"Error in {method_name}: {e}")
                drift_scores[method_name] = None

        # Calculate overall drift score
        valid_scores = [s for s in drift_scores.values() if s is not None]
        if valid_scores:
            drift_scores['overall'] = np.mean(valid_scores)
        else:
            drift_scores['overall'] = 0.0

        return drift_scores

    def get_threshold(self, drift_type: str) -> float:
        """Get threshold for drift type"""
        thresholds = {
            'kolmogorov_smirnov': 0.1,
            'population_stability_index': 0.25,
            'jensen_shannon': 0.1,
            'wasserstein': 0.1,
            'kl_divergence': 0.1,
            'chi_square': 0.1,
            'overall': self.threshold
        }
        return thresholds.get(drift_type, self.threshold)

    def ks_test(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Kolmogorov-Smirnov test for drift detection"""
        from scipy import stats

        ks_statistics = []

        for i in range(current.shape[1]):
            statistic, _ = stats.ks_2samp(current[:, i], reference[:, i])
            ks_statistics.append(statistic)

        return np.mean(ks_statistics)

    def psi(self, current: np.ndarray, reference: np.ndarray, buckets: int = 10) -> float:
        """Population Stability Index calculation"""
        psi_values = []

        for i in range(current.shape[1]):
            # Create bins
            min_val = min(reference[:, i].min(), current[:, i].min())
            max_val = max(reference[:, i].max(), current[:, i].max())
            bins = np.linspace(min_val, max_val, buckets + 1)

            # Calculate distributions
            ref_counts, _ = np.histogram(reference[:, i], bins=bins)
            curr_counts, _ = np.histogram(current[:, i], bins=bins)

            # Normalize
            ref_prop = (ref_counts + 1) / (len(reference) + buckets)
            curr_prop = (curr_counts + 1) / (len(current) + buckets)

            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            psi_values.append(psi)

        return np.mean(psi_values)

    def jensen_shannon_distance(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Jensen-Shannon distance for distribution comparison"""
        from scipy.spatial.distance import jensenshannon
        from scipy.stats import gaussian_kde

        js_distances = []

        for i in range(current.shape[1]):
            try:
                # Create probability distributions using KDE
                ref_kde = gaussian_kde(reference[:, i])
                curr_kde = gaussian_kde(current[:, i])

                # Sample points for comparison
                sample_points = np.linspace(
                    min(reference[:, i].min(), current[:, i].min()),
                    max(reference[:, i].max(), current[:, i].max()),
                    100
                )

                ref_dist = ref_kde(sample_points)
                curr_dist = curr_kde(sample_points)

                # Normalize
                ref_dist = ref_dist / ref_dist.sum()
                curr_dist = curr_dist / curr_dist.sum()

                # Calculate JS distance
                js_dist = jensenshannon(ref_dist, curr_dist)
                js_distances.append(js_dist)

            except Exception as e:
                logging.error(f"Error calculating JS distance for feature {i}: {e}")
                js_distances.append(0.0)

        return np.mean(js_distances)

    def wasserstein_distance(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Wasserstein distance for distribution comparison"""
        from scipy.stats import wasserstein_distance

        w_distances = []

        for i in range(current.shape[1]):
            try:
                w_dist = wasserstein_distance(current[:, i], reference[:, i])
                w_distances.append(w_dist)
            except Exception as e:
                logging.error(f"Error calculating Wasserstein distance for feature {i}: {e}")
                w_distances.append(0.0)

        return np.mean(w_distances)

    def kl_divergence(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Kullback-Leibler divergence calculation"""
        from scipy.stats import entropy

        kl_divergences = []

        for i in range(current.shape[1]):
            try:
                # Create histograms
                ref_hist, _ = np.histogram(reference[:, i], bins=50, density=True)
                curr_hist, _ = np.histogram(current[:, i], bins=50, density=True)

                # Calculate KL divergence
                kl_div = entropy(ref_hist, curr_hist)
                kl_divergences.append(kl_div)

            except Exception as e:
                logging.error(f"Error calculating KL divergence for feature {i}: {e}")
                kl_divergences.append(0.0)

        return np.mean(kl_divergences)

    def chi_square_test(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Chi-square test for distribution comparison"""
        from scipy.stats import chi2_contingency

        chi_square_values = []

        for i in range(current.shape[1]):
            try:
                # Create contingency table
                ref_counts, _ = np.histogram(reference[:, i], bins=10)
                curr_counts, _ = np.histogram(current[:, i], bins=10)

                # Create contingency table
                contingency_table = np.array([ref_counts, curr_counts])

                # Perform chi-square test
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                chi_square_values.append(p_value)

            except Exception as e:
                logging.error(f"Error calculating chi-square test for feature {i}: {e}")
                chi_square_values.append(1.0)  # No significant difference

        return np.mean(chi_square_values)

class AlertManager:
    """Advanced alert management system"""

    def __init__(self):
        self.alert_handlers = []
        self.alert_history = []
        self.alert_rules = {}

        # Setup default alert handlers
        self.setup_default_handlers()

    def setup_default_handlers(self):
        """Setup default alert handlers"""
        # Email handler
        self.add_alert_handler(EmailAlertHandler())

        # Slack handler
        self.add_alert_handler(SlackAlertHandler())

        # PagerDuty handler
        self.add_alert_handler(PagerDutyAlertHandler())

    def add_alert_handler(self, handler):
        """Add alert handler"""
        self.alert_handlers.append(handler)

    def send_alert(self, alert: Alert):
        """Send alert to all handlers"""
        for handler in self.alert_handlers:
            try:
                handler.send_alert(alert)
            except Exception as e:
                logging.error(f"Error sending alert with {handler.__class__.__name__}: {e}")

        # Add to history
        self.alert_history.append(alert)

    def add_alert_rule(self, rule_name: str, condition: callable, action: callable):
        """Add custom alert rule"""
        self.alert_rules[rule_name] = {
            'condition': condition,
            'action': action
        }

class PerformanceTracker:
    """Track and analyze model performance"""

    def __init__(self):
        self.performance_history = []
        self.baseline_metrics = {}

    def update_baseline(self, metrics: dict):
        """Update baseline metrics"""
        self.baseline_metrics = metrics

    def track_performance(self, metrics: dict):
        """Track performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

    def get_performance_trend(self, metric_name: str, window_hours: int = 24) -> dict:
        """Get performance trend for a metric"""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [
            p for p in self.performance_history
            if p['timestamp'] > cutoff_time
        ]

        if not recent_data:
            return {'trend': 'unknown', 'change': 0.0}

        values = [p['metrics'].get(metric_name, 0) for p in recent_data]

        # Calculate trend
        if len(values) < 2:
            return {'trend': 'stable', 'change': 0.0}

        # Simple linear regression
        x = list(range(len(values)))
        y = values

        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'degrading'

        return {
            'trend': trend,
            'change': slope,
            'current_value': values[-1],
            'baseline_value': self.baseline_metrics.get(metric_name, 0)
        }

# Example alert handlers
class EmailAlertHandler:
    def send_alert(self, alert: Alert):
        # Implementation for email alerts
        pass

class SlackAlertHandler:
    def send_alert(self, alert: Alert):
        # Implementation for Slack alerts
        pass

class PagerDutyAlertHandler:
    def send_alert(self, alert: Alert):
        # Implementation for PagerDuty alerts
        pass

# Usage example
if __name__ == "__main__":
    # Initialize monitoring
    monitor = ModelMonitoring(
        model_name="churn_prediction",
        model_version="1.0.0",
        prometheus_port=8000
    )

    # Example usage
    @monitor.track_request(endpoint="predict")
    def predict_function(features):
        # Your prediction logic here
        prediction = 1  # Example
        confidence = 0.85

        # Track prediction
        monitor.track_prediction(prediction, confidence, features)

        return prediction
```

## Logging and Observability

### Advanced Structured Logging System

```python
import logging
import json
import traceback
import sys
from datetime import datetime
from typing import Any, Dict, Optional, List
import contextvars
import uuid
import threading
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Context variables for request tracking
request_id_var = contextvars.ContextVar('request_id', default=None)
trace_id_var = contextvars.ContextVar('trace_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)
session_id_var = contextvars.ContextVar('session_id', default=None)

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    PREDICTION_MADE = "prediction_made"
    MODEL_LOADED = "model_loaded"
    ERROR_OCCURRED = "error_occurred"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_UPDATE = "performance_update"
    MODEL_UPDATED = "model_updated"
    SYSTEM_METRICS = "system_metrics"

@dataclass
class LogContext:
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    component: Optional[str] = None
    service: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None

@dataclass
class LogEvent:
    level: LogLevel
    event_type: EventType
    message: str
    context: LogContext
    data: Dict[str, Any]
    duration: Optional[float] = None
    error: Optional[Exception] = None
    stack_trace: Optional[str] = None

class AsyncLogHandler:
    """Asynchronous log handler for high-performance logging"""

    def __init__(self, max_queue_size: int = 10000, batch_size: int = 100):
        self.log_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_size = batch_size
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.subscribers = []

    async def start(self):
        """Start the log handler"""
        self.is_running = True
        await self._process_logs()

    async def stop(self):
        """Stop the log handler"""
        self.is_running = False
        # Process remaining logs
        await self._process_remaining_logs()

    async def emit(self, event: LogEvent):
        """Emit a log event"""
        try:
            await self.log_queue.put(event)
        except asyncio.QueueFull:
            # Queue full, drop the event
            print(f"Log queue full, dropping event: {event.event_type}")

    def subscribe(self, callback):
        """Subscribe to log events"""
        self.subscribers.append(callback)

    async def _process_logs(self):
        """Process logs in batches"""
        batch = []

        while self.is_running or not self.log_queue.empty():
            try:
                # Wait for logs with timeout
                event = await asyncio.wait_for(self.log_queue.get(), timeout=1.0)
                batch.append(event)

                # Process batch when full
                if len(batch) >= self.batch_size:
                    await self._process_batch(batch)
                    batch = []

            except asyncio.TimeoutError:
                # Timeout, process remaining batch
                if batch:
                    await self._process_batch(batch)
                    batch = []

    async def _process_remaining_logs(self):
        """Process remaining logs before stopping"""
        batch = []
        while not self.log_queue.empty():
            event = self.log_queue.get_nowait()
            batch.append(event)

            if len(batch) >= self.batch_size:
                await self._process_batch(batch)
                batch = []

        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[LogEvent]):
        """Process a batch of log events"""
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            self.executor, self._write_logs, batch
        )

    def _write_logs(self, batch: List[LogEvent]):
        """Write logs to subscribers"""
        for event in batch:
            for subscriber in self.subscribers:
                try:
                    subscriber(event)
                except Exception as e:
                    print(f"Error in log subscriber: {e}")

class StructuredLogger:
    """Advanced structured logging for ML systems"""

    def __init__(self,
                 logger_name: str,
                 service_name: str,
                 version: str = "1.0.0",
                 environment: str = "production",
                 enable_async: bool = True):

        self.logger_name = logger_name
        self.service_name = service_name
        self.version = version
        self.environment = environment

        # Initialize async handler
        self.async_handler = AsyncLogHandler() if enable_async else None

        # Configure Python logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add handlers
        if enable_async:
            self._setup_async_handlers()
        else:
            self._setup_sync_handlers()

        # Performance tracking
        self.log_counts = defaultdict(int)
        self.error_counts = defaultdict(int)

        # Start async handler
        if self.async_handler:
            asyncio.create_task(self.async_handler.start())

    def _setup_async_handlers(self):
        """Setup asynchronous handlers"""
        # Async handler
        self.async_handler.subscribe(self._handle_log_event)

        # Fallback console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONFormatter(self.service_name))
        self.logger.addHandler(console_handler)

    def _setup_sync_handlers(self):
        """Setup synchronous handlers"""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONFormatter(self.service_name))
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(f'{self.service_name}.log')
        file_handler.setFormatter(JSONFormatter(self.service_name))
        self.logger.addHandler(file_handler)

    def _handle_log_event(self, event: LogEvent):
        """Handle log event from async handler"""
        # Write to console
        print(self._format_event(event))

    def _format_event(self, event: LogEvent) -> str:
        """Format log event for output"""
        log_obj = {
            'timestamp': event.context.timestamp.isoformat() if event.context.timestamp else datetime.utcnow().isoformat(),
            'level': event.level.value,
            'service': self.service_name,
            'version': self.version,
            'environment': self.environment,
            'event_type': event.event_type.value,
            'message': event.message,
            'logger': self.logger_name,
            'request_id': event.context.request_id,
            'trace_id': event.context.trace_id,
            'user_id': event.context.user_id,
            'session_id': event.context.session_id,
            'component': event.context.component,
            'data': event.data
        }

        # Add duration if present
        if event.duration is not None:
            log_obj['duration_ms'] = event.duration * 1000

        # Add error info if present
        if event.error:
            log_obj['error'] = {
                'type': type(event.error).__name__,
                'message': str(event.error),
                'stack_trace': event.stack_trace
            }

        return json.dumps(log_obj)

    def _get_context(self) -> LogContext:
        """Get current context"""
        return LogContext(
            request_id=request_id_var.get(),
            trace_id=trace_id_var.get(),
            user_id=user_id_var.get(),
            session_id=session_id_var.get(),
            timestamp=datetime.utcnow(),
            service=self.service_name,
            version=self.version,
            environment=self.environment
        )

    def _log(self, level: LogLevel, event_type: EventType, message: str, data: Dict[str, Any] = None):
        """Internal logging method"""
        context = self._get_context()
        event = LogEvent(
            level=level,
            event_type=event_type,
            message=message,
            context=context,
            data=data or {}
        )

        # Update counters
        self.log_counts[event_type.value] += 1
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.error_counts[event_type.value] += 1

        # Emit to async handler or sync logger
        if self.async_handler:
            asyncio.create_task(self.async_handler.emit(event))
        else:
            self.logger.log(
                getattr(logging, level.value),
                message,
                extra={'event': event}
            )

    def debug(self, event_type: EventType, message: str, data: Dict[str, Any] = None):
        """Log debug message"""
        self._log(LogLevel.DEBUG, event_type, message, data)

    def info(self, event_type: EventType, message: str, data: Dict[str, Any] = None):
        """Log info message"""
        self._log(LogLevel.INFO, event_type, message, data)

    def warning(self, event_type: EventType, message: str, data: Dict[str, Any] = None):
        """Log warning message"""
        self._log(LogLevel.WARNING, event_type, message, data)

    def error(self, event_type: EventType, message: str, error: Exception = None, data: Dict[str, Any] = None):
        """Log error message"""
        context = self._get_context()
        event = LogEvent(
            level=LogLevel.ERROR,
            event_type=event_type,
            message=message,
            context=context,
            data=data or {},
            error=error,
            stack_trace=traceback.format_exc() if error else None
        )

        # Update counters
        self.log_counts[event_type.value] += 1
        self.error_counts[event_type.value] += 1

        # Emit to async handler or sync logger
        if self.async_handler:
            asyncio.create_task(self.async_handler.emit(event))
        else:
            self.logger.error(
                message,
                extra={'event': event},
                exc_info=error
            )

    def critical(self, event_type: EventType, message: str, error: Exception = None, data: Dict[str, Any] = None):
        """Log critical message"""
        context = self._get_context()
        event = LogEvent(
            level=LogLevel.CRITICAL,
            event_type=event_type,
            message=message,
            context=context,
            data=data or {},
            error=error,
            stack_trace=traceback.format_exc() if error else None
        )

        # Update counters
        self.log_counts[event_type.value] += 1
        self.error_counts[event_type.value] += 1

        # Emit to async handler or sync logger
        if self.async_handler:
            asyncio.create_task(self.async_handler.emit(event))
        else:
            self.logger.critical(
                message,
                extra={'event': event},
                exc_info=error
            )

    def log_prediction(self,
                      customer_id: str,
                      features: Dict[str, Any],
                      prediction: int,
                      confidence: float,
                      latency: float,
                      model_version: str = None):
        """Log prediction details"""
        self.info(
            EventType.PREDICTION_MADE,
            "prediction_made",
            data={
                'customer_id': customer_id,
                'features': features,
                'prediction': prediction,
                'confidence': confidence,
                'latency_ms': latency * 1000,
                'model_version': model_version or self.version
            }
        )

    def log_model_update(self,
                        old_version: str,
                        new_version: str,
                        metrics: Dict[str, float],
                        reason: str = None):
        """Log model update event"""
        self.info(
            EventType.MODEL_UPDATED,
            "model_updated",
            data={
                'old_version': old_version,
                'new_version': new_version,
                'metrics': metrics,
                'reason': reason
            }
        )

    def log_drift_detected(self,
                          drift_type: str,
                          drift_score: float,
                          threshold: float,
                          feature_name: str = None):
        """Log drift detection event"""
        severity = 'high' if drift_score > threshold * 1.5 else 'medium'

        self.warning(
            EventType.DRIFT_DETECTED,
            "drift_detected",
            data={
                'drift_type': drift_type,
                'drift_score': drift_score,
                'threshold': threshold,
                'severity': severity,
                'feature_name': feature_name
            }
        )

    def log_performance_update(self,
                             metrics: Dict[str, float],
                             window_size: int = None):
        """Log performance metrics update"""
        self.info(
            EventType.PERFORMANCE_UPDATE,
            "performance_updated",
            data={
                'metrics': metrics,
                'window_size': window_size
            }
        )

    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system metrics"""
        self.debug(
            EventType.SYSTEM_METRICS,
            "system_metrics_collected",
            data=metrics
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        total_logs = sum(self.log_counts.values())
        total_errors = sum(self.error_counts.values())

        return {
            'total_logs': total_logs,
            'total_errors': total_errors,
            'error_rate': total_errors / total_logs if total_logs > 0 else 0,
            'log_counts': dict(self.log_counts),
            'error_counts': dict(self.error_counts)
        }

class RequestTracker:
    """Track requests throughout their lifecycle"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def track_request(self, endpoint: str, method: str = 'POST'):
        """Context manager for request tracking"""
        class RequestContext:
            def __init__(self, logger, endpoint, method):
                self.logger = logger
                self.endpoint = endpoint
                self.method = method
                self.start_time = None
                self.request_id = None
                self.trace_id = None

            def __enter__(self):
                self.start_time = time.time()
                self.request_id = str(uuid.uuid4())
                self.trace_id = str(uuid.uuid4())

                # Set context variables
                request_id_var.set(self.request_id)
                trace_id_var.set(self.trace_id)

                # Log request start
                self.logger.info(
                    EventType.REQUEST_START,
                    "request_started",
                    data={
                        'endpoint': self.endpoint,
                        'method': self.method,
                        'request_id': self.request_id,
                        'trace_id': self.trace_id
                    }
                )

                return self.request_id

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time

                if exc_type:
                    # Log error
                    self.logger.error(
                        EventType.REQUEST_END,
                        "request_failed",
                        error=exc_val,
                        data={
                            'endpoint': self.endpoint,
                            'method': self.method,
                            'duration_ms': duration * 1000,
                            'error_type': exc_type.__name__
                        }
                    )
                else:
                    # Log success
                    self.logger.info(
                        EventType.REQUEST_END,
                        "request_completed",
                        data={
                            'endpoint': self.endpoint,
                            'method': self.method,
                            'duration_ms': duration * 1000
                        }
                    )

                # Clear context variables
                request_id_var.set(None)
                trace_id_var.set(None)

        return RequestContext(self.logger, endpoint, method)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'service': self.service_name,
            'message': record.getMessage(),
            'logger': record.name,
            'request_id': request_id_var.get(),
            'trace_id': trace_id_var.get(),
            'user_id': user_id_var.get(),
            'session_id': session_id_var.get()
        }

        # Add extra fields
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_obj)

# Usage example
if __name__ == "__main__":
    # Initialize logger
    logger = StructuredLogger(
        logger_name="ml_model_logger",
        service_name="churn_prediction_service",
        version="1.0.0",
        environment="production"
    )

    # Initialize request tracker
    request_tracker = RequestTracker(logger)

    # Example request tracking
    with request_tracker.track_request("/predict"):
        # Your prediction logic here
        logger.log_prediction(
            customer_id="12345",
            features={"age": 35, "income": 50000},
            prediction=1,
            confidence=0.85,
            latency=0.15
        )

        # Simulate error
        try:
            raise ValueError("Invalid input")
        except Exception as e:
            logger.error(EventType.ERROR_OCCURRED, "Input validation failed", error=e)
```

## Key Takeaways

### Monitoring Components
1. **Metrics Collection**: Comprehensive Prometheus metrics for all aspects
2. **Drift Detection**: Multiple algorithms for detecting data and model drift
3. **Alert Management**: Advanced alerting with multiple handlers
4. **Performance Tracking**: Real-time performance monitoring
5. **System Metrics**: Resource utilization and performance monitoring

### Logging Components
1. **Structured Logging**: JSON-formatted logs with context
2. **Async Processing**: High-performance asynchronous logging
3. **Request Tracking**: End-to-end request lifecycle tracking
4. **Event Types**: Comprehensive event categorization
5. **Performance Monitoring**: System and application performance metrics

### Best Practices
- **Comprehensive Coverage**: Monitor all aspects of ML systems
- **Real-time Alerting**: Immediate notification of issues
- **Performance Tracking**: Continuous performance monitoring
- **Context Preservation**: Maintain context across systems
- **Scalable Architecture**: Handle high-volume logging and monitoring

### Common Challenges
- **Alert Fatigue**: Too many alerts leading to ignored notifications
- **Performance Impact**: Monitoring overhead on system performance
- **Data Volume**: Managing large volumes of monitoring data
- **Complexity**: Managing complex monitoring systems
- **Integration**: Integrating with existing monitoring systems

---

## Next Steps

Continue to [Module 6: Model Management and Versioning](06_Model_Management_and_Versioning.md) to learn about comprehensive model lifecycle management.

## Quick Reference

### Key Concepts
- **Metrics Collection**: Quantitative monitoring data
- **Drift Detection**: Detect changes in data distribution
- **Alert Management**: Automated notification systems
- **Structured Logging**: Context-rich log data
- **Performance Tracking**: Continuous performance monitoring

### Essential Tools
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **ELK Stack**: Log aggregation and analysis
- **Jaeger**: Distributed tracing
- **AlertManager**: Alert routing and management

### Common Patterns
- **Request Tracking**: End-to-end request lifecycle
- **Drift Detection**: Statistical change detection
- **Performance Monitoring**: Resource utilization tracking
- **Error Tracking**: Error aggregation and analysis
- **Business Metrics**: Business value monitoring