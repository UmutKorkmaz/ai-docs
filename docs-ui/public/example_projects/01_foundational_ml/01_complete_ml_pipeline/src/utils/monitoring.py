"""
Monitoring utilities for tracking model performance and API metrics.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from .logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Comprehensive metrics collection system for ML model serving.
    """

    def __init__(self, metrics_port: int = 8001):
        """
        Initialize metrics collector.

        Args:
            metrics_port: Port for Prometheus metrics endpoint
        """
        self.metrics_port = metrics_port

        # Initialize Prometheus metrics
        self.prediction_counter = Counter(
            'predictions_total',
            'Total number of predictions',
            ['model_version', 'prediction_type']
        )

        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_version', 'prediction_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        self.error_counter = Counter(
            'prediction_errors_total',
            'Total number of prediction errors',
            ['model_version', 'error_type']
        )

        self.model_accuracy_gauge = Gauge(
            'model_accuracy',
            'Current model accuracy',
            ['model_version']
        )

        self.data_drift_score = Gauge(
            'data_drift_score',
            'Data drift score',
            ['feature_name', 'model_version']
        )

        # Internal metrics storage
        self.prediction_history = deque(maxlen=10000)
        self.error_history = deque(maxlen=1000)
        self.performance_history = defaultdict(list)
        self.drift_scores = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # Start Prometheus server
        try:
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def record_prediction(self, latency: float, prediction: int,
                         model_version: str = "unknown") -> None:
        """
        Record a single prediction.

        Args:
            latency: Prediction latency in seconds
            prediction: Prediction result
            model_version: Model version
        """
        with self._lock:
            # Update Prometheus metrics
            self.prediction_counter.labels(
                model_version=model_version,
                prediction_type="single"
            ).inc()

            self.prediction_latency.labels(
                model_version=model_version,
                prediction_type="single"
            ).observe(latency)

            # Store internal metrics
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'latency': latency,
                'prediction': prediction,
                'model_version': model_version
            })

            # Update performance history
            self.performance_history['latency'].append(latency)

    def record_batch_prediction(self, total_latency: float, batch_size: int,
                              model_version: str = "unknown") -> None:
        """
        Record batch prediction metrics.

        Args:
            total_latency: Total processing time in seconds
            batch_size: Number of predictions in batch
            model_version: Model version
        """
        avg_latency = total_latency / batch_size

        with self._lock:
            self.prediction_counter.labels(
                model_version=model_version,
                prediction_type="batch"
            ).inc(batch_size)

            self.prediction_latency.labels(
                model_version=model_version,
                prediction_type="batch"
            ).observe(avg_latency)

            # Store individual predictions
            for _ in range(batch_size):
                self.prediction_history.append({
                    'timestamp': datetime.now(),
                    'latency': avg_latency,
                    'prediction': None,  # Unknown for batch
                    'model_version': model_version
                })

            self.performance_history['latency'].extend([avg_latency] * batch_size)

    def record_error(self, error_type: str, model_version: str = "unknown") -> None:
        """
        Record a prediction error.

        Args:
            error_type: Type of error
            model_version: Model version
        """
        with self._lock:
            self.error_counter.labels(
                model_version=model_version,
                error_type=error_type
            ).inc()

            self.error_history.append({
                'timestamp': datetime.now(),
                'error_type': error_type,
                'model_version': model_version
            })

    def update_model_accuracy(self, accuracy: float, model_version: str = "unknown") -> None:
        """
        Update model accuracy metric.

        Args:
            accuracy: Model accuracy value
            model_version: Model version
        """
        self.model_accuracy_gauge.labels(model_version=model_version).set(accuracy)

        with self._lock:
            self.performance_history['accuracy'].append(accuracy)

    def record_data_drift(self, feature_name: str, drift_score: float,
                         model_version: str = "unknown") -> None:
        """
        Record data drift score for a feature.

        Args:
            feature_name: Name of the feature
            drift_score: Drift score
            model_version: Model version
        """
        self.data_drift_score.labels(
            feature_name=feature_name,
            model_version=model_version
        ).set(drift_score)

        with self._lock:
            self.drift_scores[feature_name] = drift_score

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics summary.

        Returns:
            Dictionary of current metrics
        """
        with self._lock:
            total_predictions = len(self.prediction_history)
            total_errors = len(self.error_history)

            # Calculate average latency
            if self.performance_history['latency']:
                avg_latency = sum(self.performance_history['latency']) / len(self.performance_history['latency'])
            else:
                avg_latency = 0.0

            # Calculate error rate
            error_rate = total_errors / (total_predictions + total_errors) if (total_predictions + total_errors) > 0 else 0.0

            # Get recent model accuracy
            if self.performance_history['accuracy']:
                model_accuracy = self.performance_history['accuracy'][-1]
            else:
                model_accuracy = None

            return {
                'predictions_count': total_predictions,
                'errors_count': total_errors,
                'avg_latency': avg_latency,
                'error_rate': error_rate,
                'model_accuracy': model_accuracy,
                'drift_scores': self.drift_scores.copy(),
                'last_updated': datetime.now()
            }

    def get_performance_history(self, hours: int = 24) -> Dict[str, List[float]]:
        """
        Get performance history for the specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary of performance metrics over time
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            history = {
                'timestamps': [],
                'latency': [],
                'accuracy': []
            }

            # Filter prediction history
            for pred in self.prediction_history:
                if pred['timestamp'] >= cutoff_time:
                    history['timestamps'].append(pred['timestamp'])
                    history['latency'].append(pred['latency'])

            # Filter accuracy history
            if 'accuracy' in self.performance_history:
                for i, acc in enumerate(self.performance_history['accuracy']):
                    # This is simplified - in practice, you'd store timestamps for accuracy updates
                    history['accuracy'].append(acc)

            return history

    def get_error_summary(self, hours: int = 24) -> Dict[str, int]:
        """
        Get error summary for the specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary of error counts by type
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            error_summary = defaultdict(int)

            for error in self.error_history:
                if error['timestamp'] >= cutoff_time:
                    error_summary[error['error_type']] += 1

            return dict(error_summary)

    def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive health report.

        Returns:
            Health report dictionary
        """
        metrics = self.get_metrics()
        error_summary = self.get_error_summary()

        # Calculate health status
        health_status = "healthy"
        issues = []

        if metrics['error_rate'] > 0.05:  # 5% error threshold
            health_status = "degraded"
            issues.append(f"High error rate: {metrics['error_rate']:.2%}")

        if metrics['avg_latency'] > 2.0:  # 2 second latency threshold
            health_status = "degraded"
            issues.append(f"High latency: {metrics['avg_latency']:.2f}s")

        # Check data drift
        max_drift = max(metrics['drift_scores'].values()) if metrics['drift_scores'] else 0.0
        if max_drift > 0.1:  # 10% drift threshold
            health_status = "warning"
            issues.append(f"Data drift detected: {max_drift:.3f}")

        return {
            'status': health_status,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'error_summary': error_summary,
            'issues': issues,
            'recommendations': self._generate_recommendations(metrics, error_summary)
        }

    def _generate_recommendations(self, metrics: Dict[str, Any],
                                error_summary: Dict[str, int]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        if metrics['error_rate'] > 0.05:
            recommendations.append("Investigate high error rate - check model and data quality")

        if metrics['avg_latency'] > 2.0:
            recommendations.append("Optimize prediction latency - consider model optimization or scaling")

        if not metrics['model_accuracy']:
            recommendations.append("Set up model accuracy monitoring")

        if metrics['drift_scores']:
            max_drift_feature = max(metrics['drift_scores'].items(), key=lambda x: x[1])
            if max_drift_feature[1] > 0.1:
                recommendations.append(f"Monitor feature '{max_drift_feature[0]}' for data drift")

        if error_summary.get('prediction_error', 0) > 10:
            recommendations.append("Review prediction error patterns and data validation")

        return recommendations

    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """
        Export metrics to file.

        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
        """
        metrics = self.get_metrics()
        performance_history = self.get_performance_history()
        error_summary = self.get_error_summary()

        if format == 'json':
            export_data = {
                'metrics': metrics,
                'performance_history': performance_history,
                'error_summary': error_summary,
                'export_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format == 'csv':
            # Convert to DataFrame and save
            df = pd.DataFrame(self.prediction_history)
            df.to_csv(filepath, index=False)

        logger.info(f"Metrics exported to {filepath}")

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.prediction_history.clear()
            self.error_history.clear()
            self.performance_history.clear()
            self.drift_scores.clear()

        logger.info("Metrics reset")

    def close(self) -> None:
        """Clean up resources."""
        logger.info("Metrics collector closed")


class DriftDetector:
    """
    Data drift detection using statistical methods.
    """

    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference dataset for comparison
            significance_level: Statistical significance level
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.reference_stats = self._calculate_reference_stats()

    def _calculate_reference_stats(self) -> Dict[str, Dict]:
        """Calculate reference statistics for all features."""
        stats = {}

        for column in self.reference_data.select_dtypes(include=[np.number]).columns:
            col_data = self.reference_data[column].dropna()
            if len(col_data) > 0:
                stats[column] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'median': col_data.median(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75),
                    'min': col_data.min(),
                    'max': col_data.max()
                }

        return stats

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect drift in current data compared to reference.

        Args:
            current_data: Current dataset to check

        Returns:
            Dictionary of drift scores by feature
        """
        drift_scores = {}

        numeric_columns = current_data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if column in self.reference_stats:
                current_col = current_data[column].dropna()
                if len(current_col) > 0:
                    drift_score = self._calculate_drift_score(
                        self.reference_data[column].dropna(),
                        current_col
                    )
                    drift_scores[column] = drift_score

        return drift_scores

    def _calculate_drift_score(self, ref_data: pd.Series, current_data: pd.Series) -> float:
        """
        Calculate drift score using Kolmogorov-Smirnov test.

        Args:
            ref_data: Reference data series
            current_data: Current data series

        Returns:
            Drift score (p-value from KS test)
        """
        from scipy import stats

        try:
            statistic, p_value = stats.ks_2samp(ref_data, current_data)
            # Return 1 - p_value so higher values indicate more drift
            return 1 - p_value
        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0


class AlertManager:
    """
    Alert management system for monitoring thresholds.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config
        self.alert_thresholds = config.get('thresholds', {})
        self.alert_channels = config.get('channels', [])
        self.alert_history = deque(maxlen=1000)
        self.suppressed_alerts = set()

    def check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check metrics against thresholds and return alerts.

        Args:
            metrics: Current metrics

        Returns:
            List of triggered alerts
        """
        alerts = []

        for metric_name, threshold_config in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                threshold = threshold_config.get('value')
                operator = threshold_config.get('operator', 'gt')

                triggered = False
                if operator == 'gt' and value > threshold:
                    triggered = True
                elif operator == 'lt' and value < threshold:
                    triggered = True
                elif operator == 'eq' and value == threshold:
                    triggered = True

                if triggered:
                    alert_key = f"{metric_name}_{operator}_{threshold}"
                    if alert_key not in self.suppressed_alerts:
                        alert = {
                            'metric': metric_name,
                            'value': value,
                            'threshold': threshold,
                            'operator': operator,
                            'severity': threshold_config.get('severity', 'warning'),
                            'timestamp': datetime.now(),
                            'message': threshold_config.get('message', f'{metric_name} threshold exceeded')
                        }
                        alerts.append(alert)
                        self.alert_history.append(alert)

        return alerts

    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Send alert through configured channels.

        Args:
            alert: Alert to send

        Returns:
            True if alert was sent successfully
        """
        try:
            for channel in self.alert_channels:
                if channel['type'] == 'email':
                    self._send_email_alert(alert, channel['config'])
                elif channel['type'] == 'slack':
                    self._send_slack_alert(alert, channel['config'])
                elif channel['type'] == 'webhook':
                    self._send_webhook_alert(alert, channel['config'])

            return True

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    def _send_email_alert(self, alert: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Send email alert (placeholder implementation)."""
        logger.info(f"Email alert would be sent: {alert['message']}")

    def _send_slack_alert(self, alert: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Send Slack alert (placeholder implementation)."""
        logger.info(f"Slack alert would be sent: {alert['message']}")

    def _send_webhook_alert(self, alert: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Send webhook alert (placeholder implementation)."""
        logger.info(f"Webhook alert would be sent: {alert['message']}")

    def suppress_alert(self, alert_key: str, duration_hours: int = 24) -> None:
        """
        Suppress an alert for a specified duration.

        Args:
            alert_key: Alert identifier
            duration_hours: Duration to suppress in hours
        """
        self.suppressed_alerts.add(alert_key)
        # In a real implementation, you'd set up a timer to remove the suppression

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alert history for the specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            List of alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [alert for alert in self.alert_history if alert['timestamp'] >= cutoff_time]


if __name__ == "__main__":
    # Example usage
    metrics_collector = MetricsCollector()

    # Simulate some predictions
    for i in range(100):
        latency = np.random.exponential(0.1)  # 100ms average latency
        prediction = np.random.choice([0, 1], p=[0.8, 0.2])
        metrics_collector.record_prediction(latency, prediction)

    # Get metrics
    current_metrics = metrics_collector.get_metrics()
    print("Current Metrics:")
    for key, value in current_metrics.items():
        print(f"{key}: {value}")

    # Generate health report
    health_report = metrics_collector.generate_health_report()
    print(f"\nHealth Status: {health_report['status']}")
    if health_report['issues']:
        print("Issues:")
        for issue in health_report['issues']:
            print(f"  - {issue}")

    # Export metrics
    metrics_collector.export_metrics("metrics_export.json", format="json")