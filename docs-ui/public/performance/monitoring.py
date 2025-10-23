"""
Performance Monitoring System

Comprehensive monitoring and analytics for documentation performance metrics,
including load times, cache efficiency, and user behavior patterns.

Author: AI Documentation Team
Version: 1.0.0
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import uuid

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    timestamp: float
    metric_type: str
    value: float
    module_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadTimeMetric:
    """Specific metric for load time tracking."""
    module_path: str
    load_time: float
    cache_hit: bool
    from_memory: bool
    file_size: int
    timestamp: float
    session_id: str

@dataclass
class CacheMetric:
    """Specific metric for cache performance."""
    cache_type: str  # 'memory' or 'disk'
    operation: str  # 'hit', 'miss', 'eviction'
    key: str
    size_bytes: int
    timestamp: float

@dataclass
class UserNavigationMetric:
    """Metric for user navigation patterns."""
    from_module: Optional[str]
    to_module: str
    navigation_type: str  # 'click', 'search', 'direct'
    time_spent_on_previous: float
    timestamp: float
    session_id: str

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.

    Features:
    - Real-time performance metrics collection
    - Load time tracking and analysis
    - Cache performance monitoring
    - User navigation pattern analysis
    - Statistical analysis and reporting
    - Performance trend detection
    """

    def __init__(self, max_metrics: int = 10000):
        """
        Initialize performance monitor.

        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.max_metrics = max_metrics

        # Storage
        self.general_metrics: deque = deque(maxlen=max_metrics)
        self.load_time_metrics: deque = deque(maxlen=max_metrics)
        self.cache_metrics: deque = deque(maxlen=max_metrics)
        self.navigation_metrics: deque = deque(maxlen=max_metrics)

        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)

        # Aggregated statistics
        self.aggregated_stats = {
            'total_loads': 0,
            'average_load_time': 0.0,
            'cache_hit_rate': 0.0,
            'memory_cache_hit_rate': 0.0,
            'disk_cache_hit_rate': 0.0,
            'peak_load_time': 0.0,
            'modules_accessed': set()
        }

        # Alert thresholds
        self.thresholds = {
            'slow_load_threshold': 2.0,  # seconds
            'low_cache_hit_rate': 0.5,   # below 50%
            'high_error_rate': 0.1,      # above 10%
            'memory_pressure': 0.8       # above 80% memory usage
        }

        # Performance alerts
        self.alerts: List[Dict[str, Any]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Background processing
        self._running = False
        self._analytics_thread = None

    def start_monitoring(self):
        """Start background monitoring."""
        self._running = True
        self._analytics_thread = threading.Thread(target=self._background_analytics, daemon=True)
        self._analytics_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._analytics_thread:
            self._analytics_thread.join()

    def _background_analytics(self):
        """Background analytics processing."""
        while self._running:
            try:
                time.sleep(60)  # Process every minute
                self._update_aggregated_stats()
                self._check_performance_alerts()
            except Exception:
                pass

    def track_load_time(
        self,
        module_path: str,
        load_time: float,
        cache_hit: bool,
        from_memory: bool,
        file_size: int,
        session_id: Optional[str] = None
    ):
        """
        Track module load time.

        Args:
            module_path: Path to the loaded module
            load_time: Time taken to load
            cache_hit: Whether cache was hit
            from_memory: Whether loaded from memory cache
            file_size: Size of the file in bytes
            session_id: User session identifier
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        metric = LoadTimeMetric(
            module_path=module_path,
            load_time=load_time,
            cache_hit=cache_hit,
            from_memory=from_memory,
            file_size=file_size,
            timestamp=time.time(),
            session_id=session_id
        )

        with self._lock:
            self.load_time_metrics.append(metric)

            # Update session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    'start_time': time.time(),
                    'modules_accessed': [],
                    'total_load_time': 0.0,
                    'cache_hits': 0,
                    'cache_misses': 0
                }

            session = self.active_sessions[session_id]
            session['modules_accessed'].append(module_path)
            session['total_load_time'] += load_time
            if cache_hit:
                session['cache_hits'] += 1
            else:
                session['cache_misses'] += 1

        # Check for slow load
        if load_time > self.thresholds['slow_load_threshold']:
            self._create_alert('slow_load', {
                'module_path': module_path,
                'load_time': load_time,
                'threshold': self.thresholds['slow_load_threshold']
            })

    def track_cache_operation(
        self,
        cache_type: str,
        operation: str,
        key: str,
        size_bytes: int = 0
    ):
        """
        Track cache operation.

        Args:
            cache_type: Type of cache ('memory' or 'disk')
            operation: Operation type ('hit', 'miss', 'eviction')
            key: Cache key
            size_bytes: Size of cached item
        """
        metric = CacheMetric(
            cache_type=cache_type,
            operation=operation,
            key=key,
            size_bytes=size_bytes,
            timestamp=time.time()
        )

        with self._lock:
            self.cache_metrics.append(metric)

    def track_navigation(
        self,
        to_module: str,
        from_module: Optional[str] = None,
        navigation_type: str = 'click',
        time_spent_on_previous: float = 0.0,
        session_id: Optional[str] = None
    ):
        """
        Track user navigation.

        Args:
            to_module: Module being navigated to
            from_module: Module being navigated from
            navigation_type: Type of navigation
            time_spent_on_previous: Time spent on previous module
            session_id: User session identifier
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        metric = UserNavigationMetric(
            from_module=from_module,
            to_module=to_module,
            navigation_type=navigation_type,
            time_spent_on_previous=time_spent_on_previous,
            timestamp=time.time(),
            session_id=session_id
        )

        with self._lock:
            self.navigation_metrics.append(metric)

    def track_general_metric(
        self,
        metric_type: str,
        value: float,
        module_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track general performance metric.

        Args:
            metric_type: Type of metric
            value: Metric value
            module_id: Associated module
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_type=metric_type,
            value=value,
            module_id=module_id,
            metadata=metadata or {}
        )

        with self._lock:
            self.general_metrics.append(metric)

    def _update_aggregated_stats(self):
        """Update aggregated statistics."""
        with self._lock:
            if not self.load_time_metrics:
                return

            # Load time statistics
            load_times = [m.load_time for m in self.load_time_metrics]
            self.aggregated_stats['total_loads'] = len(load_times)
            self.aggregated_stats['average_load_time'] = statistics.mean(load_times)
            self.aggregated_stats['peak_load_time'] = max(load_times)

            # Cache statistics
            cache_hits = sum(1 for m in self.load_time_metrics if m.cache_hit)
            self.aggregated_stats['cache_hit_rate'] = cache_hits / len(load_times) if load_times else 0

            memory_hits = sum(1 for m in self.load_time_metrics if m.from_memory)
            self.aggregated_stats['memory_cache_hit_rate'] = memory_hits / len(load_times) if load_times else 0

            disk_hits = sum(1 for m in self.load_time_metrics if m.cache_hit and not m.from_memory)
            self.aggregated_stats['disk_cache_hit_rate'] = disk_hits / len(load_times) if load_times else 0

            # Module tracking
            self.aggregated_stats['modules_accessed'] = set(m.module_path for m in self.load_time_metrics)

    def _check_performance_alerts(self):
        """Check for performance issues and create alerts."""
        # Check cache hit rate
        if self.aggregated_stats['cache_hit_rate'] < self.thresholds['low_cache_hit_rate']:
            self._create_alert('low_cache_hit_rate', {
                'current_rate': self.aggregated_stats['cache_hit_rate'],
                'threshold': self.thresholds['low_cache_hit_rate']
            })

        # Check average load time
        if self.aggregated_stats['average_load_time'] > self.thresholds['slow_load_threshold']:
            self._create_alert('high_average_load_time', {
                'current_average': self.aggregated_stats['average_load_time'],
                'threshold': self.thresholds['slow_load_threshold']
            })

    def _create_alert(self, alert_type: str, data: Dict[str, Any]):
        """Create performance alert."""
        alert = {
            'type': alert_type,
            'data': data,
            'timestamp': time.time(),
            'severity': self._calculate_severity(alert_type, data)
        }

        with self._lock:
            self.alerts.append(alert)

            # Keep only recent alerts (last 100)
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]

    def _calculate_severity(self, alert_type: str, data: Dict[str, Any]) -> str:
        """Calculate alert severity."""
        if alert_type == 'slow_load':
            if data['load_time'] > 5.0:
                return 'critical'
            elif data['load_time'] > 3.0:
                return 'high'
            else:
                return 'medium'

        elif alert_type == 'low_cache_hit_rate':
            if data['current_rate'] < 0.3:
                return 'high'
            elif data['current_rate'] < 0.5:
                return 'medium'
            else:
                return 'low'

        return 'medium'

    def get_performance_report(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            time_range: Time range for report (None for all time)

        Returns:
            Performance report dictionary
        """
        now = time.time()
        cutoff_time = now - time_range.total_seconds() if time_range else 0

        with self._lock:
            # Filter metrics by time range
            load_times = [
                m for m in self.load_time_metrics
                if m.timestamp >= cutoff_time
            ]

            # Basic statistics
            report = {
                'report_period': str(time_range) if time_range else 'All time',
                'generated_at': datetime.fromtimestamp(now).isoformat(),
                'summary': {
                    'total_loads': len(load_times),
                    'unique_modules': len(set(m.module_path for m in load_times)),
                    'active_sessions': len(self.active_sessions)
                },
                'performance': self._calculate_performance_stats(load_times),
                'cache_performance': self._calculate_cache_stats(load_times),
                'user_behavior': self._calculate_user_behavior_stats(),
                'module_performance': self._calculate_module_performance(load_times),
                'alerts': self._get_recent_alerts(cutoff_time),
                'recommendations': self._generate_recommendations()
            }

        return report

    def _calculate_performance_stats(self, load_times: List[LoadTimeMetric]) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not load_times:
            return {'average_load_time': 0, 'peak_load_time': 0, 'slow_loads': 0}

        load_values = [m.load_time for m in load_times]

        return {
            'average_load_time': statistics.mean(load_values),
            'median_load_time': statistics.median(load_values),
            'peak_load_time': max(load_values),
            'p95_load_time': statistics.quantiles(load_values, n=20)[18],  # 95th percentile
            'slow_loads': len([t for t in load_values if t > self.thresholds['slow_load_threshold']]),
            'slow_load_percentage': len([t for t in load_values if t > self.thresholds['slow_load_threshold']]) / len(load_times) * 100
        }

    def _calculate_cache_stats(self, load_times: List[LoadTimeMetric]) -> Dict[str, Any]:
        """Calculate cache performance statistics."""
        if not load_times:
            return {'hit_rate': 0, 'memory_hit_rate': 0, 'disk_hit_rate': 0}

        total_loads = len(load_times)
        cache_hits = sum(1 for m in load_times if m.cache_hit)
        memory_hits = sum(1 for m in load_times if m.from_memory)

        return {
            'hit_rate': cache_hits / total_loads,
            'memory_hit_rate': memory_hits / total_loads,
            'disk_hit_rate': (cache_hits - memory_hits) / total_loads if cache_hits > memory_hits else 0,
            'total_cache_hits': cache_hits,
            'total_cache_misses': total_loads - cache_hits
        }

    def _calculate_user_behavior_stats(self) -> Dict[str, Any]:
        """Calculate user behavior statistics."""
        if not self.navigation_metrics:
            return {'total_navigations': 0, 'avg_time_per_module': 0}

        total_navigations = len(self.navigation_metrics)
        time_spent = [m.time_spent_on_previous for m in self.navigation_metrics if m.time_spent_on_previous > 0]

        # Navigation patterns
        navigation_types = defaultdict(int)
        for m in self.navigation_metrics:
            navigation_types[m.navigation_type] += 1

        # Popular modules
        module_visits = defaultdict(int)
        for m in self.navigation_metrics:
            module_visits[m.to_module] += 1

        return {
            'total_navigations': total_navigations,
            'avg_time_per_module': statistics.mean(time_spent) if time_spent else 0,
            'navigation_patterns': dict(navigation_types),
            'popular_modules': dict(sorted(module_visits.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    def _calculate_module_performance(self, load_times: List[LoadTimeMetric]) -> Dict[str, Any]:
        """Calculate per-module performance statistics."""
        module_stats = defaultdict(lambda: {'loads': 0, 'total_time': 0, 'cache_hits': 0})

        for m in load_times:
            stats = module_stats[m.module_path]
            stats['loads'] += 1
            stats['total_time'] += m.load_time
            if m.cache_hit:
                stats['cache_hits'] += 1

        # Calculate averages and identify slow modules
        slow_modules = []
        for module, stats in module_stats.items():
            avg_time = stats['total_time'] / stats['loads']
            hit_rate = stats['cache_hits'] / stats['loads']

            module_stats[module]['average_load_time'] = avg_time
            module_stats[module]['cache_hit_rate'] = hit_rate

            if avg_time > self.thresholds['slow_load_threshold']:
                slow_modules.append({
                    'module': module,
                    'avg_load_time': avg_time,
                    'loads': stats['loads']
                })

        return {
            'module_stats': dict(module_stats),
            'slow_modules': sorted(slow_modules, key=lambda x: x['avg_load_time'], reverse=True)[:10]
        }

    def _get_recent_alerts(self, cutoff_time: float) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return [
            alert for alert in self.alerts
            if alert['timestamp'] >= cutoff_time
        ]

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Check cache performance
        if self.aggregated_stats['cache_hit_rate'] < 0.7:
            recommendations.append("Consider increasing cache size or implementing more aggressive caching strategies")

        # Check load times
        if self.aggregated_stats['average_load_time'] > 1.0:
            recommendations.append("Average load times are high. Consider implementing lazy loading or content compression")

        # Check for frequently accessed slow modules
        slow_modules = self._calculate_module_performance(self.load_time_metrics).get('slow_modules', [])
        if slow_modules:
            recommendations.append(f"Found {len(slow_modules)} slow modules. Consider optimizing or splitting these modules")

        # Check session patterns
        if len(self.active_sessions) > 100:
            recommendations.append("High concurrent sessions detected. Consider implementing session pooling or load balancing")

        return recommendations

    def export_metrics(self, format_type: str = 'json', time_range: Optional[timedelta] = None) -> str:
        """
        Export metrics data.

        Args:
            format_type: Export format ('json', 'csv')
            time_range: Time range for export

        Returns:
            Exported data string
        """
        cutoff_time = time.time() - time_range.total_seconds() if time_range else 0

        metrics_data = {
            'load_times': [asdict(m) for m in self.load_time_metrics if m.timestamp >= cutoff_time],
            'cache_metrics': [asdict(m) for m in self.cache_metrics if m.timestamp >= cutoff_time],
            'navigation_metrics': [asdict(m) for m in self.navigation_metrics if m.timestamp >= cutoff_time],
            'general_metrics': [asdict(m) for m in self.general_metrics if m.timestamp >= cutoff_time],
            'export_timestamp': datetime.now().isoformat()
        }

        if format_type == 'json':
            return json.dumps(metrics_data, indent=2)
        elif format_type == 'csv':
            # Convert to CSV format (simplified)
            lines = ['timestamp,module_path,load_time,cache_hit']
            for m in metrics_data['load_times']:
                lines.append(f"{m['timestamp']},{m['module_path']},{m['load_time']},{m['cache_hit']}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def clear_metrics(self, older_than: Optional[timedelta] = None):
        """
        Clear metrics data.

        Args:
            older_than: Clear metrics older than this time
        """
        cutoff_time = time.time() - older_than.total_seconds() if older_than else 0

        with self._lock:
            if older_than:
                # Clear old metrics
                self.load_time_metrics = deque(
                    [m for m in self.load_time_metrics if m.timestamp >= cutoff_time],
                    maxlen=self.max_metrics
                )
                self.cache_metrics = deque(
                    [m for m in self.cache_metrics if m.timestamp >= cutoff_time],
                    maxlen=self.max_metrics
                )
                self.navigation_metrics = deque(
                    [m for m in self.navigation_metrics if m.timestamp >= cutoff_time],
                    maxlen=self.max_metrics
                )
                self.general_metrics = deque(
                    [m for m in self.general_metrics if m.timestamp >= cutoff_time],
                    maxlen=self.max_metrics
                )
            else:
                # Clear all metrics
                self.load_time_metrics.clear()
                self.cache_metrics.clear()
                self.navigation_metrics.clear()
                self.general_metrics.clear()
                self.active_sessions.clear()
                self.session_metrics.clear()
                self.alerts.clear()

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics snapshot."""
        return {
            'current_sessions': len(self.active_sessions),
            'recent_loads': len([m for m in self.load_time_metrics if time.time() - m.timestamp < 300]),
            'cache_hit_rate_last_hour': self._calculate_recent_cache_hit_rate(3600),
            'average_load_time_last_hour': self._calculate_recent_average_load_time(3600),
            'active_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 3600])
        }

    def _calculate_recent_cache_hit_rate(self, seconds: int) -> float:
        """Calculate cache hit rate for recent period."""
        cutoff_time = time.time() - seconds
        recent_loads = [m for m in self.load_time_metrics if m.timestamp >= cutoff_time]

        if not recent_loads:
            return 0.0

        cache_hits = sum(1 for m in recent_loads if m.cache_hit)
        return cache_hits / len(recent_loads)

    def _calculate_recent_average_load_time(self, seconds: int) -> float:
        """Calculate average load time for recent period."""
        cutoff_time = time.time() - seconds
        recent_loads = [m for m in self.load_time_metrics if m.timestamp >= cutoff_time]

        if not recent_loads:
            return 0.0

        load_times = [m.load_time for m in recent_loads]
        return statistics.mean(load_times)