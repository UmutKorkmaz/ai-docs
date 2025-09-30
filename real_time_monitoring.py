#!/usr/bin/env python3
"""
Real-time Monitoring System for AI Documentation Analytics
Live monitoring, alerting, and real-time data processing pipeline.
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import numpy as np
from collections import defaultdict, deque
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import websockets
import aiofiles
from analytics_dashboard import AnalyticsDashboard, SystemMetrics, PredictiveInsight

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MonitoringEvent:
    """Real-time monitoring event"""
    timestamp: datetime
    event_type: str  # system, user, content, performance, security
    severity: str    # info, warning, error, critical
    source: str
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    condition: str  # e.g., "cpu_usage > 80"
    threshold: float
    duration: int  # seconds
    severity: str
    enabled: bool
    cooldown_period: int  # seconds between alerts

@dataclass
class AlertInstance:
    """Active alert instance"""
    id: str
    rule_id: str
    triggered_at: datetime
    current_value: float
    threshold: float
    severity: str
    status: str  # active, acknowledged, resolved
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    resolved_by: Optional[str]
    resolved_at: Optional[datetime]

class RealTimeMonitor:
    """Real-time monitoring system"""

    def __init__(self, dashboard: AnalyticsDashboard, config_path: str = "monitoring_config.json"):
        self.dashboard = dashboard
        self.config_path = Path(config_path)
        self.running = False
        self.monitoring_tasks = []

        # Real-time data streams
        self.event_queue = mp.Queue()
        self.alert_queue = mp.Queue()
        self.websocket_connections = set()

        # Monitoring data stores
        self.system_metrics = deque(maxlen=10000)
        self.user_activity = deque(maxlen=50000)
        self.performance_events = deque(maxlen=10000)
        self.active_alerts = {}

        # Monitoring intervals
        self.system_check_interval = 5  # seconds
        self.user_activity_check_interval = 10  # seconds
        self.alert_check_interval = 30  # seconds

        # Load configuration
        self.config = self._load_config()
        self.alert_rules = self._load_alert_rules()

        # Start monitoring threads
        self._start_monitoring_threads()

        logger.info("Real-time Monitor initialized successfully")

    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "enabled_monitors": {
                "system_health": True,
                "user_activity": True,
                "performance_metrics": True,
                "security_events": True,
                "content_analytics": True
            },
            "alerting": {
                "enabled": True,
                "email_notifications": False,
                "webhook_url": None,
                "slack_webhook": None
            },
            "retention": {
                "system_metrics_days": 30,
                "user_activity_days": 90,
                "alert_history_days": 365
            },
            " thresholds": {
                "cpu_usage_warning": 70,
                "cpu_usage_critical": 90,
                "memory_usage_warning": 80,
                "memory_usage_critical": 95,
                "response_time_warning": 500,
                "response_time_critical": 1000,
                "error_rate_warning": 1,
                "error_rate_critical": 5
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")

        return default_config

    def _load_alert_rules(self) -> Dict[str, AlertRule]:
        """Load alert rules"""
        default_rules = {
            "high_cpu_usage": AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                condition="cpu_usage > threshold",
                threshold=self.config["thresholds"]["cpu_usage_warning"],
                duration=300,  # 5 minutes
                severity="warning",
                enabled=True,
                cooldown_period=600  # 10 minutes
            ),
            "critical_cpu_usage": AlertRule(
                id="critical_cpu_usage",
                name="Critical CPU Usage",
                condition="cpu_usage > threshold",
                threshold=self.config["thresholds"]["cpu_usage_critical"],
                duration=60,  # 1 minute
                severity="critical",
                enabled=True,
                cooldown_period=300  # 5 minutes
            ),
            "high_memory_usage": AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                condition="memory_usage > threshold",
                threshold=self.config["thresholds"]["memory_usage_warning"],
                duration=300,
                severity="warning",
                enabled=True,
                cooldown_period=600
            ),
            "high_response_time": AlertRule(
                id="high_response_time",
                name="High Response Time",
                condition="response_time > threshold",
                threshold=self.config["thresholds"]["response_time_warning"],
                duration=180,
                severity="warning",
                enabled=True,
                cooldown_period=600
            ),
            "high_error_rate": AlertRule(
                id="high_error_rate",
                name="High Error Rate",
                condition="error_rate > threshold",
                threshold=self.config["thresholds"]["error_rate_warning"],
                duration=300,
                severity="warning",
                enabled=True,
                cooldown_period=900
            )
        }

        return default_rules

    def _start_monitoring_threads(self):
        """Start background monitoring threads"""
        # System health monitor
        system_thread = threading.Thread(target=self._system_health_monitor, daemon=True)
        system_thread.start()

        # User activity monitor
        activity_thread = threading.Thread(target=self._user_activity_monitor, daemon=True)
        activity_thread.start()

        # Alert processor
        alert_thread = threading.Thread(target=self._alert_processor, daemon=True)
        alert_thread.start()

        # Event processor
        event_thread = threading.Thread(target=self._event_processor, daemon=True)
        event_thread.start()

        logger.info("Monitoring threads started")

    def _system_health_monitor(self):
        """Monitor system health metrics"""
        while self.running:
            try:
                if self.config["enabled_monitors"]["system_health"]:
                    # Collect system metrics
                    metrics = self._collect_system_metrics()

                    # Store metrics
                    self.system_metrics.append(metrics)

                    # Create monitoring event
                    event = MonitoringEvent(
                        timestamp=datetime.now(),
                        event_type="system",
                        severity="info",
                        source="system_monitor",
                        message="System metrics collected",
                        data={
                            "cpu_usage": metrics.cpu_usage_percent,
                            "memory_usage": metrics.memory_usage_mb,
                            "response_time": metrics.response_time_ms,
                            "error_rate": metrics.error_rate_percent
                        },
                        metadata={"metrics_count": len(self.system_metrics)}
                    )

                    self.event_queue.put(event)

                    # Check for immediate alerts
                    self._check_immediate_alerts(metrics)

                time.sleep(self.system_check_interval)

            except Exception as e:
                logger.error(f"Error in system health monitor: {e}")
                time.sleep(self.system_check_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # Get process information
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB

            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Simulate response time (would be actual API response time in production)
            response_time = np.random.normal(100, 20)  # Mean 100ms, std 20ms

            # Calculate error rate from recent system metrics
            recent_errors = 0
            total_checks = 10
            for i in range(min(total_checks, len(self.system_metrics))):
                if self.system_metrics[-(i+1)].error_rate_percent > 0:
                    recent_errors += 1

            error_rate = (recent_errors / total_checks) * 100 if total_checks > 0 else 0

            return SystemMetrics(
                timestamp=datetime.now(),
                active_users=len(self.dashboard.active_user_sessions),
                memory_usage_mb=process_memory,
                cpu_usage_percent=cpu_percent,
                response_time_ms=max(0, response_time),
                database_connections=1,  # Placeholder
                error_rate_percent=error_rate,
                throughput_rps=len(self.dashboard.user_activity_stream) / 60 if self.dashboard.user_activity_stream else 0
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                active_users=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                response_time_ms=0,
                database_connections=0,
                error_rate_percent=100,  # Assume error if collection fails
                throughput_rps=0
            )

    def _check_immediate_alerts(self, metrics: SystemMetrics):
        """Check for immediate alert conditions"""
        current_time = datetime.now()

        # Check CPU usage
        if metrics.cpu_usage_percent > self.config["thresholds"]["cpu_usage_critical"]:
            self._trigger_alert("critical_cpu_usage", metrics.cpu_usage_percent, current_time)
        elif metrics.cpu_usage_percent > self.config["thresholds"]["cpu_usage_warning"]:
            self._trigger_alert("high_cpu_usage", metrics.cpu_usage_percent, current_time)

        # Check memory usage
        if metrics.memory_usage_mb > self.config["thresholds"]["memory_usage_critical"]:
            self._trigger_alert("high_memory_usage", metrics.memory_usage_mb, current_time)

        # Check response time
        if metrics.response_time_ms > self.config["thresholds"]["response_time_critical"]:
            self._trigger_alert("high_response_time", metrics.response_time_ms, current_time)

        # Check error rate
        if metrics.error_rate_percent > self.config["thresholds"]["error_rate_critical"]:
            self._trigger_alert("high_error_rate", metrics.error_rate_percent, current_time)

    def _trigger_alert(self, rule_id: str, current_value: float, timestamp: datetime):
        """Trigger an alert"""
        if rule_id not in self.alert_rules:
            return

        rule = self.alert_rules[rule_id]
        alert_id = f"{rule_id}_{int(timestamp.timestamp())}"

        # Check if alert is already active and within cooldown
        if rule_id in self.active_alerts:
            last_alert = self.active_alerts[rule_id]
            time_since_last = (timestamp - last_alert.triggered_at).total_seconds()

            if time_since_last < rule.cooldown_period:
                return  # Skip due to cooldown

        # Create alert instance
        alert = AlertInstance(
            id=alert_id,
            rule_id=rule_id,
            triggered_at=timestamp,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            status="active",
            acknowledged_by=None,
            acknowledged_at=None,
            resolved_by=None,
            resolved_at=None
        )

        self.active_alerts[rule_id] = alert
        self.alert_queue.put(alert)

        # Log alert event
        event = MonitoringEvent(
            timestamp=timestamp,
            event_type="alert",
            severity=rule.severity,
            source="monitoring_system",
            message=f"Alert triggered: {rule.name}",
            data={
                "rule_id": rule_id,
                "current_value": current_value,
                "threshold": rule.threshold,
                "alert_id": alert_id
            },
            metadata={"severity": rule.severity}
        )

        self.event_queue.put(event)

        logger.warning(f"Alert triggered: {rule.name} - Current: {current_value}, Threshold: {rule.threshold}")

    def _user_activity_monitor(self):
        """Monitor user activity patterns"""
        while self.running:
            try:
                if self.config["enabled_monitors"]["user_activity"]:
                    # Analyze user activity patterns
                    activity_analysis = self._analyze_user_activity()

                    # Create monitoring event
                    if activity_analysis["anomalies_detected"]:
                        event = MonitoringEvent(
                            timestamp=datetime.now(),
                            event_type="user",
                            severity="warning",
                            source="activity_monitor",
                            message="User activity anomaly detected",
                            data=activity_analysis,
                            metadata={"anomalies_count": len(activity_analysis["anomalies"])}
                        )
                        self.event_queue.put(event)

                time.sleep(self.user_activity_check_interval)

            except Exception as e:
                logger.error(f"Error in user activity monitor: {e}")
                time.sleep(self.user_activity_check_interval)

    def _analyze_user_activity(self) -> Dict[str, Any]:
        """Analyze user activity for anomalies"""
        analysis = {
            "total_users": len(self.dashboard.active_user_sessions),
            "total_sessions": len(self.dashboard.user_activity_stream),
            "anomalies_detected": [],
            "patterns": {}
        }

        if len(self.dashboard.user_activity_stream) < 10:
            return analysis

        # Analyze activity patterns
        recent_activity = list(self.dashboard.user_activity_stream)[-100:]  # Last 100 activities

        # Check for unusual activity spikes
        activity_timestamps = [datetime.fromisoformat(a['timestamp']) if isinstance(a['timestamp'], str) else a['timestamp'] for a in recent_activity]
        if len(activity_timestamps) > 1:
            time_diffs = [(activity_timestamps[i+1] - activity_timestamps[i]).total_seconds() for i in range(len(activity_timestamps)-1)]

            if time_diffs:
                avg_interval = np.mean(time_diffs)
                std_interval = np.std(time_diffs)

                # Detect anomalies (intervals more than 2 std from mean)
                for i, diff in enumerate(time_diffs):
                    if abs(diff - avg_interval) > 2 * std_interval:
                        analysis["anomalies_detected"].append({
                            "type": "unusual_interval",
                            "interval": diff,
                            "expected_range": f"{avg_interval - 2*std_interval:.1f} - {avg_interval + 2*std_interval:.1f}s"
                        })

        # Check for unusual user behavior
        user_activity_counts = defaultdict(int)
        for activity in recent_activity:
            user_activity_counts[activity['user_id']] += 1

        # Find users with unusually high activity
        if user_activity_counts:
            activity_values = list(user_activity_counts.values())
            avg_activity = np.mean(activity_values)
            std_activity = np.std(activity_values)

            for user_id, count in user_activity_counts.items():
                if count > avg_activity + 2 * std_activity:
                    analysis["anomalies_detected"].append({
                        "type": "high_activity_user",
                        "user_id": user_id,
                        "activity_count": count,
                        "average_activity": avg_activity
                    })

        return analysis

    def _alert_processor(self):
        """Process alerts and send notifications"""
        while self.running:
            try:
                # Process alerts from queue
                while not self.alert_queue.empty():
                    alert = self.alert_queue.get()

                    # Send notifications
                    self._send_alert_notifications(alert)

                    # Store alert in database
                    self._store_alert(alert)

                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in alert processor: {e}")
                time.sleep(5)

    def _send_alert_notifications(self, alert: AlertInstance):
        """Send alert notifications"""
        if not self.config["alerting"]["enabled"]:
            return

        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return

        message = f"Alert: {rule.name}\n"
        message += f"Severity: {alert.severity}\n"
        message += f"Current Value: {alert.current_value}\n"
        message += f"Threshold: {alert.threshold}\n"
        message += f"Time: {alert.triggered_at}"

        # Send email notification (placeholder)
        if self.config["alerting"]["email_notifications"]:
            logger.info(f"Email alert sent: {message}")

        # Send webhook notification
        if self.config["alerting"]["webhook_url"]:
            # Placeholder for webhook implementation
            logger.info(f"Webhook alert sent to: {self.config['alerting']['webhook_url']}")

        # Send Slack notification
        if self.config["alerting"]["slack_webhook"]:
            # Placeholder for Slack webhook implementation
            logger.info(f"Slack alert sent")

    def _store_alert(self, alert: AlertInstance):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.dashboard.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO performance_alerts
                (alert_type, threshold_value, current_value, severity, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert.rule_id,
                alert.threshold,
                alert.current_value,
                alert.severity,
                alert.status,
                alert.triggered_at
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing alert: {e}")

    def _event_processor(self):
        """Process monitoring events"""
        while self.running:
            try:
                # Process events from queue
                while not self.event_queue.empty():
                    event = self.event_queue.get()

                    # Store event in database
                    self._store_event(event)

                    # Send to websocket clients
                    self._broadcast_to_websockets(event)

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                time.sleep(1)

    def _store_event(self, event: MonitoringEvent):
        """Store monitoring event in database"""
        try:
            conn = sqlite3.connect(self.dashboard.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO user_activity
                (user_id, session_id, activity_type, section_id, timestamp, duration_seconds, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                "system_monitor",
                f"session_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
                event.event_type,
                event.source,
                event.timestamp,
                0,
                json.dumps({
                    "event_data": event.data,
                    "metadata": event.metadata,
                    "severity": event.severity,
                    "message": event.message
                })
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing event: {e}")

    def _broadcast_to_websockets(self, event: MonitoringEvent):
        """Broadcast event to websocket clients"""
        if not self.websocket_connections:
            return

        event_data = {
            "type": "monitoring_event",
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "severity": event.severity,
            "source": event.source,
            "message": event.message,
            "data": event.data,
            "metadata": event.metadata
        }

        # Send to all connected clients
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                asyncio.run(websocket.send(json.dumps(event_data)))
            except:
                disconnected.add(websocket)

        # Remove disconnected clients
        self.websocket_connections -= disconnected

    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.running:
            logger.warning("Monitoring is already running")
            return

        self.running = True
        logger.info("Real-time monitoring started")

        # Start asyncio event loop for websockets
        asyncio.run(self._start_websocket_server())

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        logger.info("Real-time monitoring stopped")

    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def websocket_handler(websocket, path):
            self.websocket_connections.add(websocket)
            logger.info(f"WebSocket client connected: {websocket.remote_address}")

            try:
                # Send initial data
                initial_data = {
                    "type": "initial_data",
                    "timestamp": datetime.now().isoformat(),
                    "system_metrics": [asdict(m) for m in list(self.system_metrics)[-10:]],
                    "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
                    "config": self.config
                }
                await websocket.send(json.dumps(initial_data))

                # Keep connection open and listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))
                    except json.JSONDecodeError:
                        pass

            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_connections.discard(websocket)
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")

        # Start WebSocket server
        server = await websockets.serve(websocket_handler, "localhost", 8765)
        logger.info("WebSocket server started on ws://localhost:8765")

        # Keep server running
        while self.running:
            await asyncio.sleep(1)

        server.close()
        await server.wait_closed()

    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "running": self.running,
            "timestamp": datetime.now().isoformat(),
            "system_metrics_count": len(self.system_metrics),
            "user_activity_count": len(self.dashboard.user_activity_stream),
            "active_alerts_count": len(self.active_alerts),
            "websocket_connections": len(self.websocket_connections),
            "config": self.config,
            "alert_rules": {rule_id: asdict(rule) for rule_id, rule in self.alert_rules.items()},
            "active_alerts": {rule_id: asdict(alert) for rule_id, alert in self.active_alerts.items()}
        }

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent monitoring events"""
        try:
            conn = sqlite3.connect(self.dashboard.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, activity_type, timestamp, metadata
                FROM user_activity
                WHERE user_id = 'system_monitor'
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            events = cursor.fetchall()
            conn.close()

            recent_events = []
            for event in events:
                try:
                    metadata = json.loads(event[3]) if event[3] else {}
                    recent_events.append({
                        "timestamp": event[2],
                        "event_type": event[1],
                        "message": metadata.get("message", ""),
                        "severity": metadata.get("severity", "info"),
                        "data": metadata.get("event_data", {}),
                        "metadata": metadata.get("metadata", {})
                    })
                except json.JSONDecodeError:
                    continue

            return recent_events

        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert"""
        try:
            # Find the alert
            alert_to_acknowledge = None
            for alert in self.active_alerts.values():
                if alert.id == alert_id:
                    alert_to_acknowledge = alert
                    break

            if not alert_to_acknowledge:
                return False

            # Update alert status
            alert_to_acknowledge.status = "acknowledged"
            alert_to_acknowledge.acknowledged_by = acknowledged_by
            alert_to_acknowledge.acknowledged_at = datetime.now()

            # Update in database
            conn = sqlite3.connect(self.dashboard.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE performance_alerts
                SET status = 'acknowledged', acknowledged_at = ?, acknowledged_by = ?
                WHERE alert_type = ? AND created_at = ?
            ''', (
                alert_to_acknowledge.acknowledged_at,
                acknowledged_by,
                alert_to_acknowledge.rule_id,
                alert_to_acknowledge.triggered_at
            ))

            conn.commit()
            conn.close()

            # Log acknowledgment event
            event = MonitoringEvent(
                timestamp=datetime.now(),
                event_type="alert",
                severity="info",
                source="admin_action",
                message=f"Alert acknowledged by {acknowledged_by}",
                data={
                    "alert_id": alert_id,
                    "action": "acknowledged",
                    "acknowledged_by": acknowledged_by
                },
                metadata={"alert_severity": alert_to_acknowledge.severity}
            )

            self.event_queue.put(event)

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False

    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an active alert"""
        try:
            # Find the alert
            alert_to_resolve = None
            for alert in self.active_alerts.values():
                if alert.id == alert_id:
                    alert_to_resolve = alert
                    break

            if not alert_to_resolve:
                return False

            # Update alert status
            alert_to_resolve.status = "resolved"
            alert_to_resolve.resolved_by = resolved_by
            alert_to_resolve.resolved_at = datetime.now()

            # Remove from active alerts
            if alert_to_resolve.rule_id in self.active_alerts:
                del self.active_alerts[alert_to_resolve.rule_id]

            # Update in database
            conn = sqlite3.connect(self.dashboard.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE performance_alerts
                SET status = 'resolved', resolved_at = ?, resolved_by = ?
                WHERE alert_type = ? AND created_at = ?
            ''', (
                alert_to_resolve.resolved_at,
                resolved_by,
                alert_to_resolve.rule_id,
                alert_to_resolve.triggered_at
            ))

            conn.commit()
            conn.close()

            # Log resolution event
            event = MonitoringEvent(
                timestamp=datetime.now(),
                event_type="alert",
                severity="info",
                source="admin_action",
                message=f"Alert resolved by {resolved_by}",
                data={
                    "alert_id": alert_id,
                    "action": "resolved",
                    "resolved_by": resolved_by
                },
                metadata={"alert_severity": alert_to_resolve.severity}
            )

            self.event_queue.put(event)

            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True

        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False

    def cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            retention_days = self.config["retention"]["system_metrics_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            conn = sqlite3.connect(self.dashboard.db_path)
            cursor = conn.cursor()

            # Clean up old monitoring events
            cursor.execute('''
                DELETE FROM user_activity
                WHERE user_id = 'system_monitor' AND timestamp < ?
            ''', (cutoff_date,))

            # Clean up old resolved alerts
            alert_cutoff = datetime.now() - timedelta(days=self.config["retention"]["alert_history_days"])
            cursor.execute('''
                DELETE FROM performance_alerts
                WHERE status = 'resolved' AND created_at < ?
            ''', (alert_cutoff,))

            conn.commit()
            conn.close()

            logger.info(f"Cleaned up old monitoring data (older than {retention_days} days)")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

class MonitoringClient:
    """Client for connecting to the monitoring system"""

    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.event_handlers = defaultdict(list)
        self.connected = False

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for specific event types"""
        self.event_handlers[event_type].append(handler)

    async def connect(self):
        """Connect to monitoring server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.connected = True
            logger.info(f"Connected to monitoring server: {self.server_url}")

            # Start listening for events
            asyncio.create_task(self._listen_for_events())

        except Exception as e:
            logger.error(f"Failed to connect to monitoring server: {e}")
            self.connected = False

    async def _listen_for_events(self):
        """Listen for monitoring events"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    if data["type"] == "monitoring_event":
                        # Call appropriate event handlers
                        for handler in self.event_handlers.get(data["event_type"], []):
                            try:
                                await handler(data)
                            except Exception as e:
                                logger.error(f"Error in event handler: {e}")

                    elif data["type"] == "initial_data":
                        # Handle initial data
                        for handler in self.event_handlers.get("initial_data", []):
                            try:
                                await handler(data)
                            except Exception as e:
                                logger.error(f"Error in initial data handler: {e}")

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from monitoring server")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to monitoring server closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error listening for events: {e}")
            self.connected = False

    async def disconnect(self):
        """Disconnect from monitoring server"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from monitoring server")

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Monitoring System")
    parser.add_argument("--config", default="monitoring_config.json", help="Configuration file path")
    parser.add_argument("--dashboard", help="Path to analytics dashboard data")

    args = parser.parse_args()

    # Initialize dashboard
    dashboard = AnalyticsDashboard(args.dashboard if args.dashboard else "analytics_data")

    # Initialize monitoring
    monitor = RealTimeMonitor(dashboard, args.config)

    try:
        # Start monitoring
        monitor.start_monitoring()

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down monitoring system...")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Error in monitoring system: {e}")
        monitor.stop_monitoring()