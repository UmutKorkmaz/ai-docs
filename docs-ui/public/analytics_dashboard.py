#!/usr/bin/env python3
"""
AI Documentation Project Analytics Dashboard
Comprehensive real-time analytics and monitoring system for project performance,
user engagement, and learning outcomes.
"""

import json
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from collections import defaultdict, deque
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing assessment components
from assessment.assessment_engine_core import AssessmentEngine
from assessment.progress_tracking_analytics import ProgressTracker
from assessment.achievement_certification_system import AchievementSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    timestamp: datetime
    active_users: int
    memory_usage_mb: float
    cpu_usage_percent: float
    response_time_ms: float
    database_connections: int
    error_rate_percent: float
    throughput_rps: float

@dataclass
class UserEngagementMetrics:
    """User engagement and activity metrics"""
    user_id: str
    session_duration_minutes: float
    pages_viewed: int
    sections_accessed: List[str]
    time_spent_per_section: Dict[str, float]
    interaction_events: int
    completion_rate: float
    engagement_score: float

@dataclass
class ContentAnalytics:
    """Content usage and performance analytics"""
    section_id: str
    section_name: str
    total_views: int
    unique_users: int
    average_time_spent: float
    completion_rate: float
    popularity_score: float
    difficulty_rating: float
    user_satisfaction: float
    last_updated: datetime

@dataclass
class LearningAnalytics:
    """Learning outcome analytics"""
    user_id: str
    section_id: str
    assessment_scores: List[float]
    skill_mastery_levels: Dict[str, float]
    learning_velocity: float
    knowledge_retention: float
    improvement_rate: float
    predicted_mastery_date: Optional[datetime]
    risk_factors: List[str]

@dataclass
class PredictiveInsight:
    """Predictive analytics insights"""
    insight_type: str  # 'user_churn', 'content_gap', 'performance_trend', 'system_load'
    confidence_score: float
    prediction: Any
    timeframe: str
    action_items: List[str]
    impact_level: str  # 'low', 'medium', 'high', 'critical'

class AnalyticsDashboard:
    """Main analytics dashboard engine"""

    def __init__(self, data_path: str = "analytics_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        # Initialize database
        self.db_path = self.data_path / "analytics.db"
        self._init_database()

        # Initialize Redis for real-time analytics
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_available = True
        except:
            self.redis_available = False
            logger.warning("Redis not available, using in-memory caching")
            self.cache = {}

        # Initialize existing assessment systems
        self.assessment_engine = AssessmentEngine(f"{data_path}/assessment_data")
        self.progress_tracker = ProgressTracker(f"{data_path}/assessment_data")
        self.achievement_system = AchievementSystem(f"{data_path}/assessment_data")

        # Real-time data streams
        self.user_activity_stream = deque(maxlen=10000)
        self.system_metrics_stream = deque(maxlen=10000)
        self.content_access_stream = deque(maxlen=10000)

        # ML models for predictions
        self.user_segmentation_model = KMeans(n_clusters=5, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

        # Dashboard configuration
        self.config = self._load_config()

        logger.info("Analytics Dashboard initialized successfully")

    def _init_database(self):
        """Initialize SQLite database for analytics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # User activity tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                activity_type TEXT,
                section_id TEXT,
                timestamp DATETIME,
                duration_seconds INTEGER,
                metadata TEXT
            )
        ''')

        # Content performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_id TEXT,
                view_count INTEGER DEFAULT 0,
                unique_users INTEGER DEFAULT 0,
                avg_time_spent REAL DEFAULT 0,
                completion_rate REAL DEFAULT 0,
                last_accessed DATETIME,
                difficulty_rating REAL DEFAULT 3.0,
                satisfaction_score REAL DEFAULT 3.0
            )
        ''')

        # System metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                active_users INTEGER,
                memory_usage REAL,
                cpu_usage REAL,
                response_time REAL,
                error_rate REAL,
                throughput REAL
            )
        ''')

        # User segments and predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                segment_id INTEGER,
                segment_name TEXT,
                confidence_score REAL,
                last_updated DATETIME
            )
        ''')

        conn.commit()
        conn.close()

    def _load_config(self) -> Dict[str, Any]:
        """Load dashboard configuration"""
        config_file = self.data_path / "config.json"
        default_config = {
            "real_time_update_interval": 5,  # seconds
            "cache_ttl": 3600,  # 1 hour
            "anomaly_detection_threshold": 0.1,
            "predictive_horizon_days": 30,
            "dashboard_refresh_rate": 30,  # seconds
            "retention_days": 365,
            "enable_ml_predictions": True,
            "alert_thresholds": {
                "high_error_rate": 5.0,
                "low_engagement": 30.0,
                "high_response_time": 1000,
                "low_completion_rate": 40.0
            }
        }

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        return default_config

    def track_user_activity(self, user_id: str, activity_type: str,
                          section_id: str = None, duration: int = 0,
                          metadata: Dict = None) -> bool:
        """Track user activity in real-time"""
        try:
            activity_data = {
                'user_id': user_id,
                'session_id': f"session_{user_id}_{int(datetime.now().timestamp())}",
                'activity_type': activity_type,
                'section_id': section_id,
                'timestamp': datetime.now(),
                'duration_seconds': duration,
                'metadata': json.dumps(metadata or {})
            }

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_activity
                (user_id, session_id, activity_type, section_id, timestamp, duration_seconds, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                activity_data['user_id'],
                activity_data['session_id'],
                activity_data['activity_type'],
                activity_data['section_id'],
                activity_data['timestamp'],
                activity_data['duration_seconds'],
                activity_data['metadata']
            ))
            conn.commit()
            conn.close()

            # Update content performance
            if section_id:
                self._update_content_performance(section_id, activity_data)

            # Add to real-time stream
            self.user_activity_stream.append(activity_data)

            # Cache in Redis if available
            if self.redis_available:
                cache_key = f"user_activity:{user_id}:{int(datetime.now().timestamp())}"
                self.redis_client.setex(cache_key, 3600, json.dumps(activity_data, default=str))

            return True

        except Exception as e:
            logger.error(f"Error tracking user activity: {e}")
            return False

    def _update_content_performance(self, section_id: str, activity_data: Dict):
        """Update content performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if record exists
        cursor.execute('SELECT * FROM content_performance WHERE section_id = ?', (section_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing record
            cursor.execute('''
                UPDATE content_performance
                SET view_count = view_count + 1,
                    avg_time_spent = (avg_time_spent * (view_count - 1) + ?) / view_count,
                    last_accessed = ?
                WHERE section_id = ?
            ''', (activity_data['duration_seconds'], datetime.now(), section_id))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO content_performance
                (section_id, view_count, unique_users, avg_time_spent, last_accessed)
                VALUES (?, 1, 1, ?, ?)
            ''', (section_id, activity_data['duration_seconds'], datetime.now()))

        conn.commit()
        conn.close()

    def record_system_metrics(self, metrics: SystemMetrics) -> bool:
        """Record system performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO system_metrics
                (timestamp, active_users, memory_usage, cpu_usage, response_time, error_rate, throughput)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp,
                metrics.active_users,
                metrics.memory_usage_mb,
                metrics.cpu_usage_percent,
                metrics.response_time_ms,
                metrics.error_rate_percent,
                metrics.throughput_rps
            ))

            conn.commit()
            conn.close()

            # Add to real-time stream
            self.system_metrics_stream.append(metrics)

            return True

        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")
            return False

    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            # System health
            system_health = self._get_system_health()

            # User activity (last hour)
            recent_activity = self._get_recent_activity(hours=1)

            # Content popularity
            popular_content = self._get_popular_content(limit=10)

            # User engagement metrics
            engagement_metrics = self._calculate_engagement_metrics()

            # System alerts
            alerts = self._generate_system_alerts()

            # Performance metrics
            performance_data = self._get_performance_metrics()

            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': system_health,
                'recent_activity': recent_activity,
                'popular_content': popular_content,
                'engagement_metrics': engagement_metrics,
                'alerts': alerts,
                'performance_metrics': performance_data
            }

        except Exception as e:
            logger.error(f"Error generating real-time dashboard: {e}")
            return {'error': str(e)}

    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        # Get latest system metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM system_metrics
            ORDER BY timestamp DESC LIMIT 1
        ''')

        latest_metrics = cursor.fetchone()
        conn.close()

        if not latest_metrics:
            return {'status': 'unknown', 'metrics': {}}

        # Convert tuple to dict
        columns = ['id', 'timestamp', 'active_users', 'memory_usage', 'cpu_usage',
                   'response_time', 'error_rate', 'throughput']
        metrics = dict(zip(columns, latest_metrics))

        # Determine health status
        status = 'healthy'
        issues = []

        if metrics['error_rate'] > self.config['alert_thresholds']['high_error_rate']:
            status = 'warning'
            issues.append(f"High error rate: {metrics['error_rate']}%")

        if metrics['response_time'] > self.config['alert_thresholds']['high_response_time']:
            status = 'warning'
            issues.append(f"High response time: {metrics['response_time']}ms")

        if metrics['cpu_usage'] > 80:
            status = 'critical'
            issues.append(f"High CPU usage: {metrics['cpu_usage']}%")

        return {
            'status': status,
            'issues': issues,
            'metrics': {
                'active_users': metrics['active_users'],
                'memory_usage_mb': metrics['memory_usage'],
                'cpu_usage_percent': metrics['cpu_usage'],
                'response_time_ms': metrics['response_time'],
                'error_rate_percent': metrics['error_rate'],
                'throughput_rps': metrics['throughput']
            }
        }

    def _get_recent_activity(self, hours: int = 1) -> List[Dict]:
        """Get recent user activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_time = datetime.now() - timedelta(hours=hours)

        cursor.execute('''
            SELECT user_id, activity_type, section_id, timestamp, duration_seconds
            FROM user_activity
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (cutoff_time,))

        activities = cursor.fetchall()
        conn.close()

        return [
            {
                'user_id': row[0],
                'activity_type': row[1],
                'section_id': row[2],
                'timestamp': row[3],
                'duration_seconds': row[4]
            }
            for row in activities
        ]

    def _get_popular_content(self, limit: int = 10) -> List[Dict]:
        """Get most popular content sections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT section_id, view_count, unique_users, avg_time_spent, completion_rate
            FROM content_performance
            ORDER BY view_count DESC
            LIMIT ?
        ''', (limit,))

        results = cursor.fetchall()
        conn.close()

        return [
            {
                'section_id': row[0],
                'view_count': row[1],
                'unique_users': row[2],
                'avg_time_spent': row[3],
                'completion_rate': row[4]
            }
            for row in results
        ]

    def _calculate_engagement_metrics(self) -> Dict[str, float]:
        """Calculate overall user engagement metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get activity from last 24 hours
        cutoff_time = datetime.now() - timedelta(days=1)

        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) as unique_users,
                   COUNT(*) as total_activities,
                   AVG(duration_seconds) as avg_duration,
                   SUM(duration_seconds) as total_time
            FROM user_activity
            WHERE timestamp >= ?
        ''', (cutoff_time,))

        result = cursor.fetchone()
        conn.close()

        if not result or result[0] == 0:
            return {
                'daily_active_users': 0,
                'avg_session_duration': 0,
                'total_engagement_time': 0,
                'engagement_score': 0
            }

        unique_users, total_activities, avg_duration, total_time = result

        # Calculate engagement score (0-100)
        engagement_score = min(100, (unique_users * 10) + (avg_duration / 60) * 5)

        return {
            'daily_active_users': unique_users,
            'avg_session_duration': avg_duration,
            'total_engagement_time': total_time,
            'engagement_score': engagement_score
        }

    def _generate_system_alerts(self) -> List[Dict]:
        """Generate system alerts based on current conditions"""
        alerts = []

        # Get latest metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM system_metrics
            ORDER BY timestamp DESC LIMIT 1
        ''')

        latest_metrics = cursor.fetchone()
        conn.close()

        if latest_metrics:
            columns = ['id', 'timestamp', 'active_users', 'memory_usage', 'cpu_usage',
                       'response_time', 'error_rate', 'throughput']
            metrics = dict(zip(columns, latest_metrics))

            # Error rate alert
            if metrics['error_rate'] > self.config['alert_thresholds']['high_error_rate']:
                alerts.append({
                    'type': 'error',
                    'severity': 'high',
                    'message': f"High error rate detected: {metrics['error_rate']}%",
                    'timestamp': datetime.now().isoformat()
                })

            # Response time alert
            if metrics['response_time'] > self.config['alert_thresholds']['high_response_time']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"High response time: {metrics['response_time']}ms",
                    'timestamp': datetime.now().isoformat()
                })

            # CPU usage alert
            if metrics['cpu_usage'] > 80:
                alerts.append({
                    'type': 'resource',
                    'severity': 'high',
                    'message': f"High CPU usage: {metrics['cpu_usage']}%",
                    'timestamp': datetime.now().isoformat()
                })

        return alerts

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get metrics from last 24 hours
        cutoff_time = datetime.now() - timedelta(days=1)

        cursor.execute('''
            SELECT
                AVG(active_users) as avg_active_users,
                MAX(active_users) as peak_active_users,
                AVG(memory_usage) as avg_memory_usage,
                AVG(cpu_usage) as avg_cpu_usage,
                AVG(response_time) as avg_response_time,
                AVG(error_rate) as avg_error_rate,
                AVG(throughput) as avg_throughput
            FROM system_metrics
            WHERE timestamp >= ?
        ''', (cutoff_time,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            return {}

        columns = ['avg_active_users', 'peak_active_users', 'avg_memory_usage',
                   'avg_cpu_usage', 'avg_response_time', 'avg_error_rate', 'avg_throughput']

        return dict(zip(columns, result))

    def generate_content_analytics(self, section_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive content analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if section_id:
            # Analytics for specific section
            cursor.execute('''
                SELECT * FROM content_performance
                WHERE section_id = ?
            ''', (section_id,))

            content_data = cursor.fetchone()

            if not content_data:
                return {'error': f'Section {section_id} not found'}

            # Get user activity for this section
            cursor.execute('''
                SELECT user_id, activity_type, timestamp, duration_seconds
                FROM user_activity
                WHERE section_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (section_id,))

            activities = cursor.fetchall()

            content_analytics = {
                'section_id': section_id,
                'performance': {
                    'view_count': content_data[1],
                    'unique_users': content_data[2],
                    'avg_time_spent': content_data[3],
                    'completion_rate': content_data[4],
                    'difficulty_rating': content_data[5],
                    'satisfaction_score': content_data[6]
                },
                'recent_activity': [
                    {
                        'user_id': row[0],
                        'activity_type': row[1],
                        'timestamp': row[2],
                        'duration_seconds': row[3]
                    }
                    for row in activities
                ]
            }
        else:
            # Overall content analytics
            cursor.execute('''
                SELECT section_id, view_count, unique_users, avg_time_spent, completion_rate
                FROM content_performance
                ORDER BY view_count DESC
            ''')

            all_content = cursor.fetchall()

            content_analytics = {
                'total_sections': len(all_content),
                'content_performance': [
                    {
                        'section_id': row[0],
                        'view_count': row[1],
                        'unique_users': row[2],
                        'avg_time_spent': row[3],
                        'completion_rate': row[4]
                    }
                    for row in all_content
                ]
            }

        conn.close()
        return content_analytics

    def predict_user_churn(self) -> List[PredictiveInsight]:
        """Predict user churn using ML models"""
        if not self.config['enable_ml_predictions']:
            return []

        insights = []

        try:
            # Get user activity data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, COUNT(*) as activity_count,
                       MAX(timestamp) as last_activity,
                       AVG(duration_seconds) as avg_duration
                FROM user_activity
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY user_id
            ''')

            user_data = cursor.fetchall()
            conn.close()

            if len(user_data) < 10:
                return insights

            # Prepare features for ML
            features = []
            user_ids = []

            for row in user_data:
                user_id, activity_count, last_activity, avg_duration = row

                # Calculate days since last activity
                if last_activity:
                    if isinstance(last_activity, str):
                        last_activity_dt = datetime.fromisoformat(last_activity)
                    else:
                        last_activity_dt = last_activity
                    days_inactive = (datetime.now() - last_activity_dt).days
                else:
                    days_inactive = 30

                features.append([activity_count, avg_duration or 0, days_inactive])
                user_ids.append(user_id)

            # Predict churn risk
            if features:
                feature_array = np.array(features)

                # Use isolation forest to detect anomalous (potentially churning) users
                risk_scores = self.anomaly_detector.fit_predict(feature_array)

                # Generate insights for high-risk users
                for i, risk_score in enumerate(risk_scores):
                    if risk_score == -1:  # Anomaly detected
                        user_id = user_ids[i]
                        activity_count, avg_duration, days_inactive = features[i]

                        if days_inactive > 7:  # Inactive for more than a week
                            insight = PredictiveInsight(
                                insight_type='user_churn',
                                confidence_score=0.8,
                                prediction={
                                    'user_id': user_id,
                                    'days_inactive': days_inactive,
                                    'historical_activity': activity_count
                                },
                                timeframe='next_30_days',
                                action_items=[
                                    f'Send re-engagement email to user {user_id}',
                                    'Offer personalized content recommendations',
                                    'Check for technical issues affecting this user'
                                ],
                                impact_level='medium'
                            )
                            insights.append(insight)

        except Exception as e:
            logger.error(f"Error in churn prediction: {e}")

        return insights

    def generate_content_gap_analysis(self) -> List[PredictiveInsight]:
        """Analyze content gaps and learning recommendations"""
        insights = []

        try:
            # Get all available sections (based on your documentation structure)
            available_sections = [
                "01_Foundational_Machine_Learning",
                "02_Advanced_Deep_Learning",
                "03_Natural_Language_Processing",
                "04_Computer_Vision",
                "05_Generative_AI",
                "06_AI_Agents_and_Autonomous_Systems",
                "07_AI_Ethics_Safety_and_Governance",
                "08_AI_Applications_and_Industry_Verticals"
            ]

            # Get content performance data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT section_id, view_count, completion_rate, avg_time_spent
                FROM content_performance
            ''')

            content_stats = cursor.fetchall()
            conn.close()

            # Identify underperforming content
            for section_id in available_sections:
                section_stats = next((stats for stats in content_stats if stats[0] == section_id), None)

                if not section_stats:
                    # New section with no data
                    insight = PredictiveInsight(
                        insight_type='content_gap',
                        confidence_score=0.7,
                        prediction={
                            'section_id': section_id,
                            'issue': 'no_data_available',
                            'recommendation': 'promote_section_content'
                        },
                        timeframe='immediate',
                        action_items=[
                            f'Create promotional content for section {section_id}',
                            'Add section to recommended learning paths',
                            'Create assessment questions for this section'
                        ],
                        impact_level='medium'
                    )
                    insights.append(insight)

                elif section_stats[1] < 10:  # Low view count
                    insight = PredictiveInsight(
                        insight_type='content_gap',
                        confidence_score=0.8,
                        prediction={
                            'section_id': section_id,
                            'issue': 'low_engagement',
                            'view_count': section_stats[1],
                            'completion_rate': section_stats[2]
                        },
                        timeframe='next_14_days',
                        action_items=[
                            f'Review and improve content for section {section_id}',
                            'Add interactive elements or examples',
                            'Create targeted promotion for this section'
                        ],
                        impact_level='high' if section_stats[2] < 50 else 'medium'
                    )
                    insights.append(insight)

        except Exception as e:
            logger.error(f"Error in content gap analysis: {e}")

        return insights

    def generate_performance_trends(self) -> List[PredictiveInsight]:
        """Generate performance trend predictions"""
        insights = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get system metrics trend (last 7 days)
            cursor.execute('''
                SELECT timestamp, error_rate, response_time, throughput
                FROM system_metrics
                WHERE timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp
            ''')

            metrics_data = cursor.fetchall()
            conn.close()

            if len(metrics_data) < 2:
                return insights

            # Analyze trends
            error_rates = [row[1] for row in metrics_data]
            response_times = [row[2] for row in metrics_data]
            timestamps = [row[0] for row in metrics_data]

            # Calculate trend directions
            if len(error_rates) >= 3:
                recent_error_trend = error_rates[-1] - error_rates[-3]

                if recent_error_trend > 2:  # Error rate increasing
                    insight = PredictiveInsight(
                        insight_type='performance_trend',
                        confidence_score=0.9,
                        prediction={
                            'metric': 'error_rate',
                            'current_value': error_rates[-1],
                            'trend': 'increasing',
                            'projected_value': error_rates[-1] + recent_error_trend
                        },
                        timeframe='next_7_days',
                        action_items=[
                            'Investigate recent code changes or deployments',
                            'Review error logs for patterns',
                            'Consider scaling infrastructure',
                            'Schedule maintenance window'
                        ],
                        impact_level='high' if error_rates[-1] > 5 else 'medium'
                    )
                    insights.append(insight)

            # Response time trend
            if len(response_times) >= 3:
                recent_response_trend = response_times[-1] - response_times[-3]

                if recent_response_trend > 100:  # Response time significantly increasing
                    insight = PredictiveInsight(
                        insight_type='performance_trend',
                        confidence_score=0.8,
                        prediction={
                            'metric': 'response_time',
                            'current_value': response_times[-1],
                            'trend': 'increasing',
                            'projected_value': response_times[-1] + recent_response_trend
                        },
                        timeframe='next_7_days',
                        action_items=[
                            'Profile application performance',
                            'Check database query optimization',
                            'Consider caching strategies',
                            'Monitor server resources'
                        ],
                        impact_level='high' if response_times[-1] > 1000 else 'medium'
                    )
                    insights.append(insight)

        except Exception as e:
            logger.error(f"Error in performance trend analysis: {e}")

        return insights

    def generate_comprehensive_insights(self) -> Dict[str, Any]:
        """Generate all types of predictive insights"""
        try:
            churn_insights = self.predict_user_churn()
            content_insights = self.generate_content_gap_analysis()
            performance_insights = self.generate_performance_trends()

            all_insights = churn_insights + content_insights + performance_insights

            # Sort by impact level and confidence
            impact_priority = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            all_insights.sort(
                key=lambda x: (impact_priority.get(x.impact_level, 0), x.confidence_score),
                reverse=True
            )

            return {
                'timestamp': datetime.now().isoformat(),
                'total_insights': len(all_insights),
                'insights_by_type': {
                    'user_churn': len(churn_insights),
                    'content_gap': len(content_insights),
                    'performance_trend': len(performance_insights)
                },
                'high_priority_insights': [
                    asdict(insight) for insight in all_insights[:5]
                ],
                'all_insights': [asdict(insight) for insight in all_insights]
            }

        except Exception as e:
            logger.error(f"Error generating comprehensive insights: {e}")
            return {'error': str(e)}

    def create_dashboard_visualizations(self) -> Dict[str, str]:
        """Create interactive dashboard visualizations"""
        visualizations = {}

        try:
            # System performance over time
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT timestamp, active_users, cpu_usage, memory_usage, response_time
                FROM system_metrics
                WHERE timestamp >= datetime('now', '-24 hours')
                ORDER BY timestamp
            ''')

            perf_data = cursor.fetchall()
            conn.close()

            if perf_data:
                df = pd.DataFrame(perf_data, columns=['timestamp', 'active_users', 'cpu_usage', 'memory_usage', 'response_time'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Active Users', 'CPU Usage (%)', 'Memory Usage (MB)', 'Response Time (ms)'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )

                # Add traces
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['active_users'], name='Active Users'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['cpu_usage'], name='CPU Usage'),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['memory_usage'], name='Memory Usage'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['response_time'], name='Response Time'),
                    row=2, col=2
                )

                fig.update_layout(height=600, title_text="System Performance - Last 24 Hours")
                visualizations['system_performance'] = fig.to_html(include_plotlyjs='cdn')

            # Content popularity chart
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT section_id, view_count, completion_rate
                FROM content_performance
                ORDER BY view_count DESC
                LIMIT 10
            ''')

            content_data = cursor.fetchall()
            conn.close()

            if content_data:
                df = pd.DataFrame(content_data, columns=['section_id', 'view_count', 'completion_rate'])

                fig = px.bar(df, x='section_id', y='view_count',
                           title='Top 10 Most Popular Content Sections',
                           color='completion_rate',
                           color_continuous_scale='RdYlGn')

                visualizations['content_popularity'] = fig.to_html(include_plotlyjs='cdn')

            # User engagement heatmap
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    strftime('%Y-%m-%d', timestamp) as date,
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as activity_count
                FROM user_activity
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY date, hour
                ORDER BY date, hour
            ''')

            activity_data = cursor.fetchall()
            conn.close()

            if activity_data:
                df = pd.DataFrame(activity_data, columns=['date', 'hour', 'activity_count'])

                # Create pivot table for heatmap
                pivot_df = df.pivot(index='hour', columns='date', values='activity_count').fillna(0)

                fig = px.imshow(pivot_df,
                              title='User Activity Heatmap (Last 7 Days)',
                              labels=dict(x="Date", y="Hour of Day", color="Activity Count"),
                              color_continuous_scale='Viridis')

                visualizations['engagement_heatmap'] = fig.to_html(include_plotlyjs='cdn')

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

        return visualizations

    def export_analytics_report(self, report_type: str = 'comprehensive') -> Dict[str, Any]:
        """Export analytics data in various formats"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if report_type == 'comprehensive':
                # Get all analytics data
                dashboard_data = self.get_real_time_dashboard()
                insights = self.generate_comprehensive_insights()
                content_analytics = self.generate_content_analytics()
                visualizations = self.create_dashboard_visualizations()

                report = {
                    'report_type': 'comprehensive',
                    'generated_at': datetime.now().isoformat(),
                    'dashboard_data': dashboard_data,
                    'insights': insights,
                    'content_analytics': content_analytics,
                    'visualizations': visualizations
                }

                # Save as JSON
                report_file = self.data_path / f"analytics_report_{timestamp}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

                return {
                    'status': 'success',
                    'report_file': str(report_file),
                    'summary': {
                        'total_insights': len(insights.get('all_insights', [])),
                        'high_priority_alerts': len(dashboard_data.get('alerts', [])),
                        'content_sections_analyzed': len(content_analytics.get('content_performance', []))
                    }
                }

            elif report_type == 'user_engagement':
                # User engagement specific report
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT user_id, COUNT(*) as total_activities,
                           AVG(duration_seconds) as avg_duration,
                           MAX(timestamp) as last_activity
                    FROM user_activity
                    GROUP BY user_id
                    ORDER BY total_activities DESC
                ''')

                user_data = cursor.fetchall()
                conn.close()

                report = {
                    'report_type': 'user_engagement',
                    'generated_at': datetime.now().isoformat(),
                    'total_users': len(user_data),
                    'user_statistics': [
                        {
                            'user_id': row[0],
                            'total_activities': row[1],
                            'avg_duration_seconds': row[2],
                            'last_activity': row[3]
                        }
                        for row in user_data
                    ]
                }

                # Save as CSV
                report_file = self.data_path / f"user_engagement_{timestamp}.csv"
                df = pd.DataFrame(user_data, columns=['user_id', 'total_activities', 'avg_duration', 'last_activity'])
                df.to_csv(report_file, index=False)

                return {
                    'status': 'success',
                    'report_file': str(report_file),
                    'total_users': len(user_data)
                }

            else:
                return {'status': 'error', 'message': f'Unknown report type: {report_type}'}

        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return {'status': 'error', 'message': str(e)}

    def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            retention_days = self.config['retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clean up old user activity
            cursor.execute('''
                DELETE FROM user_activity
                WHERE timestamp < ?
            ''', (cutoff_date,))

            # Clean up old system metrics
            cursor.execute('''
                DELETE FROM system_metrics
                WHERE timestamp < ?
            ''', (cutoff_date,))

            conn.commit()
            conn.close()

            logger.info(f"Cleaned up data older than {retention_days} days")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

# FastAPI Web Interface
app = FastAPI(title="AI Documentation Analytics Dashboard", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize dashboard
dashboard = AnalyticsDashboard()

# Pydantic models for API
class ActivityLog(BaseModel):
    user_id: str
    activity_type: str
    section_id: Optional[str] = None
    duration: Optional[int] = 0
    metadata: Optional[Dict] = None

class SystemMetrics(BaseModel):
    active_users: int
    memory_usage_mb: float
    cpu_usage_percent: float
    response_time_ms: float
    error_rate_percent: float
    throughput_rps: float

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Documentation Analytics Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
            .metric { font-size: 24px; font-weight: bold; color: #333; }
            .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
            .alert.warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
            .alert.error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <h1>AI Documentation Analytics Dashboard</h1>
        <div id="dashboard-content">
            <p>Loading dashboard data...</p>
        </div>

        <script>
            async function loadDashboard() {
                try {
                    const response = await fetch('/api/dashboard');
                    const data = await response.json();

                    let html = '<div class="dashboard">';

                    // System Health
                    html += '<div class="card"><h2>System Health</h2>';
                    html += `<p>Status: <span class="metric">${data.system_health.status}</span></p>`;
                    html += `<p>Active Users: ${data.system_health.metrics.active_users}</p>`;
                    html += `<p>CPU Usage: ${data.system_health.metrics.cpu_usage_percent}%</p>`;
                    html += '</div>';

                    // Engagement Metrics
                    html += '<div class="card"><h2>User Engagement</h2>';
                    html += `<p>Daily Active Users: ${data.engagement_metrics.daily_active_users}</p>`;
                    html += `<p>Engagement Score: <span class="metric">${data.engagement_metrics.engagement_score.toFixed(1)}</span></p>`;
                    html += '</div>';

                    html += '</div>';

                    // Alerts
                    if (data.alerts && data.alerts.length > 0) {
                        html += '<h2>System Alerts</h2>';
                        data.alerts.forEach(alert => {
                            html += `<div class="alert ${alert.type}">${alert.message}</div>`;
                        });
                    }

                    document.getElementById('dashboard-content').innerHTML = html;

                } catch (error) {
                    document.getElementById('dashboard-content').innerHTML =
                        `<p>Error loading dashboard: ${error.message}</p>`;
                }
            }

            // Load dashboard on page load
            loadDashboard();

            // Refresh every 30 seconds
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """

@app.get("/api/dashboard")
async def get_dashboard_data():
    """Get real-time dashboard data"""
    try:
        return dashboard.get_real_time_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/activity")
async def log_activity(activity: ActivityLog):
    """Log user activity"""
    try:
        success = dashboard.track_user_activity(
            user_id=activity.user_id,
            activity_type=activity.activity_type,
            section_id=activity.section_id,
            duration=activity.duration,
            metadata=activity.metadata
        )

        if success:
            return {"status": "success", "message": "Activity logged successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to log activity")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system-metrics")
async def log_system_metrics(metrics: SystemMetrics):
    """Log system metrics"""
    try:
        system_metrics = SystemMetrics(
            timestamp=datetime.now(),
            active_users=metrics.active_users,
            memory_usage_mb=metrics.memory_usage_mb,
            cpu_usage_percent=metrics.cpu_usage_percent,
            response_time_ms=metrics.response_time_ms,
            database_connections=0,  # Placeholder
            error_rate_percent=metrics.error_rate_percent,
            throughput_rps=metrics.throughput_rps
        )

        success = dashboard.record_system_metrics(system_metrics)

        if success:
            return {"status": "success", "message": "Metrics logged successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to log metrics")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/content-analytics/{section_id}")
async def get_content_analytics(section_id: str):
    """Get analytics for specific content section"""
    try:
        return dashboard.generate_content_analytics(section_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights")
async def get_insights():
    """Get predictive insights"""
    try:
        return dashboard.generate_comprehensive_insights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualizations")
async def get_visualizations():
    """Get dashboard visualizations"""
    try:
        return dashboard.create_dashboard_visualizations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/{report_type}")
async def export_report(report_type: str):
    """Export analytics report"""
    try:
        return dashboard.export_analytics_report(report_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()

    try:
        while True:
            # Send real-time dashboard data
            dashboard_data = dashboard.get_real_time_dashboard()
            await websocket.send_json(dashboard_data)

            # Wait for next update
            await asyncio.sleep(5)  # Update every 5 seconds

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Documentation Analytics Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Starting Analytics Dashboard on {args.host}:{args.port}")
    print(f"Dashboard URL: http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "analytics_dashboard:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="info"
    )