#!/usr/bin/env python3
"""
Administrative Management Panel for AI Documentation Analytics
Comprehensive admin tools for project management, user monitoring, and system optimization.
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fastapi import FastAPI, HTTPException, WebSocket, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum

from analytics_dashboard import AnalyticsDashboard, SystemMetrics, PredictiveInsight

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(str, Enum):
    ADMIN = "admin"
    MODERATOR = "moderator"
    ANALYST = "analyst"
    VIEWER = "viewer"

class Permission(str, Enum):
    USER_MANAGEMENT = "user_management"
    CONTENT_MANAGEMENT = "content_management"
    SYSTEM_ADMIN = "system_admin"
    ANALYTICS_VIEW = "analytics_view"
    REPORT_EXPORT = "report_export"
    ALERT_MANAGEMENT = "alert_management"

@dataclass
class AdminUser:
    """Administrator user profile"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

@dataclass
class SystemConfiguration:
    """System configuration parameters"""
    setting_key: str
    setting_value: Any
    description: str
    data_type: str  # string, integer, float, boolean, json
    is_sensitive: bool
    last_modified: datetime
    modified_by: str

@dataclass
class AuditLog:
    """Audit log entry for tracking administrative actions"""
    id: int
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    id: int
    alert_type: str
    threshold_value: float
    comparison_operator: str  # greater_than, less_than, equals
    current_value: float
    severity: str  # info, warning, error, critical
    status: str  # active, acknowledged, resolved
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]

class AdminPanel:
    """Main administrative management panel"""

    def __init__(self, dashboard: AnalyticsDashboard, data_path: str = "admin_data"):
        self.dashboard = dashboard
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        # Initialize admin database
        self.db_path = self.data_path / "admin.db"
        self._init_admin_database()

        # Load admin users
        self.admin_users = self._load_admin_users()

        # Active sessions
        self.active_sessions = {}

        logger.info("Admin Panel initialized successfully")

    def _init_admin_database(self):
        """Initialize administrative database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Admin users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                permissions TEXT,  -- JSON array
                created_at DATETIME,
                last_login DATETIME,
                is_active BOOLEAN DEFAULT 1,
                password_hash TEXT
            )
        ''')

        # System configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT,
                description TEXT,
                data_type TEXT,
                is_sensitive BOOLEAN DEFAULT 0,
                last_modified DATETIME,
                modified_by TEXT
            )
        ''')

        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                user_id TEXT,
                action TEXT,
                resource_type TEXT,
                resource_id TEXT,
                details TEXT,  -- JSON
                ip_address TEXT,
                user_agent TEXT
            )
        ''')

        # Performance alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                threshold_value REAL,
                comparison_operator TEXT,
                current_value REAL,
                severity TEXT,
                status TEXT DEFAULT 'active',
                created_at DATETIME,
                acknowledged_at DATETIME,
                resolved_at DATETIME,
                acknowledged_by TEXT,
                resolved_by TEXT
            )
        ''')

        # Create default admin user if none exists
        cursor.execute("SELECT COUNT(*) FROM admin_users")
        if cursor.fetchone()[0] == 0:
            import hashlib
            default_password = hashlib.sha256("admin123".encode()).hexdigest()

            cursor.execute('''
                INSERT INTO admin_users (user_id, username, email, role, permissions, created_at, is_active, password_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                "admin_001",
                "admin",
                "admin@ai-docs.com",
                "admin",
                json.dumps(["user_management", "content_management", "system_admin", "analytics_view", "report_export", "alert_management"]),
                datetime.now(),
                True,
                default_password
            ))

        # Insert default system configuration
        default_configs = [
            ("dashboard_refresh_rate", "30", "Dashboard refresh rate in seconds", "integer", False),
            ("data_retention_days", "365", "Data retention period in days", "integer", False),
            ("enable_ml_predictions", "true", "Enable ML-based predictions", "boolean", False),
            ("alert_email_notifications", "true", "Enable email alert notifications", "boolean", False),
            ("max_concurrent_users", "1000", "Maximum concurrent users", "integer", False),
            ("session_timeout_minutes", "60", "User session timeout in minutes", "integer", False),
            ("enable_audit_logging", "true", "Enable comprehensive audit logging", "boolean", False),
        ]

        for key, value, desc, data_type, sensitive in default_configs:
            cursor.execute('''
                INSERT OR IGNORE INTO system_config
                (setting_key, setting_value, description, data_type, is_sensitive, last_modified, modified_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (key, value, desc, data_type, sensitive, datetime.now(), "system"))

        conn.commit()
        conn.close()

    def _load_admin_users(self) -> Dict[str, AdminUser]:
        """Load administrator users from database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM admin_users')
        users_data = cursor.fetchall()
        conn.close()

        users = {}
        for user_data in users_data:
            user_id, username, email, role, permissions_json, created_at, last_login, is_active, password_hash = user_data

            permissions = json.loads(permissions_json) if permissions_json else []

            user = AdminUser(
                user_id=user_id,
                username=username,
                email=email,
                role=UserRole(role),
                permissions=[Permission(p) for p in permissions],
                created_at=datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at,
                last_login=datetime.fromisoformat(last_login) if last_login else None,
                is_active=bool(is_active)
            )
            users[user_id] = user

        return users

    def authenticate_admin(self, username: str, password: str) -> Optional[AdminUser]:
        """Authenticate administrator user"""
        import hashlib
        import sqlite3

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_id, username, email, role, permissions, created_at, last_login, is_active
            FROM admin_users
            WHERE username = ? AND password_hash = ? AND is_active = 1
        ''', (username, password_hash))

        user_data = cursor.fetchone()
        conn.close()

        if not user_data:
            return None

        user_id, username, email, role, permissions_json, created_at, last_login, is_active = user_data

        # Update last login
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE admin_users SET last_login = ? WHERE user_id = ?
        ''', (datetime.now(), user_id))
        conn.commit()
        conn.close()

        permissions = json.loads(permissions_json) if permissions_json else []

        return AdminUser(
            user_id=user_id,
            username=username,
            email=email,
            role=UserRole(role),
            permissions=[Permission(p) for p in permissions],
            created_at=datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at,
            last_login=last_login,
            is_active=bool(is_active)
        )

    def log_admin_action(self, user_id: str, action: str, resource_type: str,
                        resource_id: str, details: Dict[str, Any],
                        ip_address: str = "127.0.0.1", user_agent: str = "Admin Panel"):
        """Log administrative action for audit purposes"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO audit_log (timestamp, user_id, action, resource_type, resource_id, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            user_id,
            action,
            resource_type,
            resource_id,
            json.dumps(details),
            ip_address,
            user_agent
        ))

        conn.commit()
        conn.close()

    def get_user_management_data(self) -> Dict[str, Any]:
        """Get comprehensive user management data"""
        # Get user activity from analytics
        conn = sqlite3.connect(self.dashboard.db_path)
        cursor = conn.cursor()

        # User statistics
        cursor.execute('''
            SELECT
                COUNT(DISTINCT user_id) as total_users,
                COUNT(*) as total_activities,
                AVG(duration_seconds) as avg_session_duration,
                MAX(timestamp) as last_activity
            FROM user_activity
        ''')

        user_stats = cursor.fetchone()

        # Most active users
        cursor.execute('''
            SELECT user_id, COUNT(*) as activity_count, MAX(timestamp) as last_activity
            FROM user_activity
            GROUP BY user_id
            ORDER BY activity_count DESC
            LIMIT 20
        ''')

        active_users = cursor.fetchall()

        # User activity over time
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(DISTINCT user_id) as unique_users
            FROM user_activity
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')

        user_trends = cursor.fetchall()
        conn.close()

        return {
            'total_users': user_stats[0],
            'total_activities': user_stats[1],
            'avg_session_duration': user_stats[2],
            'last_activity': user_stats[3],
            'most_active_users': [
                {'user_id': row[0], 'activity_count': row[1], 'last_activity': row[2]}
                for row in active_users
            ],
            'user_trends': [
                {'date': row[0], 'unique_users': row[1]}
                for row in user_trends
            ]
        }

    def get_content_management_data(self) -> Dict[str, Any]:
        """Get content management analytics"""
        conn = sqlite3.connect(self.dashboard.db_path)
        cursor = conn.cursor()

        # Content performance
        cursor.execute('''
            SELECT section_id, view_count, unique_users, avg_time_spent, completion_rate
            FROM content_performance
            ORDER BY view_count DESC
        ''')

        content_performance = cursor.fetchall()

        # Content gaps (sections with low engagement)
        cursor.execute('''
            SELECT section_id, view_count, completion_rate
            FROM content_performance
            WHERE completion_rate < 50 OR view_count < 10
            ORDER BY completion_rate ASC
        ''')

        content_gaps = cursor.fetchall()

        # Content popularity trends
        cursor.execute('''
            SELECT
                section_id,
                DATE(timestamp) as date,
                COUNT(*) as daily_views
            FROM user_activity
            WHERE timestamp >= datetime('now', '-30 days')
            AND section_id IS NOT NULL
            GROUP BY section_id, DATE(timestamp)
            ORDER BY date
        ''')

        content_trends = cursor.fetchall()
        conn.close()

        return {
            'content_performance': [
                {
                    'section_id': row[0],
                    'view_count': row[1],
                    'unique_users': row[2],
                    'avg_time_spent': row[3],
                    'completion_rate': row[4]
                }
                for row in content_performance
            ],
            'content_gaps': [
                {
                    'section_id': row[0],
                    'view_count': row[1],
                    'completion_rate': row[2]
                }
                for row in content_gaps
            ],
            'content_trends': [
                {'section_id': row[0], 'date': row[1], 'daily_views': row[2]}
                for row in content_trends
            ]
        }

    def get_system_health_data(self) -> Dict[str, Any]:
        """Get system health and performance data"""
        conn = sqlite3.connect(self.dashboard.db_path)
        cursor = conn.cursor()

        # Current system metrics
        cursor.execute('''
            SELECT * FROM system_metrics
            ORDER BY timestamp DESC
            LIMIT 1
        ''')

        current_metrics = cursor.fetchone()

        # System metrics trends (last 24 hours)
        cursor.execute('''
            SELECT timestamp, cpu_usage, memory_usage, response_time, error_rate
            FROM system_metrics
            WHERE timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp
        ''')

        metrics_trends = cursor.fetchall()

        # Active alerts
        cursor.execute('''
            SELECT * FROM performance_alerts
            WHERE status = 'active'
            ORDER BY created_at DESC
        ''')

        active_alerts = cursor.fetchall()
        conn.close()

        return {
            'current_metrics': {
                'active_users': current_metrics[2] if current_metrics else 0,
                'cpu_usage': current_metrics[3] if current_metrics else 0,
                'memory_usage': current_metrics[4] if current_metrics else 0,
                'response_time': current_metrics[5] if current_metrics else 0,
                'error_rate': current_metrics[6] if current_metrics else 0,
                'throughput': current_metrics[7] if current_metrics else 0
            },
            'metrics_trends': [
                {
                    'timestamp': row[0],
                    'cpu_usage': row[1],
                    'memory_usage': row[2],
                    'response_time': row[3],
                    'error_rate': row[4]
                }
                for row in metrics_trends
            ],
            'active_alerts': [
                {
                    'id': row[0],
                    'alert_type': row[1],
                    'threshold_value': row[2],
                    'current_value': row[3],
                    'severity': row[4],
                    'created_at': row[5]
                }
                for row in active_alerts
            ]
        }

    def generate_admin_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive admin dashboard"""
        try:
            # Get data from all components
            user_data = self.get_user_management_data()
            content_data = self.get_content_management_data()
            system_data = self.get_system_health_data()
            insights = self.dashboard.generate_comprehensive_insights()

            # Get audit log (recent actions)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT timestamp, user_id, action, resource_type, resource_id
                FROM audit_log
                ORDER BY timestamp DESC
                LIMIT 10
            ''')

            recent_actions = cursor.fetchall()
            conn.close()

            return {
                'timestamp': datetime.now().isoformat(),
                'user_management': user_data,
                'content_management': content_data,
                'system_health': system_data,
                'insights': insights,
                'recent_admin_actions': [
                    {
                        'timestamp': row[0],
                        'user_id': row[1],
                        'action': row[2],
                        'resource_type': row[3],
                        'resource_id': row[4]
                    }
                    for row in recent_actions
                ]
            }

        except Exception as e:
            logger.error(f"Error generating admin dashboard: {e}")
            return {'error': str(e)}

    def manage_user_sessions(self, action: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Manage user sessions (kick, ban, etc.)"""
        if action == "list":
            return {
                'active_sessions': len(self.dashboard.active_user_sessions),
                'sessions': list(self.dashboard.active_user_sessions.keys())[:10]  # First 10 for security
            }

        elif action == "terminate":
            if user_id and user_id in self.dashboard.active_user_sessions:
                del self.dashboard.active_user_sessions[user_id]
                return {'status': 'success', 'message': f'Session terminated for user {user_id}'}
            else:
                return {'status': 'error', 'message': 'User session not found'}

        else:
            return {'status': 'error', 'message': f'Unknown action: {action}'}

    def update_system_configuration(self, setting_key: str, setting_value: Any,
                                   modified_by: str) -> Dict[str, Any]:
        """Update system configuration"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current setting
        cursor.execute('SELECT * FROM system_config WHERE setting_key = ?', (setting_key,))
        current_setting = cursor.fetchone()

        if not current_setting:
            return {'status': 'error', 'message': f'Setting {setting_key} not found'}

        # Update setting
        cursor.execute('''
            UPDATE system_config
            SET setting_value = ?, last_modified = ?, modified_by = ?
            WHERE setting_key = ?
        ''', (str(setting_value), datetime.now(), modified_by, setting_key))

        conn.commit()
        conn.close()

        # Update dashboard config
        self.dashboard.config[setting_key] = setting_value

        # Log action
        self.log_admin_action(
            user_id=modified_by,
            action="update_configuration",
            resource_type="system_config",
            resource_id=setting_key,
            details={"old_value": current_setting[1], "new_value": str(setting_value)}
        )

        return {'status': 'success', 'message': f'Configuration {setting_key} updated successfully'}

    def export_admin_report(self, report_type: str = 'comprehensive',
                           start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Export administrative reports"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if report_type == 'comprehensive':
                # Get all admin data
                admin_dashboard = self.generate_admin_dashboard()
                user_data = self.get_user_management_data()
                content_data = self.get_content_management_data()
                system_data = self.get_system_health_data()

                report = {
                    'report_type': 'admin_comprehensive',
                    'generated_at': datetime.now().isoformat(),
                    'date_range': {
                        'start': start_date,
                        'end': end_date
                    },
                    'user_management': user_data,
                    'content_management': content_data,
                    'system_health': system_data,
                    'admin_dashboard': admin_dashboard
                }

                # Save as JSON
                report_file = self.data_path / f"admin_report_{timestamp}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

                return {
                    'status': 'success',
                    'report_file': str(report_file),
                    'summary': {
                        'total_users': user_data.get('total_users', 0),
                        'total_activities': user_data.get('total_activities', 0),
                        'active_alerts': len(system_data.get('active_alerts', []))
                    }
                }

            elif report_type == 'audit_log':
                # Export audit log
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                query = '''
                    SELECT timestamp, user_id, action, resource_type, resource_id, details, ip_address
                    FROM audit_log
                '''
                params = []

                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)

                query += ' ORDER BY timestamp DESC'
                cursor.execute(query, params)

                audit_data = cursor.fetchall()
                conn.close()

                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(audit_data, columns=[
                    'timestamp', 'user_id', 'action', 'resource_type', 'resource_id', 'details', 'ip_address'
                ])

                report_file = self.data_path / f"audit_log_{timestamp}.csv"
                df.to_csv(report_file, index=False)

                return {
                    'status': 'success',
                    'report_file': str(report_file),
                    'total_entries': len(audit_data)
                }

            else:
                return {'status': 'error', 'message': f'Unknown report type: {report_type}'}

        except Exception as e:
            logger.error(f"Error exporting admin report: {e}")
            return {'status': 'error', 'message': str(e)}

    def manage_performance_alerts(self, action: str, alert_id: int = None,
                                user_id: str = None) -> Dict[str, Any]:
        """Manage performance alerts"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if action == "list":
            cursor.execute('''
                SELECT * FROM performance_alerts
                ORDER BY created_at DESC
                LIMIT 20
            ''')

            alerts = cursor.fetchall()
            conn.close()

            return {
                'alerts': [
                    {
                        'id': row[0],
                        'alert_type': row[1],
                        'threshold_value': row[2],
                        'current_value': row[3],
                        'severity': row[4],
                        'status': row[5],
                        'created_at': row[6]
                    }
                    for row in alerts
                ]
            }

        elif action == "acknowledge" and alert_id and user_id:
            cursor.execute('''
                UPDATE performance_alerts
                SET status = 'acknowledged', acknowledged_at = ?, acknowledged_by = ?
                WHERE id = ? AND status = 'active'
            ''', (datetime.now(), user_id, alert_id))

            conn.commit()
            conn.close()

            # Log action
            self.log_admin_action(
                user_id=user_id,
                action="acknowledge_alert",
                resource_type="performance_alert",
                resource_id=str(alert_id)
            )

            return {'status': 'success', 'message': f'Alert {alert_id} acknowledged'}

        elif action == "resolve" and alert_id and user_id:
            cursor.execute('''
                UPDATE performance_alerts
                SET status = 'resolved', resolved_at = ?, resolved_by = ?
                WHERE id = ? AND status IN ('active', 'acknowledged')
            ''', (datetime.now(), user_id, alert_id))

            conn.commit()
            conn.close()

            # Log action
            self.log_admin_action(
                user_id=user_id,
                action="resolve_alert",
                resource_type="performance_alert",
                resource_id=str(alert_id)
            )

            return {'status': 'success', 'message': f'Alert {alert_id} resolved'}

        else:
            return {'status': 'error', 'message': 'Invalid action or missing parameters'}

# FastAPI Admin Panel
admin_app = FastAPI(title="AI Documentation Admin Panel", version="1.0.0")

# Add CORS middleware
admin_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize admin panel
dashboard = AnalyticsDashboard()
admin_panel = AdminPanel(dashboard)

# Pydantic models for admin API
class AdminLogin(BaseModel):
    username: str
    password: str

class ConfigurationUpdate(BaseModel):
    setting_key: str
    setting_value: Any

class AlertAction(BaseModel):
    action: str
    alert_id: Optional[int] = None
    user_id: Optional[str] = None

# Session management
active_admin_sessions = {}

def get_current_admin(session_token: str) -> Optional[AdminUser]:
    """Get current admin user from session token"""
    if session_token in active_admin_sessions:
        user_id = active_admin_sessions[session_token]['user_id']
        return admin_panel.admin_users.get(user_id)
    return None

@admin_app.get("/", response_class=HTMLResponse)
async def admin_home():
    """Admin panel home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Documentation Admin Panel</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .login-form { max-width: 400px; margin: 50px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .form-group { margin-bottom: 15px; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #2980b9; }
            .dashboard { display: none; }
            .metric { background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 10px 0; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
            .alert.warning { background: #fff3cd; border: 1px solid #ffeaa7; }
            .alert.error { background: #f8d7da; border: 1px solid #f5c6cb; }
            .alert.success { background: #d4edda; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>AI Documentation Admin Panel</h1>
                <p>Comprehensive administrative management and analytics</p>
            </div>

            <div id="login-form" class="login-form">
                <h2>Administrator Login</h2>
                <form id="loginForm">
                    <div class="form-group">
                        <label for="username">Username:</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password:</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn">Login</button>
                </form>
                <div id="login-message"></div>
            </div>

            <div id="dashboard" class="dashboard">
                <h2>Admin Dashboard</h2>
                <button onclick="logout()" class="btn" style="float: right;">Logout</button>

                <div id="dashboard-content">
                    <p>Loading dashboard...</p>
                </div>
            </div>
        </div>

        <script>
            let sessionToken = localStorage.getItem('adminSessionToken');

            // Check if already logged in
            if (sessionToken) {
                showDashboard();
            }

            document.getElementById('loginForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const formData = new FormData(e.target);
                const loginData = {
                    username: formData.get('username'),
                    password: formData.get('password')
                };

                try {
                    const response = await fetch('/api/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(loginData)
                    });

                    const result = await response.json();

                    if (result.status === 'success') {
                        sessionToken = result.session_token;
                        localStorage.setItem('adminSessionToken', sessionToken);
                        showDashboard();
                    } else {
                        document.getElementById('login-message').innerHTML =
                            `<div class="alert error">${result.message}</div>`;
                    }
                } catch (error) {
                    document.getElementById('login-message').innerHTML =
                        `<div class="alert error">Login failed: ${error.message}</div>`;
                }
            });

            async function showDashboard() {
                document.getElementById('login-form').style.display = 'none';
                document.getElementById('dashboard').style.display = 'block';

                try {
                    const response = await fetch('/api/dashboard', {
                        headers: { 'Authorization': `Bearer ${sessionToken}` }
                    });

                    const data = await response.json();

                    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">';

                    // User metrics
                    if (data.user_management) {
                        html += '<div class="metric">';
                        html += `<h3>User Management</h3>`;
                        html += `<p>Total Users: <span class="metric-value">${data.user_management.total_users || 0}</span></p>`;
                        html += `<p>Total Activities: <span class="metric-value">${data.user_management.total_activities || 0}</span></p>`;
                        html += `<p>Avg Session Duration: ${Math.round(data.user_management.avg_session_duration || 0)}s</p>`;
                        html += '</div>';
                    }

                    // System health
                    if (data.system_health) {
                        html += '<div class="metric">';
                        html += `<h3>System Health</h3>`;
                        html += `<p>Active Users: <span class="metric-value">${data.system_health.current_metrics.active_users || 0}</span></p>`;
                        html += `<p>CPU Usage: ${Math.round(data.system_health.current_metrics.cpu_usage || 0)}%</p>`;
                        html += `<p>Memory Usage: ${Math.round(data.system_health.current_metrics.memory_usage || 0)}MB</p>`;
                        html += `<p>Error Rate: ${Math.round(data.system_health.current_metrics.error_rate || 0)}%</p>`;
                        html += '</div>';
                    }

                    // Content management
                    if (data.content_management) {
                        html += '<div class="metric">';
                        html += `<h3>Content Management</h3>`;
                        html += `<p>Total Sections: <span class="metric-value">${data.content_management.content_performance.length || 0}</span></p>`;
                        const lowPerforming = data.content_management.content_gaps.length || 0;
                        html += `<p>Low Performing Sections: ${lowPerforming}</p>`;
                        html += '</div>';
                    }

                    html += '</div>';

                    // Recent alerts
                    if (data.system_health && data.system_health.active_alerts.length > 0) {
                        html += '<h3>Active Alerts</h3>';
                        data.system_health.active_alerts.slice(0, 5).forEach(alert => {
                            html += `<div class="alert ${alert.severity}">${alert.alert_type}: ${alert.current_value} (threshold: ${alert.threshold_value})</div>`;
                        });
                    }

                    // Recent insights
                    if (data.insights && data.insights.high_priority_insights) {
                        html += '<h3>High Priority Insights</h3>';
                        data.insights.high_priority_insights.slice(0, 3).forEach(insight => {
                            html += `<div class="alert warning">${insight.insight_type}: ${insight.prediction}</div>`;
                        });
                    }

                    document.getElementById('dashboard-content').innerHTML = html;

                } catch (error) {
                    document.getElementById('dashboard-content').innerHTML =
                        `<div class="alert error">Error loading dashboard: ${error.message}</div>`;
                }
            }

            function logout() {
                localStorage.removeItem('adminSessionToken');
                document.getElementById('dashboard').style.display = 'none';
                document.getElementById('login-form').style.display = 'block';
                document.getElementById('login-message').innerHTML = '';
            }

            // Auto-refresh dashboard every 30 seconds
            setInterval(() => {
                if (sessionToken) {
                    showDashboard();
                }
            }, 30000);
        </script>
    </body>
    </html>
    """

# Admin API endpoints
@admin_app.post("/api/login")
async def admin_login(login_data: AdminLogin):
    """Authenticate admin user"""
    try:
        user = admin_panel.authenticate_admin(login_data.username, login_data.password)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Create session
        session_token = f"admin_{user.user_id}_{int(datetime.now().timestamp())}"
        active_admin_sessions[session_token] = {
            'user_id': user.user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }

        # Log login
        admin_panel.log_admin_action(
            user_id=user.user_id,
            action="login",
            resource_type="admin_session",
            resource_id=session_token
        )

        return {
            'status': 'success',
            'session_token': session_token,
            'user': {
                'user_id': user.user_id,
                'username': user.username,
                'role': user.role.value,
                'permissions': [p.value for p in user.permissions]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.get("/api/dashboard")
async def get_admin_dashboard(authorization: str = None):
    """Get admin dashboard data"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.ANALYTICS_VIEW not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return admin_panel.generate_admin_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.get("/api/users")
async def get_user_management_data(authorization: str = None):
    """Get user management data"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.USER_MANAGEMENT not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return admin_panel.get_user_management_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.get("/api/content")
async def get_content_management_data(authorization: str = None):
    """Get content management data"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.CONTENT_MANAGEMENT not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return admin_panel.get_content_management_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.get("/api/system")
async def get_system_health_data(authorization: str = None):
    """Get system health data"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.SYSTEM_ADMIN not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return admin_panel.get_system_health_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.post("/api/sessions/{action}")
async def manage_user_sessions_endpoint(action: str, authorization: str = None):
    """Manage user sessions"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.USER_MANAGEMENT not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        result = admin_panel.manage_user_sessions(action)

        # Log action
        admin_panel.log_admin_action(
            user_id=admin_user.user_id,
            action=f"session_{action}",
            resource_type="user_session",
            resource_id=action
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.post("/api/configuration")
async def update_configuration(config_data: ConfigurationUpdate, authorization: str = None):
    """Update system configuration"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.SYSTEM_ADMIN not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return admin_panel.update_system_configuration(
            config_data.setting_key,
            config_data.setting_value,
            admin_user.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.get("/api/report/{report_type}")
async def export_admin_report_endpoint(report_type: str, start_date: str = None,
                                     end_date: str = None, authorization: str = None):
    """Export admin reports"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.REPORT_EXPORT not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return admin_panel.export_admin_report(report_type, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.post("/api/alerts")
async def manage_alerts(alert_action: AlertAction, authorization: str = None):
    """Manage performance alerts"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]
    admin_user = get_current_admin(session_token)

    if not admin_user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    if Permission.ALERT_MANAGEMENT not in admin_user.permissions:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        return admin_panel.manage_performance_alerts(
            alert_action.action,
            alert_action.alert_id,
            admin_user.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_app.post("/api/logout")
async def admin_logout(authorization: str = None):
    """Logout admin user"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    session_token = authorization.split(' ')[1]

    if session_token in active_admin_sessions:
        session_data = active_admin_sessions[session_token]
        user_id = session_data['user_id']

        # Log logout
        admin_panel.log_admin_action(
            user_id=user_id,
            action="logout",
            resource_type="admin_session",
            resource_id=session_token
        )

        del active_admin_sessions[session_token]

        return {'status': 'success', 'message': 'Logged out successfully'}
    else:
        return {'status': 'error', 'message': 'No active session'}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Documentation Admin Panel")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Starting Admin Panel on {args.host}:{args.port}")
    print(f"Admin Panel URL: http://{args.host}:{args.port}")
    print(f"Default login: admin / admin123")

    import uvicorn
    uvicorn.run(
        "admin_panel:admin_app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="info"
    )