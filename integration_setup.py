#!/usr/bin/env python3
"""
Integration Setup for AI Documentation Analytics Dashboard
Complete setup and configuration guide for the analytics system.
"""

import os
import sys
import json
import subprocess
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

def setup_project_structure():
    """Set up the complete project structure"""
    base_dir = Path.cwd()

    # Create necessary directories
    directories = [
        "analytics_data",
        "admin_data",
        "monitoring_data",
        "logs",
        "reports",
        "static",
        "templates"
    ]

    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

    return base_dir

def check_system_requirements():
    """Check and install system requirements"""
    requirements = [
        "python>=3.8",
        "pip",
        "sqlite3"
    ]

    print("Checking system requirements...")

    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"✗ Python {python_version.major}.{python_version.minor}.{python_version.micro} (need >=3.8)")
        return False

    # Check pip
    try:
        import pip
        print(f"✓ pip {pip.__version__}")
    except ImportError:
        print("✗ pip not found")
        return False

    # Check SQLite
    try:
        import sqlite3
        print("✓ SQLite3 available")
    except ImportError:
        print("✗ SQLite3 not found")
        return False

    return True

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "websockets==12.0",
        "plotly==5.17.0",
        "pandas==2.1.4",
        "numpy==1.25.2",
        "scikit-learn==1.3.2",
        "psutil==5.9.6",
        "redis==5.0.1",
        "aiofiles==23.2.1",
        "pydantic==2.5.0"
    ]

    print("\nInstalling Python packages...")

    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✓ {package}")
            else:
                print(f"✗ {package}: {result.stderr}")

        except Exception as e:
            print(f"✗ {package}: {str(e)}")

def setup_databases():
    """Set up and initialize databases"""
    print("\nSetting up databases...")

    base_dir = Path.cwd()

    # Initialize analytics database
    analytics_db = base_dir / "analytics_data" / "analytics.db"
    if not analytics_db.exists():
        print("Creating analytics database...")

        conn = sqlite3.connect(analytics_db)
        cursor = conn.cursor()

        # Create tables
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
        print("✓ Analytics database created")

    # Initialize admin database
    admin_db = base_dir / "admin_data" / "admin.db"
    if not admin_db.exists():
        print("Creating admin database...")

        conn = sqlite3.connect(admin_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                permissions TEXT,
                created_at DATETIME,
                last_login DATETIME,
                is_active BOOLEAN DEFAULT 1,
                password_hash TEXT
            )
        ''')

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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                user_id TEXT,
                action TEXT,
                resource_type TEXT,
                resource_id TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')

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

        # Create default admin user
        import hashlib
        default_password = hashlib.sha256("admin123".encode()).hexdigest()

        cursor.execute('''
            INSERT OR IGNORE INTO admin_users (user_id, username, email, role, permissions, created_at, is_active, password_hash)
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
        print("✓ Admin database created")

def create_configuration_files():
    """Create configuration files"""
    print("\nCreating configuration files...")

    base_dir = Path.cwd()

    # Analytics dashboard configuration
    analytics_config = {
        "real_time_update_interval": 5,
        "cache_ttl": 3600,
        "anomaly_detection_threshold": 0.1,
        "predictive_horizon_days": 30,
        "dashboard_refresh_rate": 30,
        "retention_days": 365,
        "enable_ml_predictions": True,
        "alert_thresholds": {
            "high_error_rate": 5.0,
            "low_engagement": 30.0,
            "high_response_time": 1000,
            "low_completion_rate": 40.0
        }
    }

    with open(base_dir / "analytics_data" / "config.json", "w") as f:
        json.dump(analytics_config, f, indent=2)
    print("✓ Analytics configuration created")

    # Monitoring configuration
    monitoring_config = {
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
        "thresholds": {
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

    with open(base_dir / "monitoring_config.json", "w") as f:
        json.dump(monitoring_config, f, indent=2)
    print("✓ Monitoring configuration created")

def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data...")

    base_dir = Path.cwd()

    # Sample user activity
    sample_activities = [
        {
            "user_id": "user_001",
            "session_id": "session_001",
            "activity_type": "page_view",
            "section_id": "01_Foundational_Machine_Learning",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": 180,
            "metadata": {"page": "introduction"}
        },
        {
            "user_id": "user_002",
            "session_id": "session_002",
            "activity_type": "assessment_start",
            "section_id": "02_Advanced_Deep_Learning",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": 0,
            "metadata": {"assessment_type": "quiz"}
        },
        {
            "user_id": "user_001",
            "session_id": "session_001",
            "activity_type": "content_complete",
            "section_id": "01_Foundational_Machine_Learning",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": 600,
            "metadata": {"completion_score": 85}
        }
    ]

    # Insert sample data
    conn = sqlite3.connect(base_dir / "analytics_data" / "analytics.db")
    cursor = conn.cursor()

    for activity in sample_activities:
        cursor.execute('''
            INSERT INTO user_activity
            (user_id, session_id, activity_type, section_id, timestamp, duration_seconds, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            activity["user_id"],
            activity["session_id"],
            activity["activity_type"],
            activity["section_id"],
            activity["timestamp"],
            activity["duration_seconds"],
            json.dumps(activity["metadata"])
        ))

    # Sample content performance
    sample_content = [
        ("01_Foundational_Machine_Learning", 150, 25, 420.5, 0.82),
        ("02_Advanced_Deep_Learning", 120, 18, 380.2, 0.75),
        ("03_Natural_Language_Processing", 95, 15, 350.8, 0.68),
        ("04_Computer_Vision", 85, 12, 320.1, 0.72),
        ("05_Generative_AI", 110, 20, 400.3, 0.78)
    ]

    for content in sample_content:
        cursor.execute('''
            INSERT OR IGNORE INTO content_performance
            (section_id, view_count, unique_users, avg_time_spent, completion_rate, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (*content, datetime.now()))

    conn.commit()
    conn.close()
    print("✓ Sample data created")

def create_startup_scripts():
    """Create startup scripts"""
    print("\nCreating startup scripts...")

    base_dir = Path.cwd()

    # Main dashboard startup script
    dashboard_script = f'''#!/bin/bash
# AI Documentation Analytics Dashboard Startup Script

echo "Starting AI Documentation Analytics Dashboard..."

# Check if running
if pgrep -f "analytics_dashboard.py" > /dev/null; then
    echo "Dashboard is already running"
    exit 1
fi

# Start analytics dashboard
cd {base_dir}
python analytics_dashboard.py --host 0.0.0.0 --port 8000 &
DASHBOARD_PID=$!

# Start admin panel
python admin_panel.py --host 0.0.0.0 --port 8001 &
ADMIN_PID=$!

# Start monitoring system
python real_time_monitoring.py &
MONITOR_PID=$!

# Save PIDs
echo $DASHBOARD_PID > dashboard.pid
echo $ADMIN_PID > admin.pid
echo $MONITOR_PID > monitor.pid

echo "Dashboard started successfully!"
echo "Main Dashboard: http://localhost:8000"
echo "Admin Panel: http://localhost:8001"
echo "WebSocket Monitor: ws://localhost:8765"
echo ""
echo "To stop: ./stop_dashboard.sh"
'''

    with open(base_dir / "start_dashboard.sh", "w") as f:
        f.write(dashboard_script)

    # Make executable
    os.chmod(base_dir / "start_dashboard.sh", 0o755)
    print("✓ Dashboard startup script created")

    # Stop script
    stop_script = '''#!/bin/bash
# Stop Dashboard Script

echo "Stopping AI Documentation Analytics Dashboard..."

# Kill processes
if [ -f dashboard.pid ]; then
    kill $(cat dashboard.pid) 2>/dev/null
    rm dashboard.pid
fi

if [ -f admin.pid ]; then
    kill $(cat admin.pid) 2>/dev/null
    rm admin.pid
fi

if [ -f monitor.pid ]; then
    kill $(cat monitor.pid) 2>/dev/null
    rm monitor.pid
fi

# Force kill any remaining processes
pkill -f "analytics_dashboard.py"
pkill -f "admin_panel.py"
pkill -f "real_time_monitoring.py"

echo "Dashboard stopped!"
'''

    with open(base_dir / "stop_dashboard.sh", "w") as f:
        f.write(stop_script)

    # Make executable
    os.chmod(base_dir / "stop_dashboard.sh", 0o755)
    print("✓ Stop script created")

def create_documentation():
    """Create project documentation"""
    print("\nCreating documentation...")

    base_dir = Path.cwd()

    # README file
    readme_content = '''# AI Documentation Analytics Dashboard

A comprehensive real-time analytics and monitoring system for AI documentation projects.

## Features

### Real-time Analytics
- Live user activity monitoring
- System performance tracking
- Content engagement analytics
- Interactive dashboards and visualizations

### Administrative Management
- User management and session control
- Content performance optimization
- System configuration management
- Audit logging and compliance

### Predictive Insights
- Machine learning-based predictions
- User churn detection
- Content gap analysis
- Performance trend forecasting

### Real-time Monitoring
- System health monitoring
- Automated alerting
- WebSocket-based live updates
- Performance anomaly detection

## Quick Start

1. Run the setup script:
   ```bash
   python integration_setup.py
   ```

2. Start the dashboard:
   ```bash
   ./start_dashboard.sh
   ```

3. Access the interfaces:
   - Main Dashboard: http://localhost:8000
   - Admin Panel: http://localhost:8001 (login: admin/admin123)
   - WebSocket Monitor: ws://localhost:8765

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main Dashboard │    │   Admin Panel   │    │  Real-time      │
│   (FastAPI)      │    │   (FastAPI)      │    │  Monitor        │
│   Port: 8000     │    │   Port: 8001     │    │  Port: 8765     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Analytics     │
                    │   Database      │
                    │   (SQLite)      │
                    └─────────────────┘
```

## Configuration

### Analytics Dashboard
- File: `analytics_data/config.json`
- Controls data retention, alert thresholds, ML predictions

### Monitoring System
- File: `monitoring_config.json`
- Controls monitoring intervals, alert rules, notification settings

### Admin Panel
- Database: `admin_data/admin.db`
- Default login: admin / admin123

## API Endpoints

### Main Dashboard API
- `GET /api/dashboard` - Real-time dashboard data
- `POST /api/activity` - Log user activity
- `GET /api/content-analytics/{section_id}` - Content analytics
- `GET /api/insights` - Predictive insights
- `GET /api/visualizations` - Dashboard charts

### Admin Panel API
- `POST /api/login` - Admin authentication
- `GET /api/dashboard` - Admin dashboard
- `GET /api/users` - User management
- `GET /api/content` - Content management
- `GET /api/system` - System health
- `POST /api/configuration` - Update config

### WebSocket Events
- `monitoring_event` - Real-time system events
- `initial_data` - Initial connection data
- `alert` - Alert notifications

## Development

### Running in Development Mode
```bash
# Analytics Dashboard
python analytics_dashboard.py --debug

# Admin Panel
python admin_panel.py --debug

# Monitoring System
python real_time_monitoring.py
```

### Testing
```bash
# Run system tests
python -m pytest tests/

# Check system health
python health_check.py
```

## Monitoring and Alerting

The system includes comprehensive monitoring:

- **System Metrics**: CPU, memory, response time, error rates
- **User Activity**: Session tracking, engagement patterns
- **Content Performance**: View counts, completion rates, popularity
- **Predictive Analytics**: ML-based insights and recommendations

Alerts are configured through the monitoring system and can be sent via:
- Email notifications
- Webhook integrations
- Slack notifications
- WebSocket broadcasts

## Data Retention

- User activity: 90 days
- System metrics: 30 days
- Alert history: 365 days
- Analytics data: 365 days (configurable)

## Security

- Admin authentication with role-based access
- Audit logging for all administrative actions
- Secure session management
- Input validation and sanitization
- CORS protection for web APIs

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Check if services are already running: `ps aux | grep python`
   - Kill existing processes: `./stop_dashboard.sh`

2. **Database connection errors**
   - Ensure database files exist in data directories
   - Check file permissions
   - Verify SQLite installation

3. **Missing dependencies**
   - Run: `pip install -r requirements.txt`
   - Check Python version (>=3.8 required)

### Logs
- Application logs: `logs/`
- System logs: Check system journal
- Error logs: Individual service logs

## Support

For issues and questions:
- Check the troubleshooting section
- Review log files
- Verify configuration settings
- Test with sample data

## License

This project is part of the AI Documentation system.
'''

    with open(base_dir / "README_ANALYTICS.md", "w") as f:
        f.write(readme_content)
    print("✓ Documentation created")

def run_integration_tests():
    """Run basic integration tests"""
    print("\nRunning integration tests...")

    try:
        # Test imports
        import analytics_dashboard
        import admin_panel
        import real_time_monitoring
        print("✓ All modules import successfully")

        # Test database connections
        base_dir = Path.cwd()

        # Test analytics database
        conn = sqlite3.connect(base_dir / "analytics_data" / "analytics.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()

        expected_tables = {'user_activity', 'content_performance', 'system_metrics', 'user_segments'}
        found_tables = {table[0] for table in tables}

        if expected_tables.issubset(found_tables):
            print("✓ Analytics database tables created correctly")
        else:
            print(f"✗ Missing tables in analytics database: {expected_tables - found_tables}")

        # Test admin database
        conn = sqlite3.connect(base_dir / "admin_data" / "admin.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()

        expected_tables = {'admin_users', 'system_config', 'audit_log', 'performance_alerts'}
        found_tables = {table[0] for table in tables}

        if expected_tables.issubset(found_tables):
            print("✓ Admin database tables created correctly")
        else:
            print(f"✗ Missing tables in admin database: {expected_tables - found_tables}")

        # Test configuration files
        import json

        # Test analytics config
        with open(base_dir / "analytics_data" / "config.json", "r") as f:
            config = json.load(f)

        required_keys = {'real_time_update_interval', 'cache_ttl', 'alert_thresholds'}
        if required_keys.issubset(config.keys()):
            print("✓ Analytics configuration valid")
        else:
            print(f"✗ Missing keys in analytics config: {required_keys - config.keys()}")

        # Test monitoring config
        with open(base_dir / "monitoring_config.json", "r") as f:
            config = json.load(f)

        required_keys = {'enabled_monitors', 'alerting', 'thresholds'}
        if required_keys.issubset(config.keys()):
            print("✓ Monitoring configuration valid")
        else:
            print(f"✗ Missing keys in monitoring config: {required_keys - config.keys()}")

        print("✓ Integration tests completed successfully")

    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        return False

    return True

def main():
    """Main setup function"""
    print("🚀 AI Documentation Analytics Dashboard - Setup")
    print("=" * 50)

    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please install required dependencies.")
        return

    # Setup project structure
    base_dir = setup_project_structure()

    # Install Python packages
    install_python_packages()

    # Setup databases
    setup_databases()

    # Create configuration files
    create_configuration_files()

    # Create sample data
    create_sample_data()

    # Create startup scripts
    create_startup_scripts()

    # Create documentation
    create_documentation()

    # Run integration tests
    if run_integration_tests():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the dashboard: ./start_dashboard.sh")
        print("2. Access the main dashboard: http://localhost:8000")
        print("3. Access the admin panel: http://localhost:8001")
        print("4. Login to admin panel: admin / admin123")
        print("\nFor more information, see README_ANALYTICS.md")
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")

if __name__ == "__main__":
    main()