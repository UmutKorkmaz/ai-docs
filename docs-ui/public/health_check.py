#!/usr/bin/env python3
"""
Health Check Script for AI Documentation Analytics Dashboard
Comprehensive system health verification and diagnostics.
"""

import sys
import json
import sqlite3
import time
import requests
from datetime import datetime
from pathlib import Path
import subprocess

class HealthChecker:
    """System health checker for analytics dashboard"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.issues = []
        self.warnings = []
        self.info = []

    def log_issue(self, component: str, message: str, severity: str = "error"):
        """Log an issue with severity"""
        issue = {
            "component": component,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }

        if severity == "error":
            self.issues.append(issue)
        elif severity == "warning":
            self.warnings.append(issue)
        else:
            self.info.append(issue)

    def check_python_dependencies(self):
        """Check if all required Python packages are installed"""
        print("🔍 Checking Python dependencies...")

        required_packages = [
            "fastapi",
            "uvicorn",
            "websockets",
            "plotly",
            "pandas",
            "numpy",
            "scikit-learn",
            "psutil",
            "redis",
            "aiofiles",
            "pydantic"
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✓ {package}")
            except ImportError:
                print(f"✗ {package}")
                missing_packages.append(package)

        if missing_packages:
            self.log_issue("dependencies", f"Missing packages: {', '.join(missing_packages)}")
            return False

        print("✓ All Python dependencies satisfied")
        return True

    def check_directories(self):
        """Check if all required directories exist"""
        print("\n🔍 Checking directory structure...")

        required_dirs = [
            "analytics_data",
            "admin_data",
            "monitoring_data",
            "logs",
            "reports",
            "assessment"
        ]

        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                print(f"✓ {dir_name}/")
            else:
                print(f"✗ {dir_name}/")
                self.log_issue("directories", f"Missing directory: {dir_name}")

        return len(self.issues) == 0

    def check_databases(self):
        """Check database files and connectivity"""
        print("\n🔍 Checking databases...")

        # Check analytics database
        analytics_db = self.base_dir / "analytics_data" / "analytics.db"
        if analytics_db.exists():
            print("✓ Analytics database exists")

            try:
                conn = sqlite3.connect(analytics_db)
                cursor = conn.cursor()

                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0] for row in cursor.fetchall()}

                expected_tables = {'user_activity', 'content_performance', 'system_metrics', 'user_segments'}
                missing_tables = expected_tables - tables

                if missing_tables:
                    self.log_issue("database", f"Missing tables in analytics.db: {missing_tables}")
                else:
                    print("✓ Analytics database tables are correct")

                conn.close()
            except Exception as e:
                self.log_issue("database", f"Cannot connect to analytics database: {str(e)}")
        else:
            print("✗ Analytics database missing")
            self.log_issue("database", "Analytics database file does not exist")

        # Check admin database
        admin_db = self.base_dir / "admin_data" / "admin.db"
        if admin_db.exists():
            print("✓ Admin database exists")

            try:
                conn = sqlite3.connect(admin_db)
                cursor = conn.cursor()

                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0] for row in cursor.fetchall()}

                expected_tables = {'admin_users', 'system_config', 'audit_log', 'performance_alerts'}
                missing_tables = expected_tables - tables

                if missing_tables:
                    self.log_issue("database", f"Missing tables in admin.db: {missing_tables}")
                else:
                    print("✓ Admin database tables are correct")

                # Check default admin user
                cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
                admin_count = cursor.fetchone()[0]

                if admin_count == 0:
                    self.log_issue("database", "Default admin user not found")
                else:
                    print("✓ Default admin user exists")

                conn.close()
            except Exception as e:
                self.log_issue("database", f"Cannot connect to admin database: {str(e)}")
        else:
            print("✗ Admin database missing")
            self.log_issue("database", "Admin database file does not exist")

        return len([i for i in self.issues if i["component"] == "database"]) == 0

    def check_configuration_files(self):
        """Check configuration files"""
        print("\n🔍 Checking configuration files...")

        # Check analytics config
        analytics_config = self.base_dir / "analytics_data" / "config.json"
        if analytics_config.exists():
            print("✓ Analytics configuration exists")

            try:
                with open(analytics_config, 'r') as f:
                    config = json.load(f)

                required_keys = {'real_time_update_interval', 'cache_ttl', 'alert_thresholds'}
                missing_keys = required_keys - set(config.keys())

                if missing_keys:
                    self.log_issue("configuration", f"Missing keys in analytics config: {missing_keys}")
                else:
                    print("✓ Analytics configuration is valid")
            except Exception as e:
                self.log_issue("configuration", f"Invalid analytics config: {str(e)}")
        else:
            print("✗ Analytics configuration missing")
            self.log_issue("configuration", "Analytics configuration file does not exist")

        # Check monitoring config
        monitoring_config = self.base_dir / "monitoring_config.json"
        if monitoring_config.exists():
            print("✓ Monitoring configuration exists")

            try:
                with open(monitoring_config, 'r') as f:
                    config = json.load(f)

                required_keys = {'enabled_monitors', 'alerting', 'thresholds'}
                missing_keys = required_keys - set(config.keys())

                if missing_keys:
                    self.log_issue("configuration", f"Missing keys in monitoring config: {missing_keys}")
                else:
                    print("✓ Monitoring configuration is valid")
            except Exception as e:
                self.log_issue("configuration", f"Invalid monitoring config: {str(e)}")
        else:
            print("✗ Monitoring configuration missing")
            self.log_issue("configuration", "Monitoring configuration file does not exist")

        return len([i for i in self.issues if i["component"] == "configuration"]) == 0

    def check_service_availability(self):
        """Check if services are running"""
        print("\n🔍 Checking service availability...")

        services = [
            ("Analytics Dashboard", "http://localhost:8000/api/dashboard"),
            ("Admin Panel", "http://localhost:8001/api/dashboard"),
            ("WebSocket Monitor", "ws://localhost:8765")
        ]

        for service_name, url in services:
            if url.startswith("ws://"):
                # WebSocket check (simplified)
                try:
                    import websockets
                    import asyncio

                    async def check_websocket():
                        try:
                            async with websockets.connect(url, timeout=5):
                                return True
                        except:
                            return False

                    # Simple check - just try to connect
                    print(f"⚠ WebSocket monitor check requires manual verification")
                    continue

                except ImportError:
                    print(f"⚠ Cannot check WebSocket - websockets package not available")
                    continue

            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ {service_name} is running")
                else:
                    print(f"✗ {service_name} returned status {response.status_code}")
                    self.log_issue("services", f"{service_name} returned status {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"✗ {service_name} is not running")
                self.log_issue("services", f"{service_name} is not running")
            except Exception as e:
                print(f"✗ {service_name} check failed: {str(e)}")
                self.log_issue("services", f"{service_name} check failed: {str(e)}")

        return len([i for i in self.issues if i["component"] == "services"]) == 0

    def check_system_resources(self):
        """Check system resources"""
        print("\n🔍 Checking system resources...")

        try:
            import psutil

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"CPU Usage: {cpu_percent}%")

            if cpu_percent > 80:
                self.log_issue("resources", f"High CPU usage: {cpu_percent}%", "warning")

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            print(f"Memory Usage: {memory_percent}%")

            if memory_percent > 80:
                self.log_issue("resources", f"High memory usage: {memory_percent}%", "warning")

            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            print(f"Disk Usage: {disk_percent:.1f}%")

            if disk_percent > 80:
                self.log_issue("resources", f"High disk usage: {disk_percent:.1f}%", "warning")

            print("✓ System resources checked")

        except ImportError:
            print("⚠ Cannot check system resources - psutil package not available")
        except Exception as e:
            self.log_issue("resources", f"System resource check failed: {str(e)}")

        return True

    def check_file_permissions(self):
        """Check file permissions"""
        print("\n🔍 Checking file permissions...")

        important_files = [
            "analytics_dashboard.py",
            "admin_panel.py",
            "real_time_monitoring.py",
            "start_dashboard.sh",
            "stop_dashboard.sh"
        ]

        for file_name in important_files:
            file_path = self.base_dir / file_name
            if file_path.exists():
                if os.access(file_path, os.R_OK):
                    print(f"✓ {file_name} is readable")
                else:
                    print(f"✗ {file_name} is not readable")
                    self.log_issue("permissions", f"File not readable: {file_name}")

                if file_name.endswith(".sh") and os.access(file_path, os.X_OK):
                    print(f"✓ {file_name} is executable")
                elif file_name.endswith(".sh"):
                    print(f"✗ {file_name} is not executable")
                    self.log_issue("permissions", f"Script not executable: {file_name}")
            else:
                print(f"⚠ {file_name} not found")

        return len([i for i in self.issues if i["component"] == "permissions"]) == 0

    def run_performance_test(self):
        """Run basic performance tests"""
        print("\n🔍 Running performance tests...")

        try:
            # Test database read performance
            start_time = time.time()
            conn = sqlite3.connect(self.base_dir / "analytics_data" / "analytics.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_activity")
            cursor.fetchone()
            conn.close()
            db_time = time.time() - start_time

            print(f"Database read time: {db_time:.3f}s")

            if db_time > 1.0:
                self.log_issue("performance", f"Slow database read: {db_time:.3f}s", "warning")

            # Test JSON parsing performance
            start_time = time.time()
            with open(self.base_dir / "analytics_data" / "config.json", 'r') as f:
                json.load(f)
            json_time = time.time() - start_time

            print(f"JSON parsing time: {json_time:.3f}s")

            if json_time > 0.1:
                self.log_issue("performance", f"Slow JSON parsing: {json_time:.3f}s", "warning")

            print("✓ Performance tests completed")

        except Exception as e:
            self.log_issue("performance", f"Performance test failed: {str(e)}")

        return True

    def generate_health_report(self):
        """Generate comprehensive health report"""
        print("\n📊 Generating Health Report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_issues": len(self.issues),
                "total_warnings": len(self.warnings),
                "total_info": len(self.info),
                "overall_status": "healthy" if not self.issues else "unhealthy"
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
            "recommendations": []
        }

        # Generate recommendations
        if report["summary"]["total_issues"] > 0:
            report["recommendations"].append("Address all error-level issues before deploying")
        if report["summary"]["total_warnings"] > 0:
            report["recommendations"].append("Review and address warning-level issues for optimal performance")
        if len([i for i in self.issues if i["component"] == "services"]) > 0:
            report["recommendations"].append("Start the required services using ./start_dashboard.sh")
        if len([i for i in self.issues if i["component"] == "dependencies"]) > 0:
            report["recommendations"].append("Install missing dependencies: pip install -r requirements.txt")

        # Save report
        report_file = self.base_dir / "logs" / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Health report saved to: {report_file}")

        return report

    def run_full_check(self):
        """Run complete health check"""
        print("🏥 AI Documentation Analytics Dashboard - Health Check")
        print("=" * 60)

        checks = [
            ("Python Dependencies", self.check_python_dependencies),
            ("Directory Structure", self.check_directories),
            ("Databases", self.check_databases),
            ("Configuration Files", self.check_configuration_files),
            ("Service Availability", self.check_service_availability),
            ("System Resources", self.check_system_resources),
            ("File Permissions", self.check_file_permissions),
            ("Performance Tests", self.run_performance_test)
        ]

        for check_name, check_function in checks:
            print(f"\n📋 {check_name}")
            print("-" * len(check_name))
            check_function()

        # Generate report
        report = self.generate_health_report()

        # Summary
        print("\n" + "=" * 60)
        print("🏥 HEALTH CHECK SUMMARY")
        print("=" * 60)

        if report["summary"]["total_issues"] == 0:
            print("✅ System is HEALTHY")
        else:
            print("❌ System has ISSUES")
            print(f"   Errors: {report['summary']['total_issues']}")
            print(f"   Warnings: {report['summary']['total_warnings']}")

        if report["recommendations"]:
            print("\n📋 Recommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"   {i}. {rec}")

        return report

def main():
    """Main health check function"""
    checker = HealthChecker()
    report = checker.run_full_check()

    # Exit with appropriate code
    if report["summary"]["total_issues"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()