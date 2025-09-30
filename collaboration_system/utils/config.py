#!/usr/bin/env python3
"""
Configuration management for the collaboration system

This module handles environment variables, configuration settings,
and provides a centralized configuration interface.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from decouple import config

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = config('DB_HOST', default='localhost')
    port: int = config('DB_PORT', default=5432, cast=int)
    database: str = config('DB_NAME', default='ai_docs_collaboration')
    username: str = config('DB_USER', default='ai_docs_user')
    password: str = config('DB_PASSWORD', default='ai_docs_password')
    ssl_mode: str = config('DB_SSL_MODE', default='prefer')

    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = config('REDIS_HOST', default='localhost')
    port: int = config('REDIS_PORT', default=6379, cast=int)
    password: Optional[str] = config('REDIS_PASSWORD', default=None)
    db: int = config('REDIS_DB', default=0, cast=int)
    ssl: bool = config('REDIS_SSL', default=False, cast=bool)

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class WebSocketConfig:
    """WebSocket server configuration"""
    host: str = config('WS_HOST', default='0.0.0.0')
    port: int = config('WS_PORT', default=8001, cast=int)
    max_connections: int = config('WS_MAX_CONNECTIONS', default=1000, cast=int)
    ping_interval: int = config('WS_PING_INTERVAL', default=30, cast=int)
    ping_timeout: int = config('WS_PING_TIMEOUT', default=10, cast=int)
    max_message_size: int = config('WS_MAX_MESSAGE_SIZE', default=10485760, cast=int)  # 10MB

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = config('SECRET_KEY', default='your-secret-key-change-in-production')
    algorithm: str = config('JWT_ALGORITHM', default='HS256')
    access_token_expire_minutes: int = config('ACCESS_TOKEN_EXPIRE_MINUTES', default=30, cast=int)
    refresh_token_expire_days: int = config('REFRESH_TOKEN_EXPIRE_DAYS', default=7, cast=int)
    password_min_length: int = config('PASSWORD_MIN_LENGTH', default=8, cast=int)
    max_login_attempts: int = config('MAX_LOGIN_ATTEMPTS', default=5, cast=int)
    session_timeout_minutes: int = config('SESSION_TIMEOUT_MINUTES', default=60, cast=int)

@dataclass
class APIConfig:
    """API configuration"""
    host: str = config('API_HOST', default='0.0.0.0')
    port: int = config('API_PORT', default=8000, cast=int)
    workers: int = config('API_WORKERS', default=4, cast=int)
    reload: bool = config('API_RELOAD', default=False, cast=bool)
    cors_origins: list = field(default_factory=lambda: config('CORS_ORIGINS', default='*', cast=lambda v: [s.strip() for s in v.split(',')]))
    rate_limit_requests: int = config('RATE_LIMIT_REQUESTS', default=100, cast=int)
    rate_limit_window: int = config('RATE_LIMIT_WINDOW', default=60, cast=int)

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = config('LOG_LEVEL', default='INFO')
    log_format: str = config('LOG_FORMAT', default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    enable_metrics: bool = config('ENABLE_METRICS', default=True, cast=bool)
    metrics_port: int = config('METRICS_PORT', default=9090, cast=int)
    enable_tracing: bool = config('ENABLE_TRACING', default=False, cast=bool)
    tracing_service_name: str = config('TRACING_SERVICE_NAME', default='ai-docs-collaboration')

@dataclass
class EmailConfig:
    """Email configuration"""
    smtp_host: str = config('SMTP_HOST', default='smtp.gmail.com')
    smtp_port: int = config('SMTP_PORT', default=587, cast=int)
    smtp_username: Optional[str] = config('SMTP_USERNAME', default=None)
    smtp_password: Optional[str] = config('SMTP_PASSWORD', default=None)
    from_email: str = config('FROM_EMAIL', default='noreply@ai-docs.com')
    use_tls: bool = config('SMTP_USE_TLS', default=True, cast=bool)

@dataclass
class StorageConfig:
    """Storage configuration for file uploads"""
    upload_dir: str = config('UPLOAD_DIR', default='uploads')
    max_file_size: int = config('MAX_FILE_SIZE', default=10485760, cast=int)  # 10MB
    allowed_extensions: list = field(default_factory=lambda: config('ALLOWED_EXTENSIONS', default='jpg,jpeg,png,gif,pdf,txt,md', cast=lambda v: [s.strip() for s in v.split(',')]))
    storage_backend: str = config('STORAGE_BACKEND', default='local')  # local, s3, gcs

@dataclass
class SearchConfig:
    """Search configuration"""
    enabled: bool = config('SEARCH_ENABLED', default=True, cast=bool)
    backend: str = config('SEARCH_BACKEND', default='elasticsearch')  # elasticsearch, whoosh
    host: str = config('SEARCH_HOST', default='localhost')
    port: int = config('SEARCH_PORT', default=9200, cast=int)
    index_prefix: str = config('SEARCH_INDEX_PREFIX', default='ai_docs')

@dataclass
class AIConfig:
    """AI/ML configuration for advanced features"""
    openai_api_key: Optional[str] = config('OPENAI_API_KEY', default=None)
    enable_content_suggestions: bool = config('ENABLE_CONTENT_SUGGESTIONS', default=False, cast=bool)
    enable_spam_detection: bool = config('ENABLE_SPAM_DETECTION', default=True, cast=bool)
    enable_quality_scoring: bool = config('ENABLE_QUALITY_SCORING', default=True, cast=bool)
    model_cache_dir: str = config('MODEL_CACHE_DIR', default='models/cache')

@dataclass
class CollaborationConfig:
    """Collaboration-specific configuration"""
    max_session_duration_hours: int = config('MAX_SESSION_DURATION_HOURS', default=24, cast=int)
    max_participants_per_session: int = config('MAX_PARTICIPANTS_PER_SESSION', default=50, cast=int)
    enable_version_history: bool = config('ENABLE_VERSION_HISTORY', default=True, cast=bool)
    max_edit_history_size: int = config('MAX_EDIT_HISTORY_SIZE', default=1000, cast=int)
    auto_save_interval_seconds: int = config('AUTO_SAVE_INTERVAL_SECONDS', default=30, cast=int)
    conflict_resolution_strategy: str = config('CONFLICT_RESOLUTION_STRATEGY', default='operational_transformation')

@dataclass
class Config:
    """Main configuration class"""
    # Environment
    environment: str = config('ENVIRONMENT', default='development')
    debug: bool = config('DEBUG', default=False, cast=bool)
    testing: bool = config('TESTING', default=False, cast=bool)

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    collaboration: CollaborationConfig = field(default_factory=CollaborationConfig)

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration values"""
        if self.environment == 'production' and self.debug:
            raise ValueError("Debug mode should not be enabled in production")

        if self.security.secret_key == 'your-secret-key-change-in-production' and self.environment == 'production':
            raise ValueError("Please change the default secret key in production")

        if self.database.password == 'ai_docs_password' and self.environment == 'production':
            raise ValueError("Please change the default database password in production")

    @property
    def is_development(self) -> bool:
        return self.environment == 'development'

    @property
    def is_production(self) -> bool:
        return self.environment == 'production'

    @property
    def is_testing(self) -> bool:
        return self.environment == 'testing'

    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url

    def get_redis_url(self) -> str:
        """Get Redis URL"""
        return self.redis.url

    def get_websocket_url(self) -> str:
        """Get WebSocket URL"""
        return f"ws://{self.websocket.host}:{self.websocket.port}"

    def get_api_url(self) -> str:
        """Get API URL"""
        return f"http://{self.api.host}:{self.api.port}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'testing': self.testing,
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'websocket': self.websocket.__dict__,
            'security': self.security.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__,
            'email': self.email.__dict__,
            'storage': self.storage.__dict__,
            'search': self.search.__dict__,
            'ai': self.ai.__dict__,
            'collaboration': self.collaboration.__dict__,
        }

# Global configuration instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config

def set_config(config_instance: Config):
    """Set the global configuration instance (for testing)"""
    global _config
    _config = config_instance

def load_config_from_file(config_file: str):
    """Load configuration from file"""
    import json
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    # Create config instance with loaded data
    config = Config()
    for key, value in config_data.items():
        if hasattr(config, key):
            setattr(config, key, value)

    set_config(config)
    return config

def get_environment_config() -> Dict[str, str]:
    """Get environment-specific configuration"""
    env = os.getenv('ENVIRONMENT', 'development')
    config_map = {
        'development': {
            'DEBUG': 'True',
            'LOG_LEVEL': 'DEBUG',
            'API_RELOAD': 'True',
        },
        'testing': {
            'DEBUG': 'True',
            'LOG_LEVEL': 'DEBUG',
            'TESTING': 'True',
        },
        'production': {
            'DEBUG': 'False',
            'LOG_LEVEL': 'INFO',
            'API_RELOAD': 'False',
        }
    }
    return config_map.get(env, config_map['development'])

def validate_required_env_vars():
    """Validate that required environment variables are set"""
    required_vars = [
        'SECRET_KEY',
        'DB_PASSWORD',
        'ENVIRONMENT',
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize configuration on import
try:
    validate_required_env_vars()
    config = get_config()
except EnvironmentError as e:
    print(f"Configuration error: {e}")
    # Use default configuration for development
    config = Config()
    set_config(config)