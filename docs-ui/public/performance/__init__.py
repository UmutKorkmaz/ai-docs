"""
AI Documentation Performance Optimization System

This package provides comprehensive performance optimization utilities for the AI documentation system,
including lazy loading, caching strategies, content compression, and monitoring capabilities.

Author: AI Documentation Team
Version: 1.0.0
"""

from .lazy_loader import DocumentationLazyLoader
from .cache_manager import DocumentationCacheManager
from .compression import ContentOptimizer
from .monitoring import PerformanceMonitor

__version__ = "1.0.0"
__all__ = [
    "DocumentationLazyLoader",
    "DocumentationCacheManager",
    "ContentOptimizer",
    "PerformanceMonitor"
]

# Performance configuration defaults
DEFAULT_CONFIG = {
    "cache_size_mb": 512,
    "lazy_load_enabled": True,
    "compression_enabled": True,
    "monitoring_enabled": True,
    "max_file_size_mb": 1,
    "preload_dependencies": True
}

# Performance metrics
METRICS = {
    "load_times": [],
    "cache_hit_rates": [],
    "compression_ratios": [],
    "user_navigation_patterns": []
}

# Export commonly used classes and functions
for cls in __all__:
    globals()[cls] = eval(cls)