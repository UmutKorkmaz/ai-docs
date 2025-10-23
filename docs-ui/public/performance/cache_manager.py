"""
Documentation Cache Management System

Provides multi-level caching strategies for documentation content to improve
performance and reduce redundant loading operations.

Author: AI Documentation Team
Version: 1.0.0
"""

import os
import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
from functools import lru_cache
import tempfile
import shutil

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    content: str
    etag: str
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        """Calculate size after initialization."""
        self.size_bytes = len(self.content.encode('utf-8'))

class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: CacheEntry) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all values from cache."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get current cache size in bytes."""
        pass

class MemoryCache(CacheBackend):
    """In-memory cache backend."""

    def __init__(self, max_size_mb: int = 256):
        """
        Initialize memory cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from memory cache."""
        with self._lock:
            entry = self.cache.get(key)
            if entry:
                entry.access_count += 1
            return entry

    def set(self, key: str, value: CacheEntry) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Check if we need to evict
            if self.size() + value.size_bytes > self.max_size_bytes:
                self._evict_lru(value.size_bytes)

            self.cache[key] = value
            return True

    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> bool:
        """Clear all values from memory cache."""
        with self._lock:
            self.cache.clear()
            return True

    def size(self) -> int:
        """Get current cache size in bytes."""
        return sum(entry.size_bytes for entry in self.cache.values())

    def _evict_lru(self, required_space: int):
        """Evict least recently used items."""
        items = list(self.cache.items())
        items.sort(key=lambda x: x[1].access_count)

        freed_space = 0
        for key, entry in items:
            if freed_space >= required_space:
                break
            del self.cache[key]
            freed_space += entry.size_bytes

class DiskCache(CacheBackend):
    """Disk-based cache backend."""

    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 1024):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "ai_docs_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()

        # Initialize index
        self._index_file = self.cache_dir / "index.json"
        self.index: Dict[str, Dict] = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk."""
        if self._index_file.exists():
            with open(self._index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        with self._lock:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)

    def _get_cache_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from disk cache."""
        with self._lock:
            if key not in self.index:
                return None

            cache_file = self._get_cache_file_path(key)
            if not cache_file.exists():
                # Clean up index
                del self.index[key]
                self._save_index()
                return None

            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)

                # Update access count
                entry.access_count += 1
                self.index[key]["access_count"] = entry.access_count
                self.index[key]["last_accessed"] = time.time()
                self._save_index()

                return entry
            except Exception:
                # Remove corrupted cache entry
                self.delete(key)
                return None

    def set(self, key: str, value: CacheEntry) -> bool:
        """Set value in disk cache."""
        with self._lock:
            # Check if we need to evict
            if self.size() + value.size_bytes > self.max_size_bytes:
                self._evict_lru(value.size_bytes)

            cache_file = self._get_cache_file_path(key)

            try:
                # Save to disk
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)

                # Update index
                self.index[key] = {
                    "etag": value.etag,
                    "timestamp": value.timestamp,
                    "access_count": value.access_count,
                    "size_bytes": value.size_bytes,
                    "last_accessed": time.time()
                }
                self._save_index()

                return True
            except Exception:
                # Clean up on failure
                if cache_file.exists():
                    cache_file.unlink()
                return False

    def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        with self._lock:
            if key in self.index:
                cache_file = self._get_cache_file_path(key)
                if cache_file.exists():
                    cache_file.unlink()
                del self.index[key]
                self._save_index()
                return True
            return False

    def clear(self) -> bool:
        """Clear all values from disk cache."""
        with self._lock:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()

            # Clear index
            self.index.clear()
            self._save_index()

            return True

    def size(self) -> int:
        """Get current cache size in bytes."""
        return sum(entry.get("size_bytes", 0) for entry in self.index.values())

    def _evict_lru(self, required_space: int):
        """Evict least recently used items."""
        items = list(self.index.items())
        items.sort(key=lambda x: x[1].get("access_count", 0))

        freed_space = 0
        for key, entry in items:
            if freed_space >= required_space:
                break

            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                cache_file.unlink()

            del self.index[key]
            freed_space += entry.get("size_bytes", 0)

        self._save_index()

class DocumentationCacheManager:
    """
    Multi-level caching system for documentation content.

    Features:
    - Multi-level caching (memory + disk)
    - Intelligent cache eviction policies
    - Content validation with ETags
    - Performance monitoring
    - Configurable cache sizes
    """

    def __init__(
        self,
        memory_cache_mb: int = 256,
        disk_cache_mb: int = 1024,
        disk_cache_dir: Optional[str] = None,
        ttl_seconds: int = 3600
    ):
        """
        Initialize the cache manager.

        Args:
            memory_cache_mb: Memory cache size in MB
            disk_cache_mb: Disk cache size in MB
            disk_cache_dir: Directory for disk cache
            ttl_seconds: Time-to-live for cache entries
        """
        self.memory_cache = MemoryCache(memory_cache_mb)
        self.disk_cache = DiskCache(disk_cache_dir, disk_cache_mb)
        self.ttl_seconds = ttl_seconds

        # Performance tracking
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }

        # Thread safety
        self._lock = threading.RLock()

    def _generate_etag(self, content: str) -> str:
        """Generate ETag for content validation."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry.timestamp > self.ttl_seconds

    def get_content(self, content_id: str) -> Optional[str]:
        """
        Get content from cache with fallback strategy.

        Args:
            content_id: Unique identifier for the content

        Returns:
            Cached content or None if not found
        """
        with self._lock:
            self.stats["total_requests"] += 1

            # Try memory cache first
            entry = self.memory_cache.get(content_id)
            if entry and not self._is_expired(entry):
                self.stats["memory_hits"] += 1
                return entry.content

            # Try disk cache
            entry = self.disk_cache.get(content_id)
            if entry and not self._is_expired(entry):
                # Promote to memory cache
                self.memory_cache.set(content_id, entry)
                self.stats["disk_hits"] += 1
                return entry.content

            self.stats["misses"] += 1
            return None

    def set_content(
        self,
        content_id: str,
        content: str,
        etag: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Store content in multi-level cache.

        Args:
            content_id: Unique identifier for the content
            content: Content to cache
            etag: ETag for validation (generated if not provided)
            timestamp: Timestamp for TTL (current time if not provided)

        Returns:
            True if successfully cached
        """
        with self._lock:
            if etag is None:
                etag = self._generate_etag(content)

            if timestamp is None:
                timestamp = time.time()

            entry = CacheEntry(
                content=content,
                etag=etag,
                timestamp=timestamp
            )

            # Store in both caches
            memory_success = self.memory_cache.set(content_id, entry)
            disk_success = self.disk_cache.set(content_id, entry)

            return memory_success and disk_success

    def delete_content(self, content_id: str) -> bool:
        """
        Delete content from all cache levels.

        Args:
            content_id: Unique identifier for the content

        Returns:
            True if successfully deleted from at least one level
        """
        with self._lock:
            memory_deleted = self.memory_cache.delete(content_id)
            disk_deleted = self.disk_cache.delete(content_id)

            return memory_deleted or disk_deleted

    def clear_cache(self, level: str = "all") -> bool:
        """
        Clear cache at specified level.

        Args:
            level: Cache level to clear ("memory", "disk", or "all")

        Returns:
            True if successfully cleared
        """
        with self._lock:
            if level == "memory":
                return self.memory_cache.clear()
            elif level == "disk":
                return self.disk_cache.clear()
            elif level == "all":
                memory_cleared = self.memory_cache.clear()
                disk_cleared = self.disk_cache.clear()
                return memory_cleared and disk_cleared
            else:
                return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["total_requests"]
        total_hits = self.stats["memory_hits"] + self.stats["disk_hits"]

        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        memory_hit_rate = self.stats["memory_hits"] / total_requests if total_requests > 0 else 0

        return {
            "performance": {
                "total_requests": total_requests,
                "total_hits": total_hits,
                "memory_hits": self.stats["memory_hits"],
                "disk_hits": self.stats["disk_hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "memory_hit_rate": memory_hit_rate
            },
            "memory_cache": {
                "size_bytes": self.memory_cache.size(),
                "size_mb": self.memory_cache.size() / (1024 * 1024),
                "entries": len(self.memory_cache.cache)
            },
            "disk_cache": {
                "size_bytes": self.disk_cache.size(),
                "size_mb": self.disk_cache.size() / (1024 * 1024),
                "entries": len(self.disk_cache.index)
            },
            "evictions": self.stats["evictions"],
            "ttl_seconds": self.ttl_seconds
        }

    def preload_content(self, content_ids: List[str], contents: List[str]):
        """
        Preload multiple content items into cache.

        Args:
            content_ids: List of content identifiers
            contents: List of content items to cache
        """
        with self._lock:
            for content_id, content in zip(content_ids, contents):
                self.set_content(content_id, content)

    def cleanup_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            # Clean memory cache
            expired_keys = []
            for key, entry in self.memory_cache.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                self.memory_cache.delete(key)

            # Clean disk cache
            expired_keys = []
            for key, entry_data in self.disk_cache.index.items():
                if time.time() - entry_data["timestamp"] > self.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self.disk_cache.delete(key)

            self.stats["evictions"] += len(expired_keys)

    def get_cache_size_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of cache usage by content type.

        Returns:
            Dictionary with cache size breakdown
        """
        breakdown = {
            "documentation": 0,
            "images": 0,
            "code": 0,
            "other": 0
        }

        # Analyze memory cache
        for key, entry in self.memory_cache.cache.items():
            if key.endswith(".md"):
                breakdown["documentation"] += entry.size_bytes
            elif key.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                breakdown["images"] += entry.size_bytes
            elif key.endswith((".py", ".js", ".html", ".css")):
                breakdown["code"] += entry.size_bytes
            else:
                breakdown["other"] += entry.size_bytes

        return breakdown

    def optimize_cache(self):
        """Optimize cache based on usage patterns."""
        with self._lock:
            # Get cache statistics
            stats = self.get_cache_stats()

            # Adjust cache sizes based on hit rates
            if stats["performance"]["memory_hit_rate"] < 0.5:
                # Memory cache not effective, could reduce size
                pass

            # Clean up expired entries
            self.cleanup_expired()

            # Compact disk cache if needed
            if stats["disk_cache"]["size_mb"] > 500:
                self.disk_cache._evict_lru(self.disk_cache.size() * 0.1)

    def __del__(self):
        """Cleanup when destroyed."""
        try:
            self.clear_cache()
        except:
            pass