"""
Documentation Lazy Loading System

Provides lazy loading capabilities for documentation modules to improve performance
and reduce initial load times.

Author: AI Documentation Team
Version: 1.0.0
"""

import os
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class ModuleMetadata:
    """Metadata for documentation modules."""
    path: str
    size_bytes: int
    line_count: int
    last_modified: float
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    priority: int = 0
    estimated_load_time: float = 0.0

class DocumentationLazyLoader:
    """
    Advanced lazy loading system for documentation modules.

    Features:
    - On-demand module loading
    - Dependency preloading
    - Priority-based loading
    - Thread-safe operations
    - Performance monitoring
    """

    def __init__(self, base_path: str, max_cache_size: int = 100):
        """
        Initialize the lazy loader.

        Args:
            base_path: Base path for documentation files
            max_cache_size: Maximum number of modules to cache
        """
        self.base_path = Path(base_path)
        self.max_cache_size = max_cache_size

        # Storage
        self.loaded_modules: Dict[str, str] = {}
        self.module_metadata: Dict[str, ModuleMetadata] = {}
        self.load_history: List[Dict[str, Any]] = []

        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Performance tracking
        self.load_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}

        # Load metadata
        self._initialize_metadata()

    def _initialize_metadata(self):
        """Initialize module metadata from directory structure."""
        metadata_file = self.base_path / "module_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
                self.module_metadata = {
                    path: ModuleMetadata(**data)
                    for path, data in metadata_data.items()
                }
        else:
            self._scan_and_create_metadata()

    def _scan_and_create_metadata(self):
        """Scan directory structure and create metadata."""
        for md_file in self.base_path.rglob("*.md"):
            if md_file.name == "README.md":
                continue

            relative_path = md_file.relative_to(self.base_path)
            stat = md_file.stat()

            # Count lines
            with open(md_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)

            # Create metadata
            metadata = ModuleMetadata(
                path=str(relative_path),
                size_bytes=stat.st_size,
                line_count=line_count,
                last_modified=stat.st_mtime,
                dependencies=self._extract_dependencies(md_file),
                tags=self._extract_tags(md_file),
                priority=self._calculate_priority(md_file),
                estimated_load_time=self._estimate_load_time(stat.st_size)
            )

            self.module_metadata[str(relative_path)] = metadata

        # Save metadata
        self._save_metadata()

    def _extract_dependencies(self, file_path: Path) -> List[str]:
        """Extract module dependencies from file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            dependencies = []

            # Extract markdown links
            import re
            link_pattern = r'\[.*?\]\(\.\./([^)]+)/\)'
            matches = re.findall(link_pattern, content)
            dependencies.extend(matches)

            return list(set(dependencies))
        except Exception:
            return []

    def _extract_tags(self, file_path: Path) -> List[str]:
        """Extract tags from file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [next(f) for _ in range(10)]

            tags = []
            for line in first_lines:
                if line.startswith("# "):
                    # Extract from heading
                    heading = line[2:].strip()
                    tags.extend(heading.lower().split())
                elif "tags:" in line.lower():
                    # Extract explicit tags
                    tag_content = line.split(":", 1)[1].strip()
                    tags.extend([t.strip() for t in tag_content.split(",")])

            return list(set(tags))
        except Exception:
            return []

    def _calculate_priority(self, file_path: Path) -> int:
        """Calculate loading priority based on file characteristics."""
        priority = 0

        # Higher priority for overview files
        if "overview" in file_path.name.lower() or file_path.name.startswith("00_"):
            priority += 10

        # Higher priority for fundamental modules
        if "01_" in file_path.name or "fundamental" in file_path.name.lower():
            priority += 8

        # Lower priority for advanced topics
        if "advanced" in file_path.name.lower() or file_path.name.startswith("04_"):
            priority -= 3

        return max(0, priority)

    def _estimate_load_time(self, size_bytes: int) -> float:
        """Estimate load time based on file size."""
        # Base estimation: ~1ms per 1KB with minimum of 10ms
        return max(0.01, size_bytes / 1024 * 0.001)

    def _save_metadata(self):
        """Save metadata to JSON file."""
        metadata_file = self.base_path / "module_metadata.json"
        metadata_data = {
            path: {
                "path": meta.path,
                "size_bytes": meta.size_bytes,
                "line_count": meta.line_count,
                "last_modified": meta.last_modified,
                "dependencies": meta.dependencies,
                "tags": meta.tags,
                "priority": meta.priority,
                "estimated_load_time": meta.estimated_load_time
            }
            for path, meta in self.module_metadata.items()
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_data, f, indent=2)

    def load_module(self, module_path: str, force_reload: bool = False) -> Optional[str]:
        """
        Load a module with lazy loading.

        Args:
            module_path: Path to the module relative to base path
            force_reload: Force reload even if cached

        Returns:
            Module content or None if not found
        """
        start_time = time.time()

        with self._lock:
            # Check cache first
            if not force_reload and module_path in self.loaded_modules:
                self.access_counts[module_path] = self.access_counts.get(module_path, 0) + 1
                load_time = time.time() - start_time
                self._track_load_time(module_path, load_time)
                return self.loaded_modules[module_path]

            # Load from file
            file_path = self.base_path / module_path
            if not file_path.exists():
                return None

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Cache management
                if len(self.loaded_modules) >= self.max_cache_size:
                    self._evict_from_cache()

                self.loaded_modules[module_path] = content
                self.access_counts[module_path] = 1

                # Track performance
                load_time = time.time() - start_time
                self._track_load_time(module_path, load_time)

                return content

            except Exception as e:
                print(f"Error loading module {module_path}: {e}")
                return None

    def _evict_from_cache(self):
        """Evict least recently used modules from cache."""
        if not self.loaded_modules:
            return

        # Sort by access count and recency
        modules_by_access = sorted(
            self.access_counts.items(),
            key=lambda x: (x[1], self.load_times.get(x[0], 0))
        )

        # Evict bottom 20%
        to_evict = modules_by_access[:len(modules_by_access) // 5]

        for module_path, _ in to_evict:
            if module_path in self.loaded_modules:
                del self.loaded_modules[module_path]
            if module_path in self.access_counts:
                del self.access_counts[module_path]

    def _track_load_time(self, module_path: str, load_time: float):
        """Track module load time for performance monitoring."""
        self.load_times[module_path] = load_time

        self.load_history.append({
            "module": module_path,
            "load_time": load_time,
            "timestamp": time.time(),
            "from_cache": module_path in self.loaded_modules
        })

    def preload_dependencies(self, module_path: str):
        """
        Preload dependencies for a module.

        Args:
            module_path: Path to the module
        """
        if module_path not in self.module_metadata:
            return

        metadata = self.module_metadata[module_path]

        # Load dependencies in parallel
        def load_dependency(dep_path):
            if dep_path in self.module_metadata:
                self.load_module(dep_path)

        # Use thread pool for parallel loading
        futures = []
        for dep in metadata.dependencies:
            future = self._executor.submit(load_dependency, dep)
            futures.append(future)

        # Wait for all dependencies to load
        for future in futures:
            future.result()

    def get_recommended_modules(self, current_module: str, limit: int = 5) -> List[str]:
        """
        Get recommended next modules based on current module.

        Args:
            current_module: Current module path
            limit: Maximum number of recommendations

        Returns:
            List of recommended module paths
        """
        if current_module not in self.module_metadata:
            return []

        recommendations = []
        current_metadata = self.module_metadata[current_module]

        # Get modules with similar tags
        for path, metadata in self.module_metadata.items():
            if path == current_module:
                continue

            # Calculate similarity score
            score = 0

            # Tag similarity
            common_tags = set(current_metadata.tags) & set(metadata.tags)
            score += len(common_tags) * 2

            # Priority bonus
            score += metadata.priority

            # Dependency bonus
            if path in current_metadata.dependencies:
                score += 5

            recommendations.append((path, score))

        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in recommendations[:limit]]

    async def async_load_module(self, module_path: str, force_reload: bool = False) -> Optional[str]:
        """
        Asynchronously load a module.

        Args:
            module_path: Path to the module
            force_reload: Force reload even if cached

        Returns:
            Module content or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.load_module(module_path, force_reload)
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the lazy loader.

        Returns:
            Dictionary with performance metrics
        """
        total_loads = len(self.load_history)
        cache_hits = sum(1 for entry in self.load_history if entry["from_cache"])

        avg_load_time = (
            sum(entry["load_time"] for entry in self.load_history) / total_loads
            if total_loads > 0 else 0
        )

        cache_hit_rate = cache_hits / total_loads if total_loads > 0 else 0

        return {
            "total_modules": len(self.module_metadata),
            "cached_modules": len(self.loaded_modules),
            "total_loads": total_loads,
            "cache_hits": cache_hits,
            "cache_misses": total_loads - cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "average_load_time": avg_load_time,
            "cache_utilization": len(self.loaded_modules) / self.max_cache_size
        }

    def clear_cache(self):
        """Clear all cached modules."""
        with self._lock:
            self.loaded_modules.clear()
            self.access_counts.clear()

    def preload_popular_modules(self, limit: int = 10):
        """
        Preload frequently accessed modules.

        Args:
            limit: Number of modules to preload
        """
        # Sort by access count
        popular_modules = sorted(
            self.access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        # Load popular modules
        for module_path, _ in popular_modules:
            self.load_module(module_path)

    def __del__(self):
        """Cleanup when destroyed."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)