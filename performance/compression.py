"""
Content Compression and Optimization System

Provides content compression, minification, and optimization utilities
for documentation delivery.

Author: AI Documentation Team
Version: 1.0.0
"""

import re
import gzip
import zlib
import json
import base64
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import time

@dataclass
class CompressionResult:
    """Result of content compression operation."""
    original: str
    compressed: bytes
    compression_ratio: float
    size_original: int
    size_compressed: int
    algorithm: str
    time_taken: float

@dataclass
class OptimizationResult:
    """Result of content optimization operation."""
    original: str
    optimized: str
    size_reduction: float
    optimizations_applied: List[str]
    time_taken: float

class MarkdownMinifier:
    """Markdown content minifier."""

    def __init__(self):
        """Initialize minifier with optimization patterns."""
        self.optimizations = [
            self._remove_comments,
            self._normalize_whitespace,
            self._minify_code_blocks,
            self._optimize_links,
            self._normalize_headers,
            self._remove_redundant_formatting
        ]

    def minify(self, content: str) -> str:
        """
        Minify markdown content.

        Args:
            content: Raw markdown content

        Returns:
            Minified content
        """
        optimized = content

        for optimization in self.optimizations:
            optimized = optimization(optimized)

        return optimized

    def _remove_comments(self, content: str) -> str:
        """Remove HTML and markdown comments."""
        # HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        # Markdown-style comments
        content = re.sub(r'\[//]: # \(.*?\)', '', content)
        return content

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace without affecting markdown structure."""
        lines = content.split('\n')
        normalized_lines = []

        for line in lines:
            # Preserve indentation for code blocks and lists
            if line.strip() == '':
                normalized_lines.append('')
            else:
                # Remove trailing whitespace
                line = line.rstrip()
                # Normalize multiple spaces (but preserve indentation)
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def _minify_code_blocks(self, content: str) -> str:
        """Minify code blocks while preserving functionality."""
        # This is a simple implementation - in production, you'd use language-specific minifiers
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'

        def minify_code_block(match):
            language = match.group(1) or ''
            code = match.group(2)

            # Simple minification for common languages
            if language in ['python', 'py']:
                code = self._minify_python_code(code)
            elif language in ['javascript', 'js']:
                code = self._minify_javascript_code(code)

            return f'```{language}\n{code}\n```'

        return re.sub(code_block_pattern, minify_code_block, content, flags=re.DOTALL)

    def _minify_python_code(self, code: str) -> str:
        """Simple Python code minification."""
        lines = code.split('\n')
        minified_lines = []

        for line in lines:
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            # Remove empty lines but preserve structure
            if line.strip() or line == '':
                minified_lines.append(line.rstrip())

        return '\n'.join(minified_lines)

    def _minify_javascript_code(self, code: str) -> str:
        """Simple JavaScript code minification."""
        # Remove comments
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    def _optimize_links(self, content: str) -> str:
        """Optimize markdown links."""
        # Remove unnecessary parentheses in URLs
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: f'[{m.group(1)}]({m.group(2).strip()})', content)
        return content

    def _normalize_headers(self, content: str) -> str:
        """Normalize header formatting."""
        # Ensure consistent header spacing
        content = re.sub(r'^(#{1,6})\s*', r'\1 ', content, flags=re.MULTILINE)
        return content

    def _remove_redundant_formatting(self, content: str) -> str:
        """Remove redundant formatting."""
        # Remove consecutive bold/italic markers
        content = re.sub(r'\*\*\*\*', '***', content)
        content = re.sub(r'____', '___', content)
        return content

class ContentCompressor:
    """Content compression utility."""

    def __init__(self):
        """Initialize compressor."""
        self.algorithms = {
            'gzip': self._compress_gzip,
            'zlib': self._compress_zlib,
            'base64': self._compress_base64,
            'none': self._no_compression
        }

    def compress(self, content: Union[str, bytes], algorithm: str = 'gzip') -> CompressionResult:
        """
        Compress content using specified algorithm.

        Args:
            content: Content to compress
            algorithm: Compression algorithm to use

        Returns:
            Compression result
        """
        start_time = time.time()

        if isinstance(content, str):
            original_bytes = content.encode('utf-8')
            original_content = content
        else:
            original_bytes = content
            original_content = content.decode('utf-8', errors='ignore')

        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")

        compressed_bytes = self.algorithms[algorithm](original_bytes)

        end_time = time.time()

        return CompressionResult(
            original=original_content,
            compressed=compressed_bytes,
            compression_ratio=len(compressed_bytes) / len(original_bytes),
            size_original=len(original_bytes),
            size_compressed=len(compressed_bytes),
            algorithm=algorithm,
            time_taken=end_time - start_time
        )

    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress using gzip."""
        return gzip.compress(data, compresslevel=9)

    def _compress_zlib(self, data: bytes) -> bytes:
        """Compress using zlib."""
        return zlib.compress(data, level=9)

    def _compress_base64(self, data: bytes) -> bytes:
        """Compress using base64 encoding (not real compression)."""
        return base64.b64encode(data)

    def _no_compression(self, data: bytes) -> bytes:
        """No compression."""
        return data

    def decompress(self, compressed_data: bytes, algorithm: str = 'gzip') -> bytes:
        """
        Decompress content.

        Args:
            compressed_data: Compressed data
            algorithm: Compression algorithm used

        Returns:
            Decompressed data
        """
        if algorithm == 'gzip':
            return gzip.decompress(compressed_data)
        elif algorithm == 'zlib':
            return zlib.decompress(compressed_data)
        elif algorithm == 'base64':
            return base64.b64decode(compressed_data)
        elif algorithm == 'none':
            return compressed_data
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")

class ContentOptimizer:
    """
    Content optimization and compression system.

    Features:
    - Markdown minification
    - Multiple compression algorithms
    - Content analysis
    - Performance optimization
    - Delivery format selection
    """

    def __init__(self):
        """Initialize content optimizer."""
        self.minifier = MarkdownMinifier()
        self.compressor = ContentCompressor()
        self.optimization_cache = {}

    def optimize_content(self, content: str, compression_algorithms: List[str] = None) -> Dict[str, Any]:
        """
        Optimize content with various compression algorithms.

        Args:
            content: Content to optimize
            compression_algorithms: List of algorithms to try

        Returns:
            Optimization results
        """
        if compression_algorithms is None:
            compression_algorithms = ['gzip', 'zlib', 'base64', 'none']

        start_time = time.time()

        # Generate content hash for caching
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        # Check cache
        if content_hash in self.optimization_cache:
            cached_result = self.optimization_cache[content_hash]
            # Verify content hasn't changed
            if cached_result['original_length'] == len(content):
                return cached_result

        # Minify content
        minified_content = self.minifier.minify(content)

        # Test different compression algorithms
        compression_results = {}
        for algorithm in compression_algorithms:
            try:
                result = self.compressor.compress(minified_content, algorithm)
                compression_results[algorithm] = {
                    'compressed_size': result.size_compressed,
                    'compression_ratio': result.compression_ratio,
                    'time_taken': result.time_taken
                }
            except Exception as e:
                compression_results[algorithm] = {
                    'error': str(e)
                }

        # Analyze content
        content_analysis = self._analyze_content(content)

        # Select best algorithm
        best_algorithm = self._select_best_algorithm(compression_results)

        # Generate final optimized content
        optimized_result = self.compressor.compress(minified_content, best_algorithm)

        end_time = time.time()

        result = {
            'original': content,
            'minified': minified_content,
            'optimized_compressed': optimized_result.compressed,
            'original_length': len(content),
            'minified_length': len(minified_content),
            'compressed_length': optimized_result.size_compressed,
            'minification_ratio': len(minified_content) / len(content),
            'compression_ratio': optimized_result.compression_ratio,
            'total_ratio': optimized_result.size_compressed / len(content),
            'best_algorithm': best_algorithm,
            'all_algorithms': compression_results,
            'content_analysis': content_analysis,
            'time_taken': end_time - start_time,
            'content_hash': content_hash
        }

        # Cache result
        self.optimization_cache[content_hash] = result

        return result

    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content characteristics."""
        lines = content.split('\n')
        words = content.split()

        # Count different content types
        code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
        headers = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        links = len(re.findall(r'\[.*?\]\(.*?\)', content))
        images = len(re.findall(r'!\[.*?\]\(.*?\)', content))
        tables = len(re.findall(r'\|.*\|.*\|', content))

        # Calculate readability metrics
        avg_words_per_line = len(words) / len(lines) if lines else 0
        avg_chars_per_line = len(content) / len(lines) if lines else 0

        return {
            'total_lines': len(lines),
            'total_words': len(words),
            'total_characters': len(content),
            'code_blocks': code_blocks,
            'headers': headers,
            'links': links,
            'images': images,
            'tables': tables,
            'avg_words_per_line': avg_words_per_line,
            'avg_chars_per_line': avg_chars_per_line,
            'content_type': self._classify_content(content, code_blocks, headers, links)
        }

    def _classify_content(self, content: str, code_blocks: int, headers: int, links: int) -> str:
        """Classify content type."""
        if code_blocks > 5:
            return "code_heavy"
        elif headers > 10:
            return "documentation"
        elif links > 20:
            return "reference"
        elif len(content) > 5000:
            return "long_form"
        else:
            return "general"

    def _select_best_algorithm(self, compression_results: Dict[str, Dict]) -> str:
        """Select best compression algorithm based on results."""
        best_algorithm = 'none'
        best_ratio = 1.0

        for algorithm, result in compression_results.items():
            if 'error' in result:
                continue

            ratio = result['compression_ratio']
            time_taken = result['time_taken']

            # Consider both compression ratio and speed
            # Weight compression ratio more heavily for documentation
            score = ratio * 0.8 + (time_taken * 0.2)

            if score < best_ratio:
                best_ratio = score
                best_algorithm = algorithm

        return best_algorithm

    def get_delivery_format(
        self,
        content: str,
        format_type: str = 'optimized',
        compression_algorithm: str = 'gzip'
    ) -> Dict[str, Any]:
        """
        Get content in specified delivery format.

        Args:
            content: Original content
            format_type: Format type ('raw', 'minified', 'compressed', 'optimized')
            compression_algorithm: Algorithm to use for compression

        Returns:
            Formatted content with metadata
        """
        result = {
            'format': format_type,
            'algorithm': compression_algorithm,
            'content': None,
            'metadata': {}
        }

        if format_type == 'raw':
            result['content'] = content
            result['metadata']['size'] = len(content)

        elif format_type == 'minified':
            minified = self.minifier.minify(content)
            result['content'] = minified
            result['metadata']['size'] = len(minified)

        elif format_type == 'compressed':
            compressed = self.compressor.compress(content, compression_algorithm)
            result['content'] = compressed.compressed
            result['metadata']['size'] = compressed.size_compressed
            result['metadata']['compression_ratio'] = compressed.compression_ratio

        elif format_type == 'optimized':
            optimized = self.optimize_content(content, [compression_algorithm])
            result['content'] = optimized['optimized_compressed']
            result['metadata'] = {
                'size': optimized['compressed_length'],
                'compression_ratio': optimized['compression_ratio'],
                'minification_ratio': optimized['minification_ratio'],
                'total_ratio': optimized['total_ratio']
            }

        return result

    def batch_optimize(self, contents: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Optimize multiple content items.

        Args:
            contents: Dictionary of content_id -> content

        Returns:
            Dictionary of optimization results
        """
        results = {}

        for content_id, content in contents.items():
            try:
                results[content_id] = self.optimize_content(content)
            except Exception as e:
                results[content_id] = {
                    'error': str(e),
                    'content_id': content_id
                }

        return results

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_cache:
            return {
                'total_optimized': 0,
                'avg_compression_ratio': 0,
                'avg_minification_ratio': 0,
                'cache_size': 0
            }

        total_optimized = len(self.optimization_cache)
        compression_ratios = [result['compression_ratio'] for result in self.optimization_cache.values()]
        minification_ratios = [result['minification_ratio'] for result in self.optimization_cache.values()]

        return {
            'total_optimized': total_optimized,
            'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0,
            'avg_minification_ratio': sum(minification_ratios) / len(minification_ratios) if minification_ratios else 0,
            'cache_size': len(self.optimization_cache),
            'best_performing_algorithm': max(
                set(result['best_algorithm'] for result in self.optimization_cache.values()),
                key=lambda x: list(result['best_algorithm'] for result in self.optimization_cache.values()).count(x)
            ) if self.optimization_cache else 'none'
        }

    def clear_cache(self):
        """Clear optimization cache."""
        self.optimization_cache.clear()

    def preload_common_patterns(self):
        """Preload optimization patterns for common content types."""
        common_patterns = {
            'documentation': '# Documentation\n\nThis is documentation content.',
            'code_heavy': '# Code Examples\n\n```python\ndef hello():\n    print("Hello")\n```',
            'list_heavy': '# Lists\n\n- Item 1\n- Item 2\n- Item 3',
            'table_heavy': '# Tables\n\n| Header | Value |\n|--------|-------|\n| Data   | Info  |'
        }

        for pattern_name, content in common_patterns.items():
            self.optimize_content(content)