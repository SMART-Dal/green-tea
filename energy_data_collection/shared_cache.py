#!/usr/bin/env python3
"""
SharedExecutionCache - Thread-safe and multi-job-safe cache for tracking completed executions.

Features:
- Filesystem-based (works across multiple SLURM jobs)
- Atomic operations (safe for concurrent access)
- Resumable (skip already-completed executions)
- Tracks failures separately
- Zero dependencies (pure Python)

Usage:
    cache = SharedExecutionCache('/shared/cache/dir')

    if not cache.is_completed(execution_id):
        result = run_simulation(execution)
        cache.mark_completed(execution_id, result)
"""

import os
import json
import time
import fcntl
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SharedExecutionCache:
    """
    Filesystem-based cache for tracking completed/failed executions.

    Directory structure:
        cache_dir/
        ├── completed/
        │   ├── {execution_id}.done     # Marker + result metadata
        │   └── ...
        ├── failed/
        │   ├── {execution_id}.failed   # Marker + error info
        │   └── ...
        └── .lock                        # Global lock file
    """

    def __init__(self, cache_dir: str):
        """
        Initialize the shared execution cache.

        Args:
            cache_dir: Path to shared cache directory (must be accessible by all jobs)
        """
        self.cache_dir = Path(cache_dir)
        self.completed_dir = self.cache_dir / "completed"
        self.failed_dir = self.cache_dir / "failed"
        self.lock_file = self.cache_dir / ".lock"

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(exist_ok=True)
        self.failed_dir.mkdir(exist_ok=True)
        self.lock_file.touch(exist_ok=True)

        logger.info(f"SharedExecutionCache initialized at {cache_dir}")

    @contextmanager
    def _lock(self, timeout: int = 300):
        """
        Acquire exclusive lock for atomic operations.

        Args:
            timeout: Maximum seconds to wait for lock (default: 300s for high concurrency)
        """
        start_time = time.time()
        lockf = None

        try:
            lockf = open(self.lock_file, 'w')

            while True:
                try:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        raise TimeoutError(f"Could not acquire lock after {timeout}s (waited {elapsed:.1f}s)")
                    # Exponential backoff: start with 0.1s, max 1s
                    sleep_time = min(0.1 * (2 ** int(elapsed / 2)), 1.0)
                    time.sleep(sleep_time)

            yield

        finally:
            if lockf:
                try:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
                    lockf.close()
                except:
                    pass

    def is_completed(self, execution_id: str) -> bool:
        """
        Check if execution is already completed.

        Args:
            execution_id: Execution ID to check

        Returns:
            True if execution is completed, False otherwise
        """
        marker_file = self.completed_dir / f"{execution_id}.done"
        return marker_file.exists()

    def is_failed(self, execution_id: str) -> bool:
        """
        Check if execution is marked as failed.

        Args:
            execution_id: Execution ID to check

        Returns:
            True if execution failed, False otherwise
        """
        marker_file = self.failed_dir / f"{execution_id}.failed"
        return marker_file.exists()

    def mark_completed(self, execution_id: str, result: Optional[Dict[str, Any]] = None):
        """
        Mark execution as completed (lock-free for better concurrency).

        Args:
            execution_id: Execution ID
            result: Optional result metadata to store
        """
        marker_file = self.completed_dir / f"{execution_id}.done"

        # Remove from failed if present (idempotent - safe without lock)
        failed_file = self.failed_dir / f"{execution_id}.failed"
        if failed_file.exists():
            try:
                failed_file.unlink()
            except FileNotFoundError:
                pass  # Already removed or doesn't exist

        # Write completion marker atomically using temp + rename
        data = {
            'execution_id': execution_id,
            'completed_at': time.time(),
            'result': result or {}
        }

        # Atomic write (temp file + rename is atomic on POSIX filesystems)
        temp_file = marker_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        temp_file.rename(marker_file)

    def mark_failed(self, execution_id: str, error_info: Optional[Dict[str, Any]] = None):
        """
        Mark execution as failed (lock-free for better concurrency).

        Args:
            execution_id: Execution ID
            error_info: Optional error information to store
        """
        marker_file = self.failed_dir / f"{execution_id}.failed"

        data = {
            'execution_id': execution_id,
            'failed_at': time.time(),
            'error': error_info or {}
        }

        # Atomic write using temp file + rename (atomic on POSIX filesystems)
        temp_file = marker_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        temp_file.rename(marker_file)

    def get_completed_ids(self) -> Set[str]:
        """
        Get set of all completed execution IDs.

        Returns:
            Set of completed execution IDs
        """
        completed = set()
        for marker_file in self.completed_dir.glob("*.done"):
            execution_id = marker_file.stem
            completed.add(execution_id)
        return completed

    def get_failed_ids(self) -> Set[str]:
        """
        Get set of all failed execution IDs.

        Returns:
            Set of failed execution IDs
        """
        failed = set()
        for marker_file in self.failed_dir.glob("*.failed"):
            execution_id = marker_file.stem
            failed.add(execution_id)
        return failed

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with completed/failed counts
        """
        completed = len(list(self.completed_dir.glob("*.done")))
        failed = len(list(self.failed_dir.glob("*.failed")))

        return {
            'completed': completed,
            'failed': failed,
            'total': completed + failed
        }

    def get_completion_info(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get completion metadata for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Completion metadata dict, or None if not completed
        """
        marker_file = self.completed_dir / f"{execution_id}.done"

        if not marker_file.exists():
            return None

        try:
            with open(marker_file) as f:
                return json.load(f)
        except:
            return None

    def get_failure_info(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get failure metadata for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Failure metadata dict, or None if not failed
        """
        marker_file = self.failed_dir / f"{execution_id}.failed"

        if not marker_file.exists():
            return None

        try:
            with open(marker_file) as f:
                return json.load(f)
        except:
            return None

    def clear_completed(self):
        """Clear all completed markers (use with caution!)."""
        for marker_file in self.completed_dir.glob("*.done"):
            try:
                marker_file.unlink()
            except FileNotFoundError:
                pass  # Already removed
        logger.info("Cleared all completed markers")

    def clear_failed(self):
        """Clear all failed markers."""
        for marker_file in self.failed_dir.glob("*.failed"):
            try:
                marker_file.unlink()
            except FileNotFoundError:
                pass  # Already removed
        logger.info("Cleared all failed markers")

    def clear_all(self):
        """Clear all markers (completed + failed)."""
        self.clear_completed()
        self.clear_failed()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create cache
    cache = SharedExecutionCache("/tmp/test_cache")

    # Mark some as completed
    cache.mark_completed("p00000_abc123_def456", {
        'energy_joules': 0.045,
        'runtime_seconds': 1.23
    })

    # Mark some as failed
    cache.mark_failed("p00001_xyz789_ghi012", {
        'error': 'Compilation failed',
        'stderr': 'syntax error...'
    })

    # Check status
    print(f"Is p00000_abc123_def456 completed? {cache.is_completed('p00000_abc123_def456')}")
    print(f"Is p00001_xyz789_ghi012 failed? {cache.is_failed('p00001_xyz789_ghi012')}")

    # Get stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")

    # Get completion info
    info = cache.get_completion_info("p00000_abc123_def456")
    print(f"Completion info: {info}")
