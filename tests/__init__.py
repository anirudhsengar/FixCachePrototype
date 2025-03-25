#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache Test Suite

This package contains tests for the FixCache bug prediction tool.
It includes unit tests, integration tests, and test fixtures.

Author: anirudhsengar
"""

import os
import sys
import logging
import tempfile
import shutil
import contextlib
import unittest
from typing import Dict, List, Any, Optional, Tuple, Generator, Callable

# Add the parent directory to the path so we can import fixcache
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress logging during tests unless explicitly enabled
logging.getLogger('fixcache').setLevel(logging.ERROR)

# Test repositories
MOCK_REPO_PATHS = {
    'small': os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'mock_small_repo')),
    'medium': os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'mock_medium_repo')),
    'large': os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'mock_large_repo')),
}

# Default test configuration
DEFAULT_TEST_CONFIG = {
    "cache_size": 0.2,
    "policy": "BUG",
    "window_ratio": 0.25,
    "cache_seeding": True,
    "bug_keywords": ["fix", "bug", "issue", "error", "crash"]
}


@contextlib.contextmanager
def temp_directory() -> Generator[str, None, None]:
    """
    Create a temporary directory that is automatically cleaned up.

    Yields:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="fixcache_test_")
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextlib.contextmanager
def capture_output() -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Capture stdout and stderr output.

    Yields:
        Tuple of (stdout_lines, stderr_lines)
    """
    stdout_buffer = []
    stderr_buffer = []

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class StdoutCapture:
        def write(self, text):
            stdout_buffer.append(text)
            original_stdout.write(text)

        def flush(self):
            original_stdout.flush()

    class StderrCapture:
        def write(self, text):
            stderr_buffer.append(text)
            original_stderr.write(text)

        def flush(self):
            original_stderr.flush()

    sys.stdout = StdoutCapture()
    sys.stderr = StderrCapture()

    try:
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def create_mock_git_repo(
        temp_dir: str,
        files: Dict[str, List[str]],
        commits: List[Dict[str, Any]]
) -> str:
    """
    Create a mock Git repository for testing.

    Args:
        temp_dir: Temporary directory to create repository in
        files: Dictionary mapping file paths to list of content versions
        commits: List of commit information dictionaries

    Returns:
        Path to the created repository
    """
    import subprocess

    # Create directory structure
    repo_path = os.path.join(temp_dir, "mock_repo")
    os.makedirs(repo_path)

    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)

    # Configure git user
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True)

    # Create files and commits
    current_files = {}

    for commit_info in commits:
        # Update files for this commit
        for file_path, content_idx in commit_info.get("file_changes", {}).items():
            if file_path not in files or content_idx >= len(files[file_path]):
                continue

            # Create directory if needed
            file_dir = os.path.dirname(os.path.join(repo_path, file_path))
            if file_dir and not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # Write file content
            with open(os.path.join(repo_path, file_path), "w") as f:
                f.write(files[file_path][content_idx])

            current_files[file_path] = content_idx

        # Add all changes
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)

        # Commit changes
        subprocess.run(
            ["git", "commit", "-m", commit_info.get("message", "Commit")],
            cwd=repo_path, check=True, capture_output=True
        )

    return repo_path


def find_failing_tests(test_class: unittest.TestCase) -> List[str]:
    """
    Run a test class and return names of failing tests.

    Args:
        test_class: Test class to run

    Returns:
        List of failing test names
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    return [test._testMethodName for test, _ in result.failures + result.errors]


def is_git_available() -> bool:
    """
    Check if git is available on the system.

    Returns:
        True if git is available, False otherwise
    """
    try:
        import subprocess
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def setup_test_logging(level: int = logging.INFO) -> None:
    """
    Set up logging for tests.

    Args:
        level: Logging level to use
    """
    logger = logging.getLogger('fixcache')
    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)


# Make a note of test requirements
TEST_REQUIREMENTS = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mock>=4.0.0",
    "coverage>=7.0.0"
]