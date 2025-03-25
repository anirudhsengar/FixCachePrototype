#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit Tests for FixCache Repository Analysis

Tests the repository analysis functionality including commit history extraction,
bug fix identification, and file change tracking.

Author: anirudhsengar
"""

import os
import sys
import unittest
import tempfile
import shutil
import datetime
from unittest import mock
from typing import Dict, List, Any, Optional

# Import from tests package
from tests import temp_directory, capture_output, create_mock_git_repo

# Import FixCache components
from fixcache.repository import RepositoryAnalyzer
from fixcache.utils import is_code_file


class TestRepositoryAnalyzerInitialization(unittest.TestCase):
    """Test RepositoryAnalyzer initialization and parameter validation."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "file1.py": ["print('Hello World')"],
                    "file2.py": ["def foo():\n    return 'bar'"]
                },
                commits=[
                    {"message": "Initial commit", "file_changes": {"file1.py": 0, "file2.py": 0}}
                ]
            )

            # Create RepositoryAnalyzer instance
            repo_analyzer = RepositoryAnalyzer(repo_path)

            # Check initialization
            self.assertEqual(repo_analyzer.repo_path, repo_path)
            self.assertIsNone(repo_analyzer.lookback_commits)
            self.assertEqual(repo_analyzer.encoding, 'utf-8')
            self.assertEqual(repo_analyzer.fallback_encoding, 'latin-1')
            self.assertGreater(len(repo_analyzer.bug_keywords), 0)
            self.assertFalse(repo_analyzer.is_analyzed)
            self.assertEqual(len(repo_analyzer.error_messages), 0)

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Custom parameters
            bug_keywords = ["error", "crash", "exception"]
            lookback_commits = 50
            encoding = 'iso-8859-1'
            fallback_encoding = 'ascii'

            # Create RepositoryAnalyzer instance with custom parameters
            repo_analyzer = RepositoryAnalyzer(
                repo_path,
                bug_keywords=bug_keywords,
                lookback_commits=lookback_commits,
                encoding=encoding,
                fallback_encoding=fallback_encoding
            )

            # Check initialization
            self.assertEqual(repo_analyzer.repo_path, repo_path)
            self.assertEqual(repo_analyzer.lookback_commits, lookback_commits)
            self.assertEqual(repo_analyzer.encoding, encoding)
            self.assertEqual(repo_analyzer.fallback_encoding, fallback_encoding)

            # Check that bug keywords were properly set
            for keyword in bug_keywords:
                self.assertIn(keyword, repo_analyzer.bug_keywords)

    def test_init_with_invalid_repository(self):
        """Test initialization with invalid repository path."""
        # Create instance with non-existent path
        repo_analyzer = RepositoryAnalyzer("/path/does/not/exist")

        # Check initialization
        self.assertEqual(repo_analyzer.repo_path, "/path/does/not/exist")
        self.assertFalse(repo_analyzer.is_analyzed)

        # Analysis should fail
        self.assertFalse(repo_analyzer.analyze())


class TestRepositoryAnalysis(unittest.TestCase):
    """Test repository analysis functionality."""

    def setUp(self):
        """Set up a test repository."""
        self.temp_dir = tempfile.mkdtemp(prefix="fixcache_test_")

        # Create a git repository with bug fixes
        self.repo_path = create_mock_git_repo(
            self.temp_dir,
            files={
                "file1.py": ["def func1():\n    return 1",
                             "def func1():\n    return 1  # fixed"],
                "file2.py": ["def func2():\n    return 2",
                             "def func2():\n    return 2  # updated"],
                "file3.py": ["def func3():\n    return 3"],
                "README.md": ["# Test Repository",
                              "# Test Repository\n\nAdded description"]
            },
            commits=[
                {"message": "Initial commit",
                 "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0, "README.md": 0}},
                {"message": "Fix bug in file1",
                 "file_changes": {"file1.py": 1}},
                {"message": "Update file2",
                 "file_changes": {"file2.py": 1}},
                {"message": "Update documentation",
                 "file_changes": {"README.md": 1}}
            ]
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyze_success(self):
        """Test successful repository analysis."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Run analysis
        success = repo_analyzer.analyze()

        # Check analysis results
        self.assertTrue(success)
        self.assertTrue(repo_analyzer.is_analyzed)
        self.assertEqual(repo_analyzer.total_files, 3)  # Excluding README.md
        self.assertEqual(len(repo_analyzer.commit_history), 4)
        self.assertEqual(len(repo_analyzer.bug_fixes), 1)  # Only one fix commit

        # Check file stats
        self.assertIn("file1.py", repo_analyzer.file_stats)
        self.assertEqual(repo_analyzer.file_stats["file1.py"]["bug_fixes"], 1)
        self.assertIn("file2.py", repo_analyzer.file_stats)
        self.assertEqual(repo_analyzer.file_stats["file2.py"]["bug_fixes"], 0)

    def test_analyze_with_lookback(self):
        """Test analysis with lookback commits limit."""
        # Create analyzer with lookback of 2 commits
        repo_analyzer = RepositoryAnalyzer(self.repo_path, lookback_commits=2)

        # Run analysis
        success = repo_analyzer.analyze()

        # Check analysis results
        self.assertTrue(success)
        self.assertEqual(len(repo_analyzer.commit_history), 2)  # Only last 2 commits

    def test_get_all_files(self):
        """Test _get_all_files method."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Call _get_all_files directly
        repo_analyzer._get_all_files()

        # Check results
        self.assertEqual(repo_analyzer.total_files, 3)  # Excluding README.md
        self.assertIn("file1.py", repo_analyzer.file_stats)
        self.assertIn("file2.py", repo_analyzer.file_stats)
        self.assertIn("file3.py", repo_analyzer.file_stats)
        self.assertNotIn("README.md", repo_analyzer.file_stats)  # Not a code file

        # Check file stats initialization
        for file_path in ["file1.py", "file2.py", "file3.py"]:
            self.assertEqual(repo_analyzer.file_stats[file_path]["bug_fixes"], 0)
            self.assertIsNone(repo_analyzer.file_stats[file_path]["last_bug_fix"])
            self.assertEqual(repo_analyzer.file_stats[file_path]["commits"], 0)
            self.assertEqual(len(repo_analyzer.file_stats[file_path]["authors"]), 0)

    def test_is_code_file(self):
        """Test _is_code_file method."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Test code file detection
        self.assertTrue(repo_analyzer._is_code_file("file1.py"))
        self.assertTrue(repo_analyzer._is_code_file("file2.py"))
        self.assertTrue(repo_analyzer._is_code_file("src/main.cpp"))
        self.assertTrue(repo_analyzer._is_code_file("lib/util.js"))

        # Test non-code files
        self.assertFalse(repo_analyzer._is_code_file("image.png"))
        self.assertFalse(repo_analyzer._is_code_file("document.pdf"))
        self.assertFalse(repo_analyzer._is_code_file("data.bin"))

        # Test non-existent file
        self.assertFalse(repo_analyzer._is_code_file("doesnotexist.py"))

    def test_get_all_commits(self):
        """Test _get_all_commits method."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Call _get_all_commits directly
        repo_analyzer._get_all_commits()

        # Check results
        self.assertEqual(len(repo_analyzer.commit_history), 4)

        # Check commit format
        first_commit = repo_analyzer.commit_history[-1]  # First commit is last in history
        self.assertIn("sha", first_commit)
        self.assertIn("author", first_commit)
        self.assertIn("timestamp", first_commit)
        self.assertIn("message", first_commit)
        self.assertIn("is_bug_fix", first_commit)
        self.assertIn("files_changed", first_commit)

        # Check commit message
        self.assertEqual(first_commit["message"], "Initial commit")
        self.assertFalse(first_commit["is_bug_fix"])

    def test_identify_bug_fixes(self):
        """Test _identify_bug_fixes method."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Setup commit history
        repo_analyzer.commit_history = [
            {"sha": "abc123", "message": "Initial commit", "is_bug_fix": False},
            {"sha": "def456", "message": "Fix bug in file1", "is_bug_fix": False},
            {"sha": "ghi789", "message": "Update documentation", "is_bug_fix": False},
            {"sha": "jkl012", "message": "Fixed issue #123", "is_bug_fix": False},
            {"sha": "mno345", "message": "Refactor code", "is_bug_fix": False}
        ]

        # Call _identify_bug_fixes directly
        repo_analyzer._identify_bug_fixes()

        # Check results
        self.assertEqual(len(repo_analyzer.bug_fixes), 2)

        # Check bug fix identification
        bug_fix_messages = [commit["message"] for commit in repo_analyzer.bug_fixes]
        self.assertIn("Fix bug in file1", bug_fix_messages)
        self.assertIn("Fixed issue #123", bug_fix_messages)
        self.assertNotIn("Initial commit", bug_fix_messages)
        self.assertNotIn("Update documentation", bug_fix_messages)
        self.assertNotIn("Refactor code", bug_fix_messages)

        # Check is_bug_fix flag
        for commit in repo_analyzer.commit_history:
            if commit["message"] in ["Fix bug in file1", "Fixed issue #123"]:
                self.assertTrue(commit["is_bug_fix"])
            else:
                self.assertFalse(commit["is_bug_fix"])

    def test_process_bug_fixes(self):
        """Test _process_bug_fixes method."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Setup bug fixes with mocked _get_files_changed
        repo_analyzer.bug_fixes = [
            {"sha": "def456", "message": "Fix bug in file1", "author": "test_user",
             "timestamp": 12345, "is_bug_fix": True},
            {"sha": "jkl012", "message": "Fixed issue #123", "author": "test_user",
             "timestamp": 23456, "is_bug_fix": True}
        ]

        repo_analyzer.file_stats = {
            "file1.py": {"bug_fixes": 0, "last_bug_fix": None, "authors": set()},
            "file2.py": {"bug_fixes": 0, "last_bug_fix": None, "authors": set()},
            "file3.py": {"bug_fixes": 0, "last_bug_fix": None, "authors": set()}
        }

        # Mock _get_files_changed
        with mock.patch.object(repo_analyzer, '_get_files_changed') as mock_get_files:
            mock_get_files.side_effect = [
                ["file1.py"],  # First bug fix changed file1.py
                ["file1.py", "file2.py"]  # Second bug fix changed file1.py and file2.py
            ]

            # Call _process_bug_fixes directly
            repo_analyzer._process_bug_fixes()

            # Check file stats updates
            self.assertEqual(repo_analyzer.file_stats["file1.py"]["bug_fixes"], 2)
            self.assertEqual(repo_analyzer.file_stats["file1.py"]["last_bug_fix"], 23456)
            self.assertIn("test_user", repo_analyzer.file_stats["file1.py"]["authors"])

            self.assertEqual(repo_analyzer.file_stats["file2.py"]["bug_fixes"], 1)
            self.assertEqual(repo_analyzer.file_stats["file2.py"]["last_bug_fix"], 23456)
            self.assertIn("test_user", repo_analyzer.file_stats["file2.py"]["authors"])

            self.assertEqual(repo_analyzer.file_stats["file3.py"]["bug_fixes"], 0)
            self.assertIsNone(repo_analyzer.file_stats["file3.py"]["last_bug_fix"])
            self.assertEqual(len(repo_analyzer.file_stats["file3.py"]["authors"]), 0)

    def test_get_files_changed(self):
        """Test _get_files_changed method."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Run analysis to get commit SHAs
        repo_analyzer.analyze()

        # Get a commit SHA
        commit_sha = repo_analyzer.commit_history[0]["sha"]

        # Call _get_files_changed directly
        files_changed = repo_analyzer._get_files_changed(commit_sha)

        # Check results
        self.assertIsInstance(files_changed, list)

        # At least one of our test files should be in the result
        found_file = False
        for file_path in ["file1.py", "file2.py", "file3.py", "README.md"]:
            if file_path in files_changed:
                found_file = True
                break

        self.assertTrue(found_file, "No test files found in files_changed")

    def test_run_git_command(self):
        """Test _run_git_command method."""
        # Create analyzer
        repo_analyzer = RepositoryAnalyzer(self.repo_path)

        # Test successful command
        output = repo_analyzer._run_git_command(["status"])
        self.assertIsInstance(output, str)
        self.assertIn("branch", output.lower())

        # Test with encoding fallback (simulate error)
        with mock.patch.object(repo_analyzer, '_run_command_with_encoding') as mock_run:
            mock_run.side_effect = [
                UnicodeDecodeError('utf-8', b'test', 0, 1, 'test error'),  # Primary encoding fails
                "fallback output"  # Fallback encoding works
            ]

            output = repo_analyzer._run_git_command(["status"])
            self.assertEqual(output, "fallback output")

            # Verify both encodings were tried
            self.assertEqual(mock_run.call_count, 2)
            self.assertEqual(mock_run.call_args_list[0][0][1], "utf-8")
            self.assertEqual(mock_run.call_args_list[1][0][1], "latin-1")


class TestRepositoryMetrics(unittest.TestCase):
    """Test repository metrics and analysis methods."""

    def setUp(self):
        """Set up a test repository."""
        self.temp_dir = tempfile.mkdtemp(prefix="fixcache_test_")

        # Create a git repository with more complex history
        self.repo_path = create_mock_git_repo(
            self.temp_dir,
            files={
                "file1.py": ["v1", "v2", "v3", "v4"],
                "file2.py": ["v1", "v2", "v3"],
                "file3.py": ["v1", "v2"],
                "file4.py": ["v1"]
            },
            commits=[
                {"message": "Initial commit",
                 "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0, "file4.py": 0}},
                {"message": "Fix bug in file1",
                 "file_changes": {"file1.py": 1}},
                {"message": "Update file2",
                 "file_changes": {"file2.py": 1}},
                {"message": "Fix critical bug in file3",
                 "file_changes": {"file3.py": 1}},
                {"message": "Fix another bug in file1",
                 "file_changes": {"file1.py": 2}},
                {"message": "Fix regression in file2",
                 "file_changes": {"file2.py": 2}},
                {"message": "Update file1",
                 "file_changes": {"file1.py": 3}}
            ]
        )

        # Create and analyze repository
        self.repo_analyzer = RepositoryAnalyzer(self.repo_path)
        self.repo_analyzer.analyze()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_file_change_frequency(self):
        """Test get_file_change_frequency method."""
        # Get change frequency
        frequencies = self.repo_analyzer.get_file_change_frequency()

        # Check results
        self.assertIsInstance(frequencies, dict)
        self.assertGreater(len(frequencies), 0)

        # Check specific files
        self.assertIn("file1.py", frequencies)
        self.assertIn("file2.py", frequencies)
        self.assertIn("file3.py", frequencies)

        # file1.py has more bug fixes than file4.py
        self.assertGreater(frequencies["file1.py"], frequencies["file4.py"])

    def test_get_file_complexity(self):
        """Test get_file_complexity method."""
        # Get file complexity
        complexity = self.repo_analyzer.get_file_complexity()

        # Check results
        self.assertIsInstance(complexity, dict)
        self.assertGreater(len(complexity), 0)

        # Check specific files
        for file_path in ["file1.py", "file2.py", "file3.py", "file4.py"]:
            if file_path in complexity:
                file_metrics = complexity[file_path]
                self.assertIn("size_bytes", file_metrics)
                self.assertIn("line_count", file_metrics)
                self.assertIn("size_category", file_metrics)
                self.assertIn("bug_density", file_metrics)

    def test_categorize_file_size(self):
        """Test _categorize_file_size method."""
        # Test small file
        self.assertEqual(self.repo_analyzer._categorize_file_size(500), "small")

        # Test medium file
        self.assertEqual(self.repo_analyzer._categorize_file_size(5000), "medium")

        # Test large file
        self.assertEqual(self.repo_analyzer._categorize_file_size(15000), "large")

    def test_get_file_ownership(self):
        """Test get_file_ownership method."""
        # Get file ownership
        ownership = self.repo_analyzer.get_file_ownership()

        # Check results
        self.assertIsInstance(ownership, dict)
        self.assertGreater(len(ownership), 0)

        # Check specific files
        for file_path in ["file1.py", "file2.py", "file3.py", "file4.py"]:
            if file_path in ownership:
                file_metrics = ownership[file_path]
                self.assertIn("author_count", file_metrics)
                self.assertIn("has_multiple_authors", file_metrics)

    def test_get_repository_summary(self):
        """Test get_repository_summary method."""
        # Get repository summary
        summary = self.repo_analyzer.get_repository_summary()

        # Check results
        self.assertIsInstance(summary, dict)
        self.assertIn("name", summary)
        self.assertIn("total_files", summary)
        self.assertIn("total_commits", summary)
        self.assertIn("bug_fixing_commits", summary)
        self.assertIn("unique_authors", summary)
        self.assertIn("top_file_extensions", summary)
        self.assertIn("top_authors", summary)
        self.assertIn("bug_fix_ratio", summary)
        self.assertIn("avg_bug_fixes_per_file", summary)
        self.assertIn("analyzed_at", summary)

        # Check values
        self.assertEqual(summary["total_files"], 4)
        self.assertEqual(summary["total_commits"], 7)
        self.assertEqual(summary["bug_fixing_commits"], 4)  # 4 commits with 'fix' in message

    def test_get_bug_fix_distribution(self):
        """Test get_bug_fix_distribution method."""
        # Get bug fix distribution
        distribution = self.repo_analyzer.get_bug_fix_distribution()

        # Check results
        self.assertIsInstance(distribution, dict)
        self.assertGreater(len(distribution), 0)

        # Check specific extension
        self.assertIn(".py", distribution)
        self.assertGreater(distribution[".py"], 0)

    def test_get_temporal_bug_distribution(self):
        """Test get_temporal_bug_distribution method."""
        # Get temporal distribution
        distribution = self.repo_analyzer.get_temporal_bug_distribution()

        # Check results
        self.assertIsInstance(distribution, dict)
        self.assertIn("period_type", distribution)
        self.assertIn("labels", distribution)
        self.assertIn("counts", distribution)

        # Check that we have data
        self.assertGreater(len(distribution["labels"]), 0)
        self.assertEqual(len(distribution["labels"]), len(distribution["counts"]))

        # Check that some period has bug fixes
        self.assertGreater(sum(distribution["counts"]), 0)


class TestRepositoryErrorHandling(unittest.TestCase):
    """Test error handling in repository analysis."""

    def test_non_git_repository(self):
        """Test analysis of a non-git repository."""
        with temp_directory() as temp_dir:
            # Create a directory that is not a git repository
            non_git_dir = os.path.join(temp_dir, "not_a_git_repo")
            os.makedirs(non_git_dir)

            # Create analyzer
            repo_analyzer = RepositoryAnalyzer(non_git_dir)

            # Run analysis
            success = repo_analyzer.analyze()

            # Check analysis results
            self.assertFalse(success)
            self.assertFalse(repo_analyzer.is_analyzed)
            self.assertGreater(len(repo_analyzer.error_messages), 0)

    def test_repository_with_no_commits(self):
        """Test analysis of a repository with no commits."""
        with temp_directory() as temp_dir:
            # Create an empty git repository (just initialize, no commits)
            empty_repo = os.path.join(temp_dir, "empty_repo")
            os.makedirs(empty_repo)

            # Initialize git repository without commits
            import subprocess
            subprocess.run(["git", "init"], cwd=empty_repo, check=True, capture_output=True)

            # Create analyzer
            repo_analyzer = RepositoryAnalyzer(empty_repo)

            # Run analysis
            success = repo_analyzer.analyze()

            # Check analysis results - should succeed but with empty results
            self.assertTrue(success)
            self.assertTrue(repo_analyzer.is_analyzed)
            self.assertEqual(repo_analyzer.total_files, 0)
            self.assertEqual(len(repo_analyzer.commit_history), 0)
            self.assertEqual(len(repo_analyzer.bug_fixes), 0)

    def test_repository_with_no_code_files(self):
        """Test analysis of a repository with no code files."""
        with temp_directory() as temp_dir:
            # Create a git repository with only non-code files
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "image.png": ["binary_content"],
                    "document.pdf": ["binary_content"],
                    "data.bin": ["binary_content"]
                },
                commits=[
                    {"message": "Initial commit",
                     "file_changes": {"image.png": 0, "document.pdf": 0, "data.bin": 0}}
                ]
            )

            # Create analyzer
            repo_analyzer = RepositoryAnalyzer(repo_path)

            # Run analysis
            success = repo_analyzer.analyze()

            # Check analysis results
            self.assertTrue(success)
            self.assertTrue(repo_analyzer.is_analyzed)
            self.assertEqual(repo_analyzer.total_files, 0)  # No code files
            self.assertEqual(len(repo_analyzer.commit_history), 1)
            self.assertEqual(len(repo_analyzer.bug_fixes), 0)

    def test_error_in_git_command(self):
        """Test error handling when git command fails."""
        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Create analyzer
            repo_analyzer = RepositoryAnalyzer(repo_path)

            # Mock _run_git_command to raise an exception
            with mock.patch.object(repo_analyzer, '_run_git_command', side_effect=Exception("Git command failed")):
                # Run analysis
                success = repo_analyzer.analyze()

                # Check analysis results
                self.assertFalse(success)
                self.assertFalse(repo_analyzer.is_analyzed)
                self.assertGreater(len(repo_analyzer.error_messages), 0)
                self.assertIn("Git command failed", repo_analyzer.error_messages[0])


class TestRepositoryEdgeCases(unittest.TestCase):
    """Test repository analysis edge cases."""

    def test_repository_with_unicode_characters(self):
        """Test analysis of a repository with unicode characters in file names and commit messages."""
        with temp_directory() as temp_dir:
            # Create a git repository with unicode characters
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "ünicode_filé.py": ["print('Hello Unicode')"],
                    "emoji_🔥.py": ["print('Hello Emoji')"],
                    "normal.py": ["print('Hello Normal')"]
                },
                commits=[
                    {"message": "Initial commit with ünicode",
                     "file_changes": {"ünicode_filé.py": 0, "emoji_🔥.py": 0, "normal.py": 0}},
                    {"message": "Fix 🐛 in ünicode_filé.py",
                     "file_changes": {"ünicode_filé.py": 0}}
                ]
            )

            # Create analyzer
            repo_analyzer = RepositoryAnalyzer(repo_path)

            # Run analysis
            success = repo_analyzer.analyze()

            # Check analysis results
            self.assertTrue(success)
            self.assertTrue(repo_analyzer.is_analyzed)

            # Count code files (might be fewer than 3 if filesystem doesn't support unicode names)
            code_files = sum(1 for f in ["ünicode_filé.py", "emoji_🔥.py", "normal.py"]
                             if os.path.exists(os.path.join(repo_path, f)))

            self.assertEqual(repo_analyzer.total_files, code_files)
            self.assertEqual(len(repo_analyzer.commit_history), 2)
            self.assertEqual(len(repo_analyzer.bug_fixes), 1)  # One bug fix commit

    def test_repository_with_large_history(self):
        """Test analysis of a repository with many commits (simulated)."""
        with temp_directory() as temp_dir:
            # Create a git repository with a single file
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Create analyzer with lookback limit
            repo_analyzer = RepositoryAnalyzer(repo_path, lookback_commits=10)

            # Mock a large number of commits
            with mock.patch.object(repo_analyzer, '_run_git_command') as mock_git:
                # Generate 20 fake commit lines
                commit_lines = []
                for i in range(20):
                    timestamp = int(datetime.datetime.now().timestamp()) - i * 3600
                    commit_lines.append(f"commit{i}|author{i}|{timestamp}|Commit message {i}")

                mock_git.return_value = "\n".join(commit_lines)

                # Call _get_all_commits directly
                repo_analyzer._get_all_commits()

                # Should only get the last 10 commits due to lookback limit
                self.assertEqual(len(repo_analyzer.commit_history), 10)

    def test_analysis_with_custom_bug_patterns(self):
        """Test analysis with custom bug keywords."""
        with temp_directory() as temp_dir:
            # Create a git repository with custom bug keywords
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "file1.py": ["v1", "v2"],
                    "file2.py": ["v1", "v2"],
                    "file3.py": ["v1", "v2"]
                },
                commits=[
                    {"message": "Initial commit",
                     "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0}},
                    {"message": "Repair file1",
                     "file_changes": {"file1.py": 1}},
                    {"message": "Solve problem in file2",
                     "file_changes": {"file2.py": 1}},
                    {"message": "Standard update to file3",
                     "file_changes": {"file3.py": 1}}
                ]
            )

            # Create analyzer with custom keywords
            repo_analyzer = RepositoryAnalyzer(
                repo_path,
                bug_keywords=["repair", "solve", "problem"]
            )

            # Run analysis
            success = repo_analyzer.analyze()

            # Check analysis results
            self.assertTrue(success)
            self.assertEqual(len(repo_analyzer.bug_fixes), 2)  # Two bug fix commits

            # Check file stats
            self.assertEqual(repo_analyzer.file_stats["file1.py"]["bug_fixes"], 1)
            self.assertEqual(repo_analyzer.file_stats["file2.py"]["bug_fixes"], 1)
            self.assertEqual(repo_analyzer.file_stats["file3.py"]["bug_fixes"], 0)


if __name__ == '__main__':
    unittest.main()