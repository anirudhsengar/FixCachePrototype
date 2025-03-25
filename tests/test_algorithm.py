#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit Tests for FixCache Algorithm

Tests the core FixCache algorithm functionality including cache prediction,
replacement policies, and other key features.

Author: anirudhsengar
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest import mock
from typing import Dict, List, Any, Optional

# Import from tests package
from tests import temp_directory, capture_output, create_mock_git_repo

# Import FixCache components
from fixcache.algorithm import FixCache
from fixcache.repository import RepositoryAnalyzer
from fixcache.utils import is_code_file


class TestFixCacheInitialization(unittest.TestCase):
    """Test FixCache initialization and parameter validation."""

    def test_init_with_path(self):
        """Test initialization with a repository path."""
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

            # Create FixCache instance
            fix_cache = FixCache(repo_path=repo_path)

            # Check initialization
            self.assertEqual(fix_cache.repo_analyzer.repo_path, repo_path)
            self.assertEqual(fix_cache.cache_size, 0.2)  # Default value
            self.assertEqual(fix_cache.policy, "BUG")  # Default value
            self.assertEqual(fix_cache.cache, [])
            self.assertEqual(fix_cache.hit_count, 0)
            self.assertEqual(fix_cache.miss_count, 0)

    def test_init_with_repository_analyzer(self):
        """Test initialization with a RepositoryAnalyzer instance."""
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

            # Create FixCache instance with repository analyzer
            fix_cache = FixCache(repo_path=repo_analyzer)

            # Check initialization
            self.assertEqual(fix_cache.repo_analyzer, repo_analyzer)
            self.assertEqual(fix_cache.cache_size, 0.2)  # Default value
            self.assertEqual(fix_cache.policy, "BUG")  # Default value

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
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

            # Create FixCache instance with custom parameters
            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=0.3,
                policy="FIFO",
                bug_keywords=["error", "crash"],
                lookback_commits=100,
                min_file_count=5,
                window_ratio=0.3,
                cache_seeding=False
            )

            # Check initialization
            self.assertEqual(fix_cache.repo_analyzer.repo_path, repo_path)
            self.assertEqual(fix_cache.cache_size, 0.3)
            self.assertEqual(fix_cache.policy, "FIFO")
            self.assertEqual(fix_cache.min_file_count, 5)
            self.assertEqual(fix_cache.window_ratio, 0.3)
            self.assertEqual(fix_cache.cache_seeding, False)
            self.assertEqual(fix_cache.repo_analyzer.lookback_commits, 100)
            self.assertIn("error", fix_cache.repo_analyzer.bug_keywords)
            self.assertIn("crash", fix_cache.repo_analyzer.bug_keywords)

    def test_invalid_policy(self):
        """Test initialization with invalid policy."""
        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Check that invalid policy raises ValueError
            with self.assertRaises(ValueError):
                FixCache(repo_path=repo_path, policy="INVALID")


class TestFixCacheRepository(unittest.TestCase):
    """Test FixCache repository analysis."""

    def test_analyze_repository(self):
        """Test repository analysis."""
        with temp_directory() as temp_dir:
            # Create a git repository with some bug fixes
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "file1.py": ["def foo():\n    return 'bar'",
                                 "def foo():\n    return 'bar'  # Fixed"],
                    "file2.py": ["print('Hello')",
                                 "print('Hello World')"],
                    "file3.py": ["import os\n\ndef test():\n    pass"]
                },
                commits=[
                    {"message": "Initial commit",
                     "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0}},
                    {"message": "Fix bug in file1",
                     "file_changes": {"file1.py": 1}},
                    {"message": "Update file2",
                     "file_changes": {"file2.py": 1}}
                ]
            )

            # Create FixCache instance
            fix_cache = FixCache(repo_path=repo_path)

            # Run repository analysis
            success = fix_cache.analyze_repository()

            # Check analysis results
            self.assertTrue(success)
            self.assertTrue(fix_cache.repo_analyzer.is_analyzed)
            self.assertEqual(fix_cache.repo_analyzer.total_files, 3)
            self.assertEqual(len(fix_cache.repo_analyzer.commit_history), 3)
            self.assertEqual(len(fix_cache.repo_analyzer.bug_fixes), 1)  # Only one bug fix commit

            # Check that cache size was calculated
            self.assertEqual(fix_cache.cache_max_size, 1)  # 20% of 3 files, rounded up to 1

    def test_analyze_repository_with_no_files(self):
        """Test repository analysis with no files."""
        with temp_directory() as temp_dir:
            # Create an empty git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={},
                commits=[{"message": "Initial commit", "file_changes": {}}]
            )

            # Create FixCache instance with lower min_file_count
            fix_cache = FixCache(repo_path=repo_path, min_file_count=0)

            # Run repository analysis
            success = fix_cache.analyze_repository()

            # Check analysis results
            self.assertTrue(success)
            self.assertTrue(fix_cache.repo_analyzer.is_analyzed)
            self.assertEqual(fix_cache.repo_analyzer.total_files, 0)

    def test_analyze_repository_with_too_few_files(self):
        """Test repository analysis with too few files."""
        with temp_directory() as temp_dir:
            # Create a git repository with one file
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Create FixCache instance with higher min_file_count
            fix_cache = FixCache(repo_path=repo_path, min_file_count=5)

            # Run repository analysis
            success = fix_cache.analyze_repository()

            # Check analysis results
            self.assertFalse(success)
            self.assertIn("less than the required minimum", fix_cache.error_messages[0])


class TestFixCachePrediction(unittest.TestCase):
    """Test FixCache prediction algorithm."""

    def setUp(self):
        """Set up a more complex repository for prediction tests."""
        self.temp_dir = tempfile.mkdtemp(prefix="fixcache_test_")

        # Create a git repository with multiple bug fixes
        self.repo_path = create_mock_git_repo(
            self.temp_dir,
            files={
                "file1.py": ["def func1():\n    return 1",  # v0
                             "def func1():\n    return True",  # v1 - bug
                             "def func1():\n    return 1"],  # v2 - fix
                "file2.py": ["def func2():\n    return 2",  # v0
                             "def func2():\n    return None",  # v1 - bug
                             "def func2():\n    return 2"],  # v2 - fix
                "file3.py": ["def func3():\n    return 3",  # v0
                             "def func3():\n    raise Exception()",  # v1 - bug
                             "def func3():\n    return 3"],  # v2 - fix
                "file4.py": ["def func4():\n    return 4",  # v0
                             "def func4():\n    return '4'"],  # v1 - no bug
                "utils.py": ["def util():\n    return 'util'"]  # unchanged
            },
            commits=[
                {"message": "Initial commit",
                 "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0,
                                  "file4.py": 0, "utils.py": 0}},
                {"message": "Change file1",
                 "file_changes": {"file1.py": 1}},
                {"message": "Change file2",
                 "file_changes": {"file2.py": 1}},
                {"message": "Change file3",
                 "file_changes": {"file3.py": 1}},
                {"message": "Change file4",
                 "file_changes": {"file4.py": 1}},
                {"message": "Fix bug in file1",
                 "file_changes": {"file1.py": 2}},
                {"message": "Fix bug in file2",
                 "file_changes": {"file2.py": 2}},
                {"message": "Fix bug in file3",
                 "file_changes": {"file3.py": 2}}
            ]
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predict_with_bug_policy(self):
        """Test prediction with BUG policy."""
        # Create FixCache instance
        fix_cache = FixCache(
            repo_path=self.repo_path,
            cache_size=0.4,  # 40% of files -> 2 files in cache
            policy="BUG",
            window_ratio=0.5  # Use half of commits for training
        )

        # Run analysis and prediction
        fix_cache.analyze_repository()
        hit_rate = fix_cache.predict()

        # Check prediction results
        self.assertGreater(hit_rate, 0)
        self.assertTrue(0 <= hit_rate <= 100)
        self.assertGreater(fix_cache.hit_count + fix_cache.miss_count, 0)

        # Check cache contents
        self.assertEqual(len(fix_cache.cache), 2)  # 40% of 5 files = 2

        # Check results
        self.assertIn('hit_rate', fix_cache.results)
        self.assertIn('top_files', fix_cache.results)

    def test_predict_with_fifo_policy(self):
        """Test prediction with FIFO policy."""
        # Create FixCache instance
        fix_cache = FixCache(
            repo_path=self.repo_path,
            cache_size=0.4,  # 40% of files -> 2 files in cache
            policy="FIFO",
            window_ratio=0.5  # Use half of commits for training
        )

        # Run analysis and prediction
        fix_cache.analyze_repository()
        hit_rate = fix_cache.predict()

        # Check prediction results
        self.assertGreaterEqual(hit_rate, 0)
        self.assertLessEqual(hit_rate, 100)

    def test_predict_with_lru_policy(self):
        """Test prediction with LRU policy."""
        # Create FixCache instance
        fix_cache = FixCache(
            repo_path=self.repo_path,
            cache_size=0.4,  # 40% of files -> 2 files in cache
            policy="LRU",
            window_ratio=0.5  # Use half of commits for training
        )

        # Run analysis and prediction
        fix_cache.analyze_repository()
        hit_rate = fix_cache.predict()

        # Check prediction results
        self.assertGreaterEqual(hit_rate, 0)
        self.assertLessEqual(hit_rate, 100)

    def test_predict_with_no_bug_fixes(self):
        """Test prediction with no bug fixes."""
        # Create a repository with no bug fixes
        no_bugs_repo = create_mock_git_repo(
            self.temp_dir,
            files={"file1.py": ["print('Hello')"], "file2.py": ["print('World')"]},
            commits=[
                {"message": "Initial commit",
                 "file_changes": {"file1.py": 0, "file2.py": 0}},
                {"message": "Update file1",
                 "file_changes": {"file1.py": 0}}
            ]
        )

        # Create FixCache instance
        fix_cache = FixCache(repo_path=no_bugs_repo)

        # Run analysis and prediction
        fix_cache.analyze_repository()
        hit_rate = fix_cache.predict()

        # Check prediction results
        self.assertEqual(hit_rate, 0.0)
        self.assertEqual(fix_cache.hit_count, 0)
        self.assertEqual(fix_cache.miss_count, 0)
        self.assertIn('No bug fixes found', fix_cache.error_messages[0])


class TestFixCacheHelperMethods(unittest.TestCase):
    """Test FixCache helper methods."""

    def setUp(self):
        """Set up a repository for helper method tests."""
        self.temp_dir = tempfile.mkdtemp(prefix="fixcache_test_")

        # Create a git repository
        self.repo_path = create_mock_git_repo(
            self.temp_dir,
            files={
                "file1.py": ["def func1():\n    return 1",  # v0
                             "def func1():\n    return True"],  # v1
                "file2.py": ["def func2():\n    return 2"],  # v0
                "file3.py": ["def func3():\n    return 3"],  # v0
                "file4.py": ["def func4():\n    return 4"],  # v0
                "utils.py": ["def util():\n    return 'util'"]  # v0
            },
            commits=[
                {"message": "Initial commit",
                 "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0,
                                  "file4.py": 0, "utils.py": 0}},
                {"message": "Fix bug in file1",
                 "file_changes": {"file1.py": 1}}
            ]
        )

        # Create FixCache instance
        self.fix_cache = FixCache(
            repo_path=self.repo_path,
            cache_size=0.4  # 40% of files -> 2 files in cache
        )

        # Run analysis
        self.fix_cache.analyze_repository()
        self.fix_cache.predict()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_top_files(self):
        """Test get_top_files method."""
        top_files = self.fix_cache.get_top_files(3)

        # Check top files
        self.assertLessEqual(len(top_files), 3)
        self.assertGreater(len(top_files), 0)

        # Check file format
        first_file = top_files[0]
        self.assertIn('file_path', first_file)
        self.assertIn('bug_fixes', first_file)
        self.assertIn('file_type', first_file)

    def test_get_bottom_files(self):
        """Test get_bottom_files method."""
        bottom_files = self.fix_cache.get_bottom_files(3)

        # Check bottom files
        self.assertLessEqual(len(bottom_files), 3)

        # If we have files not in cache, check their format
        if bottom_files:
            first_file = bottom_files[0]
            self.assertIn('file_path', first_file)
            self.assertIn('bug_fixes', first_file)
            self.assertIn('file_type', first_file)

    def test_get_recommended_actions(self):
        """Test get_recommended_actions method."""
        recommendations = self.fix_cache.get_recommended_actions(3)

        # Check recommendations
        self.assertLessEqual(len(recommendations), 3)

        # If we have recommendations, check their format
        if recommendations:
            first_rec = recommendations[0]
            self.assertIn('type', first_rec)
            self.assertIn('message', first_rec)
            self.assertIn('priority', first_rec)

    def test_get_summary(self):
        """Test get_summary method."""
        summary = self.fix_cache.get_summary()

        # Check summary fields
        self.assertIn('repository', summary)
        self.assertIn('total_files', summary)
        self.assertIn('bug_fixing_commits', summary)
        self.assertIn('cache_size', summary)
        self.assertIn('hit_rate', summary)
        self.assertIn('policy', summary)

    def test_save_results(self):
        """Test save_results method."""
        with temp_directory() as output_dir:
            output_file = os.path.join(output_dir, "results.json")

            # Save results
            self.fix_cache.save_results(output_file)

            # Check that file was created
            self.assertTrue(os.path.exists(output_file))

            # Check file contents
            with open(output_file, 'r') as f:
                results = json.load(f)

            self.assertIn('hit_rate', results)
            self.assertIn('top_files', results)
            self.assertIn('policy', results)

    @mock.patch('fixcache.visualization.visualize_results')
    def test_visualize_results(self, mock_visualize):
        """Test visualize_results method."""
        # Mock successful visualization
        mock_visualize.return_value = True

        # Call visualize_results
        self.fix_cache.visualize_results("output.png")

        # Check that visualization function was called
        mock_visualize.assert_called_once()
        self.assertEqual(mock_visualize.call_args[0][0], self.fix_cache.results)
        self.assertEqual(mock_visualize.call_args[0][1], "output.png")


class TestFixCacheCacheBehavior(unittest.TestCase):
    """Test FixCache cache behavior."""

    def test_bug_policy_eviction(self):
        """Test cache eviction with BUG policy."""
        with temp_directory() as temp_dir:
            # Create a git repository with different bug frequencies
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "file1.py": ["v0", "v1", "v2"],  # 2 changes
                    "file2.py": ["v0", "v1"],  # 1 change
                    "file3.py": ["v0"],  # 0 changes
                    "file4.py": ["v0", "v1", "v2"],  # 2 changes
                    "file5.py": ["v0", "v1"]  # 1 change
                },
                commits=[
                    {"message": "Initial commit",
                     "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0,
                                      "file4.py": 0, "file5.py": 0}},
                    {"message": "Fix bug in file1",
                     "file_changes": {"file1.py": 1}},
                    {"message": "Fix bug in file2",
                     "file_changes": {"file2.py": 1}},
                    {"message": "Fix bug in file4",
                     "file_changes": {"file4.py": 1}},
                    {"message": "Fix bug in file5",
                     "file_changes": {"file5.py": 1}},
                    {"message": "Fix another bug in file1",
                     "file_changes": {"file1.py": 2}},
                    {"message": "Fix another bug in file4",
                     "file_changes": {"file4.py": 2}}
                ]
            )

            # Create FixCache instance with small cache
            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=0.2,  # 20% of files -> 1 file in cache
                policy="BUG",
                cache_seeding=False  # Disable seeding for this test
            )

            # We'll modify the cache and test eviction manually
            fix_cache.analyze_repository()

            # Setup for manual eviction test
            fix_cache.cache = ["file3.py"]  # Least bug-prone file
            fix_cache.cache_max_size = 1

            # Add a new file, which should trigger eviction
            fix_cache._process_commit_for_prediction({
                'files_changed': ["file1.py"]  # Most bug-prone file
            })

            # With BUG policy, file3.py should be evicted (fewest bug fixes)
            self.assertNotIn("file3.py", fix_cache.cache)
            self.assertIn("file1.py", fix_cache.cache)

    def test_fifo_policy_eviction(self):
        """Test cache eviction with FIFO policy."""
        with temp_directory() as temp_dir:
            # Create a git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "file1.py": ["v0"],
                    "file2.py": ["v0"],
                    "file3.py": ["v0"]
                },
                commits=[
                    {"message": "Initial commit",
                     "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0}}
                ]
            )

            # Create FixCache instance with small cache
            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=0.33,  # 33% of files -> 1 file in cache
                policy="FIFO"
            )

            # We'll modify the cache and test eviction manually
            fix_cache.analyze_repository()

            # Setup for manual eviction test
            fix_cache.cache = ["file1.py"]  # Oldest file in cache
            fix_cache.cache_max_size = 1

            # Add a new file, which should trigger eviction
            fix_cache._process_commit_for_prediction({
                'files_changed': ["file2.py"]
            })

            # With FIFO policy, file1.py should be evicted (first in)
            self.assertNotIn("file1.py", fix_cache.cache)
            self.assertIn("file2.py", fix_cache.cache)

    def test_lru_policy_eviction(self):
        """Test cache eviction with LRU policy."""
        with temp_directory() as temp_dir:
            # Create a git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={
                    "file1.py": ["v0"],
                    "file2.py": ["v0"],
                    "file3.py": ["v0"]
                },
                commits=[
                    {"message": "Initial commit",
                     "file_changes": {"file1.py": 0, "file2.py": 0, "file3.py": 0}}
                ]
            )

            # Create FixCache instance with small cache
            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=0.66,  # 66% of files -> 2 files in cache
                policy="LRU"
            )

            # We'll modify the cache and test eviction manually
            fix_cache.analyze_repository()

            # Setup for manual eviction test
            fix_cache.cache = ["file1.py", "file2.py"]  # file1.py is least recently used
            fix_cache.cache_max_size = 2

            # Access file2.py to make file1.py the least recently used
            fix_cache._process_commit_for_prediction({
                'files_changed': ["file2.py"]
            })

            # Now add a new file, which should trigger eviction
            fix_cache._process_commit_for_prediction({
                'files_changed': ["file3.py"]
            })

            # With LRU policy, file1.py should be evicted (least recently used)
            self.assertNotIn("file1.py", fix_cache.cache)
            self.assertIn("file2.py", fix_cache.cache)
            self.assertIn("file3.py", fix_cache.cache)


class TestFixCacheEdgeCases(unittest.TestCase):
    """Test FixCache edge cases and error handling."""

    def test_cache_size_zero(self):
        """Test with cache size of zero."""
        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Create FixCache instance with zero cache size
            with self.assertRaises(ValueError):
                FixCache(repo_path=repo_path, cache_size=0)

    def test_cache_size_too_large(self):
        """Test with cache size greater than 1."""
        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Create FixCache instance with cache size > 1
            with self.assertRaises(ValueError):
                FixCache(repo_path=repo_path, cache_size=1.5)

    def test_repository_not_analyzed(self):
        """Test prediction without prior analysis."""
        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Create FixCache instance
            fix_cache = FixCache(repo_path=repo_path)

            # Try prediction without analysis
            with mock.patch.object(fix_cache, 'analyze_repository', return_value=False):
                hit_rate = fix_cache.predict()

                # Should return 0.0 if analysis fails
                self.assertEqual(hit_rate, 0.0)

    def test_invalid_repository_path(self):
        """Test with invalid repository path."""
        # Create FixCache instance with invalid path
        fix_cache = FixCache(repo_path="/path/that/does/not/exist")

        # Analysis should fail
        success = fix_cache.analyze_repository()
        self.assertFalse(success)
        self.assertGreater(len(fix_cache.error_messages), 0)

    def test_non_git_repository(self):
        """Test with a directory that is not a git repository."""
        with temp_directory() as temp_dir:
            # Create a non-git directory
            os.makedirs(os.path.join(temp_dir, "not_a_git_repo"))

            # Create FixCache instance with non-git path
            fix_cache = FixCache(repo_path=os.path.join(temp_dir, "not_a_git_repo"))

            # Analysis should fail
            success = fix_cache.analyze_repository()
            self.assertFalse(success)
            self.assertGreater(len(fix_cache.error_messages), 0)


class TestFixCacheFactory(unittest.TestCase):
    """Test FixCache factory function."""

    def test_create_fixcache(self):
        """Test create_fixcache factory function."""
        from fixcache.algorithm import create_fixcache

        with temp_directory() as temp_dir:
            # Create a simple git repository
            repo_path = create_mock_git_repo(
                temp_dir,
                files={"file1.py": ["print('Hello World')"]},
                commits=[{"message": "Initial commit", "file_changes": {"file1.py": 0}}]
            )

            # Use factory function to create FixCache instance
            with mock.patch('fixcache.algorithm.FixCache.analyze_repository', return_value=True):
                fix_cache = create_fixcache(
                    repo_path=repo_path,
                    cache_size=0.25,
                    policy="FIFO",
                    bug_keywords=["bug", "fix"]
                )

                # Check instance properties
                self.assertEqual(fix_cache.repo_path, repo_path)
                self.assertEqual(fix_cache.cache_size, 0.25)
                self.assertEqual(fix_cache.policy, "FIFO")
                self.assertIn("bug", fix_cache.repo_analyzer.bug_keywords)
                self.assertIn("fix", fix_cache.repo_analyzer.bug_keywords)


if __name__ == '__main__':
    unittest.main()