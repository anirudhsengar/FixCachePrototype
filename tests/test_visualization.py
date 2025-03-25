#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit Tests for FixCache Visualization Module

Tests the visualization capabilities of FixCache including charts,
plots, and visual representation of results.

Author: anirudhsengar
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import datetime
from unittest import mock
from typing import Dict, List, Any, Optional

# Import from tests package
from tests import temp_directory, capture_output

# Try to import matplotlib for testing
try:
    import matplotlib

    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import FixCache visualization functions
from fixcache.visualization import (
    visualize_results,
    plot_cache_optimization,
    plot_repository_comparison,
    _create_hit_rate_gauge,
    _plot_top_files,
    _plot_bug_distribution_by_type,
    _plot_bug_timeline,
    _plot_file_complexity,
    _plot_hit_miss_pie
)


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib not available")
class TestVisualizationBasics(unittest.TestCase):
    """Test basic visualization functionality."""

    def setUp(self):
        """Set up test data."""
        # Sample results dictionary
        self.sample_results = {
            "repo_path": "/path/to/repo",
            "total_files": 100,
            "total_bug_fixes": 25,
            "cache_size": 0.2,
            "policy": "BUG",
            "hit_rate": 75.0,
            "hit_count": 15,
            "miss_count": 5,
            "timestamp": datetime.datetime.now().isoformat(),
            "top_files": [
                {"file_path": "file1.py", "bug_fixes": 5, "last_modified": 1614556800,
                 "file_type": "Python source code"},
                {"file_path": "file2.cpp", "bug_fixes": 4, "last_modified": 1614470400, "file_type": "C++ source code"},
                {"file_path": "file3.java", "bug_fixes": 3, "last_modified": 1614384000,
                 "file_type": "Java source code"},
                {"file_path": "file4.js", "bug_fixes": 2, "last_modified": 1614297600,
                 "file_type": "JavaScript source code"},
                {"file_path": "file5.py", "bug_fixes": 1, "last_modified": 1614211200,
                 "file_type": "Python source code"}
            ]
        }

        # Sample optimization results
        self.optimization_results = {
            0.1: 60.0,
            0.15: 65.0,
            0.2: 75.0,
            0.25: 72.0,
            0.3: 70.0
        }

        # Sample repository comparison results
        self.comparison_results = {
            "/path/to/repo1": {
                "repo_path": "/path/to/repo1",
                "total_files": 100,
                "total_bug_fixes": 25,
                "hit_rate": 75.0
            },
            "/path/to/repo2": {
                "repo_path": "/path/to/repo2",
                "total_files": 150,
                "total_bug_fixes": 30,
                "hit_rate": 65.0
            },
            "/path/to/repo3": {
                "repo_path": "/path/to/repo3",
                "total_files": 80,
                "total_bug_fixes": 15,
                "hit_rate": 80.0
            }
        }

    def test_visualize_results_creates_file(self):
        """Test that visualize_results creates a file."""
        with temp_directory() as temp_dir:
            output_file = os.path.join(temp_dir, "results.png")

            # Call visualize_results
            result = visualize_results(self.sample_results, output_file)

            # Check result
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)

    def test_plot_cache_optimization_creates_file(self):
        """Test that plot_cache_optimization creates a file."""
        with temp_directory() as temp_dir:
            output_file = os.path.join(temp_dir, "optimization.png")

            # Call plot_cache_optimization
            result = plot_cache_optimization(self.optimization_results, output_file)

            # Check result
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)

    def test_plot_repository_comparison_creates_file(self):
        """Test that plot_repository_comparison creates a file."""
        with temp_directory() as temp_dir:
            output_file = os.path.join(temp_dir, "comparison.png")

            # Call plot_repository_comparison
            result = plot_repository_comparison(self.comparison_results, output_file)

            # Check result
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib not available")
class TestVisualizationComponents(unittest.TestCase):
    """Test individual visualization components."""

    def setUp(self):
        """Set up test data and figure."""
        # Create a figure and axis for testing
        self.fig = plt.figure(figsize=(10, 6))

        # Sample data for plots
        self.hit_rate = 75.0

        self.top_files = [
            {"file_path": "file1.py", "bug_fixes": 5, "last_modified": 1614556800, "file_type": "Python source code"},
            {"file_path": "file2.cpp", "bug_fixes": 4, "last_modified": 1614470400, "file_type": "C++ source code"},
            {"file_path": "file3.java", "bug_fixes": 3, "last_modified": 1614384000, "file_type": "Java source code"}
        ]

        self.results = {
            "hit_rate": 75.0,
            "hit_count": 15,
            "miss_count": 5,
            "temporal_distribution": {
                "period_type": "month",
                "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
                "counts": [2, 4, 3, 5, 1]
            }
        }

    def tearDown(self):
        """Clean up figure."""
        plt.close(self.fig)

    def test_create_hit_rate_gauge(self):
        """Test _create_hit_rate_gauge function."""
        # Create axis
        ax = self.fig.add_subplot(111)

        # Call _create_hit_rate_gauge
        _create_hit_rate_gauge(ax, self.hit_rate)

        # Check that axis has been populated
        self.assertGreater(len(ax.get_children()), 0)

        # Check axis properties
        self.assertEqual(ax.get_xlim(), (-1.1, 1.1))
        self.assertEqual(ax.get_ylim(), (-1.1, 1.1))

    def test_plot_top_files(self):
        """Test _plot_top_files function."""
        # Create axis
        ax = self.fig.add_subplot(111)

        # Call _plot_top_files
        _plot_top_files(ax, self.top_files)

        # Check that axis has been populated
        self.assertGreater(len(ax.get_children()), 0)

        # Check that y-labels include file names
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        for file_info in self.top_files:
            file_name = os.path.basename(file_info["file_path"])
            file_found = any(file_name in label for label in y_labels)
            self.assertTrue(file_found, f"File name {file_name} not found in plot y-labels")

    def test_plot_bug_distribution_by_type(self):
        """Test _plot_bug_distribution_by_type function."""
        # Create axis
        ax = self.fig.add_subplot(111)

        # Call _plot_bug_distribution_by_type
        _plot_bug_distribution_by_type(ax, self.top_files)

        # Check that axis has been populated
        self.assertGreater(len(ax.get_children()), 0)

        # Check that legend has been created
        self.assertIsNotNone(ax.get_legend())

    def test_plot_bug_timeline(self):
        """Test _plot_bug_timeline function."""
        # Create axis
        ax = self.fig.add_subplot(111)

        # Call _plot_bug_timeline
        _plot_bug_timeline(ax, self.results)

        # Check that axis has been populated
        self.assertGreater(len(ax.get_children()), 0)

        # Check axis labels
        self.assertEqual(ax.get_xlabel(), "Time (Months)")
        self.assertEqual(ax.get_ylabel(), "Bug Fixes")

    def test_plot_file_complexity(self):
        """Test _plot_file_complexity function."""
        # Create axis
        ax = self.fig.add_subplot(111)

        # Call _plot_file_complexity
        _plot_file_complexity(ax, self.top_files)

        # Check that axis has been populated
        self.assertGreater(len(ax.get_children()), 0)

        # Check axis labels
        self.assertEqual(ax.get_xlabel(), "Files Grouped by Type")
        self.assertEqual(ax.get_ylabel(), "Number of Bug Fixes")

    def test_plot_hit_miss_pie(self):
        """Test _plot_hit_miss_pie function."""
        # Create axis
        ax = self.fig.add_subplot(111)

        # Call _plot_hit_miss_pie
        _plot_hit_miss_pie(ax, self.results)

        # Check that axis has been populated
        self.assertGreater(len(ax.get_children()), 0)

        # Check that legend has been created
        self.assertIsNotNone(ax.get_legend())


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib not available")
class TestVisualizationEdgeCases(unittest.TestCase):
    """Test edge cases for visualization."""

    def test_visualize_results_empty_data(self):
        """Test visualize_results with empty data."""
        with temp_directory() as temp_dir:
            output_file = os.path.join(temp_dir, "empty_results.png")

            # Empty results
            empty_results = {
                "repo_path": "/path/to/repo",
                "total_files": 0,
                "total_bug_fixes": 0,
                "cache_size": 0.2,
                "policy": "BUG",
                "hit_rate": 0.0,
                "hit_count": 0,
                "miss_count": 0,
                "top_files": []
            }

            # Call visualize_results
            result = visualize_results(empty_results, output_file)

            # Check result - should still succeed and create a file
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)

    def test_plot_cache_optimization_single_point(self):
        """Test plot_cache_optimization with a single data point."""
        with temp_directory() as temp_dir:
            output_file = os.path.join(temp_dir, "single_point.png")

            # Single data point
            single_point = {0.2: 75.0}

            # Call plot_cache_optimization
            result = plot_cache_optimization(single_point, output_file)

            # Check result - should still succeed and create a file
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)

    def test_plot_repository_comparison_one_repo(self):
        """Test plot_repository_comparison with a single repository."""
        with temp_directory() as temp_dir:
            output_file = os.path.join(temp_dir, "one_repo.png")

            # Single repository
            one_repo = {
                "/path/to/repo1": {
                    "repo_path": "/path/to/repo1",
                    "total_files": 100,
                    "total_bug_fixes": 25,
                    "hit_rate": 75.0
                }
            }

            # Call plot_repository_comparison
            result = plot_repository_comparison(one_repo, output_file)

            # Check result - should still succeed and create a file
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)

    def test_visualize_results_missing_fields(self):
        """Test visualize_results with missing fields."""
        with temp_directory() as temp_dir:
            output_file = os.path.join(temp_dir, "missing_fields.png")

            # Results with missing fields
            missing_fields = {
                "repo_path": "/path/to/repo",
                "hit_rate": 75.0  # Many fields missing
            }

            # Call visualize_results
            result = visualize_results(missing_fields, output_file)

            # Check result - should still succeed and create a file
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)


@unittest.skipIf(not MATPLOTLIB_AVAILABLE, "Matplotlib not available")
class TestVisualizationMocking(unittest.TestCase):
    """Test visualization with mocking."""

    def setUp(self):
        """Set up test data."""
        # Sample results dictionary
        self.sample_results = {
            "repo_path": "/path/to/repo",
            "total_files": 100,
            "total_bug_fixes": 25,
            "cache_size": 0.2,
            "policy": "BUG",
            "hit_rate": 75.0,
            "hit_count": 15,
            "miss_count": 5,
            "timestamp": datetime.datetime.now().isoformat(),
            "top_files": [
                {"file_path": "file1.py", "bug_fixes": 5, "last_modified": 1614556800,
                 "file_type": "Python source code"},
                {"file_path": "file2.cpp", "bug_fixes": 4, "last_modified": 1614470400, "file_type": "C++ source code"},
                {"file_path": "file3.java", "bug_fixes": 3, "last_modified": 1614384000,
                 "file_type": "Java source code"}
            ]
        }

    @mock.patch('matplotlib.pyplot.savefig')
    def test_visualize_results_calls_savefig(self, mock_savefig):
        """Test that visualize_results calls plt.savefig."""
        # Mock savefig to simply return success
        mock_savefig.return_value = None

        # Call visualize_results
        output_file = "mocked_results.png"
        result = visualize_results(self.sample_results, output_file)

        # Check that plt.savefig was called with the correct filename
        mock_savefig.assert_called_once()
        self.assertEqual(mock_savefig.call_args[0][0], output_file)

        # Check result
        self.assertTrue(result)

    @mock.patch('matplotlib.pyplot.figure')
    def test_visualize_results_with_show_plot(self, mock_figure):
        """Test visualize_results with show_plot=True."""
        # Mock the figure and figure.show
        mock_fig = mock.MagicMock()
        mock_figure.return_value = mock_fig

        # Call visualize_results with show_plot=True
        result = visualize_results(self.sample_results, "results.png", show_plot=True)

        # Check that figure was created
        mock_figure.assert_called_once()

        # Check result
        self.assertTrue(result)

    @mock.patch('matplotlib.pyplot.savefig', side_effect=Exception("Save failed"))
    def test_visualize_results_handles_errors(self, mock_savefig):
        """Test that visualize_results handles errors gracefully."""
        # Call visualize_results, which should catch the exception
        result = visualize_results(self.sample_results, "results.png")

        # Check that plt.savefig was called
        mock_savefig.assert_called_once()

        # Check result - should return False due to error
        self.assertFalse(result)


@unittest.skipIf(MATPLOTLIB_AVAILABLE, "Matplotlib is available")
class TestVisualizationWithoutMatplotlib(unittest.TestCase):
    """Test visualization functions when matplotlib is not available."""

    def setUp(self):
        """Set up test data."""
        # Sample results dictionary
        self.sample_results = {
            "repo_path": "/path/to/repo",
            "hit_rate": 75.0,
            "top_files": [{"file_path": "file1.py", "bug_fixes": 5}]
        }

        # Sample optimization results
        self.optimization_results = {0.2: 75.0}

        # Sample repository comparison results
        self.comparison_results = {
            "/path/to/repo1": {"hit_rate": 75.0},
            "/path/to/repo2": {"hit_rate": 65.0}
        }

    def test_visualize_results_no_matplotlib(self):
        """Test visualize_results without matplotlib."""
        # Mock the actual visualization module to simulate matplotlib not available
        with mock.patch.dict('sys.modules', {'matplotlib': None}):
            # Call visualize_results - should return False but not crash
            result = visualize_results(self.sample_results, "results.png")
            self.assertFalse(result)

    def test_plot_cache_optimization_no_matplotlib(self):
        """Test plot_cache_optimization without matplotlib."""
        # Mock the actual visualization module to simulate matplotlib not available
        with mock.patch.dict('sys.modules', {'matplotlib': None}):
            # Call plot_cache_optimization - should return False but not crash
            result = plot_cache_optimization(self.optimization_results, "optimization.png")
            self.assertFalse(result)

    def test_plot_repository_comparison_no_matplotlib(self):
        """Test plot_repository_comparison without matplotlib."""
        # Mock the actual visualization module to simulate matplotlib not available
        with mock.patch.dict('sys.modules', {'matplotlib': None}):
            # Call plot_repository_comparison - should return False but not crash
            result = plot_repository_comparison(self.comparison_results, "comparison.png")
            self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()