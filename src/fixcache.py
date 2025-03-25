#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache: A bug prediction algorithm based on historical bug fix patterns.

This module implements the FixCache algorithm described in the paper:
"Predicting Faults from Cached History" by Sunghun Kim, Thomas Zimmermann,
E. James Whitehead Jr., and Andreas Zeller.

The algorithm maintains a cache of files that are likely to contain bugs based on
historical bug fixes. When a bug fix occurs, the files involved are added to the cache.
If the cache is full, files are evicted based on a BUG policy.
"""

import os
import json
import re
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Any, Optional
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fixcache')


class FixCache:
    """Implementation of the FixCache algorithm for bug prediction."""

    def __init__(self, repo_path: str, cache_size: float = 0.2,
                 policy: str = "BUG", bug_keywords: List[str] = None):
        """
        Initialize the FixCache with a repository path and parameters.

        Args:
            repo_path: Path to the git repository
            cache_size: Size of the cache as a fraction of total files (0.1 = 10%)
            policy: Cache replacement policy (BUG, FIFO, or LRU)
            bug_keywords: List of keywords to identify bug-fixing commits
        """
        self.repo_path = os.path.abspath(repo_path)
        self.cache_size = cache_size
        self.policy = policy

        # Default bug keywords if none provided
        self.bug_keywords = bug_keywords or [
            'fix', 'bug', 'defect', 'issue', 'error', 'crash', 'problem',
            'fail', 'failure', 'segfault', 'fault', 'patch'
        ]

        # Initialize cache and repository data
        self.cache = []  # List of files currently in the cache
        self.file_stats = {}  # Stats for each file
        self.commit_history = []  # List of all commits
        self.bug_fixes = []  # List of bug-fixing commits
        self.total_files = 0  # Total number of files in repository
        self.cache_max_size = 0  # Maximum cache size (calculated from cache_size)
        self.hit_count = 0  # Count of cache hits
        self.miss_count = 0  # Count of cache misses
        self.results = {}  # Results of prediction

        logger.info(f"Initialized FixCache with cache size {cache_size * 100}%")

    def analyze_repository(self) -> None:
        """
        Analyze the git repository to extract commit history and identify bug-fixing commits.
        """
        logger.info(f"Analyzing repository: {self.repo_path}")
        start_time = time.time()

        try:
            # Change to repository directory
            os.chdir(self.repo_path)

            # Get list of all files in the repository
            self._get_all_files()

            # Calculate max cache size based on total files
            self.cache_max_size = max(1, int(self.total_files * self.cache_size))
            logger.info(f"Total files: {self.total_files}, Cache max size: {self.cache_max_size}")

            # Get all commits
            self._get_all_commits()

            # Identify bug-fixing commits based on commit messages
            self._identify_bug_fixes()

            # Process each bug fix commit to extract changed files
            self._process_bug_fixes()

            elapsed_time = time.time() - start_time
            logger.info(f"Repository analysis completed in {elapsed_time:.2f} seconds")
            logger.info(f"Identified {len(self.bug_fixes)} bug-fixing commits")

        except Exception as e:
            logger.error(f"Error during repository analysis: {str(e)}")
            raise

    def _get_all_files(self) -> None:
        """Get all files in the repository and initialize file stats."""
        try:
            # Use git to get all tracked files
            cmd = ["git", "ls-files"]
            output = subprocess.check_output(cmd, universal_newlines=True)
            all_files = output.strip().split('\n')

            # Filter out empty entries and non-code files
            code_files = [f for f in all_files if self._is_code_file(f)]
            self.total_files = len(code_files)

            # Initialize file stats for each file
            for file_path in code_files:
                self.file_stats[file_path] = {
                    'bug_fixes': 0,
                    'last_bug_fix': None,
                    'added_to_cache_count': 0,
                    'in_cache_time': 0,
                }

            logger.info(f"Found {self.total_files} code files in repository")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting files from repository: {str(e)}")
            raise

    def _is_code_file(self, file_path: str) -> bool:
        """
        Check if a file is a code file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            True if it's a code file, False otherwise
        """
        # List of common code file extensions
        code_extensions = [
            '.py', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.js', '.ts',
            '.go', '.rb', '.php', '.swift', '.kt', '.scala', '.rs', '.sh'
        ]

        # Check if the file has a code extension
        _, ext = os.path.splitext(file_path)
        return ext.lower() in code_extensions

    def _get_all_commits(self) -> None:
        """Get all commits from the repository."""
        try:
            # Use git to get all commits with metadata
            cmd = ["git", "log", "--pretty=format:%H|%an|%at|%s"]
            output = subprocess.check_output(cmd, universal_newlines=True)
            commits = output.strip().split('\n')

            # Parse commit information
            for commit in commits:
                parts = commit.split('|', 3)
                if len(parts) == 4:
                    sha, author, timestamp, message = parts
                    commit_info = {
                        'sha': sha,
                        'author': author,
                        'timestamp': int(timestamp),
                        'message': message,
                        'is_bug_fix': False,
                        'files_changed': []
                    }
                    self.commit_history.append(commit_info)

            logger.info(f"Retrieved {len(self.commit_history)} commits from repository")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting commits from repository: {str(e)}")
            raise

    def _identify_bug_fixes(self) -> None:
        """Identify bug-fixing commits based on commit messages."""
        bug_pattern = re.compile(r'\b(' + '|'.join(self.bug_keywords) + r')\b', re.IGNORECASE)

        for commit in self.commit_history:
            if bug_pattern.search(commit['message']):
                commit['is_bug_fix'] = True
                self.bug_fixes.append(commit)

        logger.info(f"Identified {len(self.bug_fixes)} bug-fixing commits")

    def _process_bug_fixes(self) -> None:
        """Process each bug-fixing commit to extract changed files."""
        for commit in self.bug_fixes:
            # Get files changed in this commit
            files_changed = self._get_files_changed(commit['sha'])
            commit['files_changed'] = files_changed

            # Update file stats for each file
            for file_path in files_changed:
                if file_path in self.file_stats:
                    self.file_stats[file_path]['bug_fixes'] += 1
                    self.file_stats[file_path]['last_bug_fix'] = commit['timestamp']

    def _get_files_changed(self, commit_sha: str) -> List[str]:
        """
        Get the list of files changed in a commit.

        Args:
            commit_sha: The SHA of the commit

        Returns:
            List of file paths changed in the commit
        """
        try:
            # Use git to get files changed in this commit
            cmd = ["git", "show", "--name-only", "--pretty=format:", commit_sha]
            output = subprocess.check_output(cmd, universal_newlines=True)

            # Filter out empty lines and non-code files
            files = [f for f in output.strip().split('\n') if f and self._is_code_file(f)]

            return files

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting files changed in commit {commit_sha}: {str(e)}")
            return []

    def predict(self) -> float:
        """
        Run the FixCache algorithm to predict fault-prone files.

        Returns:
            Hit rate as a percentage
        """
        logger.info("Starting FixCache prediction")
        start_time = time.time()

        # Initialize cache and counters
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0

        # Process commits chronologically
        sorted_commits = sorted(self.bug_fixes, key=lambda x: x['timestamp'])

        for commit in sorted_commits:
            for file_path in commit['files_changed']:
                if file_path not in self.file_stats:
                    continue

                # Check if file is in cache (hit) or not (miss)
                if file_path in self.cache:
                    self.hit_count += 1
                    # Move to end of cache (for LRU policy)
                    if self.policy == "LRU":
                        self.cache.remove(file_path)
                        self.cache.append(file_path)
                else:
                    self.miss_count += 1
                    # Add to cache
                    self.file_stats[file_path]['added_to_cache_count'] += 1
                    self.cache.append(file_path)

                    # If cache is full, evict according to policy
                    if len(self.cache) > self.cache_max_size:
                        self._evict_from_cache()

        # Calculate hit rate
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0

        # Store prediction results
        self.results = {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_bug_fixes': len(self.bug_fixes),
            'total_files': self.total_files,
            'cache_size': self.cache_size,
            'cache_max_size': self.cache_max_size,
            'policy': self.policy,
            'top_files': self._get_top_files(10),
            'timestamp': datetime.datetime.now().isoformat()
        }

        elapsed_time = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed_time:.2f} seconds")
        logger.info(f"Hit rate: {hit_rate:.2f}%")

        return hit_rate

    def _evict_from_cache(self) -> None:
        """Evict a file from the cache based on the selected policy."""
        if not self.cache:
            return

        if self.policy == "FIFO":
            # First In First Out - remove oldest file
            self.cache.pop(0)

        elif self.policy == "LRU":
            # Least Recently Used - already handled by moving hits to end
            self.cache.pop(0)

        elif self.policy == "BUG":
            # BUG policy - remove file with fewest bug fixes
            min_bugs = float('inf')
            min_index = 0

            for i, file_path in enumerate(self.cache):
                bug_fixes = self.file_stats[file_path]['bug_fixes']
                if bug_fixes < min_bugs:
                    min_bugs = bug_fixes
                    min_index = i

            self.cache.pop(min_index)

    def _get_top_files(self, n: int = 10) -> List[Dict]:
        """
        Get the top N files most likely to contain bugs.

        Args:
            n: Number of files to return

        Returns:
            List of dictionaries with file info
        """
        # Sort files by bug fix count
        top_files = []

        for file_path in self.cache:
            if file_path in self.file_stats:
                stats = self.file_stats[file_path]
                top_files.append({
                    'file_path': file_path,
                    'bug_fixes': stats['bug_fixes'],
                    'added_to_cache_count': stats['added_to_cache_count']
                })

        # Sort by bug fix count (descending)
        top_files.sort(key=lambda x: x['bug_fixes'], reverse=True)

        # Return top N files
        return top_files[:n]

    def save_results(self, output_file: str) -> None:
        """
        Save prediction results to a JSON file.

        Args:
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def visualize_results(self, output_file: str = "fixcache_results.png") -> None:
        """
        Visualize the prediction results.

        Args:
            output_file: Path to output image file
        """
        if not self.results:
            logger.error("No results to visualize. Run predict() first.")
            return

        try:
            plt.figure(figsize=(12, 10))

            # Create subplot for hit rate
            plt.subplot(2, 1, 1)
            labels = ['Hits', 'Misses']
            sizes = [self.hit_count, self.miss_count]
            colors = ['#66b3ff', '#ff9999']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'FixCache Hit Rate: {self.results["hit_rate"]:.2f}%')

            # Create subplot for top files
            plt.subplot(2, 1, 2)
            top_files = self.results['top_files']
            file_names = [os.path.basename(f['file_path']) for f in top_files]
            bug_counts = [f['bug_fixes'] for f in top_files]

            y_pos = np.arange(len(file_names))
            plt.barh(y_pos, bug_counts, align='center', color='#66b3ff')
            plt.yticks(y_pos, file_names)
            plt.xlabel('Number of Bug Fixes')
            plt.title('Top Files by Bug Fix Count')

            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()

            logger.info(f"Visualization saved to {output_file}")

        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")


def optimize_cache_size(repo_path: str, output_prefix: str, cache_sizes: List[float] = None) -> Dict[float, float]:
    """
    Run FixCache with multiple cache sizes and compare results.

    Args:
        repo_path: Path to the git repository
        output_prefix: Prefix for output files
        cache_sizes: List of cache sizes to test (as decimal percentages)

    Returns:
        Dictionary mapping cache sizes to hit rates
    """
    cache_sizes = cache_sizes or [0.1, 0.2, 0.3]
    results = {}

    for size in cache_sizes:
        logger.info(f"Testing cache size: {size * 100}%")

        # Initialize the FixCache with the current size
        fix_cache = FixCache(repo_path, cache_size=size)

        # Analyze repository
        fix_cache.analyze_repository()

        # Run prediction
        hit_rate = fix_cache.predict()

        # Store results
        results[size] = hit_rate

        # Save detailed results to a file
        fix_cache.save_results(f"{output_prefix}_cache_{int(size * 100)}pct.json")
        fix_cache.visualize_results(f"{output_prefix}_cache_{int(size * 100)}pct.png")

    # Create comparative visualization
    plot_cache_optimization(results, f"{output_prefix}_cache_optimization.png")

    return results


def plot_cache_optimization(optimization_results: Dict[float, float],
                            output_file: str = "cache_optimization.png") -> None:
    """
    Create a line chart visualizing cache size optimization results.

    Args:
        optimization_results: Dictionary mapping cache sizes to hit rates
        output_file: Path to output image file
    """
    try:
        # Sort cache sizes
        cache_sizes = sorted(optimization_results.keys())
        hit_rates = [optimization_results[size] for size in cache_sizes]

        # Convert cache sizes to percentages for display
        cache_size_labels = [f"{int(size * 100)}%" for size in cache_sizes]

        plt.figure(figsize=(10, 6))
        plt.plot(cache_size_labels, hit_rates, marker='o', linestyle='-', color='#66b3ff', linewidth=2, markersize=8)

        # Add labels and title
        plt.xlabel('Cache Size')
        plt.ylabel('Hit Rate (%)')
        plt.title('FixCache Performance by Cache Size')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add hit rate values as text above points
        for i, hit_rate in enumerate(hit_rates):
            plt.text(i, hit_rate + 1, f"{hit_rate:.2f}%", ha='center')

        # Find and mark optimal cache size
        optimal_size_index = hit_rates.index(max(hit_rates))
        plt.scatter([cache_size_labels[optimal_size_index]], [hit_rates[optimal_size_index]],
                    color='red', s=100, zorder=5, label='Optimal')

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

        logger.info(f"Cache optimization plot saved to {output_file}")

    except Exception as e:
        logger.error(f"Error creating cache optimization plot: {str(e)}")


def main():
    """Main function to run FixCache from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='FixCache: Bug prediction from cached history')
    parser.add_argument('--repo-path', required=True, help='Path to git repository')
    parser.add_argument('--cache-size', type=float, default=0.2, help='Cache size as fraction of total files')
    parser.add_argument('--policy', default='BUG', choices=['BUG', 'FIFO', 'LRU'], help='Cache replacement policy')
    parser.add_argument('--output-file', default='fixcache_results.json', help='Output file for results')
    parser.add_argument('--optimize', action='store_true', help='Run cache size optimization')
    parser.add_argument('--small-repo', action='store_true', help='Indicate this is a small repository (for reporting)')

    args = parser.parse_args()

    try:
        if args.optimize:
            # Run optimization with multiple cache sizes
            cache_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]
            output_prefix = os.path.splitext(args.output_file)[0]

            results = optimize_cache_size(args.repo_path, output_prefix, cache_sizes)

            # Print summary
            print("\nCache Size Optimization Results:")
            for size, hit_rate in sorted(results.items()):
                print(f"  Cache Size {size * 100}%: Hit Rate {hit_rate:.2f}%")

            # Find optimal cache size
            optimal_size = max(results.items(), key=lambda x: x[1])[0]
            print(f"\nOptimal cache size: {optimal_size * 100}%")

        else:
            # Run standard prediction with single cache size
            fix_cache = FixCache(args.repo_path, args.cache_size, args.policy)
            fix_cache.analyze_repository()
            hit_rate = fix_cache.predict()

            # Save results
            fix_cache.save_results(args.output_file)
            fix_cache.visualize_results(os.path.splitext(args.output_file)[0] + '.png')

            # Print summary
            print(f"\nRepository: {os.path.basename(args.repo_path)}")
            print(f"Total files: {fix_cache.total_files}")
            print(f"Bug-fixing commits: {len(fix_cache.bug_fixes)}")
            print(f"Cache size: {args.cache_size * 100}% ({fix_cache.cache_max_size} files)")
            print(f"Hit rate: {hit_rate:.2f}%")
            print(f"Results saved to {args.output_file}")

            # Print top files
            print("\nTop files most likely to contain bugs:")
            for i, file_info in enumerate(fix_cache.results['top_files']):
                print(f"{i + 1}. {file_info['file_path']} - {file_info['bug_fixes']} bug fixes")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()