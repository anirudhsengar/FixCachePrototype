#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic Usage Example for FixCache

This example demonstrates how to use the FixCache bug prediction tool
for analyzing a Git repository and predicting bug-prone files.

Author: anirudhsengar
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the parent directory to the path so we can import fixcache
# Note: This is only needed for running the example directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FixCache components
from fixcache import FixCache, optimize_cache_size, visualize_results
from fixcache.utils import is_code_file, format_timestamp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fixcache-example')


def run_basic_prediction(repo_path):
    """
    Run basic FixCache prediction on a repository.

    Args:
        repo_path: Path to Git repository
    """
    print(f"\n{'-' * 60}")
    print(f"FixCache Basic Usage Example")
    print(f"{'-' * 60}")
    print(f"Repository: {repo_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 60}\n")

    # Step 1: Create FixCache instance with default parameters
    print("Step 1: Creating FixCache instance...")
    fix_cache = FixCache(
        repo_path=repo_path,  # Path to the repository
        cache_size=0.2,  # Cache size: 20% of total files
        policy="BUG",  # Cache replacement policy: BUG (default)
        window_ratio=0.25  # Use 25% of commits for training
    )

    # Step 2: Analyze the repository
    print("\nStep 2: Analyzing repository...")
    if not fix_cache.analyze_repository():
        print(f"Error: Repository analysis failed! {'; '.join(fix_cache.error_messages)}")
        return

    print(f"Analysis complete:")
    print(f"  - Total files: {fix_cache.repo_analyzer.total_files}")
    print(f"  - Bug-fixing commits: {len(fix_cache.repo_analyzer.bug_fixes)}")
    print(f"  - Cache size: {fix_cache.cache_size * 100}% ({fix_cache.cache_max_size} files)")

    # Step 3: Run prediction
    print("\nStep 3: Running bug prediction...")
    hit_rate = fix_cache.predict()

    print(f"Prediction complete:")
    print(f"  - Hit rate: {hit_rate:.2f}%")
    print(f"  - Hits: {fix_cache.hit_count}")
    print(f"  - Misses: {fix_cache.miss_count}")

    # Step 4: Display top bug-prone files
    print("\nStep 4: Top files most likely to contain bugs:")
    for i, file_info in enumerate(fix_cache.get_top_files(5)):
        print(f"  {i + 1}. {file_info['file_path']}")
        print(f"     - Bug fixes: {file_info['bug_fixes']}")
        last_modified = format_timestamp(file_info['last_modified']) if file_info['last_modified'] else 'N/A'
        print(f"     - Last bug fix: {last_modified}")

    # Step 5: Get recommendations
    print("\nStep 5: Recommendations:")
    recommendations = fix_cache.get_recommended_actions()
    if recommendations:
        for i, rec in enumerate(recommendations):
            print(f"  {i + 1}. {rec['message']} (Priority: {rec['priority']})")
    else:
        print("  No specific recommendations.")

    # Step 6: Save results to JSON file
    output_file = "fixcache_results.json"
    print(f"\nStep 6: Saving results to {output_file}...")
    fix_cache.save_results(output_file)
    print(f"  Results saved successfully!")

    # Step 7: Generate visualization
    visualization_file = "fixcache_results.png"
    print(f"\nStep 7: Generating visualization to {visualization_file}...")
    fix_cache.visualize_results(visualization_file)
    print(f"  Visualization saved successfully!")

    print(f"\n{'-' * 60}")
    print(f"Example completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 60}")


def optimize_cache_example(repo_path):
    """
    Demonstrate cache size optimization.

    Args:
        repo_path: Path to Git repository
    """
    print(f"\n{'-' * 60}")
    print(f"FixCache Cache Size Optimization Example")
    print(f"{'-' * 60}")
    print(f"Repository: {repo_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 60}\n")

    # Define cache sizes to test
    cache_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]
    print(f"Testing cache sizes: {[f'{size * 100}%' for size in cache_sizes]}")

    # Run optimization (this may take some time)
    print("\nRunning optimization (this may take some time)...")
    results = optimize_cache_size(
        repo_path=repo_path,
        output_prefix="cache_optimization",
        cache_sizes=cache_sizes,
        policy="BUG"
    )

    # Print results
    print("\nOptimization results:")
    for size, hit_rate in sorted(results.items()):
        print(f"  Cache Size {size * 100:.1f}%: Hit Rate {hit_rate:.2f}%")

    # Find optimal cache size
    optimal_size = max(results.items(), key=lambda x: x[1])[0]
    optimal_hit_rate = results[optimal_size]

    print(f"\nOptimal cache size: {optimal_size * 100:.1f}% (Hit Rate: {optimal_hit_rate:.2f}%)")
    print(f"Optimization visualization saved to: cache_optimization_chart.png")

    print(f"\n{'-' * 60}")
    print(f"Example completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 60}")


def file_stats_example(repo_path):
    """
    Demonstrate how to extract and use file statistics.

    Args:
        repo_path: Path to Git repository
    """
    print(f"\n{'-' * 60}")
    print(f"FixCache File Statistics Example")
    print(f"{'-' * 60}")
    print(f"Repository: {repo_path}")
    print(f"{'-' * 60}\n")

    # Create FixCache instance just for repository analysis
    fix_cache = FixCache(repo_path=repo_path)

    # Analyze repository
    print("Analyzing repository...")
    if not fix_cache.analyze_repository():
        print(f"Error: Repository analysis failed! {'; '.join(fix_cache.error_messages)}")
        return

    # Get file statistics
    repo_analyzer = fix_cache.repo_analyzer
    file_stats = repo_analyzer.file_stats

    # Calculate some aggregate statistics
    total_files = len(file_stats)
    total_bug_fixes = sum(stats['bug_fixes'] for stats in file_stats.values())
    avg_bug_fixes = total_bug_fixes / max(1, total_files)

    print(f"\nRepository Statistics:")
    print(f"  - Total files: {total_files}")
    print(f"  - Total bug fixes: {total_bug_fixes}")
    print(f"  - Average bug fixes per file: {avg_bug_fixes:.2f}")

    # Group files by extension
    extensions = {}
    for file_path, stats in file_stats.items():
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = "(no extension)"

        if ext not in extensions:
            extensions[ext] = {
                'count': 0,
                'bug_fixes': 0,
                'files': []
            }

        extensions[ext]['count'] += 1
        extensions[ext]['bug_fixes'] += stats['bug_fixes']
        extensions[ext]['files'].append(file_path)

    # Display extension statistics
    print("\nFile Extension Statistics:")
    for ext, stats in sorted(extensions.items(), key=lambda x: x[1]['bug_fixes'], reverse=True):
        if stats['count'] > 0:
            avg = stats['bug_fixes'] / stats['count']
            print(f"  {ext}: {stats['count']} files, {stats['bug_fixes']} bug fixes, {avg:.2f} bugs/file")

    # Find the most bug-prone directory
    directories = {}
    for file_path, stats in file_stats.items():
        directory = os.path.dirname(file_path)
        if not directory:
            directory = "(root)"

        if directory not in directories:
            directories[directory] = {
                'count': 0,
                'bug_fixes': 0
            }

        directories[directory]['count'] += 1
        directories[directory]['bug_fixes'] += stats['bug_fixes']

    # Display top directories
    print("\nTop Bug-Prone Directories:")
    top_dirs = sorted(directories.items(), key=lambda x: x[1]['bug_fixes'], reverse=True)[:5]
    for directory, stats in top_dirs:
        if stats['count'] > 0:
            avg = stats['bug_fixes'] / stats['count']
            print(f"  {directory}: {stats['count']} files, {stats['bug_fixes']} bug fixes, {avg:.2f} bugs/file")

    print(f"\n{'-' * 60}")
    print(f"Example completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FixCache Basic Usage Example')
    parser.add_argument('--repo-path', '-r', required=True, help='Path to Git repository')
    parser.add_argument('--optimize', '-o', action='store_true', help='Run cache size optimization')
    parser.add_argument('--stats', '-s', action='store_true', help='Show file statistics')

    args = parser.parse_args()

    if args.optimize:
        optimize_cache_example(args.repo_path)
    elif args.stats:
        file_stats_example(args.repo_path)
    else:
        run_basic_prediction(args.repo_path)