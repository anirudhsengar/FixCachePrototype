#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cache Optimization Example for FixCache

This example demonstrates how to optimize the cache size parameter
for the FixCache bug prediction algorithm on a given repository.
It tests multiple cache sizes and policies to find the optimal configuration.

Author: anirudhsengar
"""

import os
import sys
import time
import argparse
import logging
import json
import multiprocessing
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the path so we can import fixcache
# Note: This is only needed for running the example directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FixCache components
from fixcache.algorithm import FixCache
from fixcache.visualization import plot_cache_optimization
from fixcache.utils import optimize_cache_size

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fixcache-optimization')


def optimize_cache_for_repository(
        repo_path: str,
        cache_sizes: Optional[List[float]] = None,
        policy: str = "BUG",
        window_ratio: float = 0.25,
        output_dir: str = ".",
        parallel: bool = True,
        fine_grained: bool = False,
        compare_policies: bool = False
) -> Dict[str, Dict[float, float]]:
    """
    Find the optimal cache size for a repository.

    Args:
        repo_path: Path to Git repository
        cache_sizes: List of cache sizes to test (as fraction of total files)
        policy: Cache replacement policy to use
        window_ratio: Ratio of commits to use for training window
        output_dir: Directory to save output files
        parallel: Whether to use parallel processing
        fine_grained: Whether to use fine-grained cache size steps
        compare_policies: Whether to compare different cache policies

    Returns:
        Dictionary mapping policies to {cache_size: hit_rate} dictionaries
    """
    # Create timestamp string for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate cache sizes to test if not provided
    if cache_sizes is None:
        if fine_grained:
            # More granular steps for fine-grained optimization
            cache_sizes = [round(i * 0.05, 2) for i in range(1, 11)]  # 0.05 to 0.50 in steps of 0.05
        else:
            # Standard steps for regular optimization
            cache_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # Policies to test
    policies = ["BUG", "FIFO", "LRU"] if compare_policies else [policy]

    # Store results for each policy
    all_results = {}

    # Track time for benchmark
    start_time = time.time()

    # Display optimization parameters
    print(f"\n{'-' * 70}")
    print(f" FixCache Cache Size Optimization")
    print(f"{'-' * 70}")
    print(f"Repository: {repo_path}")
    print(f"Cache sizes to test: {[f'{size * 100:.1f}%' for size in cache_sizes]}")
    print(f"Policies to test: {policies}")
    print(f"Window ratio: {window_ratio * 100:.1f}%")
    print(f"Parallel processing: {'Enabled' if parallel else 'Disabled'}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 70}\n")

    # Test each policy
    for current_policy in policies:
        print(f"\nTesting policy: {current_policy}")

        # Create output prefix for this policy
        output_prefix = os.path.join(output_dir, f"cache_opt_{current_policy.lower()}_{timestamp}")

        # Run optimization for this policy
        policy_results = optimize_cache_size(
            repo_path=repo_path,
            output_prefix=output_prefix,
            cache_sizes=cache_sizes,
            policy=current_policy,
            window_ratio=window_ratio,
            parallel=parallel
        )

        # Store results
        all_results[current_policy] = policy_results

        # Print results for this policy
        print(f"\nResults for {current_policy} policy:")
        for size, hit_rate in sorted(policy_results.items()):
            print(f"  Cache Size {size * 100:.1f}%: Hit Rate {hit_rate:.2f}%")

        # Find optimal cache size for this policy
        if policy_results:
            optimal_size = max(policy_results.items(), key=lambda x: x[1])[0]
            optimal_hit_rate = policy_results[optimal_size]
            print(f"\n  Optimal cache size for {current_policy}: {optimal_size * 100:.1f}% "
                  f"(Hit Rate: {optimal_hit_rate:.2f}%)")
            print(f"  Results saved to: {output_prefix}_*.json")
            print(f"  Visualization saved to: {output_prefix}_chart.png")

    # Track total execution time
    elapsed_time = time.time() - start_time

    # Save combined results
    combined_results = {
        "repository": repo_path,
        "timestamp": datetime.now().isoformat(),
        "window_ratio": window_ratio,
        "policies_tested": policies,
        "cache_sizes_tested": cache_sizes,
        "results": all_results,
        "execution_time": elapsed_time
    }

    combined_file = os.path.join(output_dir, f"cache_optimization_summary_{timestamp}.json")
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    # Create comparative visualization if multiple policies tested
    if compare_policies and len(policies) > 1:
        create_comparative_visualization(all_results, os.path.join(output_dir, f"policy_comparison_{timestamp}.png"))

    # Print summary
    print(f"\n{'-' * 70}")
    print(f" Optimization Summary")
    print(f"{'-' * 70}")

    # Print best overall configuration
    best_policy = None
    best_size = None
    best_rate = 0

    for policy, results in all_results.items():
        if results:
            optimal_size = max(results.items(), key=lambda x: x[1])[0]
            optimal_hit_rate = results[optimal_size]

            if optimal_hit_rate > best_rate:
                best_rate = optimal_hit_rate
                best_policy = policy
                best_size = optimal_size

    if best_policy:
        print(f"Best overall configuration:")
        print(f"  Policy: {best_policy}")
        print(f"  Cache size: {best_size * 100:.1f}%")
        print(f"  Hit rate: {best_rate:.2f}%")

    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"Combined results saved to: {combined_file}")
    print(f"{'-' * 70}\n")

    return all_results


def create_comparative_visualization(
        policy_results: Dict[str, Dict[float, float]],
        output_file: str
) -> None:
    """
    Create a visualization comparing different policies.

    Args:
        policy_results: Dictionary mapping policies to results dictionaries
        output_file: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(12, 8))

        markers = ['o', 's', '^', 'D', 'x']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Plot each policy
        for i, (policy, results) in enumerate(policy_results.items()):
            if not results:
                continue

            # Convert to lists for plotting
            sizes = [size * 100 for size in results.keys()]  # Convert to percentages
            hit_rates = list(results.values())

            # Sort by size
            sorted_data = sorted(zip(sizes, hit_rates))
            sorted_sizes, sorted_hit_rates = zip(*sorted_data)

            # Plot line
            plt.plot(
                sorted_sizes, sorted_hit_rates,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linestyle='-',
                linewidth=2,
                markersize=8,
                label=f"{policy} Policy"
            )

            # Find and mark optimal point
            optimal_idx = sorted_hit_rates.index(max(sorted_hit_rates))
            optimal_size = sorted_sizes[optimal_idx]
            optimal_hit_rate = sorted_hit_rates[optimal_idx]

            plt.plot(
                optimal_size, optimal_hit_rate,
                marker='o',
                color=colors[i % len(colors)],
                markersize=12,
                markeredgecolor='white',
                markeredgewidth=2
            )

            # Add annotation
            plt.annotate(
                f"{optimal_size:.1f}%, {optimal_hit_rate:.2f}%",
                (optimal_size, optimal_hit_rate),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight='bold'
            )

        # Configure plot
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Cache Size (% of total files)', fontsize=12)
        plt.ylabel('Hit Rate (%)', fontsize=12)
        plt.title('Cache Policy Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=11)

        # Add timestamp
        plt.figtext(
            0.98, 0.02,
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='right', fontsize=8, color='gray'
        )

        # Add footer with tool information
        plt.figtext(
            0.5, 0.01,
            f'Generated by FixCachePrototype | https://github.com/anirudhsengar/FixCachePrototype',
            ha='center', fontsize=8, color='gray'
        )

        # Save and close
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Comparative visualization saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error creating comparative visualization: {str(e)}")


def run_detailed_analysis(
        repo_path: str,
        best_policy: str,
        best_cache_size: float,
        output_dir: str = "."
) -> None:
    """
    Run a detailed analysis using the optimal configuration.

    Args:
        repo_path: Path to Git repository
        best_policy: Best cache replacement policy
        best_cache_size: Best cache size
        output_dir: Directory to save output files
    """
    print(f"\n{'-' * 70}")
    print(f" Detailed Analysis with Optimal Configuration")
    print(f"{'-' * 70}")
    print(f"Policy: {best_policy}")
    print(f"Cache size: {best_cache_size * 100:.1f}%")
    print(f"{'-' * 70}\n")

    # Create FixCache instance with optimal parameters
    fix_cache = FixCache(
        repo_path=repo_path,
        cache_size=best_cache_size,
        policy=best_policy
    )

    # Analyze repository
    if not fix_cache.analyze_repository():
        print(f"Error analyzing repository: {'; '.join(fix_cache.error_messages)}")
        return

    # Run prediction
    hit_rate = fix_cache.predict()

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"optimal_analysis_{timestamp}.json")
    fix_cache.save_results(output_file)

    # Generate visualization
    visualization_file = os.path.join(output_dir, f"optimal_analysis_{timestamp}.png")
    fix_cache.visualize_results(visualization_file)

    # Display results
    print(f"Detailed analysis results:")
    print(f"  Hit rate: {hit_rate:.2f}%")
    print(f"  Hits: {fix_cache.hit_count}")
    print(f"  Misses: {fix_cache.miss_count}")
    print(f"  Results saved to: {output_file}")
    print(f"  Visualization saved to: {visualization_file}")

    # Display top bug-prone files
    print(f"\nTop bug-prone files:")
    for i, file_info in enumerate(fix_cache.get_top_files(10)):
        print(f"  {i + 1}. {file_info['file_path']} - {file_info['bug_fixes']} bug fixes")

    print(f"\n{'-' * 70}")


def main():
    """Main entry point for the example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FixCache Cache Size Optimization Example')

    # Required arguments
    parser.add_argument('--repo-path', '-r', required=True, help='Path to Git repository')

    # Cache optimization options
    parser.add_argument('--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
                        help='Cache replacement policy (default: BUG)')
    parser.add_argument('--fine-grained', '-f', action='store_true',
                        help='Use fine-grained cache size steps')
    parser.add_argument('--compare-policies', '-c', action='store_true',
                        help='Compare different cache policies')
    parser.add_argument('--window-ratio', '-w', type=float, default=0.25,
                        help='Ratio of commits to use for training window (default: 0.25)')

    # Output options
    parser.add_argument('--output-dir', '-o', default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Run detailed analysis with optimal parameters')

    # Performance options
    parser.add_argument('--sequential', '-s', action='store_true',
                        help='Disable parallel processing')

    # Parse arguments
    args = parser.parse_args()

    # Run optimization
    results = optimize_cache_for_repository(
        repo_path=args.repo_path,
        policy=args.policy,
        window_ratio=args.window_ratio,
        output_dir=args.output_dir,
        parallel=not args.sequential,
        fine_grained=args.fine_grained,
        compare_policies=args.compare_policies
    )

    # Run detailed analysis if requested
    if args.detailed and results:
        # Find best policy
        best_policy = args.policy
        best_size = None
        best_rate = 0

        if args.compare_policies:
            # Find best across all policies
            for policy, policy_results in results.items():
                if policy_results:
                    optimal_size = max(policy_results.items(), key=lambda x: x[1])[0]
                    optimal_hit_rate = policy_results[optimal_size]

                    if optimal_hit_rate > best_rate:
                        best_rate = optimal_hit_rate
                        best_policy = policy
                        best_size = optimal_size
        else:
            # Just use the provided policy
            policy_results = results.get(args.policy, {})
            if policy_results:
                best_size = max(policy_results.items(), key=lambda x: x[1])[0]

        # Run detailed analysis if we found a best size
        if best_size is not None:
            run_detailed_analysis(args.repo_path, best_policy, best_size, args.output_dir)


if __name__ == "__main__":
    main()