#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repository Comparison Example for FixCache

This example demonstrates how to compare multiple repositories
using the FixCache bug prediction tool to identify patterns in
bug-prone files and evaluate different codebases.

Author: anirudhsengar
"""

import os
import sys
import time
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the path so we can import fixcache
# Note: This is only needed for running the example directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FixCache components
from fixcache.algorithm import FixCache
from fixcache.repository import RepositoryAnalyzer
from fixcache.utils import compare_repositories, simplify_path, safe_divide
from fixcache.visualization import plot_repository_comparison

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fixcache-comparison')


def compare_repos(
        repo_paths: List[str],
        cache_size: float = 0.2,
        policy: str = "BUG",
        lookback_commits: Optional[int] = None,
        output_dir: str = ".",
        detailed_analysis: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple repositories using FixCache.

    Args:
        repo_paths: List of repository paths
        cache_size: Cache size as fraction of total files
        policy: Cache replacement policy
        lookback_commits: Number of recent commits to analyze
        output_dir: Directory to save output files
        detailed_analysis: Whether to perform detailed analysis

    Returns:
        Dictionary mapping repository paths to their results
    """
    # Create timestamp string for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Output file for results
    output_file = os.path.join(output_dir, f"repo_comparison_{timestamp}.json")

    # Display comparison parameters
    print(f"\n{'-' * 70}")
    print(f" FixCache Repository Comparison")
    print(f"{'-' * 70}")
    print(f"Repositories to compare: {len(repo_paths)}")
    for i, repo_path in enumerate(repo_paths):
        print(f"  {i + 1}. {repo_path}")
    print(f"Cache size: {cache_size * 100:.1f}%")
    print(f"Policy: {policy}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 70}\n")

    # Track time for benchmark
    start_time = time.time()

    # Run comparison using utility function
    results = compare_repositories(
        repo_paths=repo_paths,
        output_file=output_file,
        cache_size=cache_size,
        policy=policy,
        lookback_commits=lookback_commits
    )

    # Track total execution time
    elapsed_time = time.time() - start_time

    # Generate additional comparison visualizations
    if results and len(results) >= 2:
        create_additional_visualizations(results, output_dir, timestamp)

    # Run detailed comparison if requested
    if detailed_analysis and results and len(results) >= 2:
        run_detailed_comparison(results, output_dir, timestamp)

    # Print summary
    print(f"\n{'-' * 70}")
    print(f" Comparison Summary")
    print(f"{'-' * 70}")

    if results:
        # Format as table
        headers = ["Repository", "Files", "Bug Fixes", "Hit Rate", "Bug Density"]
        rows = []

        for repo_path, repo_results in results.items():
            repo_name = os.path.basename(os.path.abspath(repo_path))
            total_files = repo_results.get('total_files', 0)
            total_bug_fixes = repo_results.get('total_bug_fixes', 0)
            hit_rate = repo_results.get('hit_rate', 0)

            # Calculate bug density (bugs per file)
            bug_density = safe_divide(total_bug_fixes, total_files) * 100

            rows.append([
                repo_name,
                str(total_files),
                str(total_bug_fixes),
                f"{hit_rate:.2f}%",
                f"{bug_density:.2f}%"
            ])

        # Get column widths
        col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) + 2
                      for i in range(len(headers))]

        # Print headers
        header_row = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_row)
        print("-" * len(header_row))

        # Print rows
        for row in rows:
            print("".join(cell.ljust(w) for cell, w in zip(row, col_widths)))

        # Identify best performing repository (by hit rate)
        best_repo = max(results.items(), key=lambda x: x[1].get('hit_rate', 0))[0]
        best_repo_name = os.path.basename(os.path.abspath(best_repo))
        best_hit_rate = results[best_repo].get('hit_rate', 0)

        print(f"\nBest performing repository: {best_repo_name} (Hit Rate: {best_hit_rate:.2f}%)")
    else:
        print("No comparison results available.")

    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'-' * 70}\n")

    return results


def create_additional_visualizations(
        results: Dict[str, Dict[str, Any]],
        output_dir: str,
        timestamp: str
) -> None:
    """
    Create additional visualizations for repository comparison.

    Args:
        results: Dictionary mapping repository paths to their results
        output_dir: Directory to save visualizations
        timestamp: Timestamp string for filenames
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import PercentFormatter

        # 1. Bug Type Distribution Comparison
        try:
            plt.figure(figsize=(12, 8))

            # Extract repositories and their file extension distributions
            repo_names = [os.path.basename(path) for path in results.keys()]

            # Collect extension data
            ext_data = {}
            for repo_path, repo_results in results.items():
                repo_name = os.path.basename(repo_path)

                if 'top_files' in repo_results:
                    # Count bugs by extension
                    ext_bugs = {}
                    for file_info in repo_results['top_files']:
                        ext = os.path.splitext(file_info['file_path'])[1]
                        if not ext:
                            ext = 'unknown'

                        ext_bugs[ext] = ext_bugs.get(ext, 0) + file_info.get('bug_fixes', 0)

                    # Store extension data
                    ext_data[repo_name] = ext_bugs

            # Find common extensions
            all_extensions = set()
            for ext_bugs in ext_data.values():
                all_extensions.update(ext_bugs.keys())

            # Limit to top 5 extensions by total bug count
            ext_totals = {}
            for ext in all_extensions:
                ext_totals[ext] = sum(ext_bugs.get(ext, 0) for ext_bugs in ext_data.values())

            top_extensions = sorted(ext_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            top_ext_names = [ext for ext, _ in top_extensions]

            # Prepare data for plotting
            x = np.arange(len(repo_names))
            width = 0.15
            offsets = np.linspace(-(len(top_ext_names) - 1) / 2 * width, (len(top_ext_names) - 1) / 2 * width,
                                  len(top_ext_names))

            # Plot bars for each extension
            for i, ext in enumerate(top_ext_names):
                values = [ext_data.get(repo, {}).get(ext, 0) for repo in repo_names]
                plt.bar(x + offsets[i], values, width, label=f'{ext} files')

            # Configure plot
            plt.xlabel('Repository', fontsize=12)
            plt.ylabel('Number of Bug Fixes', fontsize=12)
            plt.title('Bug Distribution by File Type', fontsize=16, fontweight='bold')
            plt.xticks(x, repo_names)
            plt.legend(title="File Types")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Save plot
            plt.tight_layout()
            filename = os.path.join(output_dir, f"bug_type_distribution_{timestamp}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Bug type distribution visualization saved to: {filename}")

        except Exception as e:
            logger.error(f"Error creating bug type distribution visualization: {str(e)}")

        # 2. Bug Density vs Hit Rate Scatter Plot
        try:
            plt.figure(figsize=(10, 8))

            # Extract data points
            repo_names = []
            hit_rates = []
            bug_densities = []
            file_counts = []

            for repo_path, repo_results in results.items():
                repo_name = os.path.basename(repo_path)
                repo_names.append(repo_name)

                hit_rate = repo_results.get('hit_rate', 0)
                hit_rates.append(hit_rate)

                total_files = repo_results.get('total_files', 0)
                total_bug_fixes = repo_results.get('total_bug_fixes', 0)

                # Calculate bug density (bugs per file)
                bug_density = safe_divide(total_bug_fixes, total_files) * 100
                bug_densities.append(bug_density)

                # Store file count for bubble size
                file_counts.append(total_files)

            # Normalize file counts for bubble size (between 100 and 1000)
            if file_counts:
                min_count = min(file_counts)
                max_count = max(file_counts)
                if min_count < max_count:
                    norm_file_counts = [100 + 900 * (count - min_count) / (max_count - min_count)
                                        for count in file_counts]
                else:
                    norm_file_counts = [500] * len(file_counts)
            else:
                norm_file_counts = [500] * len(file_counts)

            # Create scatter plot
            scatter = plt.scatter(
                bug_densities, hit_rates,
                s=norm_file_counts,
                alpha=0.7,
                c=np.arange(len(repo_names)),
                cmap='viridis',
                edgecolors='white',
                linewidth=1
            )

            # Add labels to points
            for i, repo_name in enumerate(repo_names):
                plt.annotate(
                    repo_name,
                    (bug_densities[i], hit_rates[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold'
                )

            # Configure plot
            plt.xlabel('Bug Density (% of files with bugs)', fontsize=12)
            plt.ylabel('Hit Rate (%)', fontsize=12)
            plt.title('Bug Density vs. Hit Rate by Repository', fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Format axes as percentages
            plt.gca().xaxis.set_major_formatter(PercentFormatter())
            plt.gca().yaxis.set_major_formatter(PercentFormatter())

            # Add legend for bubble size
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=3)
            size_legend = plt.legend(
                handles,
                [f"Small (~{min(file_counts):.0f} files)",
                 f"Medium (~{(min(file_counts) + max(file_counts)) / 2:.0f} files)",
                 f"Large (~{max(file_counts):.0f} files)"],
                loc="upper left",
                title="Repository Size"
            )
            plt.gca().add_artist(size_legend)

            # Save plot
            plt.tight_layout()
            filename = os.path.join(output_dir, f"bug_density_vs_hit_rate_{timestamp}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Bug density vs hit rate visualization saved to: {filename}")

        except Exception as e:
            logger.error(f"Error creating bug density vs hit rate visualization: {str(e)}")

    except ImportError:
        logger.warning("matplotlib or numpy not available for creating additional visualizations")


def run_detailed_comparison(
        results: Dict[str, Dict[str, Any]],
        output_dir: str,
        timestamp: str
) -> None:
    """
    Run a detailed comparison analysis of the repositories.

    Args:
        results: Dictionary mapping repository paths to their results
        output_dir: Directory to save output files
        timestamp: Timestamp string for filenames
    """
    print(f"\n{'-' * 70}")
    print(f" Detailed Repository Comparison")
    print(f"{'-' * 70}\n")

    # 1. Common bug patterns
    print("Analyzing common bug patterns across repositories...")

    # Collect all bug-prone files with their extensions
    all_buggy_extensions = {}
    for repo_path, repo_results in results.items():
        repo_name = os.path.basename(repo_path)

        if 'top_files' in repo_results:
            # Count bugs by extension
            for file_info in repo_results['top_files']:
                ext = os.path.splitext(file_info['file_path'])[1]
                if not ext:
                    ext = 'unknown'

                if ext not in all_buggy_extensions:
                    all_buggy_extensions[ext] = {
                        'total_bugs': 0,
                        'repos': set(),
                        'avg_bugs_per_file': 0,
                        'file_count': 0
                    }

                all_buggy_extensions[ext]['total_bugs'] += file_info.get('bug_fixes', 0)
                all_buggy_extensions[ext]['repos'].add(repo_name)
                all_buggy_extensions[ext]['file_count'] += 1

    # Calculate average bugs per file
    for ext, stats in all_buggy_extensions.items():
        stats['avg_bugs_per_file'] = safe_divide(stats['total_bugs'], stats['file_count'])
        stats['repos'] = list(stats['repos'])

    # Sort by prevalence (number of repositories) and then by total bugs
    sorted_extensions = sorted(
        all_buggy_extensions.items(),
        key=lambda x: (len(x[1]['repos']), x[1]['total_bugs']),
        reverse=True
    )

    # Display common bug patterns
    print("\nCommon Bug-Prone File Types:")
    print(f"{'Extension':<10} {'Total Bugs':<12} {'Avg/File':<10} {'Repos':<5} {'Present In'}")
    print('-' * 70)

    for ext, stats in sorted_extensions[:10]:  # Show top 10
        repos_str = ', '.join(stats['repos']) if len(
            stats['repos']) <= 3 else f"{', '.join(stats['repos'][:2])} +{len(stats['repos']) - 2} more"
        print(
            f"{ext:<10} {stats['total_bugs']:<12} {stats['avg_bugs_per_file']:.2f}{' ' * 6} {len(stats['repos']):<5} {repos_str}")

    # 2. Repository correlation analysis
    print("\nAnalyzing repository correlations...")

    if len(results) >= 3:  # Need at least 3 repositories for meaningful correlation
        # Create correlation dataframe
        data = {
            'Repository': [],
            'Files': [],
            'Bug Fixes': [],
            'Hit Rate': [],
            'Bug Density': []
        }

        for repo_path, repo_results in results.items():
            repo_name = os.path.basename(repo_path)
            total_files = repo_results.get('total_files', 0)
            total_bug_fixes = repo_results.get('total_bug_fixes', 0)
            hit_rate = repo_results.get('hit_rate', 0)
            bug_density = safe_divide(total_bug_fixes, total_files) * 100

            data['Repository'].append(repo_name)
            data['Files'].append(total_files)
            data['Bug Fixes'].append(total_bug_fixes)
            data['Hit Rate'].append(hit_rate)
            data['Bug Density'].append(bug_density)

        # Create dataframe
        try:
            df = pd.DataFrame(data)

            # Calculate correlations
            corr = df[['Files', 'Bug Fixes', 'Hit Rate', 'Bug Density']].corr()

            # Display correlations
            print("\nMetric Correlations:")
            print(corr.round(2))

            # Save correlations to CSV
            corr_file = os.path.join(output_dir, f"repo_correlations_{timestamp}.csv")
            corr.to_csv(corr_file)
            print(f"Correlation matrix saved to: {corr_file}")

            # Create correlation heatmap
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                plt.figure(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
                plt.title('Repository Metric Correlations', fontsize=14, fontweight='bold')

                # Save heatmap
                corr_img = os.path.join(output_dir, f"correlation_heatmap_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(corr_img, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"Correlation heatmap saved to: {corr_img}")

            except Exception as e:
                logger.error(f"Error creating correlation heatmap: {str(e)}")

        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
    else:
        print("Need at least 3 repositories for correlation analysis.")

    # 3. Project health ranking
    print("\nGenerating project health ranking...")

    # Calculate health score based on hit rate and bug density
    health_scores = []
    for repo_path, repo_results in results.items():
        repo_name = os.path.basename(repo_path)
        hit_rate = repo_results.get('hit_rate', 0)
        total_files = repo_results.get('total_files', 0)
        total_bug_fixes = repo_results.get('total_bug_fixes', 0)
        bug_density = safe_divide(total_bug_fixes, total_files) * 100

        # Higher hit rate is good, lower bug density is good
        # Normalize both metrics to 0-100 scale
        norm_hit_rate = hit_rate  # Already 0-100
        norm_bug_density = max(0, 100 - bug_density)  # Invert so higher is better

        # Calculate health score (weighted average)
        health_score = 0.6 * norm_hit_rate + 0.4 * norm_bug_density

        health_scores.append({
            'repo_name': repo_name,
            'repo_path': repo_path,
            'health_score': health_score,
            'hit_rate': hit_rate,
            'bug_density': bug_density
        })

    # Sort by health score (descending)
    health_scores.sort(key=lambda x: x['health_score'], reverse=True)

    # Display health ranking
    print("\nProject Health Ranking (higher score = healthier project):")
    print(f"{'Rank':<5} {'Repository':<20} {'Health Score':<15} {'Hit Rate':<10} {'Bug Density':<10}")
    print('-' * 70)

    for i, score in enumerate(health_scores):
        print(f"{i + 1:<5} {score['repo_name']:<20} {score['health_score']:.2f}{' ' * 9} "
              f"{score['hit_rate']:.2f}%{' ' * 4} {score['bug_density']:.2f}%")

    # Save health ranking to CSV
    try:
        health_df = pd.DataFrame(health_scores)
        health_file = os.path.join(output_dir, f"project_health_ranking_{timestamp}.csv")
        health_df.to_csv(health_file, index=False)
        print(f"Project health ranking saved to: {health_file}")
    except Exception as e:
        logger.error(f"Error saving health ranking: {str(e)}")

    print(f"\n{'-' * 70}")


def main():
    """Main entry point for the example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FixCache Repository Comparison Example')

    # Repository paths (multiple)
    parser.add_argument('--repo-paths', '-r', required=True,
                        help='Comma-separated list of repository paths')

    # Analysis options
    parser.add_argument('--cache-size', '-c', type=float, default=0.2,
                        help='Cache size as fraction of total files (default: 0.2)')
    parser.add_argument('--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
                        help='Cache replacement policy (default: BUG)')
    parser.add_argument('--lookback', '-l', type=int,
                        help='Number of recent commits to analyze (default: all)')

    # Output options
    parser.add_argument('--output-dir', '-o', default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Run detailed comparison analysis')

    # Parse arguments
    args = parser.parse_args()

    # Split repository paths
    repo_paths = [path.strip() for path in args.repo_paths.split(',')]

    # Check that we have at least 2 repositories
    if len(repo_paths) < 2:
        parser.error("At least two repository paths must be provided for comparison")

    # Check that all repositories exist
    for repo_path in repo_paths:
        if not os.path.exists(repo_path):
            parser.error(f"Repository path does not exist: {repo_path}")

        git_dir = os.path.join(repo_path, '.git')
        if not os.path.exists(git_dir):
            parser.error(f"Not a git repository: {repo_path}")

    # Run comparison
    compare_repos(
        repo_paths=repo_paths,
        cache_size=args.cache_size,
        policy=args.policy,
        lookback_commits=args.lookback,
        output_dir=args.output_dir,
        detailed_analysis=args.detailed
    )


if __name__ == "__main__":
    main()