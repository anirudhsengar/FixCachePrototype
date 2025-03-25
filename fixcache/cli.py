#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache Command Line Interface

This module provides the command-line interface for the FixCache tool.

Author: anirudhsengar
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from . import __version__
from .algorithm import FixCache
from .repository import RepositoryAnalyzer
from .visualization import visualize_results, plot_cache_optimization, plot_repository_comparison


def setup_logging(verbose=False, quiet=False):
    """Set up logging configuration."""
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.ERROR

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_command(args):
    """Handle the analyze command."""
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger('fixcache.analyze')

    logger.info(f"Analyzing repository: {args.repo_path}")

    # Create FixCache instance
    try:
        fix_cache = FixCache(
            repo_path=args.repo_path,
            cache_size=args.cache_size,
            policy=args.policy,
            window_ratio=args.window_ratio,
            lookback_commits=args.lookback if args.lookback else None,
            cache_seeding=not args.no_seeding
        )

        # Analyze repository
        logger.info("Analyzing repository history...")
        analyze_success = fix_cache.analyze_repository()

        if not analyze_success:
            logger.error("Repository analysis failed. Check error messages.")
            return 1

        # Run prediction
        logger.info("Running prediction...")
        hit_rate = fix_cache.predict()

        # Get summary
        results = fix_cache.get_summary()

        # Display results
        logger.info(f"Hit rate: {hit_rate:.2f}%")
        logger.info(f"Cache size: {args.cache_size * 100:.1f}% of {results['total_files']} files")

        # Generate visualization if requested
        if args.visualize:
            output_file = args.output if args.output else "fixcache_results.png"
            logger.info(f"Generating visualization to {output_file}")
            visualize_success = visualize_results(results, output_file)
            if not visualize_success:
                logger.warning("Failed to generate visualization. Matplotlib may not be installed.")

        # Generate report if requested
        if args.report:
            output_file = args.output if args.output else "fixcache_report.md"
            format_type = args.format if args.format else "md"
            logger.info(f"Generating report to {output_file}")

            report_content = fix_cache.generate_report(format_type)
            with open(output_file, 'w') as f:
                f.write(report_content)

        # Output results in the requested format
        if not args.report and not args.visualize and args.output:
            format_type = args.format if args.format else "json"
            logger.info(f"Writing results to {args.output}")

            if format_type == "json":
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            elif format_type == "yaml":
                import yaml
                with open(args.output, 'w') as f:
                    yaml.dump(results, f)
            else:
                with open(args.output, 'w') as f:
                    f.write(str(results))

        return 0

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=args.verbose)
        return 1


def optimize_command(args):
    """Handle the optimize command."""
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger('fixcache.optimize')

    logger.info(f"Optimizing cache size for repository: {args.repo_path}")

    try:
        # Create cache sizes to test
        cache_sizes = []
        current_size = args.min_size
        while current_size <= args.max_size:
            cache_sizes.append(round(current_size, 4))
            current_size += args.step_size

        logger.info(f"Testing cache sizes: {cache_sizes}")

        # Initialize results dictionary
        results = {}

        # Test each cache size
        for size in cache_sizes:
            logger.info(f"Testing cache size: {size * 100:.1f}%")

            fix_cache = FixCache(
                repo_path=args.repo_path,
                cache_size=size,
                policy=args.policy,
                window_ratio=0.25  # Use default window ratio for optimization
            )

            analyze_success = fix_cache.analyze_repository()
            if not analyze_success:
                logger.error("Repository analysis failed. Check error messages.")
                return 1

            hit_rate = fix_cache.predict()
            results[size] = hit_rate

            logger.info(f"Cache size {size * 100:.1f}% hit rate: {hit_rate:.2f}%")

        # Find best cache size
        best_size = max(results, key=results.get)
        best_hit_rate = results[best_size]

        logger.info(f"Optimal cache size: {best_size * 100:.1f}% with hit rate: {best_hit_rate:.2f}%")

        # Generate visualization if output is provided
        if args.output:
            if args.output.endswith('.png'):
                logger.info(f"Generating visualization to {args.output}")
                plot_cache_optimization(results, args.output)
            else:
                logger.info(f"Writing results to {args.output}")
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)

        return 0

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=args.verbose)
        return 1


def compare_command(args):
    """Handle the compare command."""
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger('fixcache.compare')

    repos = args.repos.split(',')
    logger.info(f"Comparing {len(repos)} repositories")

    try:
        results = {}

        for repo_path in repos:
            repo_path = repo_path.strip()
            logger.info(f"Analyzing repository: {repo_path}")

            fix_cache = FixCache(
                repo_path=repo_path,
                cache_size=args.cache_size,
                policy=args.policy
            )

            analyze_success = fix_cache.analyze_repository()
            if not analyze_success:
                logger.warning(f"Repository analysis failed for {repo_path}. Skipping.")
                continue

            hit_rate = fix_cache.predict()
            summary = fix_cache.get_summary()

            repo_name = os.path.basename(os.path.abspath(repo_path))
            results[repo_name] = summary

            logger.info(f"{repo_name}: hit rate {hit_rate:.2f}%")

        # Generate comparison visualization if output is provided
        if args.output:
            if args.output.endswith('.png'):
                logger.info(f"Generating comparison visualization to {args.output}")
                plot_repository_comparison(results, args.output)
            else:
                logger.info(f"Writing comparison results to {args.output}")
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)

        return 0

    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}", exc_info=args.verbose)
        return 1


def visualize_command(args):
    """Handle the visualize command."""
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger('fixcache.visualize')

    try:
        # Load input file
        logger.info(f"Loading results from {args.input}")
        with open(args.input, 'r') as f:
            results = json.load(f)

        # Generate visualization
        output_file = args.output if args.output else "fixcache_visualization.png"
        logger.info(f"Generating visualization to {output_file}")

        if args.type == "summary":
            visualize_success = visualize_results(results, output_file)
        elif args.type == "optimization" and isinstance(results, dict):
            visualize_success = plot_cache_optimization(results, output_file)
        elif args.type == "comparison":
            visualize_success = plot_repository_comparison(results, output_file)
        else:
            logger.error(f"Unsupported visualization type: {args.type}")
            return 1

        if not visualize_success:
            logger.warning("Failed to generate visualization. Matplotlib may not be installed.")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}", exc_info=args.verbose)
        return 1


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="FixCache - A tool for bug prediction using repository history"
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='Increase output verbosity'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true', help='Suppress output'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a repository')
    analyze_parser.add_argument('repo_path', help='Path to the repository to analyze')
    analyze_parser.add_argument(
        '--cache-size', '-c', type=float, default=0.2,
        help='Cache size as fraction of total files (default: 0.2)'
    )
    analyze_parser.add_argument(
        '--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
        help='Cache replacement policy (default: BUG)'
    )
    analyze_parser.add_argument(
        '--window-ratio', '-w', type=float, default=0.25,
        help='History window ratio (default: 0.25)'
    )
    analyze_parser.add_argument(
        '--lookback', '-l', type=int,
        help='Number of commits to look back (default: all)'
    )
    analyze_parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization'
    )
    analyze_parser.add_argument(
        '--report', '-r', action='store_true',
        help='Generate report'
    )
    analyze_parser.add_argument(
        '--no-seeding', action='store_true',
        help='Disable cache seeding'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    analyze_parser.add_argument(
        '--format', '-f', choices=['json', 'yaml', 'md', 'txt'], default='json',
        help='Output format (default: json)'
    )

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize cache size')
    optimize_parser.add_argument('repo_path', help='Path to the repository to analyze')
    optimize_parser.add_argument(
        '--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
        help='Cache replacement policy (default: BUG)'
    )
    optimize_parser.add_argument(
        '--min-size', type=float, default=0.05,
        help='Minimum cache size to test (default: 0.05)'
    )
    optimize_parser.add_argument(
        '--max-size', type=float, default=0.5,
        help='Maximum cache size to test (default: 0.5)'
    )
    optimize_parser.add_argument(
        '--step-size', type=float, default=0.05,
        help='Step size for cache testing (default: 0.05)'
    )
    optimize_parser.add_argument(
        '--fine-grained', action='store_true',
        help='Use fine-grained steps'
    )
    optimize_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare repositories')
    compare_parser.add_argument(
        '--repos', '-r', required=True,
        help='Comma-separated list of repository paths'
    )
    compare_parser.add_argument(
        '--cache-size', '-c', type=float, default=0.2,
        help='Cache size to use for comparison (default: 0.2)'
    )
    compare_parser.add_argument(
        '--policy', '-p', choices=['BUG', 'FIFO', 'LRU'], default='BUG',
        help='Cache replacement policy (default: BUG)'
    )
    compare_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )

    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    visualize_parser.add_argument(
        '--input', '-i', required=True,
        help='Input JSON file with analysis results'
    )
    visualize_parser.add_argument(
        '--type', '-t', choices=['summary', 'optimization', 'comparison'], default='summary',
        help='Visualization type (default: summary)'
    )
    visualize_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'optimize':
        return optimize_command(args)
    elif args.command == 'compare':
        return compare_command(args)
    elif args.command == 'visualize':
        return visualize_command(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())