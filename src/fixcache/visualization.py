#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module for FixCache

This module provides visualization capabilities for the FixCache algorithm results,
including charts for hit rates, bug distributions, and comparative analyses.

Author: anirudhsengar
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter, defaultdict

# Setup logging
logger = logging.getLogger(__name__)

# Define color schemes
THEME_COLORS = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'tertiary': '#2ca02c',  # Green
    'quaternary': '#d62728',  # Red
    'background': '#f8f9fa',
    'text': '#333333',
}


def visualize_results(
        results: Dict[str, Any],
        output_file: str = "fixcache_results.png",
        show_plot: bool = False
) -> bool:
    """
    Visualize FixCache prediction results with multiple charts.

    Args:
        results: Results dictionary from FixCache.predict()
        output_file: Path to save the visualization image
        show_plot: Whether to display the plot on screen

    Returns:
        True if visualization was successful, False otherwise
    """
    try:
        # Extract key metrics
        hit_rate = results.get('hit_rate', 0)
        total_files = results.get('total_files', 0)
        total_bugs = results.get('total_bug_fixes', 0)
        top_files = results.get('top_files', [])
        cache_size = results.get('cache_size', 0)
        policy = results.get('policy', 'UNKNOWN')
        repo_path = results.get('repo_path', 'Unknown Repository')
        timestamp = results.get('timestamp', datetime.datetime.now().isoformat())

        # Parse timestamp
        try:
            if isinstance(timestamp, str):
                parsed_time = datetime.datetime.fromisoformat(timestamp)
            else:
                parsed_time = datetime.datetime.now()
        except:
            parsed_time = datetime.datetime.now()

        # Create figure with subplots
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 10), dpi=100)

        # Set up grid for subplots
        grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.3)

        # Create title and repository information
        repo_name = os.path.basename(repo_path)
        fig.suptitle(f'FixCache Bug Prediction Results: {repo_name}',
                     fontsize=16, fontweight='bold', y=0.98)

        # Add subtitle with metadata
        plt.figtext(
            0.5, 0.94,
            f'Generated on {parsed_time.strftime("%Y-%m-%d %H:%M:%S")} | '
            f'Cache Size: {cache_size * 100:.1f}% | Policy: {policy} | '
            f'Total Files: {total_files} | Bug Fixes: {total_bugs}',
            ha='center', fontsize=10, fontstyle='italic'
        )

        # 1. Hit Rate Gauge (top left)
        ax1 = fig.add_subplot(grid[0, 0:2])
        _create_hit_rate_gauge(ax1, hit_rate)

        # 2. Top Bug-Prone Files (top right, spans 2 columns)
        ax2 = fig.add_subplot(grid[0, 2:])
        _plot_top_files(ax2, top_files)

        # 3. Bug Fix Distribution by File Type (middle left)
        ax3 = fig.add_subplot(grid[1, 0:2])
        _plot_bug_distribution_by_type(ax3, top_files)

        # 4. Bug Fix Timeline (middle right, spans 2 columns)
        ax4 = fig.add_subplot(grid[1, 2:])
        _plot_bug_timeline(ax4, results)

        # 5. File Complexity vs. Bug Count (bottom left & middle, spans 3 columns)
        ax5 = fig.add_subplot(grid[2, 0:3])
        _plot_file_complexity(ax5, top_files)

        # 6. Hit/Miss Distribution (bottom right)
        ax6 = fig.add_subplot(grid[2, 3])
        _plot_hit_miss_pie(ax6, results)

        # Add footer with tool information
        plt.figtext(
            0.5, 0.01,
            f'Generated by FixCachePrototype | https://github.com/anirudhsengar/FixCachePrototype',
            ha='center', fontsize=8, color='gray'
        )

        # Save the figure
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        logger.info(f"Visualization saved to {output_file}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return True

    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return False


def _create_hit_rate_gauge(ax: plt.Axes, hit_rate: float) -> None:
    """
    Create a gauge chart showing hit rate.

    Args:
        ax: Matplotlib axes to draw on
        hit_rate: Hit rate percentage
    """
    # Set up gauge parameters
    gauge_min = 0
    gauge_max = 100
    gauge_range = gauge_max - gauge_min

    # Create gauge background
    theta = np.linspace(np.pi, 0, 100)
    r = 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Draw gauge background
    ax.plot(x, y, color='lightgray', linewidth=10, solid_capstyle='round')

    # Calculate hit rate position
    hit_rate_normalized = max(0, min(1, hit_rate / gauge_range))
    hit_theta = np.linspace(np.pi, np.pi - hit_rate_normalized * np.pi, 100)
    hit_x = r * np.cos(hit_theta)
    hit_y = r * np.sin(hit_theta)

    # Determine color based on hit rate
    if hit_rate < 20:
        color = 'red'
    elif hit_rate < 40:
        color = 'orange'
    elif hit_rate < 60:
        color = 'yellow'
    else:
        color = 'green'

    # Draw hit rate gauge segment
    ax.plot(hit_x, hit_y, color=color, linewidth=10, solid_capstyle='round')

    # Add hit rate text
    ax.text(0, 0, f"{hit_rate:.1f}%", ha='center', va='center', fontsize=24,
            fontweight='bold', color=THEME_COLORS['text'])
    ax.text(0, -0.3, "Hit Rate", ha='center', va='center', fontsize=14,
            color=THEME_COLORS['text'])

    # Add gauge ticks and labels
    for val, label in [(0, '0%'), (25, '25%'), (50, '50%'), (75, '75%'), (100, '100%')]:
        val_normalized = val / gauge_range
        tick_theta = np.pi - val_normalized * np.pi
        tick_x = 0.9 * np.cos(tick_theta)
        tick_y = 0.9 * np.sin(tick_theta)
        tick_x2 = 0.82 * np.cos(tick_theta)
        tick_y2 = 0.82 * np.sin(tick_theta)
        ax.plot([tick_x, tick_x2], [tick_y, tick_y2], color='gray', linewidth=1.5)
        ax.text(
            0.75 * np.cos(tick_theta),
            0.75 * np.sin(tick_theta),
            label,
            ha='center', va='center',
            fontsize=8, color='gray'
        )

    # Add interpretation text
    if hit_rate < 20:
        interpretation = "Poor"
    elif hit_rate < 40:
        interpretation = "Fair"
    elif hit_rate < 60:
        interpretation = "Good"
    elif hit_rate < 80:
        interpretation = "Very Good"
    else:
        interpretation = "Excellent"

    ax.text(0, -0.6, f"Performance: {interpretation}", ha='center', va='center',
            fontsize=10, fontweight='bold', color=color)

    # Set up axes
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Prediction Performance', fontsize=14, pad=10)


def _plot_top_files(ax: plt.Axes, top_files: List[Dict[str, Any]], max_files: int = 7) -> None:
    """
    Plot top bug-prone files.

    Args:
        ax: Matplotlib axes to draw on
        top_files: List of top files with bug info
        max_files: Maximum number of files to show
    """
    # Prepare data
    if not top_files:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('Top Bug-Prone Files', fontsize=14)
        ax.axis('off')
        return

    # Limit to max_files
    if len(top_files) > max_files:
        files_to_plot = top_files[:max_files]
    else:
        files_to_plot = top_files

    # Extract file names and bug counts
    file_names = [os.path.basename(file_info['file_path']) for file_info in files_to_plot]
    bug_counts = [file_info['bug_fixes'] for file_info in files_to_plot]

    # Create horizontal bar chart
    y_pos = np.arange(len(file_names))
    ax.barh(y_pos, bug_counts, color=THEME_COLORS['primary'], alpha=0.8)

    # Add file names and bug counts
    for i, count in enumerate(bug_counts):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=10,
                fontweight='bold', color=THEME_COLORS['text'])

    # Add file extensions as colored markers
    for i, file_info in enumerate(files_to_plot):
        ext = os.path.splitext(file_info['file_path'])[1]
        if ext:
            ax.text(-0.8, i, ext, va='center', ha='center', fontsize=8,
                    color='white', fontweight='bold',
                    bbox=dict(facecolor=THEME_COLORS['secondary'],
                              alpha=0.8, boxstyle='round,pad=0.2'))

    # Set up axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(file_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Number of Bug Fixes', fontsize=10)
    ax.set_title('Top Bug-Prone Files', fontsize=14)

    # Add grid lines for readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_bug_distribution_by_type(ax: plt.Axes, files: List[Dict[str, Any]]) -> None:
    """
    Plot bug distribution by file type.

    Args:
        ax: Matplotlib axes to draw on
        files: List of files with bug info
    """
    # Check for data
    if not files:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('Bug Distribution by File Type', fontsize=14)
        ax.axis('off')
        return

    # Extract file extensions and count bugs
    ext_bugs = defaultdict(int)
    for file_info in files:
        ext = os.path.splitext(file_info['file_path'])[1]
        if not ext:
            ext = 'unknown'
        ext_bugs[ext] += file_info['bug_fixes']

    # Sort by bug count
    ext_bugs = dict(sorted(ext_bugs.items(), key=lambda x: x[1], reverse=True))

    # Combine small categories into "Other" if many file types
    if len(ext_bugs) > 6:
        main_exts = dict(list(ext_bugs.items())[:5])
        other_count = sum(list(ext_bugs.values())[5:])
        if other_count > 0:
            main_exts['Other'] = other_count
        ext_bugs = main_exts

    # Create data for plotting
    extensions = list(ext_bugs.keys())
    bug_counts = list(ext_bugs.values())
    total_bugs = sum(bug_counts)

    # Generate colors for the pie chart
    colors = plt.cm.tab10(np.linspace(0, 1, len(extensions)))

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        bug_counts,
        labels=None,
        autopct=None,
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True}
    )

    # Add percentage labels inside pie
    for i, autotext in enumerate(autotexts):
        percentage = bug_counts[i] / total_bugs * 100
        autotext.set_text(f"{percentage:.1f}%")
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Create legend with file extensions and counts
    legend_labels = [f"{ext} ({count} bugs, {count / total_bugs * 100:.1f}%)"
                     for ext, count in zip(extensions, bug_counts)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(-0.1, 0, 0.5, 1),
              fontsize=9)

    # Add title
    ax.set_title('Bug Distribution by File Type', fontsize=14)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')


def _plot_bug_timeline(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot bug timeline if available in results.

    Args:
        ax: Matplotlib axes to draw on
        results: Results dictionary
    """
    # Check if we have time distribution data
    if 'temporal_distribution' not in results:
        # Simulate time data if not available
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        bug_counts = np.random.randint(1, 10, size=12)

        # Plot simulated data
        ax.plot(months, bug_counts, marker='o', linestyle='-', color=THEME_COLORS['tertiary'],
                linewidth=2, markersize=8)

        # Add area under the curve
        ax.fill_between(months, bug_counts, alpha=0.3, color=THEME_COLORS['tertiary'])

        # Set up axes
        ax.set_xlabel('Time (Months)', fontsize=10)
        ax.set_ylabel('Bug Fixes', fontsize=10)
        ax.set_title('Bug Fix Timeline (Simulated)', fontsize=14)

        # Add note about simulated data
        ax.text(0.5, 0.95, "Note: Using simulated data for demonstration",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, fontstyle='italic', color='gray')
    else:
        # Use real temporal distribution data
        temporal = results['temporal_distribution']

        # Plot real data
        ax.plot(temporal['labels'], temporal['counts'], marker='o', linestyle='-',
                color=THEME_COLORS['tertiary'], linewidth=2, markersize=8)

        # Add area under the curve
        ax.fill_between(temporal['labels'], temporal['counts'], alpha=0.3,
                        color=THEME_COLORS['tertiary'])

        # Set up axes
        period_type = temporal.get('period_type', 'time')
        ax.set_xlabel(f'Time ({period_type.capitalize()}s)', fontsize=10)
        ax.set_ylabel('Bug Fixes', fontsize=10)
        ax.set_title('Bug Fix Timeline', fontsize=14)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_file_complexity(ax: plt.Axes, files: List[Dict[str, Any]]) -> None:
    """
    Plot file complexity vs bug count.

    Args:
        ax: Matplotlib axes to draw on
        files: List of files with bug info
    """
    # Check for data
    if not files:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('File Complexity vs. Bug Count', fontsize=14)
        ax.axis('off')
        return

    # Extract data - we'll use file type as a proxy for complexity group
    file_types = {}
    for file_info in files:
        ext = os.path.splitext(file_info['file_path'])[1]
        if not ext:
            ext = 'unknown'

        if ext not in file_types:
            file_types[ext] = []

        file_types[ext].append({
            'name': os.path.basename(file_info['file_path']),
            'bugs': file_info['bug_fixes']
        })

    # Set up colors for different file types
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_types)))
    color_map = {ext: colors[i] for i, ext in enumerate(file_types.keys())}

    # Plot each file as a bubble
    for ext, files in file_types.items():
        # Extract x and y coordinates for this file type
        x = np.arange(len(files))
        y = [f['bugs'] for f in files]
        sizes = [f['bugs'] * 50 for f in files]  # Scale bubble size

        # Plot bubbles
        scatter = ax.scatter(
            x, y, s=sizes, c=[color_map[ext]],
            alpha=0.7, edgecolors='w', linewidth=0.5,
            label=ext
        )

        # Add file names as annotations
        for i, file in enumerate(files):
            if file['bugs'] > 1:  # Only label significant files
                ax.annotate(
                    file['name'],
                    (x[i], y[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )

    # Create legend
    legend = ax.legend(
        title="File Types",
        loc="upper right",
        fontsize=9
    )
    legend.get_title().set_fontsize(10)

    # Set up axes
    ax.set_xlabel('Files Grouped by Type', fontsize=10)
    ax.set_ylabel('Number of Bug Fixes', fontsize=10)
    ax.set_title('File Type vs. Bug Count', fontsize=14)

    # Remove actual x-ticks as they're not meaningful in this context
    ax.set_xticks([])

    # Add grid for y-axis only
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def _plot_hit_miss_pie(ax: plt.Axes, results: Dict[str, Any]) -> None:
    """
    Plot hit/miss distribution as a pie chart.

    Args:
        ax: Matplotlib axes to draw on
        results: Results dictionary
    """
    # Extract hit and miss counts
    hit_count = results.get('hit_count', 0)
    miss_count = results.get('miss_count', 0)

    # Check for data
    if hit_count == 0 and miss_count == 0:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title('Hit/Miss Distribution', fontsize=14)
        ax.axis('off')
        return

    # Prepare data for pie chart
    labels = ['Hits', 'Misses']
    sizes = [hit_count, miss_count]
    colors = [THEME_COLORS['tertiary'], THEME_COLORS['quaternary']]
    explode = (0.05, 0)  # Explode the first slice (Hits)

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=None,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True}
    )

    # Customize autopct text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    # Add legend
    legend_labels = [f"{label} ({size})" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='center', fontsize=9)

    # Add title
    ax.set_title('Hit/Miss Distribution', fontsize=14)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')


def plot_cache_optimization(
        results: Dict[float, float],
        output_file: str = "cache_optimization.png",
        show_plot: bool = False
) -> bool:
    """
    Plot cache size optimization results.

    Args:
        results: Dictionary mapping cache sizes to hit rates
        output_file: Path to save the visualization image
        show_plot: Whether to display the plot on screen

    Returns:
        True if visualization was successful, False otherwise
    """
    try:
        # Check for data
        if not results:
            logger.error("No cache optimization results to visualize")
            return False

        # Extract cache sizes and hit rates
        cache_sizes = [size * 100 for size in results.keys()]  # Convert to percentages
        hit_rates = list(results.values())

        # Create figure
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Plot line
        ax.plot(cache_sizes, hit_rates, 'o-', linewidth=2, markersize=8,
                color=THEME_COLORS['primary'])

        # Add area under the curve
        ax.fill_between(cache_sizes, hit_rates, alpha=0.3, color=THEME_COLORS['primary'])

        # Find optimal cache size
        optimal_idx = hit_rates.index(max(hit_rates))
        optimal_size = cache_sizes[optimal_idx]
        optimal_hit_rate = hit_rates[optimal_idx]

        # Highlight optimal point
        ax.plot(optimal_size, optimal_hit_rate, 'o', markersize=12,
                markerfacecolor='red', markeredgecolor='w', markeredgewidth=2)

        # Add annotation for optimal point
        ax.annotate(
            f"Optimal: {optimal_size:.1f}% ({optimal_hit_rate:.2f}% hit rate)",
            (optimal_size, optimal_hit_rate),
            xytext=(10, 20),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black')
        )

        # Set up axes
        ax.set_xlabel('Cache Size (% of total files)', fontsize=12)
        ax.set_ylabel('Hit Rate (%)', fontsize=12)
        ax.set_title('Cache Size Optimization Results', fontsize=16, fontweight='bold')

        # Set axis limits
        ax.set_xlim(min(cache_sizes) * 0.9, max(cache_sizes) * 1.1)
        ax.set_ylim(0, max(hit_rates) * 1.2)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add footer with tool information
        plt.figtext(
            0.5, 0.01,
            f'Generated by FixCachePrototype | https://github.com/anirudhsengar/FixCachePrototype',
            ha='center', fontsize=8, color='gray'
        )

        # Add timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(
            0.98, 0.02,
            f'Generated: {current_time}',
            ha='right', fontsize=8, color='gray'
        )

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        logger.info(f"Optimization visualization saved to {output_file}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return True

    except Exception as e:
        logger.error(f"Error generating optimization visualization: {str(e)}")
        return False


def plot_repository_comparison(
        results: Dict[str, Dict[str, Any]],
        output_file: str = "repo_comparison.png",
        show_plot: bool = False
) -> bool:
    """
    Plot comparison of multiple repositories.

    Args:
        results: Dictionary mapping repository paths to their results
        output_file: Path to save the visualization image
        show_plot: Whether to display the plot on screen

    Returns:
        True if visualization was successful, False otherwise
    """
    try:
        # Check for data
        if not results or len(results) < 2:
            logger.error("Not enough repositories to compare (need at least 2)")
            return False

        # Extract repository names and hit rates
        repo_names = [os.path.basename(repo_path) for repo_path in results.keys()]
        hit_rates = [repo_results.get('hit_rate', 0) for repo_results in results.values()]
        bug_counts = [repo_results.get('total_bug_fixes', 0) for repo_results in results.values()]
        file_counts = [repo_results.get('total_files', 0) for repo_results in results.values()]

        # Calculate bug density (bugs per file)
        bug_densities = [
            count / max(1, files) * 100 for count, files in zip(bug_counts, file_counts)
        ]

        # Create figure with subplots
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

        # 1. Hit Rate Comparison (Bar Chart)
        bars = ax1.bar(repo_names, hit_rates, color=THEME_COLORS['primary'], alpha=0.8)

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        # Set up first chart
        ax1.set_xlabel('Repository', fontsize=10)
        ax1.set_ylabel('Hit Rate (%)', fontsize=10)
        ax1.set_title('Hit Rate Comparison', fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim(0, max(hit_rates) * 1.2)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # 2. Bug Count Comparison (Bar Chart)
        bars = ax2.bar(repo_names, bug_counts, color=THEME_COLORS['secondary'], alpha=0.8)

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        # Set up second chart
        ax2.set_xlabel('Repository', fontsize=10)
        ax2.set_ylabel('Bug Fix Count', fontsize=10)
        ax2.set_title('Bug Fix Comparison', fontsize=14)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, max(bug_counts) * 1.2)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. Bug Density Comparison (Bar Chart)
        bars = ax3.bar(repo_names, bug_densities, color=THEME_COLORS['tertiary'], alpha=0.8)

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(
                f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        # Set up third chart
        ax3.set_xlabel('Repository', fontsize=10)
        ax3.set_ylabel('Bug Density (% of files with bugs)', fontsize=10)
        ax3.set_title('Bug Density Comparison', fontsize=14)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.set_ylim(0, max(bug_densities) * 1.2)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # Add main title
        fig.suptitle('Repository Comparison', fontsize=16, fontweight='bold', y=0.98)

        # Add footer with tool information
        plt.figtext(
            0.5, 0.01,
            f'Generated by FixCachePrototype | https://github.com/anirudhsengar/FixCachePrototype',
            ha='center', fontsize=8, color='gray'
        )

        # Add timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(
            0.98, 0.02,
            f'Generated: {current_time}',
            ha='right', fontsize=8, color='gray'
        )

        # Save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        logger.info(f"Repository comparison visualization saved to {output_file}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return True

    except Exception as e:
        logger.error(f"Error generating repository comparison visualization: {str(e)}")
        return False