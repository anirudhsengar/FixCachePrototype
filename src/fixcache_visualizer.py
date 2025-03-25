import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fixcache_visualizer')


class FixCacheVisualizer:
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize the FixCache Visualizer.

        Args:
            output_dir (str): Directory to save visualizations (default: 'visualizations').
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, summary_file: str, timeline_file: str, top_files_file: str) -> Dict[str, Any]:
        """
        Load data from FixCache output files.

        Args:
            summary_file (str): Path to summary JSON file.
            timeline_file (str): Path to hit rate timeline JSON file.
            top_files_file (str): Path to top fault-prone files JSON file.

        Returns:
            Dict[str, Any]: Dictionary of loaded data.
        """
        data = {}

        try:
            with open(summary_file, 'r') as f:
                data['summary'] = json.load(f)

            with open(timeline_file, 'r') as f:
                data['timeline'] = json.load(f)

            with open(top_files_file, 'r') as f:
                data['top_files'] = json.load(f)

            return data

        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            raise

    def plot_hit_rate_timeline(self, timeline_data: List[Tuple[str, float]],
                               output_file: Optional[str] = None) -> str:
        """
        Plot hit rate evolution over commits.

        Args:
            timeline_data (List[Tuple[str, float]]): List of (commit_hash, hit_rate) tuples.
            output_file (str, optional): Output file path (default: None).

        Returns:
            str: Path to the generated visualization file.
        """
        plt.figure(figsize=(12, 6))

        # Extract data
        commit_indices = list(range(len(timeline_data)))
        hit_rates = [hr for _, hr in timeline_data]
        commit_hashes = [commit[:7] for commit, _ in timeline_data]

        # Plot
        plt.plot(commit_indices, hit_rates, marker='o', linestyle='-', markersize=3)

        # Add labels and title
        plt.xlabel('Commits (Chronological Order)')
        plt.ylabel('Hit Rate (%)')
        plt.title('FixCache Hit Rate Evolution')

        # Add grid and limit y-axis
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 100)

        # Add some commit hash labels but avoid overcrowding
        label_indices = np.linspace(0, len(commit_indices) - 1, min(10, len(commit_indices)), dtype=int)
        plt.xticks(label_indices, [commit_hashes[i] for i in label_indices], rotation=45)

        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Save the figure
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'hit_rate_timeline.png')

        plt.savefig(output_file, dpi=300)
        plt.close()

        logger.info(f"Hit rate timeline visualization saved to {output_file}")
        return output_file

    def plot_top_fault_prone_files(self, top_files_data: List[Tuple[str, int]],
                                   output_file: Optional[str] = None,
                                   max_files: int = 15) -> str:
        """
        Plot top fault-prone files.

        Args:
            top_files_data (List[Tuple[str, int]]): List of (file_path, bug_fix_count) tuples.
            output_file (str, optional): Output file path (default: None).
            max_files (int): Maximum number of files to show (default: 15).

        Returns:
            str: Path to the generated visualization file.
        """
        # Limit to max_files
        top_files_data = top_files_data[:max_files]

        plt.figure(figsize=(12, 8))

        # Extract data
        files = [os.path.basename(file) for file, _ in top_files_data]
        counts = [count for _, count in top_files_data]

        # Create horizontal bar chart
        bars = plt.barh(range(len(files)), counts, align='center')

        # Add count labels to the right of bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'{width:.0f}', ha='left', va='center')

        # Add labels and title
        plt.xlabel('Number of Bug Fixes')
        plt.ylabel('Files')
        plt.title('Top Fault-Prone Files')

        # Set y-ticks
        plt.yticks(range(len(files)), files)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')

        # Tight layout
        plt.tight_layout()

        # Save the figure
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'top_fault_prone_files.png')

        plt.savefig(output_file, dpi=300)
        plt.close()

        logger.info(f"Top fault-prone files visualization saved to {output_file}")
        return output_file

    def plot_hit_rate_by_policy(self, policy_data: Dict[str, float],
                                output_file: Optional[str] = None) -> str:
        """
        Plot comparison of hit rates between different replacement policies.

        Args:
            policy_data (Dict[str, float]): Dictionary of {policy_name: hit_rate} pairs.
            output_file (str, optional): Output file path (default: None).

        Returns:
            str: Path to the generated visualization file.
        """
        plt.figure(figsize=(10, 6))

        # Extract data
        policies = list(policy_data.keys())
        hit_rates = list(policy_data.values())

        # Create bar chart
        bars = plt.bar(policies, hit_rates, width=0.6)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.2f}%', ha='center', va='bottom')

        # Add labels and title
        plt.xlabel('Replacement Policy')
        plt.ylabel('Hit Rate (%)')
        plt.title('FixCache Hit Rate by Replacement Policy')

        # Limit y-axis
        plt.ylim(0, max(hit_rates) * 1.2)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Tight layout
        plt.tight_layout()

        # Save the figure
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'hit_rate_by_policy.png')

        plt.savefig(output_file, dpi=300)
        plt.close()

        logger.info(f"Hit rate by policy visualization saved to {output_file}")
        return output_file

    def create_file_heatmap(self, bug_fix_counts: Dict[str, int],
                            output_file: Optional[str] = None) -> str:
        """
        Create a treemap heatmap of files by bug fix frequency.

        Args:
            bug_fix_counts (Dict[str, int]): Dictionary of {file_path: bug_fix_count} pairs.
            output_file (str, optional): Output file path (default: None).

        Returns:
            str: Path to the generated visualization file.
        """
        try:
            import squarify  # Check if squarify is installed
        except ImportError:
            logger.warning("squarify package not found, installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "squarify"])
            import squarify

        plt.figure(figsize=(16, 9))

        # Filter out files with no bug fixes and sort by count
        filtered_data = {k: v for k, v in bug_fix_counts.items() if v > 0}
        sorted_data = sorted(filtered_data.items(), key=lambda x: x[1], reverse=True)

        # Limit data to top 50 files to avoid overcrowding
        if len(sorted_data) > 50:
            sorted_data = sorted_data[:50]

        # Extract data
        files = [os.path.basename(file) for file, _ in sorted_data]
        counts = [count for _, count in sorted_data]

        # Define colormap - red for high bug count, green for low
        norm = plt.Normalize(min(counts), max(counts))
        colors = plt.cm.YlOrRd(norm(counts))

        # Create treemap
        squarify.plot(sizes=counts, label=files, alpha=0.8, color=colors)

        # Add title
        plt.title('Bug Fix Heatmap (File Size Proportional to Bug Fix Count)')
        plt.axis('off')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Bug Fix Count')

        # Tight layout
        plt.tight_layout()

        # Save the figure
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'bug_fix_heatmap.png')

        plt.savefig(output_file, dpi=300)
        plt.close()

        logger.info(f"Bug fix heatmap saved to {output_file}")
        return output_file

    def generate_all_visualizations(self, fixcache_data_dir: str) -> Dict[str, str]:
        """
        Generate all visualizations from FixCache data.

        Args:
            fixcache_data_dir (str): Directory containing FixCache output files.

        Returns:
            Dict[str, str]: Dictionary of {visualization_name: file_path} pairs.
        """
        logger.info(f"Generating all visualizations from data in {fixcache_data_dir}")

        # Paths to input files
        summary_file = os.path.join(fixcache_data_dir, 'fixcache_summary.json')
        timeline_file = os.path.join(fixcache_data_dir, 'hit_rate_timeline.json')
        top_files_file = os.path.join(fixcache_data_dir, 'top_fault_prone_files.json')

        # Load data
        data = self.load_data(summary_file, timeline_file, top_files_file)

        # Generate visualizations
        visualizations = {}

        # Hit rate timeline
        visualizations['hit_rate_timeline'] = self.plot_hit_rate_timeline(data['timeline'])

        # Top fault-prone files
        visualizations['top_fault_prone_files'] = self.plot_top_fault_prone_files(data['top_files'])

        # Create bug fix heatmap if we have the data
        if 'top_files' in data:
            bug_fix_counts = {file: count for file, count in data['top_files']}
            visualizations['bug_fix_heatmap'] = self.create_file_heatmap(bug_fix_counts)

        return visualizations


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize visualizer
        visualizer = FixCacheVisualizer(output_dir='visualizations')

        # Generate all visualizations from FixCache output
        visualizations = visualizer.generate_all_visualizations('output')

        print("Generated visualizations:")
        for name, path in visualizations.items():
            print(f"  {name}: {path}")

    except Exception as e:
        print(f"Error: {e}")