from git import Repo
import os
import re
import time
import json
import logging
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Pattern, Union
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fixcache')


class ReplacementPolicy(Enum):
    """Enumeration of cache replacement policies."""
    BUG = "bug"  # Belady's Optimal for bug fixes
    LRU = "lru"  # Least Recently Used
    FIFO = "fifo"  # First In, First Out


class FixCache:
    def __init__(
            self,
            local_repo_path: str,
            cache_size_percent: float = 0.2,
            bug_fix_patterns: List[str] = None,
            replacement_policy: ReplacementPolicy = ReplacementPolicy.BUG,
            commit_cache_size: int = 1000
    ):
        """
        Initialize the FixCache with a local repository path and configuration options.

        Args:
            local_repo_path (str): Path to the local Git repository.
            cache_size_percent (float): Percentage of total files to use as cache size (default: 0.2).
            bug_fix_patterns (List[str], optional): Custom regex patterns for bug fix detection.
            replacement_policy (ReplacementPolicy): Cache replacement policy (default: BUG).
            commit_cache_size (int): Size of commit cache to reduce redundant processing (default: 1000).
        """
        if not os.path.exists(local_repo_path):
            raise ValueError(f"Repository path {local_repo_path} does not exist.")

        self.repo = Repo(local_repo_path)
        self.local_repo_path = local_repo_path
        self.replacement_policy = replacement_policy

        # Initialize commit cache
        self.commit_cache_size = commit_cache_size
        self._commit_cache = OrderedDict()  # Cache for commit analysis to reduce redundant processing

        # Compile bug fix patterns for faster matching
        self.bug_fix_patterns = bug_fix_patterns or [
            r"fix", r"bug", r"defect", r"issue", r"error", r"crash",
            r"closes\s+#\d+", r"fixes\s+#\d+", r"resolves\s+#\d+",
            r"close\s+#\d+", r"resolve\s+#\d+", r"corrected"
        ]
        self.bug_fix_regex = self._compile_patterns(self.bug_fix_patterns)

        # Calculate total number of files in HEAD and determine cache size
        total_files = self._count_total_files()
        if total_files == 0:
            raise ValueError("No files found in the repository.")

        self.cache_size = max(1, int(total_files * cache_size_percent))  # Ensure at least 1 slot
        logger.info(f"Repository contains {total_files} files. Cache size set to {self.cache_size} files.")

        # Core data structures
        self.cache: Set[str] = set()  # Cache stores file paths
        self.bug_fix_counts: Dict[str, int] = {}  # Tracks number of bug fixes per file
        self.hit_count: int = 0  # Number of times a fixed file was in cache
        self.miss_count: int = 0  # Number of times a fixed file was not in cache

        # For LRU policy
        self.access_order: OrderedDict = OrderedDict()  # Tracks access order for LRU policy

        # For FIFO policy
        self.entry_order: List[str] = []  # Tracks entry order for FIFO policy

        # File size cache
        self.file_size_cache: Dict[str, int] = {}  # Cache file sizes to reduce redundant calculations

        # Metrics collection
        self.hit_rate_timeline: List[Tuple[str, float]] = []  # [(commit_hash, hit_rate), ...]
        self.fixed_files_history: Dict[str, List[str]] = {}  # {commit_hash: [fixed_files], ...}

        # Preload cache
        self.preload_large_files()

    def _count_total_files(self) -> int:
        """Count the total number of files in the repository HEAD."""
        start_time = time.time()
        total_files = sum(1 for blob in self.repo.commit('HEAD').tree.traverse() if blob.type == 'blob')
        logger.info(f"Counted {total_files} files in {time.time() - start_time:.2f} seconds")
        return total_files

    def _compile_patterns(self, patterns: List[str]) -> List[Pattern]:
        """Compile regex patterns for faster matching."""
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def preload_large_files(self) -> None:
        """
        Preload the cache with the largest files based on byte size, as a proxy for fault-proneness.
        """
        logger.info("Preloading cache with largest files...")
        start_time = time.time()

        # Collect all files and their sizes from HEAD
        files = []
        for blob in self.repo.commit('HEAD').tree.traverse():
            if blob.type == 'blob':
                file_size = blob.size
                self.file_size_cache[blob.path] = file_size  # Cache file size
                files.append((blob.path, file_size))

        # Sort by size descending
        files.sort(key=lambda x: x[1], reverse=True)

        # Add the largest files up to cache_size
        for file_path, _ in files[:self.cache_size]:
            self.cache.add(file_path)
            self.bug_fix_counts[file_path] = 0  # Initialize bug fix count

            # Update data structures for different replacement policies
            if self.replacement_policy == ReplacementPolicy.LRU:
                self.access_order[file_path] = time.time()
            elif self.replacement_policy == ReplacementPolicy.FIFO:
                self.entry_order.append(file_path)

        logger.info(f"Preloaded {len(self.cache)} files in {time.time() - start_time:.2f} seconds")

    def is_bug_fix_commit(self, commit) -> bool:
        """
        Determine if a commit is a bug fix based on regex patterns in the message.

        Args:
            commit: GitPython Commit object.

        Returns:
            bool: True if the commit is a bug fix, False otherwise.
        """
        # Check if this commit has already been analyzed
        if commit.hexsha in self._commit_cache:
            return self._commit_cache[commit.hexsha]

        message = commit.message.lower()
        is_bug_fix = any(pattern.search(message) for pattern in self.bug_fix_regex)

        # Store result in cache
        self._commit_cache[commit.hexsha] = is_bug_fix

        # Maintain cache size
        if len(self._commit_cache) > self.commit_cache_size:
            self._commit_cache.popitem(last=False)  # Remove oldest item (FIFO)

        return is_bug_fix

    def get_file_size(self, file_path: str, commit='HEAD') -> int:
        """
        Get the size of a file with caching.

        Args:
            file_path (str): Path to the file.
            commit: The commit to check (default: 'HEAD').

        Returns:
            int: File size in bytes, or 0 if file not found.
        """
        cache_key = f"{commit}:{file_path}"

        if cache_key in self.file_size_cache:
            return self.file_size_cache[cache_key]

        try:
            blob = self.repo.commit(commit).tree / file_path
            size = blob.size
            self.file_size_cache[cache_key] = size
            return size
        except (KeyError, AttributeError):
            self.file_size_cache[cache_key] = 0
            return 0

    def process_commits(self, max_commits: Optional[int] = None, branch: str = 'master',
                        collect_metrics: bool = True) -> None:
        """
        Process commits in chronological order to update the cache and calculate hit rate.

        Args:
            max_commits (int, optional): Maximum number of commits to process (default: None, all commits).
            branch (str): Branch to analyze (default: 'master').
            collect_metrics (bool): Whether to collect detailed metrics during processing (default: True).
        """
        # Fetch commits in chronological order (oldest to newest)
        try:
            logger.info(f"Processing commits from branch '{branch}'...")
            start_time = time.time()

            # Try to get commits from specified branch, fall back to 'main' if 'master' doesn't exist
            try:
                commits = list(self.repo.iter_commits(branch, max_count=max_commits, reverse=True))
            except Exception as e:
                if branch == 'master':
                    logger.warning(f"Branch '{branch}' not found, trying 'main' instead: {e}")
                    commits = list(self.repo.iter_commits('main', max_count=max_commits, reverse=True))
                else:
                    raise

            if not commits:
                logger.warning("No commits found to process.")
                return

            logger.info(f"Found {len(commits)} commits to process")

            for i, commit in enumerate(commits):
                if i % 100 == 0:
                    logger.info(f"Processing commit {i + 1}/{len(commits)}: {commit.hexsha[:8]}")

                if self.is_bug_fix_commit(commit):
                    # Get files changed in this commit
                    fixed_files = self._get_fixed_files(commit)

                    if not fixed_files:
                        continue  # Skip if no files were changed

                    # Record for metrics
                    if collect_metrics:
                        self.fixed_files_history[commit.hexsha] = fixed_files.copy()

                    # Check for hits and misses before updating cache
                    for file in fixed_files:
                        self._process_fixed_file(file)

                    # Update cache with all fixed files
                    for file in fixed_files:
                        self._add_to_cache(file)

                    # Collect hit rate at this point
                    if collect_metrics:
                        current_hit_rate = self.get_hit_rate()
                        self.hit_rate_timeline.append((commit.hexsha, current_hit_rate))

            elapsed_time = time.time() - start_time
            logger.info(f"Processed {len(commits)} commits in {elapsed_time:.2f} seconds")
            logger.info(f"Final hit rate: {self.get_hit_rate():.2f}%")

        except Exception as e:
            logger.error(f"Error processing commits: {e}", exc_info=True)
            raise

    def _get_fixed_files(self, commit) -> List[str]:
        """
        Get files that were fixed in a commit.

        Args:
            commit: GitPython Commit object.

        Returns:
            List[str]: List of fixed file paths.
        """
        fixed_files = []
        if commit.parents:
            # Compare with first parent (handles merges by focusing on mainline)
            diff = commit.diff(commit.parents[0])
            fixed_files = [item.a_path for item in diff if item.a_path and not item.deleted_file]
        return fixed_files

    def _process_fixed_file(self, file: str) -> None:
        """
        Process a fixed file, updating hit/miss counts and bug fix counts.

        Args:
            file (str): Path to the fixed file.
        """
        if file in self.cache:
            self.hit_count += 1
            # Update access time for LRU policy
            if self.replacement_policy == ReplacementPolicy.LRU:
                self.access_order[file] = time.time()
        else:
            self.miss_count += 1

        # Update bug fix count
        self.bug_fix_counts[file] = self.bug_fix_counts.get(file, 0) + 1

    def _add_to_cache(self, file: str) -> None:
        """
        Add a file to the cache, applying the appropriate replacement policy if needed.

        Args:
            file (str): Path to the file to add to cache.
        """
        if file not in self.cache:
            self.cache.add(file)

            # Update data structures for different replacement policies
            if self.replacement_policy == ReplacementPolicy.LRU:
                self.access_order[file] = time.time()
            elif self.replacement_policy == ReplacementPolicy.FIFO:
                self.entry_order.append(file)

            # Apply replacement policy if cache exceeds size
            while len(self.cache) > self.cache_size:
                self._apply_replacement_policy()

    def _apply_replacement_policy(self) -> None:
        """Apply the selected cache replacement policy to remove a file."""
        if self.replacement_policy == ReplacementPolicy.BUG:
            self._remove_least_faulty()
        elif self.replacement_policy == ReplacementPolicy.LRU:
            self._remove_least_recently_used()
        elif self.replacement_policy == ReplacementPolicy.FIFO:
            self._remove_first_in()

    def _remove_least_faulty(self) -> None:
        """
        Remove the least faulty file from the cache based on bug fix counts (BUG policy).
        In case of ties, remove the smallest file by size.
        """
        # Find minimum bug fix count among cached files
        min_count = min(self.bug_fix_counts.get(file, 0) for file in self.cache)
        candidates = [file for file in self.cache if self.bug_fix_counts.get(file, 0) == min_count]

        if len(candidates) == 1:
            file_to_remove = candidates[0]
        else:
            # Resolve tie by removing the smallest file (by byte size)
            file_sizes = [(file, self.get_file_size(file)) for file in candidates]
            file_to_remove = min(file_sizes, key=lambda x: x[1])[0]

        self._remove_from_cache(file_to_remove)

    def _remove_least_recently_used(self) -> None:
        """Remove the least recently used file from the cache (LRU policy)."""
        if not self.access_order:
            # Fallback to BUG policy if access_order is empty
            self._remove_least_faulty()
            return

        # Get the least recently used file
        file_to_remove = min(self.access_order.items(), key=lambda x: x[1])[0]
        self._remove_from_cache(file_to_remove)

    def _remove_first_in(self) -> None:
        """Remove the first file added to the cache (FIFO policy)."""
        if not self.entry_order:
            # Fallback to BUG policy if entry_order is empty
            self._remove_least_faulty()
            return

        file_to_remove = self.entry_order.pop(0)
        self._remove_from_cache(file_to_remove)

    def _remove_from_cache(self, file: str) -> None:
        """
        Remove a file from the cache and update related data structures.

        Args:
            file (str): Path to the file to remove.
        """
        self.cache.remove(file)

        # Update other data structures
        if self.replacement_policy == ReplacementPolicy.LRU and file in self.access_order:
            del self.access_order[file]
        elif self.replacement_policy == ReplacementPolicy.FIFO:
            # File should already be removed from entry_order by _remove_first_in
            pass

    def get_hit_rate(self) -> float:
        """
        Calculate the hit rate as a percentage.

        Returns:
            float: Hit rate percentage, or 0 if no bug fixes were processed.
        """
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return (self.hit_count / total) * 100

    def get_top_fault_prone_files(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the top N fault-prone files based on bug fix counts.

        Args:
            n (int): Number of top files to return (default: 10).

        Returns:
            List[Tuple[str, int]]: List of (file_path, bug_fix_count) tuples.
        """
        sorted_files = sorted(self.bug_fix_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:n]

    def export_metrics(self, output_dir: str = '.') -> Dict[str, str]:
        """
        Export metrics to JSON files.

        Args:
            output_dir (str): Directory to save the output files (default: current directory).

        Returns:
            Dict[str, str]: Paths to the generated output files.
        """
        os.makedirs(output_dir, exist_ok=True)

        output_files = {}

        # Export hit rate timeline
        hit_rate_file = os.path.join(output_dir, 'hit_rate_timeline.json')
        with open(hit_rate_file, 'w') as f:
            json.dump(self.hit_rate_timeline, f, indent=2)
        output_files['hit_rate_timeline'] = hit_rate_file

        # Export top fault-prone files
        top_files_file = os.path.join(output_dir, 'top_fault_prone_files.json')
        with open(top_files_file, 'w') as f:
            json.dump(self.get_top_fault_prone_files(20), f, indent=2)
        output_files['top_fault_prone_files'] = top_files_file

        # Export summary statistics
        summary_file = os.path.join(output_dir, 'fixcache_summary.json')
        summary = {
            'hit_rate': self.get_hit_rate(),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_size': self.cache_size,
            'replacement_policy': self.replacement_policy.value,
            'total_files_analyzed': len(self.bug_fix_counts),
            'total_bug_fixes': sum(self.bug_fix_counts.values())
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        output_files['summary'] = summary_file

        return output_files

    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary report.

        Returns:
            str: Summary report.
        """
        report = [
            "FixCache Analysis Summary",
            "========================",
            f"Repository: {os.path.basename(self.local_repo_path)}",
            f"Cache Size: {self.cache_size} files ({self.cache_size / len(self.bug_fix_counts) * 100:.1f}% of total)",
            f"Replacement Policy: {self.replacement_policy.value.upper()}",
            f"Bug Fix Patterns: {', '.join(self.bug_fix_patterns)}",
            "",
            "Performance Metrics:",
            f"  Hit Rate: {self.get_hit_rate():.2f}%",
            f"  Hit Count: {self.hit_count}",
            f"  Miss Count: {self.miss_count}",
            f"  Total Files Analyzed: {len(self.bug_fix_counts)}",
            f"  Total Bug Fixes: {sum(self.bug_fix_counts.values())}",
            "",
            "Top 10 Fault-Prone Files:",
        ]

        for i, (file, count) in enumerate(self.get_top_fault_prone_files(10), 1):
            report.append(f"  {i}. {file} ({count} bug fixes)")

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    local_repo_path = r"C:\Users\aniru\OneDrive\Desktop\openjdk"
    try:
        # Initialize FixCache with custom configuration
        cache = FixCache(
            local_repo_path=local_repo_path,
            cache_size_percent=0.2,
            bug_fix_patterns=None,  # Use default patterns
            replacement_policy=ReplacementPolicy.BUG,
            commit_cache_size=1000
        )

        # Process commits
        cache.process_commits(max_commits=500, branch='master', collect_metrics=True)

        # Generate and print summary report
        print(cache.get_summary_report())

        # Export metrics to files
        output_files = cache.export_metrics('output')
        print(f"\nMetrics exported to: {', '.join(output_files.values())}")

    except Exception as e:
        print(f"Error: {e}")