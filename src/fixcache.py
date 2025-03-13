from git import Repo
import os
import re


class FixCache:
    def __init__(self, local_repo_path, cache_size_percent=0.2):
        """
        Initialize the FixCache with a local repository path and cache size percentage.

        Args:
            local_repo_path (str): Path to the local Git repository.
            cache_size_percent (float): Percentage of total files to use as cache size (default: 0.2).
        """
        if not os.path.exists(local_repo_path):
            raise ValueError(f"Repository path {local_repo_path} does not exist.")

        self.repo = Repo(local_repo_path)
        # Calculate total number of files in HEAD
        total_files = sum(1 for blob in self.repo.commit('HEAD').tree.traverse() if blob.type == 'blob')
        if total_files == 0:
            raise ValueError("No files found in the repository.")

        self.cache_size = max(1, int(total_files * cache_size_percent))  # Ensure at least 1 slot
        self.cache = set()  # Cache stores file paths
        self.bug_fix_counts = {}  # Tracks number of bug fixes per file
        self.hit_count = 0  # Number of times a fixed file was in cache
        self.miss_count = 0  # Number of times a fixed file was not in cache
        self.preload_large_files()

    def preload_large_files(self):
        """
        Preload the cache with the largest files based on byte size, as a proxy for fault-proneness.
        """
        # Collect all files and their sizes from HEAD
        files = []
        for blob in self.repo.commit('HEAD').tree.traverse():
            if blob.type == 'blob':
                files.append((blob.path, blob.size))

        # Sort by size descending
        files.sort(key=lambda x: x[1], reverse=True)

        # Add the largest files up to cache_size
        for file_path, _ in files[:self.cache_size]:
            self.cache.add(file_path)
            self.bug_fix_counts[file_path] = 0  # Initialize bug fix count

    def is_bug_fix_commit(self, commit):
        """
        Determine if a commit is a bug fix based on keywords in the message.

        Args:
            commit: GitPython Commit object.

        Returns:
            bool: True if the commit is a bug fix, False otherwise.
        """
        message = commit.message.lower()

        # Check for issue references (e.g., "JDK-123456" or "123456: ")
        if re.search(r'jdk-\d+', message) or re.match(r'^\d+:', message):
            # Optionally refine further with keywords or title analysis
            keywords = ["fix", "fixed", "bug", "resolved", "corrected"]
            if any(keyword in message for keyword in keywords):
                return True
            # Assume issue reference alone is sufficient in OpenJDK context
            return True

        return False

    def process_commits(self, max_commits=None):
        """
        Process commits in chronological order to update the cache and calculate hit rate.

        Args:
            max_commits (int, optional): Maximum number of commits to process (default: None, all commits).
        """
        # Fetch commits in chronological order (oldest to newest)
        commits = list(self.repo.iter_commits('master', max_count=max_commits, reverse=True))
        if not commits:
            print("No commits found to process.")
            return

        for commit in commits:
            if self.is_bug_fix_commit(commit):
                # Get files changed in this commit
                fixed_files = []
                if commit.parents:
                    # Compare with first parent (handles merges by focusing on mainline)
                    diff = commit.diff(commit.parents[0])
                    fixed_files = [item.a_path for item in diff if item.a_path and not item.deleted_file]

                if not fixed_files:
                    continue  # Skip if no files were changed

                # Check for hits and misses before updating cache
                for file in fixed_files:
                    if file in self.cache:
                        self.hit_count += 1
                    else:
                        self.miss_count += 1
                    # Update bug fix count
                    self.bug_fix_counts[file] = self.bug_fix_counts.get(file, 0) + 1

                # Update cache with all fixed files
                for file in fixed_files:
                    if file not in self.cache:
                        self.cache.add(file)
                        # Apply BUG replacement policy if cache exceeds size
                        while len(self.cache) > self.cache_size:
                            self._remove_least_faulty()

    def _remove_least_faulty(self):
        """
        Remove the least faulty file from the cache based on bug fix counts (BUG policy).
        In case of ties, remove the smallest file by size.
        """
        # Find minimum bug fix count among cached files
        min_count = min(self.bug_fix_counts.get(file, 0) for file in self.cache)
        candidates = [file for file in self.cache if self.bug_fix_counts.get(file, 0) == min_count]

        if len(candidates) == 1:
            self.cache.remove(candidates[0])
        else:
            # Resolve tie by removing the smallest file (by byte size)
            file_sizes = []
            for file in candidates:
                try:
                    blob = self.repo.commit('HEAD').tree / file
                    file_sizes.append((file, blob.size))
                except KeyError:
                    # File might not exist in HEAD; assign minimal size
                    file_sizes.append((file, 0))
            file_to_remove = min(file_sizes, key=lambda x: x[1])[0]
            self.cache.remove(file_to_remove)

    def get_hit_rate(self):
        """
        Calculate the hit rate as a percentage.

        Returns:
            float: Hit rate percentage, or 0 if no bug fixes were processed.
        """
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return (self.hit_count / total) * 100


if __name__ == "__main__":
    # Example usage
    local_repo_path = r"C:\Users\aniru\OneDrive\Desktop\openjdk"
    try:
        cache = FixCache(local_repo_path, cache_size_percent=0.2)
        cache.process_commits(max_commits=500)  # Limit to 500 commits for large repos
        hit_rate = cache.get_hit_rate()
        print(f"Hit rate: {hit_rate:.2f}%")
    except Exception as e:
        print(f"Error: {e}")