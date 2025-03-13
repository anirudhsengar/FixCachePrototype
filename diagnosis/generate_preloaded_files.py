import os
from git import Repo

class FixCache:
    def __init__(self, local_repo_path, cache_size_percent=0.2):
        self.repo = Repo(local_repo_path)
        self.cache_size_percent = cache_size_percent
        self.cache_size = int(self.get_total_files() * self.cache_size_percent)

    def get_total_files(self):
        """Get the total number of files in the repository."""
        tree = self.repo.tree()
        return sum(1 for _ in tree.traverse() if _.type == 'blob')

    def get_largest_files(self):
        """Get the largest files in the repository."""
        tree = self.repo.tree()
        files = [(blob.path, blob.size) for blob in tree.traverse() if blob.type == 'blob']
        # Sort files by size in descending order
        files.sort(key=lambda x: x[1], reverse=True)
        return [file[0] for file in files]

    def preload_large_files(self):
        """Preload the largest files into the cache."""
        largest_files = self.get_largest_files()
        preloaded_files = largest_files[:self.cache_size]
        with open("openjdk_preloaded_files.txt", "w") as f:
            f.write("\n".join(preloaded_files))
        print(f"Generated 'openjdk_preloaded_files.txt' with {len(preloaded_files)} files.")

if __name__ == "__main__":
    local_repo_path = r"C:\Users\aniru\OneDrive\Desktop\openjdk"  # Adjust this path
    cache = FixCache(local_repo_path, cache_size_percent=0.2)
    cache.preload_large_files()