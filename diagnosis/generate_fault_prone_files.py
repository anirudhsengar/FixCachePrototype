import re
from git import Repo

class FixCache:
    def __init__(self, local_repo_path):
        self.repo = Repo(local_repo_path)

    def is_bug_fix_commit(self, commit):
        """Check if a commit is a bug fix based on its message."""
        message = commit.message.lower()
        # Look for patterns like 'JDK-123456' or '123456: ' indicating a bug fix
        if re.search(r'jdk-\d+', message) or re.match(r'^\d+:', message):
            return True
        return False

    def get_fault_prone_files(self, max_commits=500):
        """Identify files modified in bug fix commits."""
        fault_prone_files = set()
        # Iterate through the most recent commits in the 'master' branch
        commits = list(self.repo.iter_commits('master', max_count=max_commits, reverse=True))
        for commit in commits:
            if self.is_bug_fix_commit(commit) and commit.parents:  # Ensure commit has a parent
                # Get files modified in this commit (exclude deleted files)
                fixed_files = [item.a_path for item in commit.diff(commit.parents[0])
                              if item.a_path and not item.deleted_file]
                fault_prone_files.update(fixed_files)
        # Write the list of fault-prone files to a text file
        with open("openjdk_fault_prone_files.txt", "w") as f:
            f.write("\n".join(sorted(fault_prone_files)))
        print(f"Generated 'openjdk_fault_prone_files.txt' with {len(fault_prone_files)} files.")
        return fault_prone_files

if __name__ == "__main__":
    # Use the OpenJDK repository path you provided
    local_repo_path = r"C:\Users\aniru\OneDrive\Desktop\openjdk"
    cache = FixCache(local_repo_path)
    cache.get_fault_prone_files()