from github import Github
import pandas as pd
import subprocess

class FixCache:
    def __init__(self, repo, cache_size=0.1):
        self.g = Github("your-access-token")  # Replace with a valid token
        self.repo = self.g.get_repo(repo)
        self.cache_size = int(len(self.repo.get_contents("")) * cache_size)
        self.cache = {}  # {file: hit_count}

    def fetch_commits(self):
        try:
            return self.repo.get_commits()
        except Exception as e:
            print(f"Error fetching commits: {e}")
            return []

    def detect_bug_fixes(self, commits):
        bug_fixes = []
        for commit in commits:
            if any(keyword in commit.commit.message.lower() for keyword in ["fix", "bug"]):
                bug_fixes.append(commit)
        return bug_fixes

    def trace_bug_introducing(self, file, commit_sha):
        # Simulate git blame (simplified, replace with actual Git command)
        result = subprocess.run(["git", "blame", "-L", "1,+1", file, "-w", commit_sha],
                               capture_output=True, text=True)
        return result.stdout.split()[0] if result.stdout else commit_sha

    def update_cache(self, file, co_changed_files):
        if file not in self.cache:
            self.cache[file] = 0
        self.cache[file] += 1
        for co_file in co_changed_files[:2]:  # Spatial locality, block size = 2
            if co_file not in self.cache:
                self.cache[co_file] = 0
            self.cache[co_file] += 1
        # BUG policy: Remove least faulty if over size
        if len(self.cache) > self.cache_size:
            self.cache = dict(sorted(self.cache.items(), key=lambda x: x[1])[:self.cache_size])

    def predict_faults(self):
        return sorted(self.cache.items(), key=lambda x: x[1], reverse=True)[:10]

    def run(self):
        commits = self.fetch_commits()
        bug_fixes = self.detect_bug_fixes(commits)
        for fix in bug_fixes:
            for file in fix.files:
                co_changed = [f.filename for f in fix.files if f.filename != file.filename]
                self.update_cache(file.filename, co_changed)
        return self.predict_faults()

if __name__ == "__main__":
    cache = FixCache("your-username/test-repo")  # Replace with your repo
    top_files = cache.run()
    print("Top 10 Fault-Prone Files:")
    for file, hits in top_files:
        print(f"{file}: {hits} hits")