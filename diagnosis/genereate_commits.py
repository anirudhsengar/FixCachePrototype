from git import Repo

def sample_commits(repo_path, num_commits=100):
    repo = Repo(repo_path)
    commits = list(repo.iter_commits('master', max_count=num_commits))
    with open("openjdk_commit_samples.txt", "w") as f:
        for commit in commits:
            f.write(f"Commit {commit.hexsha}: {commit.message.strip()}\n")

sample_commits(r"C:\Users\aniru\OneDrive\Desktop\openjdk")