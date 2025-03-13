# compare_files.py
with open("openjdk_preloaded_files.txt", "r") as f:
    preloaded_files = set(line.strip() for line in f)

with open("openjdk_fault_prone_files.txt", "r") as f:
    fault_prone_files = set(line.strip() for line in f)

# Calculate overlap
overlap = preloaded_files.intersection(fault_prone_files)
print(f"Number of preloaded files: {len(preloaded_files)}")
print(f"Number of fault-prone files: {len(fault_prone_files)}")
print(f"Number of overlapping files: {len(overlap)}")
print(f"Overlap percentage: {len(overlap) / len(fault_prone_files) * 100:.2f}%")
print("Overlapping files:", overlap)