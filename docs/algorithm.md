# FixCache Algorithm Technical Documentation

**Author:** anirudhsengar

## Algorithm Overview

The FixCache algorithm predicts which files in a software repository are most likely to contain bugs in the future by analyzing the project's historical bug-fixing patterns. Unlike traditional bug prediction approaches that rely on complex code metrics, FixCache uses a memory-based model inspired by CPU cache algorithms to track and predict bug-prone files.

The core insight is that files with a history of bug fixes tend to have higher chances of containing bugs in the future. The algorithm maintains a "cache" of files predicted to be buggy, updating this cache as new bug fixes are encountered and applying cache replacement policies when the cache becomes full.

## Algorithm Components

FixCache consists of three major components:

1. **Repository Analysis**: Extracts and processes historical data from the version control system
2. **Cache Prediction**: Simulates the cache behavior over historical changes
3. **Risk Assessment**: Evaluates and ranks files by their bug proneness

## Detailed Algorithm

### Repository Analysis Stage

```
function ANALYZE_REPOSITORY(repo_path, bug_keywords, lookback_commits):
    files ← GET_ALL_CODE_FILES(repo_path)
    commits ← GET_COMMIT_HISTORY(repo_path, lookback_commits)
    bug_fixes ← []
    
    for each commit in commits:
        if CONTAINS_BUG_KEYWORD(commit.message, bug_keywords):
            commit.is_bug_fix ← true
            bug_fixes.append(commit)
        else:
            commit.is_bug_fix ← false
    
    file_stats ← Initialize empty map
    for each file in files:
        file_stats[file] ← {bug_fixes: 0, last_modified: null, authors: []}
    
    for each commit in bug_fixes:
        changed_files ← GET_FILES_CHANGED(commit)
        for each file in changed_files:
            if file in file_stats:
                file_stats[file].bug_fixes += 1
                file_stats[file].last_modified ← commit.timestamp
                file_stats[file].authors.add(commit.author)
    
    return {files, commits, bug_fixes, file_stats}
```

### Cache Prediction Stage

```
function PREDICT(repo_analysis, cache_size, policy, window_ratio):
    total_files ← length of repo_analysis.files
    cache_max_size ← ceiling(total_files × cache_size)
    
    # Split commits into training and testing
    split_index ← floor(length(repo_analysis.commits) × window_ratio)
    training_commits ← repo_analysis.commits[0 to split_index]
    testing_commits ← repo_analysis.commits[split_index+1 to end]
    
    # Initialize cache
    cache ← []
    
    # Seed cache with most bug-prone files if cache_seeding is enabled
    if cache_seeding is enabled:
        sorted_files ← SORT_BY_BUG_FIXES(repo_analysis.file_stats)
        for file in sorted_files (up to cache_max_size):
            cache.append(file)
    
    # Process commits for prediction
    hit_count ← 0
    miss_count ← 0
    
    for each commit in testing_commits:
        if commit.is_bug_fix:
            files_changed ← GET_FILES_CHANGED(commit)
            
            for each file in files_changed:
                if file in cache:
                    hit_count += 1
                else:
                    miss_count += 1
                    
                    # Add to cache
                    if file is a code file:
                        if length(cache) == cache_max_size:
                            EVICT_FILE(cache, repo_analysis.file_stats, policy)
                        cache.append(file)
    
    # Calculate hit rate
    if hit_count + miss_count > 0:
        hit_rate ← (hit_count / (hit_count + miss_count)) × 100
    else:
        hit_rate ← 0
    
    return {hit_rate, hit_count, miss_count, cache}
```

### Cache Replacement Policies

```
function EVICT_FILE(cache, file_stats, policy):
    if policy == "FIFO":
        # First-In-First-Out: remove oldest file in cache
        remove cache[0]
    
    else if policy == "LRU":
        # Least Recently Used: remove least recently accessed file
        oldest_file ← file with minimum last_modified in cache
        remove oldest_file from cache
    
    else if policy == "BUG":
        # Bug-based: remove file with fewest bug fixes
        min_bugs ← infinity
        file_to_remove ← null
        
        for each file in cache:
            if file_stats[file].bug_fixes < min_bugs:
                min_bugs ← file_stats[file].bug_fixes
                file_to_remove ← file
        
        remove file_to_remove from cache
```

## Cache Optimization

FixCache includes a cache size optimization algorithm that identifies the optimal cache size for a given repository:

```
function OPTIMIZE_CACHE_SIZE(repo_path, cache_sizes, policy):
    results ← {}
    
    for each size in cache_sizes:
        fix_cache ← CREATE_FIXCACHE(repo_path, size, policy)
        fix_cache.analyze_repository()
        hit_rate ← fix_cache.predict()
        results[size] ← hit_rate
    
    optimal_size ← size with maximum hit_rate in results
    return {results, optimal_size}
```

## Time Complexity Analysis

For a repository with:
- n files
- m commits
- k bug-fixing commits
- c files in cache (c = cache_size × n)

**Repository Analysis:**
- Extracting commit history: O(m)
- Identifying bug fixes: O(m)
- Processing bug fixes: O(k × avg_files_per_commit)

**Cache Prediction:**
- Cache simulation: O(m × avg_files_per_commit)
- Cache replacement (worst case):
  - FIFO: O(1)
  - LRU: O(c)
  - BUG: O(c)

**Overall complexity:** O(m × avg_files_per_commit + n)

## Space Complexity Analysis

- Repository metadata: O(n + m)
- Commit history: O(m)
- Bug fixes: O(k)
- File statistics: O(n)
- Cache: O(c)

**Overall space complexity:** O(n + m)

## Implementation Details

### Memory Optimization

The implementation uses several techniques to optimize memory usage:

1. **Incremental Processing**: Processes commits incrementally rather than loading all at once
2. **Data Structures**: Uses efficient data structures for lookups and cache management
3. **File Filtering**: Excludes non-code files and binary files early in the analysis
4. **Partial History**: Supports analyzing only recent history (lookback_commits parameter)

### Thread Safety

The main FixCache algorithm is designed to be thread-safe during analysis:

1. Repository analysis can be parallelized by commit batches
2. File stats are maintained in thread-local storage and combined at completion
3. Cache simulation is sequential by design due to temporal dependencies

### Error Handling

The implementation includes comprehensive error handling:

1. **Repository Access**: Gracefully handles repository connectivity issues
2. **File Encoding**: Supports fallback encoding for non-UTF-8 files
3. **Invalid Cache Parameters**: Validates all parameters before execution
4. **Performance Monitoring**: Tracks execution times and provides warnings for long operations

## Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| cache_size | float (0-1) | Fraction of total files to keep in cache | 0.2 (20%) |
| policy | string | Cache replacement policy: "BUG", "FIFO", or "LRU" | "BUG" |
| window_ratio | float (0-1) | Fraction of history to use for training | 0.25 (25%) |
| bug_keywords | list of strings | Terms used to identify bug-fixing commits | ["fix", "bug", ...] |
| lookback_commits | int or null | Number of recent commits to analyze | null (all) |
| cache_seeding | boolean | Whether to seed cache with most bug-prone files | true |
| min_file_count | int | Minimum files required for analysis | 10 |

## Performance Tuning

For optimal performance, consider the following guidelines:

1. **Repository Size**:
   - Small repos (<1,000 files): Use default parameters
   - Medium repos (1,000-10,000 files): Consider lookback_commits limit
   - Large repos (>10,000 files): Use lookback_commits and parallelize analysis

2. **Cache Size**:
   - Start with 20% and optimize based on your repository
   - Smaller repos may benefit from larger cache sizes (30-40%)
   - Larger repos often work well with smaller cache sizes (10-15%)

3. **Window Ratio**:
   - Use 25% for balanced training/testing
   - Increase to 50% for more aggressive training with newer projects
   - Decrease to 10% for projects with very long history

## Technical Considerations

### Bug Identification

The algorithm identifies bug-fixing commits through keyword matching in commit messages. This approach has limitations:

1. **False Positives**: Not all commits with keywords are actual bug fixes
2. **False Negatives**: Not all bug fixes explicitly mention bug-related terms
3. **Language Dependence**: Keywords assume English commit messages

Advanced implementations can supplement this with:

- Issue tracker integration for more accurate bug identification
- Natural language processing to detect bug-fixing intent
- Author-specific commit patterns analysis

### File Identity Tracking

The algorithm tracks files by path, which presents challenges when files are:

1. Renamed or moved
2. Split into multiple files
3. Merged from multiple files

Current implementations address this by:

- Using Git's rename detection capabilities
- Maintaining file identity heuristics based on content similarity
- Linking files with common ancestry

## Algorithm Limitations

1. **Temporal Assumption**: Assumes past bug distribution predicts future bugs
2. **Coarse Granularity**: Works at file level, not function or line level
3. **Language Agnostic**: Doesn't leverage language-specific insights
4. **Project Structure**: Doesn't consider architectural relationships between files
5. **Developer Factors**: Limited consideration of developer expertise and ownership

## Future Enhancements

1. **Hybrid Approaches**: Combining with code metrics for improved predictions
2. **Deep Learning Integration**: Using neural networks to identify complex bug patterns
3. **Cross-Repository Learning**: Transferring knowledge between similar projects
4. **Contextual Analysis**: Considering code review history and test coverage
5. **Fine-Grained Prediction**: Moving to function or line-level predictions