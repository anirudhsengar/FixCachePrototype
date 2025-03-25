# The Original FixCache: Historical Background

**Author:** anirudhsengar

## Introduction

This document provides an overview of the original FixCache bug prediction concept as introduced by Kim et al. in their seminal 2007 paper. The current implementation of FixCache builds upon and extends this foundational work, addressing several limitations while preserving the core insights of the original approach.

## Historical Context

Bug prediction has been a significant area of software engineering research since the early 2000s. Before FixCache was introduced, most bug prediction approaches fell into two categories:

1. **Code Metrics-Based**: Using source code metrics like complexity, size, and coupling to predict bug-prone files.
2. **Process Metrics-Based**: Using development history metrics like number of changes, age, and developer count to predict bugs.

In 2007, Sunghun Kim, Thomas Zimmermann, E. James Whitehead Jr., and Andreas Zeller introduced FixCache as a novel approach that incorporated change history patterns, particularly focusing on bug-fixing changes rather than simply counting all changes.

## The Original FixCache Approach

The fundamental insight of the original FixCache was elegantly simple yet powerful:

> **Files that have needed bug fixes in the past are likely to need bug fixes in the future.**

Rather than relying on complex metrics, FixCache tracks a simple "cache" of files most likely to contain bugs, based on their bug fix history. The cache is maintained using a cache replacement policy (similar to those used in CPU memory caches), where files can enter and leave the cache as the project evolves.

## Algorithm Details

The original FixCache algorithm operated as follows:

1. **Training Phase**:
   - Analyze a project's source code repository to identify bug-fixing commits
   - Extract the files changed in these bug-fixing commits
   
2. **Cache Initialization**:
   - Initialize an empty cache with capacity of k files (where k is typically 10% of all files)
   
3. **Cache Simulation**:
   - For each time step t:
     - If a file f is involved in a bug fix at time t, check if f is in the cache
     - If f is in the cache, count a "hit"
     - If f is not in the cache, count a "miss" and add f to the cache
     - If the cache is full, apply a replacement policy to evict a file

4. **Replacement Policies**:
   - **FIFO (First-In-First-Out)**: Evict the file that has been in the cache the longest
   - **LRU (Least Recently Used)**: Evict the file that hasn't been used for the longest time
   - **BUG**: Evict the file with the fewest bug fixes in its history

## Evaluation in Original Paper

The original paper evaluated FixCache on seven open-source projects:

- ArgoUML
- Columba
- Eclipse JDT Core
- Eclipse PDE UI
- Lucene
- PostgreSQL
- Rhino

The results showed that FixCache achieved hit rates of 73-95% with a cache size of only 10% of the total files, significantly outperforming random selection (10%) and demonstrating that a small subset of files is responsible for most bugs.

## Key Findings from Original Research

1. **Bug Locality**: Bugs tend to cluster in a small subset of files
2. **Bug Recurrence**: Files that have had bugs before tend to have bugs again
3. **Policy Importance**: The BUG replacement policy generally outperformed FIFO and LRU
4. **Stability**: Prediction accuracy remained stable over long periods

## Limitations of the Original Approach

Despite its innovation, the original FixCache had several limitations:

1. **Binary Prediction**: Files were either in the cache (predicted buggy) or not (predicted clean), with no nuanced risk levels

2. **Limited Context**: Did not consider file relationships, developer information, or project structure

3. **Static Cache Size**: Used a fixed cache size rather than adapting to project characteristics

4. **Single Granularity**: Operated only at the file level, not considering functions or components

5. **No Integration**: Provided as a research prototype without integration into development workflows

6. **Simplistic Bug Identification**: Relied on simple keyword matching in commit messages

7. **Memory Limitations**: The original implementation had practical memory constraints when scaling to very large repositories

## The Evolution to Current FixCache

The current FixCache implementation addresses many of these limitations by:

- Providing calibrated risk scores rather than binary predictions
- Supporting variable cache sizes and optimization
- Incorporating file relationship analysis
- Offering visualization and reporting capabilities
- Enabling integration with development workflows
- Using more sophisticated bug identification techniques
- Implementing memory-efficient data structures for large repositories
- Supporting multiple programming languages and version control systems

## References

1. Kim, S., Zimmermann, T., Whitehead Jr, E. J., & Zeller, A. (2007, May). Predicting faults from cached history. In Proceedings of the 29th international conference on Software Engineering (pp. 489-498). IEEE Computer Society.

2. Zimmermann, T., & Nagappan, N. (2008, May). Predicting defects using network analysis on dependency graphs. In Proceedings of the 30th international conference on Software engineering (pp. 531-540).

3. Rahman, F., & Devanbu, P. (2013). How, and why, process metrics are better. In Proceedings of the 2013 International Conference on Software Engineering (pp. 432-441).

4. D'Ambros, M., Lanza, M., & Robbes, R. (2012). Evaluating defect prediction approaches: a benchmark and an extensive comparison. Empirical Software Engineering, 17(4), 531-577.

5. Hassan, A. E. (2009). Predicting faults using the complexity of code changes. In Proceedings of the 31st International Conference on Software Engineering (pp. 78-88).