# FixCachePrototype: Enhanced Bug Prediction Tool

![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)

## Overview

FixCachePrototype is an enhanced implementation of the FixCache algorithm for predicting fault-prone files in software repositories. This project revives and extends the original [BugTools](https://github.com/adoptium/aqa-test-tools/tree/master/BugPredict/BugTool) prototype developed by the Adoptium project, with significant improvements in accuracy, robustness, and usability.

This implementation is part of the GlitchWitcher project for Google Summer of Code 2025 with Eclipse Foundation, focusing on AI-assisted bug prediction using multiple approaches.

## Key Features

- **Accurate Bug Prediction**: Identifies files most likely to contain bugs based on historical bug-fixing patterns
- **Multiple Cache Policies**: Supports BUG, FIFO, and LRU cache replacement policies
- **Cache Size Optimization**: Automatically finds the optimal cache size for maximum hit rate
- **Rich Visualizations**: Generates charts and visualizations of prediction results
- **Repository Comparison**: Compare bug prediction across different repositories
- **GitHub Action Integration**: Run predictions automatically on pull requests
- **Robust Error Handling**: Gracefully handles encoding issues and edge cases

## How It Works

FixCachePrototype implements the FixCache algorithm described in the paper ["Predicting Faults from Cached History"](https://web.cs.ucdavis.edu/~devanbu/teaching/289/Schedule_files/Kim-Predicting.pdf) by Sunghun Kim et al. The algorithm works as follows:

1. **Repository Analysis**: Extract commit history and identify bug-fixing commits
2. **Cache Initialization**: Seed the cache with the most bug-prone files
3. **Sliding Window Training**: Learn from historical bug patterns
4. **Bug Prediction**: Predict future bug-prone files based on learned patterns
5. **Performance Evaluation**: Calculate hit rate and identify top fault-prone files

## Enhancements Over Original Implementation

This implementation includes several significant enhancements over the original BugTools prototype:

1. **Improved Bug Detection**: Enhanced heuristics and patterns for identifying bug-fixing commits
2. **Sliding Window Approach**: Better prediction through temporal learning
3. **Robust Encoding Handling**: Properly handles repositories with non-standard character encodings
4. **Visualization Components**: Rich graphical representations of results
5. **Cache Size Optimization**: Automatically finds the optimal cache configuration
6. **Comprehensive Documentation**: Detailed explanation of algorithm and usage
7. **Modernized Codebase**: Follows Python best practices with type hints and modular design
8. **Repository Comparison Tools**: Compare prediction performance across repositories
9. **GitHub Action Integration**: Automated analysis for continuous integration

## Installation

```bash
# Clone the repository
git clone https://github.com/anirudhsengar/FixCachePrototype.git
cd FixCachePrototype

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
