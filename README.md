# FixCache Prototype

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-prototype-orange.svg)

A Python implementation of the FixCache algorithm for bug prediction in software repositories. This is a prototype developed for the GlitchWitcher GSOC 2025 project.

## Overview

FixCache is a novel approach to bug prediction based on the principle that files with a history of bug fixes tend to have higher chances of containing bugs in the future. Unlike traditional bug prediction methods that rely on complex code metrics, FixCache uses a memory-based model inspired by CPU cache algorithms to track and predict bug-prone files.

The algorithm was originally introduced by Kim et al. in their 2007 paper ["Predicting Faults from Cached History"](https://dl.acm.org/doi/10.1109/ICSE.2007.66).

## Features

- **Repository Analysis**: Analyzes Git repositories to identify bug-fixing commits and affected files
- **Bug Prediction**: Predicts which files are most likely to contain bugs in the future
- **Multiple Policies**: Supports different cache replacement policies (BUG, FIFO, LRU)
- **Visualization**: Generates visual representations of prediction results
- **Reporting**: Creates detailed reports of bug prediction analysis
- **Optimization**: Identifies optimal cache size for maximum hit rate
- **Comparison**: Compares bug prediction across multiple repositories

## Installation

### Prerequisites

- Python 3.8 or higher
- Git 2.23 or higher
- Access to a Git repository

### Install via pip

```bash
pip install fixcache
```

### Install from source

```bash
git clone https://github.com/anirudhsengar/FixCachePrototype.git
cd FixCachePrototype
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Basic analysis with default parameters
fixcache analyze /path/to/repository

# Generate a visualization
fixcache analyze /path/to/repository --visualize --output results.png

# Generate a detailed report
fixcache analyze /path/to/repository --report --output report.md

# Find optimal cache size
fixcache optimize /path/to/repository

# Compare multiple repositories
fixcache compare --repos /path/to/repo1,/path/to/repo2
```

### Python API Usage

```python
from fixcache.algorithm import FixCache

# Create an instance
fix_cache = FixCache(
    repo_path="/path/to/repository",
    cache_size=0.2,
    policy="BUG"
)

# Analyze repository
fix_cache.analyze_repository()

# Run prediction
hit_rate = fix_cache.predict()

# Get results
results = fix_cache.get_summary()
risky_files = fix_cache.get_top_files(10)

# Generate visualization
fix_cache.visualize_results("output.png")
```

## How It Works

FixCache operates on the principle that bugs tend to occur in files that have already had bugs in the past. The algorithm tracks a "cache" of files predicted to be buggy and updates this cache as new bug fixes are encountered in the repository history.

The basic workflow:

1. **Repository Analysis**: Extract commit history and identify bug-fixing commits
2. **Cache Simulation**: Simulate cache behavior over the repository timeline
3. **Hit Rate Calculation**: Calculate prediction accuracy based on hit rate
4. **Risk Assessment**: Evaluate and rank files by their bug proneness

## Documentation

For more detailed documentation, see:

- [Usage Guide](docs/usage.md) - Complete usage instructions
- [Algorithm Details](docs/algorithm.md) - Technical explanation of the algorithm
- [Original Research](docs/original_bugtool.md) - Background on the original FixCache paper

## Examples

### Basic Repository Analysis

```bash
# Run analysis on a local repository
fixcache analyze /path/to/local/repo

# Analyze with custom cache size
fixcache analyze /path/to/local/repo --cache-size 0.15

# Use FIFO policy instead of default BUG
fixcache analyze /path/to/local/repo --policy FIFO
```

### Generate Reports and Visualizations

```bash
# Generate a markdown report
fixcache analyze /path/to/repo --report --format md --output report.md

# Create visualization
fixcache analyze /path/to/repo --visualize --output visualization.png
```

### Optimize Cache Size

```bash
# Find optimal cache size
fixcache optimize /path/to/repo

# Compare different policies
fixcache optimize /path/to/repo --compare-policies
```

## Project Structure

```
FixCachePrototype/
├── LICENSE                         # License file
├── README.md                       # Main documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup script
├── .github/
│   └── workflows/
│       └── fixcache_action.yml     # GitHub action for PR analysis
│
├── fixcache/
│       ├── __init__.py             # Package initialization
│       ├── main.py                 # Main entry point script
│       ├── algorithm.py            # Core FixCache algorithm (enhanced)
│       ├── repository.py           # Repository analysis utilities
│       ├── visualization.py        # Result visualization components
|       ├── cli.py                   # Command-line interface
│       ├── cache_policies.py       # Cache Policies
│       ├── utils.py                # Utility functions
│       └── config.py               # Configuration management
│
├── examples/
│   ├── basic_usage.py              # Basic usage example
│   ├── cache_optimization.py       # Cache size optimization example
│   └── compare_repositories.py     # Repository comparison example
│
├── tests/
│   ├── __init__.py                 # Test initialization
│   ├── test_algorithm.py           # Tests for algorithm.py
│   ├── test_repository.py          # Tests for repository.py
│   └── test_visualization.py       # Tests for visualization.py
│
└── docs/
    ├── diagnosis_summary.md        # Analysis of different repositories
    ├── original_bugtool.md         # Documentation of original BugTool
    ├── enhancements.md             # Our enhancements documentation
    ├── algorithm.md                # Algorithm explanation
    └── usage.md                    # Detailed usage instructions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sunghun Kim, Thomas Zimmermann, E. James Whitehead Jr., and Andreas Zeller for the original FixCache algorithm
- GlitchWitcher team for the GSOC 2025 project opportunity

## Contact

anirudhsengar - [GitHub Profile](https://github.com/anirudhsengar)

Project Link: [https://github.com/anirudhsengar/FixCachePrototype](https://github.com/anirudhsengar/FixCachePrototype)
