# FixCache Usage Guide

**Author:** anirudhsengar

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Command-Line Interface](#command-line-interface)
- [Configuration File](#configuration-file)
- [Usage Examples](#usage-examples)
- [Integration](#integration)
- [Advanced Scenarios](#advanced-scenarios)
- [Troubleshooting](#troubleshooting)

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

### Verify installation
```bash
fixcache --version
```

## Quick Start

For the impatient, here's a quick way to analyze a repository:

```bash
# Basic analysis with default parameters
fixcache analyze /path/to/repository

# Generate a visualization
fixcache analyze /path/to/repository --visualize --output results.png

# Generate a detailed report
fixcache analyze /path/to/repository --report --output report.md
```

## Basic Usage

FixCache can be used both as a command-line tool and as a Python library.

### Command-Line Usage

The basic syntax is:

```bash
fixcache <command> [options]
```

Main commands:
- `analyze`: Analyze a repository for bug prediction
- `optimize`: Find optimal cache size for a repository
- `compare`: Compare multiple repositories
- `visualize`: Generate visualizations from previous analysis

### Library Usage

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

## Command-Line Interface

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--verbose`, `-v` | Increase verbosity | False |
| `--quiet`, `-q` | Suppress output | False |
| `--config` | Path to configuration file | None |
| `--output`, `-o` | Output file path | stdout |
| `--format`, `-f` | Output format (json, yaml, md, txt) | json |

### `analyze` Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-size`, `-c` | Cache size as fraction of total files | 0.2 |
| `--policy`, `-p` | Cache replacement policy (BUG, FIFO, LRU) | BUG |
| `--window-ratio`, `-w` | History window ratio | 0.25 |
| `--lookback`, `-l` | Number of commits to look back | All |
| `--visualize`, `-v` | Generate visualization | False |
| `--report`, `-r` | Generate report | False |
| `--no-seeding` | Disable cache seeding | False |

### `optimize` Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--policy`, `-p` | Cache replacement policy to use | BUG |
| `--min-size` | Minimum cache size to test | 0.05 |
| `--max-size` | Maximum cache size to test | 0.5 |
| `--step-size` | Step size for cache testing | 0.05 |
| `--fine-grained` | Use fine-grained steps | False |
| `--compare-policies` | Compare different policies | False |

### `compare` Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--repos`, `-r` | Comma-separated list of repositories | Required |
| `--cache-size`, `-c` | Cache size to use for comparison | 0.2 |
| `--policy`, `-p` | Cache replacement policy to use | BUG |
| `--detailed`, `-d` | Run detailed analysis | False |

### `visualize` Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input JSON file with analysis results | Required |
| `--type`, `-t` | Visualization type (summary, timeline, files, heatmap) | summary |
| `--width` | Image width in pixels | 1200 |
| `--height` | Image height in pixels | 800 |

## Configuration File

FixCache supports YAML configuration files to specify options:

```yaml
# fixcache.yml example
general:
  verbose: true
  output_dir: ./results

analyze:
  cache_size: 0.15
  policy: BUG
  window_ratio: 0.3
  lookback: 1000
  bug_keywords:
    - fix
    - bug
    - issue
    - crash
    - error
    - fault
  cache_seeding: true

optimize:
  min_size: 0.05
  max_size: 0.4
  step_size: 0.05
  fine_grained: true
  compare_policies: true

visualize:
  width: 1200
  height: 800
  theme: light
  dpi: 150
```

Use with:
```bash
fixcache analyze /path/to/repo --config fixcache.yml
```

## Usage Examples

### Basic Repository Analysis

```bash
# Run analysis on a local repository
fixcache analyze /path/to/local/repo

# Run analysis on a specific branch
fixcache analyze /path/to/local/repo --branch develop

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

# Create multiple visualizations
fixcache visualize --input results.json --type summary --output summary.png
fixcache visualize --input results.json --type timeline --output timeline.png
fixcache visualize --input results.json --type files --output files.png
```

### Optimize Cache Size

```bash
# Find optimal cache size
fixcache optimize /path/to/repo

# Compare different policies
fixcache optimize /path/to/repo --compare-policies

# Use fine-grained cache size steps
fixcache optimize /path/to/repo --fine-grained
```

### Compare Multiple Repositories

```bash
# Compare two repositories
fixcache compare --repos /path/to/repo1,/path/to/repo2

# Run detailed comparison
fixcache compare --repos /path/to/repo1,/path/to/repo2,/path/to/repo3 --detailed
```

### Cache Size vs. Hit Rate Analysis

```bash
# Test multiple cache sizes and save results
fixcache optimize /path/to/repo --min-size 0.05 --max-size 0.5 --step-size 0.05 --output cache_analysis.json
```

## Integration

### Continuous Integration

FixCache can be integrated into CI/CD pipelines:

#### GitHub Actions Example

```yaml
name: FixCache Bug Prediction

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Fetch full history
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install FixCache
      run: pip install fixcache
        
    - name: Run FixCache analysis
      run: fixcache analyze . --report --format md --output bug_prediction.md
        
    - name: Upload report artifact
      uses: actions/upload-artifact@v3
      with:
        name: bug-prediction-report
        path: bug_prediction.md
```

#### GitLab CI Example

```yaml
fixcache:
  stage: analyze
  image: python:3.10
  script:
    - pip install fixcache
    - fixcache analyze . --report --format md --output bug_prediction.md
  artifacts:
    paths:
      - bug_prediction.md
```

### IDE Integration

The FixCache library can be integrated with popular IDEs:

#### VS Code Extension Usage

If using the FixCache VS Code extension:

1. Install the extension from marketplace
2. Open a Git repository
3. Open command palette (Ctrl+Shift+P)
4. Type "FixCache: Analyze Repository"
5. View results in FixCache panel

## Advanced Scenarios

### Custom Bug Keywords

Define custom terms to identify bug fixes:

```bash
fixcache analyze /path/to/repo --bug-keywords "error,crash,fix,defect,issue"
```

### Limiting Repository History

For large repositories, limit the history to analyze:

```bash
# Only analyze the last 1000 commits
fixcache analyze /path/to/repo --lookback 1000
```

### Adjusting the Analysis Window

Change how much history is used for training vs. testing:

```bash
# Use 50% of history for training
fixcache analyze /path/to/repo --window-ratio 0.5
```

### Comparing Cache Policies

Compare effectiveness of different cache replacement strategies:

```bash
# Compare all policies
fixcache optimize /path/to/repo --compare-policies

# Run analysis with specific policy
fixcache analyze /path/to/repo --policy LRU
```

### Excluding Files or Directories

Exclude specific patterns from analysis:

```bash
fixcache analyze /path/to/repo --exclude "tests/*,docs/*,*.md"
```

### Parallel Processing

Enable parallel processing for faster analysis of large repositories:

```bash
fixcache analyze /path/to/repo --parallel --threads 4
```

## Troubleshooting

### Common Issues

#### Repository Analysis Fails

```
Error: Cannot analyze repository: Not a git repository
```

**Solution**: Ensure the path is a valid Git repository with a .git directory.

#### No Bug Fixes Found

```
Warning: No bug fixes found in repository
```

**Solutions**:
- Check the bug keywords used
- Ensure commit messages follow conventions (mentioning "fix", "bug", etc.)
- Try analyzing more history with `--lookback` option

#### Memory Issues with Large Repositories

```
Error: Memory error during analysis
```

**Solutions**:
- Limit history with `--lookback`
- Use `--exclude` to ignore irrelevant files
- Increase system memory or use a machine with more RAM

#### Visualization Errors

```
Error: Failed to generate visualization
```

**Solutions**:
- Ensure matplotlib is installed: `pip install matplotlib`
- Try a different output format: `--format png`
- Check if the output directory is writable

### Getting Help

For more detailed help on any command:

```bash
fixcache <command> --help
```

For full documentation and examples:

```bash
fixcache docs
```

For support:
- GitHub Issues: https://github.com/anirudhsengar/FixCachePrototype/issues
- Documentation: https://fixcache.readthedocs.io/