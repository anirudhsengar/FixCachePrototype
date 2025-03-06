# FixCache Prototype
A Python implementation of the FixCache algorithm from "Predicting Faults from Cached History" (Kim et al., 2007) for the GlitchWitcher GSoC 2025 project.

## Features
- Predicts fault-prone files using temporal and spatial localities.
- Cache size: 10% of files, BUG replacement policy.
- Tested on a sample GitHub repository.

## Installation
```bash
pip install PyGithub pandas numpy