# GlitchWitcher: AI-Assisted Bug Prediction
A Python-based project for the GlitchWitcher GSoC 2025 initiative, focusing on predicting source code defects using two approaches: the FixCache algorithm (Approach A) and the Reconstruction Error Probability Distribution (REPD) model (Approach B). The project aims to trial these methods, compare their effectiveness, and integrate them into development workflows to enhance code quality during reviews and testing.

## Project Overview
The "GlitchWitcher: AI-assisted Bug Prediction" project is part of Google Summer of Code (GSoC) 2025, hosted under the Eclipse Adoptium and OpenJ9 projects. The primary goal is to implement and compare two defect prediction approaches to identify fault-prone files in source code repositories before code reviews or testing phases. By integrating these approaches as static analysis utilities, the project seeks to flag areas needing scrutiny, ultimately improving software reliability.

### Objectives
- **Trial Two Approaches**:
  - **Approach A**: Implement the FixCache algorithm from "Predicting Faults from Cached History" (Kim et al., 2007) to predict defect-prone files using temporal and spatial localities.
  - **Approach B**: Reproduce the REPD model (a supervised anomaly detection/classification method) as described in research paper, applying it to datasets and C/C++ codebases. (TODO)
- **Compare Approaches**: Evaluate whether FixCache and REPD identify the same files as defect-prone in a given codebase.
- **Integrate into Workflows**: Develop a verification check (GlitchWitcher workflow) that runs these utilities against pull requests in repositories.
- **Enhance Code Quality**: Provide actionable insights for code reviews and testing by flagging high-risk files.

### Expected Outcomes
- Two static analysis utilities for defect prediction.
- A comparison of FixCache and REPD approaches, assessing their overlap in identifying defect-prone files.
- An automated verification check integrated into repository workflows (e.g., GitHub Actions).
- Improved scrutiny during code reviews and testing, focusing on high-risk areas.

## Features
- **FixCache Algorithm (Approach A)**:
  - Predicts fault-prone files based on temporal and spatial localities.
  - Configurable cache size: 20% of files, with a BUG (Belady’s Optimal) replacement policy.
  - Reports the top 10 files most likely to contain defects.
- **Planned REPD Model (Approach B)**:
  - Utilizes supervised anomaly detection/classification to categorize defective and non-defective code.
  - Will be trained on NASA ESDS Data Metrics datasets and applied to C/C++ repositories like OpenJ9 or OpenJDK.
- **Workflow Integration**:
  - Designed to run as a verification check on pull requests, with potential cadence at every new tag.
- **Multi-Repository Support**:
  - Tested on large, active repositories (e.g., `https://github.com/openjdk/jdk`) and small, less active ones (e.g., hypothetical `https://github.com/example/small-jdk-utils`).

## Findings
Preliminary results from applying the FixCache algorithm (Approach A) to two repositories reveal significant variations in hit rates: a hit rate of 92.02% for the OpenJ9 repository (`https://github.com/eclipse-openj9/openj9`) and 63.51% for the OpenJDK repository (`https://github.com/adoptium/jdk`). These findings, detailed further in the `diagnosis_summary.md` file, suggest that FixCache performs exceptionally well in the highly active OpenJ9 codebase but shows moderate effectiveness in the OpenJDK repository, potentially indicating a dependency on repository activity levels. Additional analysis and comparison with Approach B are planned to validate these observations.

## Installation
To set up the project environment, install the required dependencies:

```bash
pip install PyGithub pandas numpy