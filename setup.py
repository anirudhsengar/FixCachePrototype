#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for FixCachePrototype package.
"""

from setuptools import setup, find_packages

setup(
    name="fixcache",
    version="0.1.0",
    author="anirudhsengar",
    author_email="anirudhsengar@gmail.com",
    description="A tool for bug prediction using the FixCache algorithm",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/anirudhsengar/FixCachePrototype",
    packages=find_packages(),  # No 'where' parameter needed now
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gitpython>=3.1.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.64.0",
    ],
    entry_points={
        "console_scripts": [
            "fixcache=fixcache.cli:main",
        ],
    },
)