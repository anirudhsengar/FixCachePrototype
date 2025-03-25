#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for FixCachePrototype package.
"""

import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Version information
VERSION = '0.1.0'

# Package requirements
REQUIRED = [
    'matplotlib>=3.5.0',
    'numpy>=1.20.0',
    'GitPython>=3.1.0',
    'pyyaml>=6.0',
]

# Development requirements
EXTRAS = {
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'mypy>=0.990',
        'sphinx>=4.0.0',
        'sphinx_rtd_theme>=1.0.0',
    ],
    'ml': [
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
    ],
}

setup(
    name='fixcache',
    version=VERSION,
    description='Enhanced implementation of the FixCache algorithm for bug prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/anirudhsengar/FixCachePrototype',
    author='Anirudh Sengar',
    author_email='anirudhsengar@example.com',  # Update with your email
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Bug Tracking',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Software Development :: Testing',
        'License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='bug prediction, fixcache, code analysis, repository analysis',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7, <4',
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    entry_points={
        'console_scripts': [
            'fixcache=fixcache.main:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/anirudhsengar/FixCachePrototype/docs',
        'Source': 'https://github.com/anirudhsengar/FixCachePrototype',
        'Original BugTool': 'https://github.com/adoptium/aqa-test-tools/tree/master/BugPredict/BugTool',
    },
    # Include package data files
    include_package_data=True,
    zip_safe=False,
    # Metadata creation date
    package_data={
        'fixcache': ['py.typed'],
    },
)