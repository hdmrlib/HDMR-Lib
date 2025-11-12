"""
Setup configuration for HDMR-Lib
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hdmr-lib",
    version="0.1.0",
    author="Your Name",  # Update this
    author_email="your.email@example.com",  # Update this
    description="High Dimensional Model Representation (HDMR) and Enhanced Multivariate Products Representation (EMPR) library with multi-backend support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/HDMR-Lib",  # Update this
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22",
    ],
    extras_require={
        "torch": ["torch>=2.2"],
        "tensorflow": ["tensorflow>=2.14"],
        "all": ["torch>=2.2", "tensorflow>=2.14"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "matplotlib>=3.5",
        ],
    },
    keywords="hdmr empr tensor decomposition sensitivity analysis multivariate",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/HDMR-Lib/issues",
        "Source": "https://github.com/yourusername/HDMR-Lib",
        "Documentation": "https://github.com/yourusername/HDMR-Lib#readme",
    },
)

