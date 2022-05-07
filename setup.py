#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script."""

import importlib.util
from pathlib import Path
from setuptools import setup, find_packages
from codecs import open


# Get the long description from the relevant file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Get the base version from the library.  (We'll find it in the `version.py`
# file in the src directory, but we'll bypass actually loading up the library.)
vspec = importlib.util.spec_from_file_location(
    "version", str(Path(__file__).resolve().parent / "npc_engine" / "version.py")
)
vmod = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(vmod)
version = getattr(vmod, "__version__")

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.readlines()

with open("requirements_dev.txt", encoding="utf-8") as f:
    requirements_dev = f.readlines()

with open("requirements_doc.txt", encoding="utf-8") as f:
    requirements_doc = f.readlines()


setup(
    name="npc-engine",
    description="Deep learning inference and NLP toolkit for game development.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=version,
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
        "doc": requirements_doc,
        "benchmarks": ["py3nvml"],
        "dml": ["onnxruntime-directml>=1.8.0,<2.0.0"],
        "cpu": ["onnxruntime>=1.8.0,<2.0.0"],
        "export": ["torch==1.11.0", "transformers==4.17.0"],
    },
    entry_points="""
    [console_scripts]
    npc-engine=npc_engine.cli:cli
    """,
    python_requires=">=3.7.0",
    license="MIT",
    author="eublefar",
    author_email="evil.unicorn1@gmail.com",
    # Use the URL to the github repo.
    url="https://github.com/npc-engine/npc-engine",
    download_url=(
        f"https://github.com/npc-engine/" f"npc_engine/archive/{version}.tar.gz"
    ),
    keywords=["npc", "AI", "inference", "deep-learning"],
    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
)
