#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="hadml",
    version="0.1.0",
    description="Machine Learning Project for Hadronizaton",
    author="",
    author_email="",
    url="https://github.com/hep-lbdl/hadml",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
