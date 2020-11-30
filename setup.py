#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="gaussflow",
    version="0.0.0",
    description="Gaussianization Flows (RBIG2.0)",
    author="J. Emmanuel Johnson",
    author_email="jemanjohnson34@gmail.com",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/IPL-UV/gaussflow",
    install_requires=["pytorch-lightning", "pytorch", "nflows"],
    packages=find_packages(),
)
