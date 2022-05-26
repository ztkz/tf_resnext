# -*- coding: utf-8 -*-
#
"""
setup.py: For installing via pip
"""

from setuptools import setup, find_packages

setup(
    name="tf-resnext",
    version="1.0",
    description="TensorFlow reimplementation of ResNext",
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["absl-py", "gast==0.2.2", "numpy", "tensorflow-gpu==2.7.2"],
    entry_points={
        "console_scripts": ["tf_resnext = tf_resnext.__main__:run_app"]
    },
)
