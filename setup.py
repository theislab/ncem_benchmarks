from setuptools import setup, find_packages

author = 'theislab'
author_email = 'anna.schaar@helmholtz-muenchen.de'
description = ""

with open("README.rst", "r") as fh:
     long_description = fh.read()

setup(
    name='ncem_benchmarks',
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'ncem'
    ],
    extras_require={},
    version="0.2.0",
)
