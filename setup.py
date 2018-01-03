#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '0.0.1'

requirements = [
    'numpy',
    'torch',
]

setup(
    # Metadata
    name='lith',
    version=VERSION,
    author='Lyken',
    author_email='lykensyu+github@gmail.com',
    url='https://github.com/lyken17/lith',
    description='''Ligeng's extension for PyTorch''',
    long_description=readme,
    license='MIT',

    # Package info
    packages=find_packages(exclude=('test',)),

    #
    zip_safe=True,
    install_requires=requirements,
)
