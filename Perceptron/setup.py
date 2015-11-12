#!/usr/bin/python3

"""setup.py: Installs the modules required to run perceptron.py."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

from setuptools import setup


setup(
    name = 'Perceptron',
    version = '0.1.0',
    description = 'Predicts whether some data belongs to one class or another.',
    author = 'Andrei Muntean',
    license = 'MIT',
    keywords = 'perceptron classify classifier binary machine learning ml predict',
    install_requires = ['numpy', 'matplotlib']
)