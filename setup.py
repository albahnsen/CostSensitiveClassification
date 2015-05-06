#!/usr/bin/env python

from setuptools import setup, find_packages
import re

for line in open('costcla/__init__.py'):
    match = re.match("__version__ *= *'(.*)'", line)
    if match:
        __version__, = match.groups()

setup(name='costcla',
      version=__version__,
      description='costcla is a Python module for cost-sensitive machine learning (classification) ',
      long_description=open('README.rst').read(),
      author='Alejandro CORREA BAHNSEN',
      author_email='al.bahnsen@gmail.com',
      url='https://github.com/albahnsen/CostSensitiveClassification',
      license='new BSD',
      packages=find_packages(),
      include_package_data = True,
      keywords=['machine learning', 'classification', 'cost-sensitive'],
      install_requires=['scikit-learn>=0.15.0b2','pandas>=0.14.0','numpy>=1.8.0', 'pyea>=0.1'],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',],
      )
