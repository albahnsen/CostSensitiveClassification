"""
CostSensitiveClassification
===========================

costcla is a Python module for cost-sensitive machine learning (classification)
built on top of `Scikit-Learn <http://scikit-learn.org/stable/>`__, `SciPy <http://www.scipy.org/>`__
and distributed under the 3-Clause BSD license.

In particular, it provides:

1. A set of example-dependent cost-sensitive algorithms
2. Different reald-world example-dependent cost-sensitive datasets.

The project is part of the PhD research of `Alejandro Correa Bahnsen <http://albahnsen.com>`__.

Installation
============

You can install ``costcla`` with ``pip``::

    # pip install costcla
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

__version__ = '0.04.dev1'

from metrics import *
from datasets import *
from models import *
