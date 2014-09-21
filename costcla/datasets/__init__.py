"""
The :mod:`costcla.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""

from base import load_bankmarketing
from base import load_creditscoring1
from base import load_creditscoring2

__all__ = ['load_bankmarketing',
           'load_creditscoring1',
           'load_creditscoring2']
