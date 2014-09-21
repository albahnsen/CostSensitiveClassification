"""
The :mod:`costcla.sampling` module includes methods for cost-sensitive sampling

In particular:

- :mod:`costcla.sampling.cost_sampling` methods for cost-proportionate sampling
- :mod:`costcla.sampling.undersampling` traditional undersampling
- :mod:`costcla.sampling.smote` SMOTE method for synthetic over-sampling

"""

from cost_sampling import cost_sampling
from sampling import undersampling
from sampling import smote

__all__ = ['cost_sampling',
           'undersampling',
           'smote']
