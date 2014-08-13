"""
This module include the cost sensitive ensemble methods.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

from sklearn.cross_validation import train_test_split
from costcla.datasets import load_creditscoring2, load_bankmarketing, load_creditscoring1
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
import numpy as np
from costcla.models.bagging import BaggingClassifier

#TODO add documentation for all methods
class CostSensitiveRandomForestClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 combination='majority_voting',
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(max_features='auto', pruned=pruned),
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            combination=combination,
            n_jobs=1,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class BaggingCostSensitiveDecisionTreeClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 combination='majority_voting',
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(pruned=pruned),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            combination=combination,
            n_jobs=1,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class PastingCostSensitiveDecisionTreeClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 combination='majority_voting',
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(pruned=pruned),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            combination=combination,
            n_jobs=1,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class RandomPatchesCostSensitiveDecisionTreeClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 max_features=0.5,
                 combination='majority_voting',
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(pruned=pruned),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=False,
            bootstrap_features=False,
            combination=combination,
            n_jobs=1,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


