"""
This module include the cost sensitive ensemble methods.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

from sklearn.cross_validation import train_test_split
from ..models import CostSensitiveDecisionTreeClassifier
from ..models.bagging import BaggingClassifier

#TODO add documentation for all methods
class CostSensitiveRandomForestClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 combination='majority_voting',
                 max_features='auto',
                 n_jobs=1,
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
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class CostSensitiveBaggingClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 combination='majority_voting',
                 n_jobs=1,
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
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class CostSensitivePastingClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 combination='majority_voting',
                 n_jobs=1,
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
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class CostSensitiveRandomPatchesClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 max_features=0.5,
                 combination='majority_voting',
                 n_jobs=1,
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
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


#TODO not working in parallel, without error

# from costcla.datasets import load_creditscoring1
# data = load_creditscoring1()
# x=data.data
# y=data.target
# c=data.cost_mat
#
# print 'start'
# f = BaggingClassifier(n_estimators=10, verbose=100, n_jobs=2)
# f.fit(x[0:1000],y[0:1000],c[0:1000])
# print 'predict proba'
# f.__setattr__('n_jobs', 4)
# f.predict(x)
# print 'predict END'
