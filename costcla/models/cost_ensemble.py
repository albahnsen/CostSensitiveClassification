__author__ = 'al'


from sklearn.cross_validation import train_test_split
from costcla.datasets import load_creditscoring2
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
import numpy as np
from costcla.models.bagging import BaggingClassifier

data = load_creditscoring2()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)

X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets

# f = BaggingClassifier(base_estimator=CostSensitiveDecisionTreeClassifier(),
#                       n_estimators=3, max_samples=0.2, max_features=0.5, verbose=2,
#                       bootstrap=False, bootstrap_features=False, n_jobs=1)



class BaggingCostSensitiveDecisionTreeClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=100,
                 max_samples=0.5,
                 verbose=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=1,
            random_state=None,
            verbose=verbose)

class PastingCostSensitiveDecisionTreeClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=100,
                 max_samples=0.5,
                 verbose=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            n_jobs=1,
            random_state=None,
            verbose=verbose)

class RandomPatchesCostSensitiveDecisionTreeClassifier(BaggingClassifier):
    def __init__(self,
                 n_estimators=100,
                 max_samples=0.5,
                 max_features=0.5,
                 verbose=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=True,
            bootstrap_features=True,
            n_jobs=1,
            random_state=None,
            verbose=verbose)


algos = [BaggingCostSensitiveDecisionTreeClassifier(verbose=2),
         PastingCostSensitiveDecisionTreeClassifier(verbose=2),
         RandomPatchesCostSensitiveDecisionTreeClassifier(verbose=2),
         CostSensitiveDecisionTreeClassifier()]

predict = []
for algo in algos:
    print algo
    algo.fit(X_train, y_train, cost_mat_train)
    predict.append(algo.predict(X_test))

for i, pred in enumerate(predict):
    print algos[i]
    print savings_score(y_test, pred, cost_mat_test)

for e in range(3):
    for i, estimator in enumerate(algos[e].estimators_):
        print algos[e].__class__, i, savings_score(y_test, estimator.predict(X_test), cost_mat_test)


check https://github.com/scikit-learn/scikit-learn/blob/9c8f6561874ae6ba45136ddf41c54a63ce62c616/sklearn/tree/_tree.pyx#L1079