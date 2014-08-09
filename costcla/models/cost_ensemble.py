__author__ = 'al'


from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score

data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)

X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets


X = X_train
y = y_train
cost_mat = cost_mat_train


import numpy as np


from costcla.models.bagging import BaggingClassifier

f = BaggingClassifier(base_estimator=CostSensitiveDecisionTreeClassifier(),
                      n_estimators=3, max_samples=0.2, max_features=0.5, verbose=2,
                      bootstrap=False, bootstrap_features=False, n_jobs=1)

f.fit(X, y, cost_mat)

y_pred_test_csdt_b = f.predict_proba(X_test)



n_classes = 2


(estimators, estimators_features) = f.estimators_, f.estimators_features_

temp = zip(estimators, estimators_features)
estimator, features = temp[0]

temp = f.predict_proba(X_test)


#
# y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
#
# f = CostSensitiveDecisionTreeClassifier()
# y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
#
# # Savings using only RandomForest
# print savings_score(y_test, y_pred_test_rf, cost_mat_test)
#
# # Savings using CSDecisionTree
# print savings_score(y_test, y_pred_test_csdt, cost_mat_test)
#
#
# def test_classification():
#     """Check classification for various parameter settings."""
#     rng = check_random_state(0)
#     X_train, X_test, y_train, y_test = train_test_split(iris.data,
#                                                         iris.target,
#                                                         random_state=rng)
#     grid = ParameterGrid({"max_samples": [0.5, 1.0],
#                           "max_features": [1, 2, 4],
#                           "bootstrap": [True, False],
#                           "bootstrap_features": [True, False]})
#
# params = dict(base_estimator=None,
#                  n_estimators=10,
#                  max_samples=1.0,
#                  max_features=1.0,
#                  bootstrap=True,
#                  bootstrap_features=False,
#                  oob_score=False,
#                  n_jobs=1,
#                  random_state=None,
#                  verbose=0)
#
#     for base_estimator in [None,
#                            DummyClassifier(),
#                            Perceptron(),
#                            DecisionTreeClassifier(),
#                            KNeighborsClassifier(),
#                            SVC()]:
#         for params in grid:
#             BaggingClassifier(base_estimator=base_estimator,
#                               random_state=rng,
#                               **params).fit(X_train, y_train).predict(X_test)