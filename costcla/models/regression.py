"""
This module include the cost-sensitive logistic regression method.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

from pyea import GeneticAlgorithmOptimizer
from ..metrics import cost_loss

import numpy as np
import math
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.linear_model.logistic import _intercept_dot


# Not in sklearn 0.15, is in 0.16-git
#TODO: replace once sklearn 0.16 is release
def _intercept_dot(w, X):
    """Computes y * np.dot(X, w).

    It takes into consideration if the intercept should be fit or not.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    """
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = np.dot(X, w) + c
    return w, c, z


def _sigmoid(z):
    """ Private function that calculate the sigmoid function """
    return 1 / (1 + np.exp(-z))


def _logistic_cost_loss(w, X, y, cost_mat, alpha):
    """Computes the logistic loss.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : array-like, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    Returns
    -------
    out : float
        Logistic loss.
    """
    n_samples = X.shape[0]
    w, c, z = _intercept_dot(w, X)
    y_prob = _sigmoid(z)

    out = y * (y_prob * cost_mat[:, 2] + (1 - y_prob) * cost_mat[:, 1])
    out += (1 - y) * (y_prob * cost_mat[:, 0] + (1 - y_prob) * cost_mat[:, 3])

    out = out.sum() / n_samples
    out += .5 * alpha * np.dot(w, w)

    return out

class CostSensitiveLogisticRegression(BaseEstimator):
    """A example-dependent cost-sensitive Logistic Regression classifier.

    Parameters
    ----------

    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added the decision function.

    max_iter : int
        Useful only for the newton-cg and lbfgs solvers. Maximum number of
        iterations taken for the solvers to converge.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    solver : {'ga', 'bfgs'}
        Algorithm to use in the optimization problem.

    tol : float, optional
        Tolerance for stopping criteria.

    Attributes
    ----------
    `coef_` : array, shape (n_classes, n_features)
        Coefficient of the features in the decision function.

    `intercept_` : array, shape (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier

    References
    ----------

    .. [1] A.Correa, "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           submitted.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveLogisticRegression
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = LogisticRegression(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveLogisticRegression()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print savings_score(y_test, y_pred_test_rf, cost_mat_test)
    0.12454256594
    >>> # Savings using CSDecisionTree
    >>> print savings_score(y_test, y_pred_test_csdt, cost_mat_test)
    0.481916135529
    """
    def __init__(self,
                 C=1.0,
                 fit_intercept=True,
                 max_iter=100,
                 random_state=None,
                 solver='ga',
                 tol=1e-4):

        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.tol = tol


    def fit(self, x, y, cost_mat, reg=0, method='BFGS', range1=None, params_ga=[100, 100, 10, 0.25]):
        """ Build a example-dependent cost-sensitive decision tree from the training set (X, y, cost_mat)

        Parameters
        ----------
        y : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        self : object
            Returns self.
        """

        #TODO: Check input

        setattr(self, 'intercept', intercept)
        if intercept == True:
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        n = x.shape[1]
        initial_theta = np.zeros((n, 1))
        if method == "GA":
            if range1 == None:
                range1 = np.vstack((-10 * np.ones((1, n)), 10 * np.ones((1, n))))
            res = GAcont(n, self.fitnessfunc, params_ga[0], params_ga[1], CS=params_ga[2], MP=params_ga[3],
                         range=range1,
                         fargs=[y, x, cost_mat, reg])
            res.evaluate()
        else:
            res = minimize(self.fitnessfunc, initial_theta, (y, x, cost_mat, reg,), method=method,
                           options={'maxiter': 100, 'disp': True})

        setattr(self, 'theta', res.x)
        setattr(self, 'opti', res)
        setattr(self, 'hist', res.hist)
        setattr(self, 'full_hist', res.full_hist)

    def predict_proba(self, x_test):
        #Calculate the prediction of a LogRegression
        if self.intercept == True:
            x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
        p = np.zeros((x_test.shape[0], 2))
        p[:, 1] = self.sigmoid(np.dot(x_test, self.theta))
        p[:, 0] = 1 - p[:, 1]
        return p

    def predict(self, x_test, cut_point=0.5):
        #Calculate the prediction of a LogRegression
        p = np.floor(self.predict_proba(x_test)[:, 1] + (1 - cut_point))
        return p.reshape(p.shape[0], )
