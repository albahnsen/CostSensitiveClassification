"""
This module include the cost-sensitive logistic regression method.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

import numpy as np
import math
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
# from sklearn.linear_model.logistic import _intercept_dot
from pyea import GeneticAlgorithmOptimizer
from ..metrics import cost_loss

# Not in sklearn 0.15, is in 0.16-git
#TODO: replace once sklearn 0.16 is release
# The one in sklearn 0.16 return yz instead of z, therefore,
# the impact on the code should be addressed before making the change.
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


def _logistic_cost_loss_i(w, X, y, cost_mat, alpha):
    n_samples = X.shape[0]
    w, c, z = _intercept_dot(w, X)
    y_prob = _sigmoid(z)

    out = cost_loss(y, y_prob, cost_mat) / n_samples
    out += .5 * alpha * np.dot(w, w)
    return out


def _logistic_cost_loss(w, X, y, cost_mat, alpha):
    """Computes the logistic loss.

    Parameters
    ----------
    w : array-like, shape (n_w, n_features,) or (n_w, n_features + 1,)
        Coefficient vector or matrix of coefficient.

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

    if w.shape[0] == w.size:
        # Only evaluating one w
        return _logistic_cost_loss_i(w, X, y, cost_mat, alpha)

    else:
        # Evaluating a set of w
        n_w = w.shape[0]
        out = np.zeros(n_w)

        for i in range(n_w):
            out[i] = _logistic_cost_loss_i(w[i], X, y, cost_mat, alpha)

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
        Useful only for the ga and bfgs solvers. Maximum number of
        iterations taken for the solvers to converge.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    solver : {'ga', 'bfgs'}
        Algorithm to use in the optimization problem.

    tol : float, optional
        Tolerance for stopping criteria.

    verbose : int, optional (default=0)
        Controls the verbosity of the optimization process.

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

    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring2
    >>> from costcla.models import CostSensitiveLogisticRegression
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring2()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_lr = LogisticRegression(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveLogisticRegression()
    >>> f.fit(X_train, y_train, cost_mat_train)
    >>> y_pred_test_cslr = f.predict(X_test)
    >>> # Savings using Logistic Regression
    >>> print savings_score(y_test, y_pred_test_lr, cost_mat_test)
    0.00283419465107
    >>> # Savings using Cost Sensitive Logistic Regression
    >>> print savings_score(y_test, y_pred_test_cslr, cost_mat_test)
    0.142872237978
    """
    def __init__(self,
                 C=1.0,
                 fit_intercept=True,
                 max_iter=100,
                 random_state=None,
                 solver='ga',
                 tol=1e-4,
                 verbose=0):

        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.
        self.verbose = verbose

    def fit(self, X, y, cost_mat):
        """ Build a example-dependent cost-sensitive logistic regression from the training set (X, y, cost_mat)

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y : array indicator matrix
            Ground truth (correct) labels.

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

        n_features = X.shape[1]
        if self.fit_intercept:
            w0 = np.zeros(n_features + 1)
        else:
            w0 = np.zeros(n_features)

        if self.solver == 'ga':
            #TODO: add n_jobs
            res = GeneticAlgorithmOptimizer(_logistic_cost_loss,
                                            w0.shape[0],
                                            iters=self.max_iter,
                                            type_='cont',
                                            n_chromosomes=100,
                                            per_mutations=0.25,
                                            n_elite=10,
                                            fargs=(X, y, cost_mat, 1. / self.C),
                                            range_=(-5, 5),
                                            n_jobs=1,
                                            verbose=self.verbose)
            res.fit()

        elif self.solver == 'bfgs':

            if self.verbose > 0:
                disp = True
            else:
                disp = False

            res = minimize(_logistic_cost_loss,
                           w0,
                           method='BFGS',
                           args=(X, y, cost_mat, 1. / self.C),
                           tol=self.tol,
                           options={'maxiter': self.max_iter, 'disp': disp})

        if self.fit_intercept:
            self.coef_ = res.x[:-1]
            self.intercept_ = res.x[-1]
        else:
            self.coef_ = res.x

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, 2]
            Returns the probability of the sample for each class in the model.
        """
        y_prob = np.zeros((X.shape[0], 2))
        y_prob[:, 1] = _sigmoid(np.dot(X, self.coef_) + self.intercept_)
        y_prob[:, 0] = 1 - y_prob[:, 1]
        return y_prob

    def predict(self, X, cut_point=0.5):
        """Predicted class.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples]
            Returns the prediction of the sample..
        """
        return np.floor(self.predict_proba(X)[:, 1] + (1 - cut_point))
