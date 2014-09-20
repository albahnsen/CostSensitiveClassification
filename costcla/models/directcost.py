"""
This module include the cost sensitive Bayes minimum risk method.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause


import numpy as np
from sklearn.base import BaseEstimator
from ..probcal import ROCConvexHull
from ..metrics import cost_loss


class BayesMinimumRiskClassifier(BaseEstimator):
    """A example-dependent cost-sensitive binary Bayes minimum risk classifier.

    Parameters
    ----------
    calibration : bool, optional (default=True)
        Whenever or not to calibrate the probabilities.

    References
    ----------

    .. [1] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import BayesMinimumRiskClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> f = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    >>> y_prob_test = f.predict_proba(X_test)
    >>> y_pred_test_rf = f.predict(X_test)
    >>> f_bmr = BayesMinimumRiskClassifier()
    >>> f_bmr.fit(y_test, y_prob_test)
    >>> y_pred_test_bmr = f_bmr.predict(y_prob_test, cost_mat_test)
    >>> # Savings using only RandomForest
    >>> print savings_score(y_test, y_pred_test_rf, cost_mat_test)
    0.12454256594
    >>> # Savings using RandomForest and Bayes Minimum Risk
    >>> print savings_score(y_test, y_pred_test_bmr, cost_mat_test)
    0.413425845555
    """
    def __init__(self, calibration=True):
        self.calibration = calibration

    def fit(self,y_true_cal=None, y_prob_cal=None):
        """ If calibration, then train the calibration of probabilities

        Parameters
        ----------
        y_true_cal : array-like of shape = [n_samples], optional default = None
            True class to be used for calibrating the probabilities

        y_prob_cal : array-like of shape = [n_samples, 2], optional default = None
            Predicted probabilities to be used for calibrating the probabilities

        Returns
        -------
        self : object
            Returns self.
        """
        if self.calibration:
            self.cal = ROCConvexHull()
            self.cal.fit(y_true_cal, y_prob_cal[:, 1])

    def predict(self, y_prob, cost_mat):
        """ Calculate the prediction using the Bayes minimum risk classifier.

        Parameters
        ----------
        y_prob : array-like of shape = [n_samples, 2]
            Predicted probabilities.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        y_pred : array-like of shape = [n_samples]
            Predicted class
        """
        if self.calibration:

            y_prob[:, 1] = self.cal.predict_proba(y_prob[:, 1])
            y_prob[:, 0] = 1 - y_prob[:, 1]

        # t_BMR = (cost_fp - cost_tn) / (cost_fn - cost_tn - cost_tp + cost_fp)
        # cost_mat[FP,FN,TP,TN]
        t_bmr = (cost_mat[:, 0] - cost_mat[:, 3]) / (cost_mat[:, 1] - cost_mat[:, 3] - cost_mat[:, 2] + cost_mat[:, 0])

        y_pred = np.greater(y_prob[:, 1], t_bmr).astype(np.float)

        return y_pred

    def fit_predict(self, y_prob, cost_mat, y_true_cal=None, y_prob_cal=None):
        """ Calculate the prediction using the Bayes minimum risk classifier.

        Parameters
        ----------
        y_prob : array-like of shape = [n_samples, 2]
            Predicted probabilities.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        y_true_cal : array-like of shape = [n_samples], optional default = None
            True class to be used for calibrating the probabilities

        y_prob_cal : array-like of shape = [n_samples, 2], optional default = None
            Predicted probabilities to be used for calibrating the probabilities

        Returns
        -------
        y_pred : array-like of shape = [n_samples]
            Predicted class
        """

        #TODO: Check input

        if self.calibration:
            self.cal = ROCConvexHull()

            if y_prob_cal is None:
                y_prob_cal = y_prob

            self.cal.fit(y_true_cal, y_prob_cal[:, 1])
            y_prob[:, 1] = self.cal.predict_proba(y_prob[:, 1])
            y_prob[:, 0] = 1 - y_prob[:, 1]

        # t_BMR = (cost_fp - cost_tn) / (cost_fn - cost_tn - cost_tp + cost_fp)
        # cost_mat[FP,FN,TP,TN]
        t_bmr = (cost_mat[:, 0] - cost_mat[:, 3]) / (cost_mat[:, 1] - cost_mat[:, 3] - cost_mat[:, 2] + cost_mat[:, 0])

        y_pred = np.greater(y_prob[:, 1], t_bmr).astype(np.float)

        return y_pred


class ThresholdingOptimization():
    """ Classifier based on finding the threshold that minimizes the total cost on a given set.

    Parameters
    ----------
    calibration : bool, optional (default=True)
        Whenever or not to calibrate the probabilities.

    Attributes
    ----------
    `threshold_` : float
        Selected threshold.

    References
    ----------

    .. [1] V. Sheng, C. Ling, "Thresholding for making classifiers cost-sensitive",
           in Proceedings of the National Conference on Artificial Intelligence, 2006.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import ThresholdingOptimization
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> f = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    >>> y_prob_train = f.predict_proba(X_train)
    >>> y_prob_test = f.predict_proba(X_test)
    >>> y_pred_test_rf = f.predict(X_test)
    >>> f_t = ThresholdingOptimization().fit(y_prob_train, cost_mat_train, y_train)
    >>> y_pred_test_rf_t = f_t.predict(y_prob_test)
    >>> # Savings using only RandomForest
    >>> print savings_score(y_test, y_pred_test_rf, cost_mat_test)
    0.12454256594
    >>> # Savings using RandomForest and Bayes Minimum Risk
    >>> print savings_score(y_test, y_pred_test_rf_t, cost_mat_test)
    0.401816361581
    """

    def __init__(self, calibration=True):
        self.calibration = calibration

    def fit(self, y_prob, cost_mat, y_true):
        """ Calculate the optimal threshold using the ThresholdingOptimization.

        Parameters
        ----------
        y_prob : array-like of shape = [n_samples, 2]
            Predicted probabilities.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        y_true : array-like of shape = [n_samples]
            True class

        Returns
        -------
        self
        """

        #TODO: Check input

        if self.calibration:
            cal = ROCConvexHull()

            cal.fit(y_true, y_prob[:, 1])
            y_prob[:, 1] = cal.predict_proba(y_prob[:, 1])
            y_prob[:, 0] = 1 - y_prob[:, 1]

        thresholds = np.unique(y_prob)

        cost = np.zeros(thresholds.shape)
        for i in range(thresholds.shape[0]):
            pred = np.floor(y_prob[:, 1]+(1-thresholds[i]))
            cost[i] = cost_loss(y_true, pred, cost_mat)

        self.threshold_ = thresholds[np.argmin(cost)]

        return self

    def predict(self, y_prob):
        """ Calculate the prediction using the ThresholdingOptimization.

        Parameters
        ----------
        y_prob : array-like of shape = [n_samples, 2]
            Predicted probabilities.

        Returns
        -------
        y_pred : array-like of shape = [n_samples]
            Predicted class
        """
        y_pred = np.floor(y_prob[:, 1] + (1 - self.threshold_))

        return y_pred
