# coding=utf-8
"""Methods to calibrate the estimated probabilities.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

from sklearn.metrics import roc_curve
import numpy as np

#TODO: Add isotonic regression from sklearn
#TODO: Add Platt calibration
# http://ift.tt/XuMk3s

class ROCConvexHull:
    """Implementation the the calibration method ROCConvexHull

    Attributes
    ----------
    `calibration_map` : array-like
        calibration map for maping the raw probabilities to the calibrated probabilities.

    See also
    --------
    sklearn.IsotonicRegression

    References
    ----------

    .. [1] J. Hernandez-Orallo, P. Flach, C. Ferri, 'A Unified View of Performance Metrics :
           Translating Threshold Choice into Expected Classification Loss', Journal of
           Machine Learning Research, 13, 2813–2869, 2012.

    Examples
    --------
    >>> from costcla.probcal import ROCConvexHull
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.metrics import brier_score_loss
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> f = RandomForestClassifier()
    >>> f.fit(X_train, y_train)
    >>> y_prob_test = f.predict_proba(X_test)
    >>> f_cal = ROCConvexHull()
    >>> f_cal.fit(y_test, y_prob_test)
    >>> y_prob_test_cal = f_cal.predict_proba(y_prob_test)
    >>> # Brier score using only RandomForest
    >>> print brier_score_loss(y_test, y_prob_test[:, 1])
    0.0577615264881
    >>> # Brier score using calibrated RandomForest
    >>> print brier_score_loss(y_test, y_prob_test_cal)
    0.0553677407894
    """

    def __init__(self):
        self.calibration_map = []

    def fit(self, y, p):
        """ Fit the calibration map

        Parameters
        ----------
        y_true : array-like of shape = [n_samples]
            True class to be used for calibrating the probabilities

        y_prob : array-like of shape = [n_samples, 2]
            Predicted probabilities to be used for calibrating the probabilities

        Returns
        -------
        self : object
            Returns self.
        """

        # TODO: Check input
        if p.size != p.shape[0]:
            p = p[:, 1]

        fpr, tpr, thresholds = roc_curve(y, p)
        #works with sklearn 0.11
        if fpr.min() > 0 or tpr.min() > 0:
            fpr = np.hstack((0, fpr))
            tpr = np.hstack((0, tpr))
            thresholds = np.hstack((1.01, thresholds))

        def prob_freq(y, predict_proba):
            #calculate distribution and return in inverse order
            proba_bins = np.unique(predict_proba)
            freq_all = np.bincount(proba_bins.searchsorted(predict_proba))
            freq_0_tempa = np.unique(predict_proba[np.nonzero(y == 0)[0]])
            freq_0_tempb = np.bincount(freq_0_tempa.searchsorted(predict_proba[np.nonzero(y == 0)[0]]))
            freq = np.zeros((proba_bins.shape[0], 3))
            freq[:, 0] = proba_bins
            for i in range(freq_0_tempa.shape[0]):
                freq[np.nonzero(proba_bins == freq_0_tempa[i])[0], 1] = freq_0_tempb[i]
            freq[:, 2] = freq_all - freq[:, 1]
            freq = freq[proba_bins.argsort()[::-1], :]
            pr = freq[:, 2] / freq[:, 1:].sum(axis=1)
            pr = pr.reshape(freq.shape[0], 1)
            #fix when no negatives in range
            pr[pr == 1.0] = 0
            freq = np.hstack((freq, pr))
            return freq

        f = prob_freq(y, p)
        temp_hull = []
        for i in range(fpr.shape[0]):
            temp_hull.append((fpr[i], tpr[i]))
        #close the plane
        temp_hull.append((1, 0))
        rocch_ = _convexhull(temp_hull)
        rocch = np.array([(a, b) for (a, b) in rocch_[:-1]])
        rocch_find = np.zeros(fpr.shape[0], dtype=np.bool)
        for i in range(rocch.shape[0]):
            rocch_find[np.intersect1d(np.nonzero(rocch[i, 0] == fpr)[0],
                                      np.nonzero(rocch[i, 1] == tpr)[0])] = True
        rocch_thresholds = thresholds[rocch_find]
        #calibrated probabilities using ROCCH
        f_cal = np.zeros((rocch_thresholds.shape[0] - 1, 5))
        for i in range(rocch_thresholds.shape[0] - 1):
            f_cal[i, 0] = rocch_thresholds[i]
            f_cal[i, 1] = rocch_thresholds[i + 1]
            join_elements = np.logical_and(f_cal[i, 1] <= f[:, 0], f_cal[i, 0] > f[:, 0])
            f_cal[i, 2] = f[join_elements, 1].sum()
            f_cal[i, 3] = f[join_elements, 2].sum()
        f_cal[:, 4] = f_cal[:, 3] / f_cal[:, [2, 3]].sum(axis=1)
        #fix to add 0
        f_cal[-1, 1] = 0
        calibrated_map = f_cal[:, [0, 1, 4]]

        self.calibration_map = calibrated_map

    def predict_proba(self, p):
        """ Calculate the calibrated probabilities

        Parameters
        ----------
        y_prob : array-like of shape = [n_samples, 2]
            Predicted probabilities to be calibrated using calibration map

        Returns
        -------
        y_prob_cal : array-like of shape = [n_samples, 1]
            Predicted calibrated probabilities
        """

        # TODO: Check input
        if p.size != p.shape[0]:
            p = p[:, 1]

        calibrated_proba = np.zeros(p.shape[0])
        for i in range(self.calibration_map.shape[0]):
            calibrated_proba[np.logical_and(self.calibration_map[i, 1] <= p, self.calibration_map[i, 0] > p)] = \
                self.calibration_map[i, 2]

        # TODO: return 2D and refactor
        return calibrated_proba


def _convexhull(P):
    """ Private function that calculate the convex hull of a set of points
    The algorithm was taken from [1].
    http://code.activestate.com/recipes/66527-finding-the-convex-hull-of-a-set-of-2d-points/

    References
    ----------

    .. [1] Alex Martelli, Anna Ravenscroft, David Ascher, 'Python Cookbook', O'Reilly Media, Inc., 2005.
    """

    def mydet(p, q, r):
        """Calc. determinant of a special matrix with three 2D points.

        The sign, "-" or "+", determines the side, right or left,
        respectivly, on which the point r lies, when measured against
        a directed vector from p to q.
        """

        # We use Sarrus' Rule to calculate the determinant.
        # (could also use the Numeric package...)
        sum1 = q[0] * r[1] + p[0] * q[1] + r[0] * p[1]
        sum2 = q[0] * p[1] + r[0] * q[1] + p[0] * r[1]

        return sum1 - sum2


    def isrightturn((p, q, r)):
        "Do the vectors pq:qr form a right turn, or not?"

        assert p != q and q != r and p != r

        if mydet(p, q, r) < 0:
            return 1
        else:
            return 0

    # Get a local list copy of the points and sort them lexically.
    points = map(None, P)
    points.sort()

    # Build upper half of the hull.
    upper = [points[0], points[1]]
    for p in points[2:]:
        upper.append(p)
        while len(upper) > 2 and not isrightturn(upper[-3:]):
            del upper[-2]

    # Build lower half of the hull.
    points.reverse()
    lower = [points[0], points[1]]
    for p in points[2:]:
        lower.append(p)
        while len(lower) > 2 and not isrightturn(lower[-3:]):
            del lower[-2]

    # Remove duplicates.
    del lower[0]
    del lower[-1]

    # Concatenate both halfs and return.
    return tuple(upper + lower)
