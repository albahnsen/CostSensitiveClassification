"""Metrics to assess performance on cost-sensitive classification tasks
given class prediction and cost-matrix

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

import numpy as np
from sklearn.utils import column_or_1d
# from sklearn.utils import check_consistent_length  # Not in 0.15.1


def cost_loss(y_true, y_pred, cost_mat):
    #TODO: update description
    """Cost classification loss.

    This function calculates the cost of using y_pred on y_true with
    cost-matrix cost-mat. It differ from traditional classification evaluation
    measures since measures such as accuracy asing the same cost to different
    errors, but that is not the real case in several real-world classification
    problems as they are example-dependent cost-sensitive in nature, where the
    costs due to misclassification vary between examples.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or label indicator matrix
        Predicted labels, as returned by a classifier.

    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.

    Returns
    -------
    loss : float
        Cost of a using y_pred on y_true with cost-matrix cost-mat

    References
    ----------
    .. [1] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.

    .. [2] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    See also
    --------
    savings_score

    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> cost_mat = np.array([[4, 1, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0], [2, 1, 0, 0]])
    >>> cost_loss(y_true, y_pred, cost_mat)
    3
    """

    #TODO: Check consistency of cost_mat

    y_true = column_or_1d(y_true)
    y_true = (y_true == 1).astype(np.float)
    y_pred = column_or_1d(y_pred)
    y_pred = (y_pred == 1).astype(np.float)
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1] + y_pred * cost_mat[:, 2])
    cost += (1 - y_true) * (y_pred * cost_mat[:, 0] + (1 - y_pred) * cost_mat[:, 3])
    return np.sum(cost)


def savings_score(y_true, y_pred, cost_mat):
    #TODO: update description
    """Savings score.

    This function calculates the savings cost of using y_pred on y_true with
    cost-matrix cost-mat, as the difference of y_pred and the cost_loss of a naive
    classification model.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or label indicator matrix
        Predicted labels, as returned by a classifier.

    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.

    Returns
    -------
    score : float
        Savings of a using y_pred on y_true with cost-matrix cost-mat

        The best performance is 1.

    References
    ----------
    .. [1] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    See also
    --------
    cost_loss

    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import savings_score, cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> cost_mat = np.array([[4, 1, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0], [2, 1, 0, 0]])
    >>> savings_score(y_true, y_pred, cost_mat)
    0.5
    """

    #TODO: Check consistency of cost_mat
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    n_samples = len(y_true)

    # Calculate the cost of naive prediction
    cost_base = min(cost_loss(y_true, np.zeros(n_samples), cost_mat),
                    cost_loss(y_true, np.ones(n_samples), cost_mat))

    cost = cost_loss(y_true, y_pred, cost_mat)
    return 1.0 - cost / cost_base


# from https://github.com/agramfort/scikit-learn/blob/isotonic_calibration/sklearn/metrics/classification.py
# Waiting for https://github.com/scikit-learn/scikit-learn/pull/1176
#TODO: Remove when #1176 is merged
def brier_score_loss(y_true, y_prob):
    """Compute the Brier score

    The smaller the Brier score, the better, hence the naming with "loss".

    Across all items in a set N predictions, the Brier score measures the
    mean squared difference between (1) the predicted probability assigned
    to the possible outcomes for item i, and (2) the actual outcome.
    Therefore, the lower the Brier score is for a set of predictions, the
    better the predictions are calibrated. Note that the Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1).

    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another).

    Parameters
    ----------
    y_true : array, shape (n_samples,)
    True targets.

    y_prob : array, shape (n_samples,)
    Probabilities of the positive class.

    Returns
    -------
    score : float
    Brier score

    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import brier_score_loss
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.1, 0.9, 0.8, 0.3]
    >>> brier_score_loss(y_true, y_prob) # doctest: +ELLIPSIS
    0.037...
    >>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
    0.0

    References
    ----------
    http://en.wikipedia.org/wiki/Brier_score
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    return np.mean((y_true - y_prob) ** 2)
