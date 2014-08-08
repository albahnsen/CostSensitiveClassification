"""
This module include the sampling methods
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

import numpy as np
from costcla.sampling._smote import _SMOTE

def undersampling(X, y, cost_mat=None, per=0.5):
    """Under-sampling.

    Parameters
    ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y : array-like of shape = [n_samples]
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4], optional (default=None)
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        per: float, optional (default = 0.5)
            Percentage of the minority class in the under-sampled data
    """

    n_samples = X.shape[0]
    #TODO: allow y different from (0, 1)
    num_y1 = y.sum()
    num_y0 = n_samples - num_y1

    filter_rand = np.random.rand(int(num_y1 + num_y0))

    #TODO: rewrite in a more readable way
    if num_y1 < num_y0:
        num_y0_new = num_y1 * 1.0 / per - num_y1
        num_y0_new_per = num_y0_new * 1.0 / num_y0
        filter_0 = np.logical_and(y == 0, filter_rand <= num_y0_new_per)
        filter_ = np.nonzero(np.logical_or(y == 1, filter_0))[0]
    else:
        num_y1_new = num_y0 * 1.0 / per - num_y0
        num_y1_new_per = num_y1_new * 1.0 / num_y1
        filter_1 = np.logical_and(y == 1, filter_rand <= num_y1_new_per)
        filter_ = np.nonzero(np.logical_or(y == 0, filter_1))[0]

    X_u = X[filter_, :]
    y_u = y[filter_]

    if not cost_mat is None:
        cost_mat_u = cost_mat[filter_, :]
        return X_u, y_u, cost_mat_u
    else:
        return X_u, y_u
    

def smote(X, y, cost_mat=None, per=0.5):
    """SMOTE: synthetic minority over-sampling technique

    Parameters
    ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y : array-like of shape = [n_samples]
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4], optional (default=None)
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        per: float, optional (default = 0.5)
            Percentage of the minority class in the over-sampled data

    References
    ----------

    .. [1] N. Chawla, K. Bowyer, L. Hall, W. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique",
           Journal of Artificial Intelligence Research, 16, 321-357, 2002.

    Examples
    --------
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.sampling import smote
    >>> data = load_creditscoring1()
    >>> data_smote, target_smote = smote(data.data, data.target, per=0.7)
    >>> # Size of each training set
    >>> print data.data.shape[0], data_smote.shape[0]
    112915 204307
    >>> # Percentage of positives in each training set
    >>> print data.target.mean(), target_smote.mean()
    0.0674489660364 0.484604051746
    """
    #TODO: Add random state
    #TODO: Check input
    n_samples = X.shape[0]
    #TODO: allow y different from (0, 1)
    num_y1 = y.sum()
    num_y0 = n_samples - num_y1

    #TODO: rewrite in a more readable way
    if num_y1 < num_y0:
        N = int((num_y0 * 1.0 / (1 - per) - num_y0) / num_y1) * 100
        X_m = X[y == 1]
        X_majority = X[y == 0]
        minority = 1
    else:
        N = int((num_y1 * 1.0 / (1 - per) - num_y1) / num_y0) * 100
        X_m = X[y == 0]
        X_majority = X[y == 1]
        minority = 0

    X_m_o = _SMOTE(X_m, N, k=3)

    X_s = np.vstack((X_majority, X_m_o))

    n_samples_s = X_s.shape[0]

    y_s = np.ones(n_samples_s) * (minority - 1)**2
    y_s[max(num_y1, num_y0):] = minority

    #TODO: Include cost_mat

    return X_s, y_s