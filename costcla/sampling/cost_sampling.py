"""
This module include the cost proportionate sampling methods
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

import numpy as np


def cost_sampling(X, y, cost_mat, method='RejectionSampling', oversampling_norm=0.1, max_wc=97.5):
    """Cost-proportionate sampling.

    Parameters
    ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y : array-like of shape = [n_samples]
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        method : str, optional (default = RejectionSampling)
            Method to perform the cost-proportionate sampling,
            either 'RejectionSampling' or 'OverSampling'.

        oversampling_norm: float, optional (default = 0.1)
            normalize value of wc, the smaller the biggest the data.

        max_wc: float, optional (default = 97.5)
            outlier adjustment for the cost.

    References
    ----------

    .. [1] B. Zadrozny, J. Langford, N. Naoki, "Cost-sensitive learning by
           cost-proportionate example weighting", in Proceedings of the
           Third IEEE International Conference on Data Mining, 435-442, 2003.

    .. [2] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.sampling import cost_sampling, undersampling
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> X_cps_o, y_cps_o, cost_mat_cps_o =  cost_sampling(X_train, y_train, cost_mat_train, method='OverSampling')
    >>> X_cps_r, y_cps_r, cost_mat_cps_r =  cost_sampling(X_train, y_train, cost_mat_train, method='RejectionSampling')
    >>> X_u, y_u, cost_mat_u = undersampling(X_train, y_train, cost_mat_train)
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> y_pred_test_rf_cps_o = RandomForestClassifier(random_state=0).fit(X_cps_o, y_cps_o).predict(X_test)
    >>> y_pred_test_rf_cps_r = RandomForestClassifier(random_state=0).fit(X_cps_r, y_cps_r).predict(X_test)
    >>> y_pred_test_rf_u = RandomForestClassifier(random_state=0).fit(X_u, y_u).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print savings_score(y_test, y_pred_test_rf, cost_mat_test)
    0.12454256594
    >>> # Savings using RandomForest with cost-proportionate over-sampling
    >>> print savings_score(y_test, y_pred_test_rf_cps_o, cost_mat_test)
    0.192480226286
    >>> # Savings using RandomForest with cost-proportionate rejection-sampling
    >>> print savings_score(y_test, y_pred_test_rf_cps_r, cost_mat_test)
    0.465830173459
    >>> # Savings using RandomForest with under-sampling
    >>> print savings_score(y_test, y_pred_test_rf_u, cost_mat_test)
    0.466630646543
    >>> # Size of each training set
    >>> print X_train.shape[0], X_cps_o.shape[0], X_cps_r.shape[0], X_u.shape[0]
    75653 109975 8690 10191
    >>> # Percentage of positives in each training set
    >>> print y_train.mean(), y_cps_o.mean(), y_cps_r.mean(), y_u.mean()
    0.0668182358928 0.358054103205 0.436939010357 0.49602590521
    """

    #TODO: Check consistency of input

    # The methods are construct only for the misclassification costs, not the full cost matrix.
    cost_mis = cost_mat[:, 0]
    cost_mis[y == 1] = cost_mat[y == 1, 1]

    # wc = cost_mis / cost_mis.max()
    wc = np.minimum(cost_mis / np.percentile(cost_mis, max_wc), 1)

    n_samples = X.shape[0]

    filter_ = range(n_samples)

    if method == 'RejectionSampling':
        # under-sampling by rejection [1]
        #TODO: Add random state
        rej_rand = np.random.rand(n_samples)

        filter_ = rej_rand <= wc

    elif method == 'OverSampling':
        # over-sampling with normalized wn [2]
        wc_n = np.ceil(wc / oversampling_norm).astype(np.int)

        new_n = wc_n.sum()

        filter_ = np.ones(new_n, dtype=np.int)

        e = 0
        #TODO replace for
        for i in range(n_samples):
            filter_[e: e + wc_n[i]] = i
            e += wc_n[i]

    x_cps = X[filter_]
    y_cps = y[filter_]
    cost_mat_cps = cost_mat[filter_]

    return x_cps, y_cps, cost_mat_cps