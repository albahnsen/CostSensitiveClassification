"""
Base IO code for all datasets
https://github.com/scikit-learn/scikit-learn/blob/56057c9630dd13f3c61fbb4c7debdff6ba8e9e8c/sklearn/datasets/base.py
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
#               2014 Alejandro CORREA BAHNSEN <al.bahnsen@gmail.com>
# License: BSD 3 clause

from os.path import dirname
from os.path import join
import numpy as np
import pandas as pd


class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_bankmarketing(cost_mat_parameters=None):
    """Load and return the bank marketing dataset (classification).

    The bank marketing is a easily transformable example-dependent cost-sensitive classification dataset.

    Parameters
    ----------
    cost_mat_parameters : Dictionary-like object, optional (default=None)
        If not None, must include 'per_balance', 'ca', and 'int_r'

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'cost_mat', the cost matrix of each example,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the full description of the dataset.

    References
    ----------
    .. [1] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           "Improving Credit Card Fraud Detection with Calibrated Probabilities",
           in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50

    >>> from costcla.datasets import load_bankmarketing
    >>> data = load_bankmarketing()
    >>> data.target[[10, 25, 319]]
    array([0, 0, 1])
    >>> data.cost_mat[[10, 25, 319]]
    array([[ 1.        ,  1.66274977,  1.        ,  0.        ],
           [ 1.        ,  1.63195811,  1.        ,  0.        ],
           [ 1.        ,  5.11141597,  1.        ,  0.        ]])
    """
    module_path = dirname(__file__)
    raw_data = pd.read_csv(join(module_path, 'data', 'bankmarketing.csv.gz'), delimiter=';', compression='gzip')
    descr = open(join(module_path, 'descr', 'bankmarketing.rst')).read()

    #only use features pre-contact:
    # 1 - age (numeric)
    # 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur",
    #                        "student","blue-collar","self-employed","retired","technician","services")
    # 3 - marital : marital status (categorical: "married","divorced","single";
    #                               note: "divorced" means divorced or widowed)
    # 4 - education (categorical: "unknown","secondary","primary","tertiary")
    # 5 - default: has credit in default? (binary: "yes","no")
    # 6 - balance: average yearly balance, in euros (numeric)
    # 7 - housing: has housing loan? (binary: "yes","no")
    # 8 - loan: has personal loan? (binary: "yes","no")
    # 15 - previous: number of contacts performed before this campaign and for this client (numeric)
    # 16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

    #Folloring features exclude because are collected after the contact event
    # # related with the last contact of the current campaign:
    # 9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
    # 10 - day: last contact day of the month (numeric)
    # 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
    # 12 - duration: last contact duration, in seconds (numeric)
    # # other attributes:
    # 13 - campaign: number of contacts performed during this campaign and for this client
    # 14 - pdays: number of days that passed by after the client was last contacted from a
    #       previous campaign (numeric, -1 means client was not previously contacted)

    #Filter if balance>0
    raw_data = raw_data.loc[raw_data['balance'] > 0]

    n_samples = raw_data.shape[0]

    target = np.zeros((n_samples,), dtype=np.int)
    target[raw_data['y'].values == 'yes'] = 1
    raw_data = raw_data.drop('y', 1)

    # Create dummies
    data = raw_data[['age', 'balance', 'previous']]
    cols_dummies = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']

    for col_ in cols_dummies:
        temp_ = pd.get_dummies(raw_data[col_], prefix=col_)
        data = data.join(temp_)

    # Calculate cost_mat (see[1])
    if cost_mat_parameters is None:
        cost_mat_parameters = {'per_balance': 0.25, 'ca': 1, 'int_r': 0.02463333}

    per_balance = cost_mat_parameters['per_balance']
    ca = cost_mat_parameters['ca']
    int_r = cost_mat_parameters['int_r']

    cost_mat = np.zeros((n_samples, 4))  # cost_mat[FP,FN,TP,TN]
    cost_mat[:, 0] = ca
    cost_mat[:, 1] = np.maximum(data['balance'].values * int_r * per_balance, ca)  # C_FN >= C_TN Elkan
    cost_mat[:, 2] = ca
    cost_mat[:, 3] = 0.0

    return Bunch(data=data.values, target=target, cost_mat=cost_mat,
                 target_names=['no', 'yes'], DESCR=descr,
                 feature_names=data.columns.values)