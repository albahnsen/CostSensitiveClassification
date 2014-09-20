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
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
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
                 feature_names=data.columns.values, name='DirectMarketing')


def load_creditscoring1(cost_mat_parameters=None):
    """Load and return the credit scoring Kaggle Credit competition dataset (classification).

    The credit scoring is a easily transformable example-dependent cost-sensitive classification dataset.

    Parameters
    ----------
    cost_mat_parameters : Dictionary-like object, optional (default=None)
        If not None, must include 'int_r', 'int_cf', 'cl_max', 'n_term', 'k','lgd'

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
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50

    >>> from costcla.datasets import load_creditscoring1
    >>> data = load_creditscoring1()
    >>> data.target[[10, 17, 400]]
    array([0, 1, 0])
    >>> data.cost_mat[[10, 17, 400]]
    array([[  1023.73054104,  18750.        ,      0.        ,      0.        ],
           [   717.25781516,   6749.25      ,      0.        ,      0.        ],
           [  1004.32819923,  17990.25      ,      0.        ,      0.        ]])
    """
    module_path = dirname(__file__)
    raw_data = pd.read_csv(join(module_path, 'data', 'creditscoring1.csv.gz'), delimiter=',', compression='gzip')
    descr = open(join(module_path, 'descr', 'creditscoring1.rst')).read()

    # Exclude MonthlyIncome = nan or =0 or DebtRatio >1
    raw_data = raw_data.dropna()
    raw_data = raw_data.loc[(raw_data['MonthlyIncome'] > 0)]
    raw_data = raw_data.loc[(raw_data['DebtRatio'] < 1)]

    target = raw_data['SeriousDlqin2yrs'].values.astype(np.int)

    data = raw_data.drop(['SeriousDlqin2yrs', 'id'], 1)

    # Calculate cost_mat (see[1])
    if cost_mat_parameters is None:
        cost_mat_parameters = {'int_r': 0.0479 / 12,
                               'int_cf': 0.0294 / 12,
                               'cl_max': 25000,
                               'n_term': 24,
                               'k': 3,
                               'lgd': .75}

    pi_1 = target.mean()
    cost_mat = _creditscoring_costmat(data['MonthlyIncome'].values, data['DebtRatio'].values, pi_1, cost_mat_parameters)

    return Bunch(data=data.values, target=target, cost_mat=cost_mat,
                 target_names=['no', 'yes'], DESCR=descr,
                 feature_names=data.columns.values, name='CreditScoring_Kaggle2011')


def load_creditscoring2(cost_mat_parameters=None):
    """Load and return the credit scoring PAKDD 2009 competition dataset (classification).

    The credit scoring is a easily transformable example-dependent cost-sensitive classification dataset.

    Parameters
    ----------
    cost_mat_parameters : Dictionary-like object, optional (default=None)
        If not None, must include 'int_r', 'int_cf', 'cl_max', 'n_term', 'k','lgd'

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
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50

    >>> from costcla.datasets import load_creditscoring2
    >>> data = load_creditscoring2()
    >>> data.target[[10, 17, 50]]
    array([1, 0, 0])
    >>> data.cost_mat[[10, 17, 50]]
    array([[ 209.   ,  547.965,    0.   ,    0.   ],
           [  24.   ,  274.725,    0.   ,    0.   ],
           [  89.   ,  371.25 ,    0.   ,    0.   ]])
    """
    module_path = dirname(__file__)
    raw_data = pd.read_csv(join(module_path, 'data', 'creditscoring2.csv.gz'), delimiter='\t', compression='gzip')
    descr = open(join(module_path, 'descr', 'creditscoring2.rst')).read()

    # Exclude TARGET_LABEL_BAD=1 == 'N'
    raw_data = raw_data.loc[raw_data['TARGET_LABEL_BAD=1'] != 'N']

    # Exclude 100<PERSONAL_NET_INCOME<10000
    raw_data = raw_data.loc[(raw_data['PERSONAL_NET_INCOME'].values.astype(np.float) > 100)]
    raw_data = raw_data.loc[(raw_data['PERSONAL_NET_INCOME'].values.astype(np.float) < 10000)]

    target = raw_data['TARGET_LABEL_BAD=1'].values.astype(np.int)

    # Continuous features
    cols_con = ['ID_SHOP', 'AGE', 'AREA_CODE_RESIDENCIAL_PHONE', 'PAYMENT_DAY', 'SHOP_RANK',
                'MONTHS_IN_RESIDENCE', 'MONTHS_IN_THE_JOB', 'PROFESSION_CODE', 'MATE_INCOME',
                'QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION', 'PERSONAL_NET_INCOME']
    data = raw_data[cols_con].astype(float)

    cols_dummies = ['SEX', 'MARITAL_STATUS', 'FLAG_RESIDENCIAL_PHONE', 'RESIDENCE_TYPE',
                    'FLAG_MOTHERS_NAME', 'FLAG_FATHERS_NAME', 'FLAG_RESIDENCE_TOWN_eq_WORKING_TOWN',
                    'FLAG_RESIDENCE_STATE_eq_WORKING_STATE', 'FLAG_RESIDENCIAL_ADDRESS_eq_POSTAL_ADDRESS']
    for col_ in cols_dummies:
        temp_ = pd.get_dummies(raw_data[col_], prefix=col_)
        data = data.join(temp_)

    # Calculate cost_mat (see[1])
    if cost_mat_parameters is None:
        cost_mat_parameters = {'int_r': 0.63 / 12,
                               'int_cf': 0.165 / 12,
                               'cl_max': 25000 * 0.33,
                               'n_term': 24,
                               'k': 3,
                               'lgd': .75}

    n_samples = data.shape[0]
    pi_1 = target.mean()
    monthly_income = data['PERSONAL_NET_INCOME'].values * 0.33
    cost_mat = _creditscoring_costmat(monthly_income, np.zeros(n_samples), pi_1, cost_mat_parameters)

    return Bunch(data=data.values, target=target, cost_mat=cost_mat,
                 target_names=['no', 'yes'], DESCR=descr,
                 feature_names=data.columns.values, name='CreditScoring_PAKDD2009')


def _creditscoring_costmat(income, debt, pi_1, cost_mat_parameters):
    """ Private function to calculate the cost matrix of credit scoring models.

    Parameters
    ----------
    income : array of shape = [n_samples]
        Monthly income of each example

    debt : array of shape = [n_samples]
        Debt ratio each example

    pi_1 : float
        Percentage of positives in the training set

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Returns
    -------
    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.
    """
    def calculate_a(cl_i, int_, n_term):
        """ Private function """
        return cl_i * ((int_ * (1 + int_) ** n_term) / ((1 + int_) ** n_term - 1))

    def calculate_pv(a, int_, n_term):
        """ Private function """
        return a / int_ * (1 - 1 / (1 + int_) ** n_term)

    #Calculate credit line Cl
    def calculate_cl(k, inc_i, cl_max, debt_i, int_r, n_term):
        """ Private function """
        cl_k = k * inc_i
        A = calculate_a(cl_k, int_r, n_term)
        Cl_debt = calculate_pv(inc_i * min(A / inc_i, 1 - debt_i), int_r, n_term)
        return min(cl_k, cl_max, Cl_debt)

    #calculate costs
    def calculate_cost_fn(cl_i, lgd):
        return cl_i * lgd

    def calculate_cost_fp(cl_i, int_r, n_term, int_cf, pi_1, lgd, cl_avg):
        a = calculate_a(cl_i, int_r, n_term)
        pv = calculate_pv(a, int_cf, n_term)
        r = pv - cl_i
        r_avg = calculate_pv(calculate_a(cl_avg, int_r, n_term), int_cf, n_term) - cl_avg
        cost_fp = r - (1 - pi_1) * r_avg + pi_1 * calculate_cost_fn(cl_avg, lgd)
        return max(0, cost_fp)

    v_calculate_cost_fp = np.vectorize(calculate_cost_fp)
    v_calculate_cost_fn = np.vectorize(calculate_cost_fn)

    v_calculate_cl = np.vectorize(calculate_cl)

    # Parameters
    k = cost_mat_parameters['k']
    int_r = cost_mat_parameters['int_r']
    n_term = cost_mat_parameters['n_term']
    int_cf = cost_mat_parameters['int_cf']
    lgd = cost_mat_parameters['lgd']
    cl_max = cost_mat_parameters['cl_max']

    cl = v_calculate_cl(k, income, cl_max, debt, int_r, n_term)
    cl_avg = cl.mean()

    n_samples = income.shape[0]
    cost_mat = np.zeros((n_samples, 4))  #cost_mat[FP,FN,TP,TN]
    cost_mat[:, 0] = v_calculate_cost_fp(cl, int_r, n_term, int_cf, pi_1, lgd, cl_avg)
    cost_mat[:, 1] = v_calculate_cost_fn(cl, lgd)
    cost_mat[:, 2] = 0.0
    cost_mat[:, 3] = 0.0

    return cost_mat
