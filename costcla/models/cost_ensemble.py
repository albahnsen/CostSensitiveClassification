"""
This module include the cost sensitive ensemble methods.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

from sklearn.cross_validation import train_test_split
from ..models import CostSensitiveDecisionTreeClassifier
from ..models.bagging import BaggingClassifier


class CostSensitiveRandomForestClassifier(BaggingClassifier):
    """A example-dependent cost-sensitive random forest  classifier.

    Parameters
    ----------

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination,.
          - If "stacking_bmr" then a Cost Sensitive Logistic Regression is used
            to learn the probabilities and a BayesMinimumRisk for the prediction.
          - If "stacking_proba_bmr" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the probabilities,
            and a BayesMinimumRisk for the prediction.
          - If "majority_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of majority_voting
          - If "weighted_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of weighted_voting

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split in each tree:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    See also
    --------
    costcla.models.CostSensitiveDecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveRandomForestClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveRandomForestClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CostSensitiveRandomForestClassifier
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.499390945808
    """
    def __init__(self,
                 n_estimators=10,
                 combination='majority_voting',
                 max_features='auto',
                 n_jobs=1,
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(max_features=max_features, pruned=pruned),
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            combination=combination,
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class CostSensitiveBaggingClassifier(BaggingClassifier):
    """A example-dependent cost-sensitive bagging  classifier.

    Parameters
    ----------

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=0.5)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination,.
          - If "stacking_bmr" then a Cost Sensitive Logistic Regression is used
            to learn the probabilities and a BayesMinimumRisk for the prediction.
          - If "stacking_proba_bmr" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the probabilities,
            and a BayesMinimumRisk for the prediction.
          - If "majority_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of majority_voting
          - If "weighted_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of weighted_voting

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    See also
    --------
    costcla.models.CostSensitiveDecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveBaggingClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveBaggingClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CostSensitiveRandomForestClassifier
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.478964004931
    """
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 combination='majority_voting',
                 n_jobs=1,
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(pruned=pruned),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            combination=combination,
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class CostSensitivePastingClassifier(BaggingClassifier):
    """A example-dependent cost-sensitive pasting  classifier.

    Parameters
    ----------

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=0.5)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination,.
          - If "stacking_bmr" then a Cost Sensitive Logistic Regression is used
            to learn the probabilities and a BayesMinimumRisk for the prediction.
          - If "stacking_proba_bmr" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the probabilities,
            and a BayesMinimumRisk for the prediction.
          - If "majority_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of majority_voting
          - If "weighted_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of weighted_voting

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    See also
    --------
    costcla.models.CostSensitiveDecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitivePastingClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitivePastingClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CostSensitiveRandomForestClassifier
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.479633754848
    """
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 combination='majority_voting',
                 n_jobs=1,
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(pruned=pruned),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            combination=combination,
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class CostSensitiveRandomPatchesClassifier(BaggingClassifier):
    """A example-dependent cost-sensitive pasting  classifier.

    Parameters
    ----------

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=0.5)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=0.5)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    combination : string, optional (default="majority_voting")
        Which combination method to use:
          - If "majority_voting" then combine by majority voting
          - If "weighted_voting" then combine by weighted voting using the
            out of bag savings as the weight for each estimator.
          - If "stacking" then a Cost Sensitive Logistic Regression is used
            to learn the combination.
          - If "stacking_proba" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the combination,.
          - If "stacking_bmr" then a Cost Sensitive Logistic Regression is used
            to learn the probabilities and a BayesMinimumRisk for the prediction.
          - If "stacking_proba_bmr" then a Cost Sensitive Logistic Regression trained
            with the estimated probabilities is used to learn the probabilities,
            and a BayesMinimumRisk for the prediction.
          - If "majority_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of majority_voting
          - If "weighted_bmr" then the BayesMinimumRisk algorithm is used to make the
            prediction using the predicted probabilities of weighted_voting

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    `base_estimator_`: list of estimators
        The base estimator from which the ensemble is grown.

    `estimators_`: list of estimators
        The collection of fitted base estimators.

    `estimators_samples_`: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    `estimators_features_`: list of arrays
        The subset of drawn features for each base estimator.

    See also
    --------
    costcla.models.CostSensitiveDecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Ensemble of Example-Dependent Cost-Sensitive Decision Trees" <http://arxiv.org/abs/1505.04637>`__,
           2015, http://arxiv.org/abs/1505.04637.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveRandomPatchesClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveRandomPatchesClassifier(combination='weighted_voting')
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CostSensitiveRandomForestClassifier
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.499548618518
    """
    def __init__(self,
                 n_estimators=10,
                 max_samples=0.5,
                 max_features=0.5,
                 combination='majority_voting',
                 n_jobs=1,
                 verbose=False,
                 pruned=False):
        super(BaggingClassifier, self).__init__(
            base_estimator=CostSensitiveDecisionTreeClassifier(pruned=pruned),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=False,
            bootstrap_features=False,
            combination=combination,
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


#TODO not working in parallel, without error

# from costcla.datasets import load_creditscoring1
# data = load_creditscoring1()
# x=data.data
# y=data.target
# c=data.cost_mat
#
# print 'start'
# f = BaggingClassifier(n_estimators=10, verbose=100, n_jobs=2)
# f.fit(x[0:1000],y[0:1000],c[0:1000])
# print 'predict proba'
# f.__setattr__('n_jobs', 4)
# f.predict(x)
# print 'predict END'
