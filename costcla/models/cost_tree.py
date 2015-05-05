"""
This module include the cost sensitive decision tree method.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

import numpy as np
import copy
from ..metrics import cost_loss
from sklearn.base import BaseEstimator
from sklearn.externals import six
import numbers

class CostSensitiveDecisionTreeClassifier(BaseEstimator):
    """A example-dependent cost-sensitive binary decision tree classifier.

    Parameters
    ----------
    criterion : string, optional (default="direct_cost")
        The function to measure the quality of a split. Supported criteria are
        "direct_cost" for the Direct Cost impurity measure, "pi_cost", "gini_cost",
        and "entropy_cost".

    criterion_weight : bool, optional (default=False)
        Whenever or not to weight the gain according to the population distribution.

    num_pct : int, optional (default=100)
        Number of percentiles to evaluate the splits for each feature.



    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
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

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_samples_leaf`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_gain : float, optional (default=0.001)
        The minimum gain that a split must produce in order to be taken into account.

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           "Example-Dependent Cost-Sensitive Decision Trees. Expert Systems with Applications",
           Expert Systems with Applications, in press, 2015,
           http://doi.org/10.1016/j.eswa.2015.04.042

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveDecisionTreeClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveDecisionTreeClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print savings_score(y_test, y_pred_test_rf, cost_mat_test)
    0.12454256594
    >>> # Savings using CSDecisionTree
    >>> print savings_score(y_test, y_pred_test_csdt, cost_mat_test)
    0.481916135529
    """
    def __init__(self,
                 criterion='direct_cost',
                 criterion_weight=False,
                 num_pct=100,
                 max_features=None,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_gain=0.001,
                 pruned=True,
                 ):

        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.num_pct = num_pct
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.pruned = pruned

        self.n_features_ = None
        self.max_features_ = None

        self.tree_ = []

    def set_param(self, attribute, value):
        setattr(self, attribute, value)

    def _node_cost(self, y_true, cost_mat):
        """ Private function to calculate the cost of a node.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        tuple(cost_loss : float, node prediction : int, node predicted probability : float)

        """
        n_samples = len(y_true)

        # Evaluates the cost by predicting the node as positive and negative
        costs = np.zeros(2)
        costs[0] = cost_loss(y_true, np.zeros(y_true.shape), cost_mat)
        costs[1] = cost_loss(y_true, np.ones(y_true.shape), cost_mat)

        pi = np.array([1 - y_true.mean(), y_true.mean()])

        if self.criterion == 'direct_cost':
            costs = costs
        elif self.criterion == 'pi_cost':
            costs *= pi
        elif self.criterion == 'gini_cost':
            costs *= pi ** 2
        elif self.criterion in 'entropy_cost':
            if pi[0] == 0 or pi[1] == 0:
                costs *= 0
            else:
                costs *= -np.log(pi)

        y_pred = np.argmin(costs)

        # Calculate the predicted probability of a node using laplace correction.
        n_positives = y_true.sum()
        y_prob = (n_positives + 1.0) / (n_samples + 2.0)

        return costs[y_pred], y_pred, y_prob

    def _calculate_gain(self, cost_base, y_true, X, cost_mat, split):
        """ Private function to calculate the gain in cost of using split in the
         current node.

        Parameters
        ----------
        cost_base : float
            Cost of the naive prediction

        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        split : tuple of len = 2
            split[0] = feature to split = j
            split[1] = where to split = l

        Returns
        -------
        tuple(gain : float, left node prediction : int)

        """

        # Check if cost_base == 0, then no gain is possible
        #TODO: This must be check in _best_split
        if cost_base == 0.0:
            return 0.0, int(np.sign(y_true.mean() - 0.5) == 1)  # In case cost_b==0 and pi_1!=(0,1)

        j, l = split
        filter_Xl = (X[:, j] <= l)
        filter_Xr = ~filter_Xl
        n_samples, n_features = X.shape

        # Check if one of the leafs is empty
        #TODO: This must be check in _best_split
        if np.nonzero(filter_Xl)[0].shape[0] in [0, n_samples]:  # One leaft is empty
            return 0.0, 0.0

        # Split X in Xl and Xr according to rule split
        Xl_cost, Xl_pred, _ = self._node_cost(y_true[filter_Xl], cost_mat[filter_Xl, :])
        Xr_cost, _, _ = self._node_cost(y_true[filter_Xr], cost_mat[filter_Xr, :])

        if self.criterion_weight:
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            Xl_w = n_samples_Xl * 1.0 / n_samples
            Xr_w = 1 - Xl_w
            gain = round((cost_base - (Xl_w * Xl_cost + Xr_w * Xr_cost)) / cost_base, 6)
        else:
            gain = round((cost_base - (Xl_cost + Xr_cost)) / cost_base, 6)

        return gain, Xl_pred

    def _best_split(self, y_true, X, cost_mat):
        """ Private function to calculate the split that gives the best gain.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        tuple(split : tuple(j, l), gain : float, left node prediction : int,
              y_pred : int, y_prob : float)

        """

        n_samples, n_features = X.shape
        num_pct = self.num_pct

        cost_base, y_pred, y_prob = self._node_cost(y_true, cost_mat)

        # Calculate the gain of all features each split in num_pct
        gains = np.zeros((n_features, num_pct))
        pred = np.zeros((n_features, num_pct))
        splits = np.zeros((n_features, num_pct))

        # Selected features
        selected_features = np.arange(0, self.n_features_)
        # Add random state
        np.random.shuffle(selected_features)
        selected_features = selected_features[:self.max_features_]
        selected_features.sort()

        #TODO:  # Skip the CPU intensive evaluation of the impurity criterion for
                # features that were already detected as constant (hence not suitable
                # for good splitting) by ancestor nodes and save the information on
                # newly discovered constant features to spare computation on descendant
                # nodes.

        # For each feature test all possible splits
        for j in selected_features:
            splits[j, :] = np.percentile(X[:, j], np.arange(0, 100, 100.0 / num_pct).tolist())

            for l in range(num_pct):
                # Avoid repeated values, since np.percentile may return repeated values
                if l == 0 or (l > 0 and splits[j, l] != splits[j, l - 1]):
                    split = (j, splits[j, l])
                    gains[j, l], pred[j, l] = self._calculate_gain(cost_base, y_true, X, cost_mat, split)

        best_split = np.unravel_index(gains.argmax(), gains.shape)

        return (best_split[0], splits[best_split]), gains.max(), pred[best_split], y_pred, y_prob

    def _tree_grow(self, y_true, X, cost_mat, level=0):
        """ Private recursive function to grow the decision tree.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        Tree : Object
            Container of the decision tree
            NOTE: it is not the same structure as the sklearn.tree.tree object

        """

        #TODO: Find error, add min_samples_split
        if len(X.shape) == 1:
            tree = dict(y_pred=y_true, y_prob=0.5, level=level, split=-1, n_samples=1, gain=0)
            return tree

        # Calculate the best split of the current node
        split, gain, Xl_pred, y_pred, y_prob = self._best_split(y_true, X, cost_mat)

        n_samples, n_features = X.shape

        # Construct the tree object as a dictionary

        #TODO: Convert tree to be equal to sklearn.tree.tree object
        tree = dict(y_pred=y_pred, y_prob=y_prob, level=level, split=-1, n_samples=n_samples, gain=gain)

        # Check the stopping criteria
        if gain < self.min_gain:
            return tree
        if self.max_depth is not None:
            if level >= self.max_depth:
                return tree
        if n_samples <= self.min_samples_split:
            return tree
        
        j, l = split
        filter_Xl = (X[:, j] <= l)
        filter_Xr = ~filter_Xl
        n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
        n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

        if min(n_samples_Xl, n_samples_Xr) <= self.min_samples_leaf:
            return tree

        # No stooping criteria is met
        tree['split'] = split
        tree['node'] = self.tree_.n_nodes
        self.tree_.n_nodes += 1

        tree['sl'] = self._tree_grow(y_true[filter_Xl], X[filter_Xl], cost_mat[filter_Xl], level + 1)
        tree['sr'] = self._tree_grow(y_true[filter_Xr], X[filter_Xr], cost_mat[filter_Xr], level + 1)

        return tree

    class _tree_class():
        def __init__(self):
            self.n_nodes = 0
            self.tree = dict()
            self.tree_pruned = dict()
            self.nodes = []
            self.n_nodes_pruned = 0

    def fit(self, X, y, cost_mat, check_input=False):
        """ Build a example-dependent cost-sensitive decision tree from the training set (X, y, cost_mat)

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.


        Returns
        -------
        self : object
            Returns self.
        """

        #TODO: Check input
        #TODO: Add random state
        n_samples, self.n_features_ = X.shape

        self.tree_ = self._tree_class()

        # Maximum number of features to be taken into account per split
        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 1  # On sklearn is 0.
        self.max_features_ = max_features

        self.tree_.tree = self._tree_grow(y, X, cost_mat)

        if self.pruned:
            self.pruning(X, y, cost_mat)

        self.classes_ = np.array([0, 1])

        return self

    def _nodes(self, tree):
        """ Private function that find the number of nodes in a tree.

        Parameters
        ----------
        tree : object

        Returns
        -------
        nodes : array like of shape [n_nodes]
        """
        def recourse(temp_tree_, nodes):
            if isinstance(temp_tree_, dict):
                if temp_tree_['split'] != -1:
                    nodes.append(temp_tree_['node'])
                    if temp_tree_['split'] != -1:
                        for k in ['sl', 'sr']:
                            recourse(temp_tree_[k], nodes)
            return None

        nodes_ = []
        recourse(tree, nodes_)
        return nodes_

    def _classify(self, X, tree, proba=False):
        """ Private function that classify a dataset using tree.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        tree : object

        proba : bool, optional (default=False)
            If True then return probabilities else return class

        Returns
        -------
        prediction : array of shape = [n_samples]
            If proba then return the predicted positive probabilities, else return
            the predicted class for each example in X
        """

        n_samples, n_features = X.shape
        predicted = np.ones(n_samples)

        # Check if final node
        if tree['split'] == -1:
            if not proba:
                predicted = predicted * tree['y_pred']
            else:
                predicted = predicted * tree['y_prob']
        else:
            j, l = tree['split']
            filter_Xl = (X[:, j] <= l)
            filter_Xr = ~filter_Xl
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

            if n_samples_Xl == 0:  # If left node is empty only continue with right
                predicted[filter_Xr] = self._classify(X[filter_Xr, :], tree['sr'], proba)
            elif n_samples_Xr == 0:  # If right node is empty only continue with left
                predicted[filter_Xl] = self._classify(X[filter_Xl, :], tree['sl'], proba)
            else:
                predicted[filter_Xl] = self._classify(X[filter_Xl, :], tree['sl'], proba)
                predicted[filter_Xr] = self._classify(X[filter_Xr, :], tree['sr'], proba)

        return predicted

    def predict(self, X):
        """ Predict class of X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes,
        """
        #TODO: Check consistency of X
        if self.pruned:
            tree_ = self.tree_.tree_pruned
        else:
            tree_ = self.tree_.tree

        return self._classify(X, tree_, proba=False)

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        prob : array of shape = [n_samples, 2]
            The class probabilities of the input samples.
        """
        #TODO: Check consistency of X
        n_samples, n_features = X.shape
        prob = np.zeros((n_samples, 2))

        if self.pruned:
            tree_ = self.tree_.tree_pruned
        else:
            tree_ = self.tree_.tree

        prob[:, 1] = self._classify(X, tree_, proba=True)
        prob[:, 0] = 1 - prob[:, 1]

        return prob

    def _delete_node(self, tree, node):
        """ Private function that eliminate node from tree.

        Parameters
        ----------

        tree : object

        node : int
            node to be eliminated from tree

        Returns
        -------

        pruned_tree : object
        """
        # Calculate gains
        temp_tree = copy.deepcopy(tree)

        def recourse(temp_tree_, del_node):
            if isinstance(temp_tree_, dict):
                if temp_tree_['split'] != -1:
                    if temp_tree_['node'] == del_node:
                        del temp_tree_['sr']
                        del temp_tree_['sl']
                        del temp_tree_['node']
                        temp_tree_['split'] = -1
                    else:
                        for k in ['sl', 'sr']:
                            recourse(temp_tree_[k], del_node)
            return None

        recourse(temp_tree, node)
        return temp_tree

    def _pruning(self, X, y_true, cost_mat):
        """ Private function that prune the decision tree.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        """
        # Calculate gains
        nodes = self._nodes(self.tree_.tree_pruned)
        n_nodes = len(nodes)
        gains = np.zeros(n_nodes)

        y_pred = self._classify(X, self.tree_.tree_pruned)
        cost_base = cost_loss(y_true, y_pred, cost_mat)

        for m, node in enumerate(nodes):

            # Create temporal tree by eliminating node from tree_pruned
            temp_tree = self._delete_node(self.tree_.tree_pruned, node)
            y_pred = self._classify(X, temp_tree)

            nodes_pruned = self._nodes(temp_tree)

            # Calculate %gain
            gain = (cost_base - cost_loss(y_true, y_pred, cost_mat)) / cost_base

            # Calculate %gain_size
            gain_size = (len(nodes) - len(nodes_pruned)) * 1.0 / len(nodes)

            # Calculate weighted gain
            gains[m] = gain * gain_size

        best_gain = np.max(gains)
        best_node = nodes[int(np.argmax(gains))]

        if best_gain > self.min_gain:
            self.tree_.tree_pruned = self._delete_node(self.tree_.tree_pruned, best_node)

            # If best tree is not root node, then recursively pruning the tree
            if best_node != 0:
                self._pruning(X, y_true, cost_mat)

    def pruning(self, X, y, cost_mat):
        """ Function that prune the decision tree.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        """
        self.tree_.tree_pruned = copy.deepcopy(self.tree_.tree)
        if self.tree_.n_nodes > 0:
            self._pruning(X, y, cost_mat)
            nodes_pruned = self._nodes(self.tree_.tree_pruned)
            self.tree_.n_nodes_pruned = len(nodes_pruned)