"""
The :mod:`costcla.metrics` module includes metrics to assess performance on cost-sensitive classification tasks given class prediction and cost-matrix

Functions named as ``*_score`` return a scalar value to maximize: the higher the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize: the lower the better
"""

from costs import cost_loss
from costs import savings_score
from costs import brier_score_loss

import numpy as np
from sklearn.utils import column_or_1d
from sklearn.metrics import roc_auc_score

__all__ = ['cost_loss',
           'savings_score',
           'brier_score_loss',
           'binary_classification_metrics']

def binary_classification_metrics(y_true, y_pred, y_prob):
    #TODO: update description
    """classification_metrics.

    This function cal...

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels, as returned by a classifier.

    y_prob : array-like
        Predicted probabilities, as returned by a classifier.

    Returns
    -------
    dict(tp, fp, fn, tn, accuracy, recall, precision, f1score, auc, brier_loss)

    Examples
    --------
    >>> from costcla.metrics import binary_classification_metrics
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.2, 0.8, 0.4, 0.3]
    >>> binary_classification_metrics(y_true, y_pred, y_prob)
    {'accuracy': 0.75,
     'auc': 0.75,
     'brier_loss': 0.13249999999999998,
     'f1score': 0.6666666666666666,
     'fn': 1.0,
     'fp': 0.0,
     'precision': 1.0,
     'recall': 0.5,
     'tn': 2.0,
     'tp': 1.0}
    """

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    y_prob = column_or_1d(y_prob)

    n_samples = y_true.shape[0]

    tp = float((y_pred * y_true).sum())
    fp = float((y_pred[np.nonzero(y_true == 0)[0]]).sum())
    fn = float((y_true[np.nonzero(y_pred == 0)[0]]).sum())
    tn = float(n_samples - tp - fn - fp)

    accuracy = (tp + tn) / n_samples
    auc = roc_auc_score(y_true, y_pred)
    brier_loss = brier_score_loss(y_true, y_prob)


    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if (recall + precision) == 0:
        f1score = 0
    else:
        f1score = 2 * (precision * recall) / (precision + recall)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=accuracy, recall=recall,
                precision=precision, f1score=f1score, auc=auc, brier_loss=brier_loss)
