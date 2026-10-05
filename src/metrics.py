"""
Pure NumPy / PyTorch Metric Functions.
Zero external DLL dependencies (avoids Windows Application Control DLL blocks with scipy/sklearn).
Implements exact standard formulas for:
- accuracy_score
- precision_recall_fscore_support
- roc_curve
- auc (trapezoidal integration)
"""

import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates binary precision, recall, and F1-score.
    Returns: (precision, recall, f1, support)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    support = int(np.sum(y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, support


def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Computes Receiver Operating Characteristic (ROC) curve: False Positive Rate and True Positive Rate.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores)

    # Sort scores descending
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[y_scores_sorted[0] + 1.0, y_scores_sorted[threshold_idxs]]

    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Calculates Area Under Curve using Trapezoidal rule.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    
    # Sort by fpr
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    # Trapezoid rule
    return float(np.sum((fpr_sorted[1:] - fpr_sorted[:-1]) * (tpr_sorted[1:] + tpr_sorted[:-1]) / 2.0))
