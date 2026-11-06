import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def comp_sensitivity(true_labels, pred_labels):
    true_positive = sum(1 for pred, true in zip(pred_labels, true_labels) if pred is True and true is True)
    actual_positive = sum(1 for label in true_labels if label is True)
    return 0.0 if actual_positive == 0 else true_positive / actual_positive


def comp_specificity(true_labels, pred_labels):
    true_negatives = sum(1 for pred, true in zip(pred_labels, true_labels) if pred is False and true is False)
    actual_negatives = sum(1 for label in true_labels if label is False)
    return 0.0 if actual_negatives == 0 else true_negatives / actual_negatives


def comp_multi_auc(y_true, y_score, n_classes):
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fpr = {}
    tpr = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], [row[1] for row in y_score])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    return auc(all_fpr, mean_tpr)
