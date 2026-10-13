from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def choose_threshold(y_true, probabilities):
    candidates = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])

def binary_metrics(y_true, probabilities, threshold=0.5):
    y_true, probabilities = np.asarray(y_true).astype(int), np.asarray(probabilities)
    pred = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"support": int(len(y_true)), "positive": int(y_true.sum()), "threshold": float(threshold),
            "auroc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
            "average_precision": float(average_precision_score(y_true, probabilities)) if y_true.sum() else None,
            "precision": float(precision_score(y_true, pred, zero_division=0)), "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)), "sensitivity": float(tp/(tp+fn)) if tp+fn else None,
            "specificity": float(tn/(tn+fp)) if tn+fp else None, "confusion_matrix": [[int(tn),int(fp)],[int(fn),int(tp)]]}
