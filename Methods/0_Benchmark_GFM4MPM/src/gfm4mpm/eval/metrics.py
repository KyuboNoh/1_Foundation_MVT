# src/gfm4mpm/eval/metrics.py
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, average_precision_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'F1': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUPRC': average_precision_score(y_true, y_prob),
        'AUROC': roc_auc_score(y_true, y_prob),
        'B.ACC': balanced_accuracy_score(y_true, y_pred),
        'ACC': (y_true==y_pred).mean(),
    }
