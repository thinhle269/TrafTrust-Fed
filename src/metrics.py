
import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from .data_utils import speed_to_class

def regression_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return mae, rmse

def classification_metrics(y_true, y_pred):
    y_true_cls = np.array([speed_to_class(v) for v in y_true])
    y_pred_cls = np.array([speed_to_class(v) for v in y_pred])
    acc = float((y_true_cls == y_pred_cls).mean())
    macro_f1 = float(f1_score(y_true_cls, y_pred_cls, average="macro"))
    recall   = float(recall_score(y_true_cls, y_pred_cls, average="macro"))
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0,1,2,3,4])
    return acc, macro_f1, recall, cm
