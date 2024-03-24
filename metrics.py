from sklearn import metrics
import numpy as np


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int32)
    # 计算 AUC
    auc = metrics.roc_auc_score(y_true, y_pred)

    # 预测标签 (类型: numpy.ndarray)
    # 基于概率生成二元预测标签（通常使用 0.5 作为阈值）
    y_pred_label = np.array([0 if score < 0.5 else 1 for score in y_pred])

    # 计算 accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred_label)

    # 计算 Precision
    precision = metrics.precision_score(y_true, y_pred_label)  # 类型: float

    # 计算 Recall
    recall = metrics.recall_score(y_true, y_pred_label)  # 类型: float

    # 计算 F1 Score
    f1 = metrics.f1_score(y_true, y_pred_label)  # 类型: float

    ret = {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return ret
