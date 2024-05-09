'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-05-08 23:01:16
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-05-08 23:19:23
FilePath: /HPV_test/metrics.py
Description: 
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''

import numpy as np
from sklearn import metrics


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    print(y_true.shape, y_pred.shape)
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

    sensitivity = metrics.recall_score(y_true, y_pred_label)

    specifity = metrics.recall_score(y_true, y_pred_label, pos_label=0)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_label)
    auc_score = metrics.auc(fpr, tpr)
    # 计算 F1 Score
    f1 = metrics.f1_score(y_true, y_pred_label)  # 类型: float

    ret = {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specifity': specifity,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'auc_score': auc_score,
    }

    return ret
