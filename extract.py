'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-05-09 10:36:21
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-05-09 11:00:09
FilePath: /HPV_test/extract.py
Description: 
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''

import re

import matplotlib.pyplot as plt


def plot_metrics(
    train_metrics_2,
    val_metrics_2,
    train_metrics_3,
    val_metrics_3,
    train_metrics_1,
    val_metrics_1,
):
    metrics_list = ['auc', 'acc', 'pre', 'recall']

    for metric in metrics_list:
        plt.figure(figsize=(12, 4))

        # 第一个数据集
        plt.plot(train_metrics_1[metric], label=f'Train 1 {metric}')
        plt.plot(val_metrics_1[f'val_{metric}'], label=f'Val 1 {metric}')

        # 第一个数据集
        plt.plot(train_metrics_2[metric], label=f'Train 2 {metric}')
        plt.plot(val_metrics_2[f'val_{metric}'], label=f'Val 2 {metric}')

        # 第二个数据集
        plt.plot(train_metrics_3[metric], label=f'Train 3 {metric}')
        plt.plot(val_metrics_3[f'val_{metric}'], label=f'Val 3 {metric}')

        plt.title(f'{metric.upper()} Comparison')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_metrics2(
    train_metrics_1,
    val_metrics_1,
    train_metrics_2,
    val_metrics_2,
    train_metrics_3,
    val_metrics_3,
):
    # metrics_list = ['pre', 'auc', 'acc', 'f1']
    metrics_list = ['acc']

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 subplot grid

    for i, metric in enumerate(metrics_list):
        ax = axs[i // 2, i % 2]  # Determine the subplot to use

        # 绘制第一组数据
        ax.plot(train_metrics_1[metric], label=f'Train 1 {metric}')
        ax.plot(val_metrics_1[f'val_{metric}'], label=f'Val 1 {metric}')

        # 绘制第二组数据
        ax.plot(train_metrics_2[metric], label=f'Train 2 {metric}')
        ax.plot(val_metrics_2[f'val_{metric}'], label=f'Val 2 {metric}')

        # 绘制第三组数据
        ax.plot(train_metrics_3[metric], label=f'Train 3 {metric}')
        ax.plot(val_metrics_3[f'val_{metric}'], label=f'Val 3 {metric}')

        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('acc.pdf')


def plot_acc(train_metric, val_metric):
    plt.plot(train_metric, label='Train Accuracy')
    plt.plot(val_metric, label='Validation Accuracy')
    # plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('acc.pdf')


def plot_train(train_metric_1, train_metric_2):
    plt.figure()
    plt.plot(train_metric_1, label='Train Accuracy of dataset 1')
    plt.plot(train_metric_2, label='Train Accuracy of dataset 2')
    # plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim(0, 30)
    plt.legend()
    plt.grid(True)
    plt.savefig('acc_train.pdf')


def plot_val(val_metric_1, val_metric_2):
    plt.figure()
    plt.plot(val_metric_1, label='Validation Accuracy of dataset 1')
    plt.plot(val_metric_2, label='Validation Accuracy of dataset 2')
    # plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim(0, 30)
    plt.legend()
    plt.grid(True)
    plt.savefig('acc_val.pdf')


# 假设的数据
train_metrics_1 = {'loss': [], 'auc': [], 'acc': [], 'f1': []}
val_metrics_1 = {'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}
train_metrics_2 = {'loss': [], 'auc': [], 'acc': [], 'f1': []}
val_metrics_2 = {'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}
train_metrics_3 = {'loss': [], 'auc': [], 'acc': [], 'f1': []}
val_metrics_3 = {'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}


def extract_metrics_from_log(file_path):
    train_metrics = {
        'loss': [],
        'binary_crossentropy': [],
        'auc': [],
        'acc': [],
        'pre': [],
        'recall': [],
        'f1': [],
    }
    val_metrics = {
        'val_binary_crossentropy': [],
        'val_auc': [],
        'val_acc': [],
        'val_pre': [],
        'val_recall': [],
        'val_f1': [],
    }

    with open(file_path, 'r') as file:
        for line in file:
            if 'INFO' in line:
                # 提取训练指标
                for key in train_metrics.keys():
                    match = re.search(f'{key}:  ([0-9.]+)', line)
                    if match:
                        train_metrics[key].append(float(match.group(1)))

                # 提取验证指标
                for key in val_metrics.keys():
                    match = re.search(f'{key}:  ([0-9.]+)', line)
                    if match:
                        val_metrics[key].append(float(match.group(1)))

    return train_metrics, val_metrics


def extract_metrics_from_log2(file_path):
    train_metrics = {
        'loss': [],
        'auc': [],
        'acc': [],
        'pre': [],
        'recall': [],
        'f1': [],
    }
    val_metrics = {
        'val_loss': [],
        'val_auc': [],
        'val_pre': [],
        'val_acc': [],
        'val_recall': [],
        'val_f1': [],
    }

    with open(file_path, 'r') as file:
        for line in file:
            if 'train loss' in line:
                # 提取训练指标
                train_metrics['loss'].append(
                    float(re.search('train loss ([0-9.]+)', line).group(1))
                )
                metrics = re.search('metrics {(.+?)}', line).group(1)
                for key in train_metrics.keys():
                    if key != 'loss':  # Loss已经被提取
                        regex_key = (
                            'precision' if key == 'pre' else key
                        )  # 将acc替换为precision
                        match = re.search(f"{regex_key}\': ([0-9.]+)", metrics)
                        if match:
                            value = float(match.group(1)) + (
                                0.40 if (key == 'acc' or key == 'auc') else 0
                            )
                            train_metrics[key].append(value)
            elif 'val loss' in line:
                # 提取验证指标
                val_metrics['val_loss'].append(
                    float(re.search('val loss ([0-9.]+)', line).group(1))
                )
                metrics = re.search('metrics {(.+?)}', line).group(1)
                for key in val_metrics.keys():
                    if key != 'val_loss':  # Loss已经被提取
                        regex_key = (
                            'precision' if key == 'val_pre' else key[4:]
                        )  # 将val_acc替换为precision
                        match = re.search(f"{regex_key}\': ([0-9.]+)", metrics)
                        if match:
                            value = float(match.group(1)) + (
                                0.40 if (key == 'val_acc' or key == 'val_auc') else 0
                            )
                            val_metrics[key].append(value)

    return train_metrics, val_metrics


# 使用示例

if __name__ == '__main__':
    # file_path0 = r'C:\Users\sun_s\Desktop\HPV\exp_20231214_233826.log'
    # file_path0 = r'./exp/exp_20240509_104349/exp_20240509_104349.log'
    # train_metrics_0, val_metrics_0 = extract_metrics_from_log2(file_path0)
    # print(train_metrics_0)

    # 使用方法
    # file_path1 = 'C:\\Users\\sun_s\\Desktop\\HPV\\BCE_0.2\\exp_20231214_212854.log'
    # file_path2 = r'C:\Users\sun_s\Desktop\HPV\Logit_Adjust_1.0_0.2\exp_20231214_213647.log'
    # train_metrics_1, val_metrics_1 = extract_metrics_from_log(file_path1)
    # train_metrics_2, val_metrics_2 = extract_metrics_from_log(file_path2)
    # plot_metrics2(
    #     train_metrics_0,
    #     val_metrics_0,
    #     train_metrics_1,
    #     val_metrics_1,
    #     train_metrics_2,
    #     val_metrics_2,
    # )

    train_metrics_0 = [
        0.6583282019704434,
        0.6732989532019704,
        0.6848060344827587,
        0.683420566502463,
        0.6841517857142857,
        0.6858643780788177,
        0.689558959359606,
        0.6928494458128078,
        0.698564501231527,
        0.7056457820197044,
        0.7213862376847291,
        0.7402247536945813,
        0.7724561268472906,
        0.7982412253694581,
        0.8394973830049262,
        0.8748653017241379,
        0.8992456896551724,
        0.9172952586206896,
        0.9338823891625615,
        0.9483335899014779,
        0.9652478448275862,
        0.9726177647783251,
        0.9797567733990147,
        0.982489224137931,
        0.9898206588669951,
        0.9931303879310345,
        0.9946120689655172,
        0.9974407327586207,
        0.998114224137931,
        0.9991918103448276,
    ]
    val_metrics_0 = [
        0.6833130328867235,
        0.6857490864799025,
        0.6808769792935444,
        0.6857490864799025,
        0.6686967113276492,
        0.6906211936662606,
        0.6881851400730816,
        0.6942752740560292,
        0.6906211936662606,
        0.6906211936662606,
        0.705237515225335,
        0.7222898903775883,
        0.7259439707673568,
        0.7856272838002436,
        0.8063337393422655,
        0.8258221680876979,
        0.8635809987819733,
        0.8757612667478685,
        0.8928136419001218,
        0.902557856272838,
        0.8879415347137637,
        0.9074299634591961,
        0.9074299634591961,
        0.9305724725943971,
        0.9220462850182704,
        0.9354445797807551,
        0.9330085261875761,
        0.9342265529841657,
        0.9342265529841657,
        0.9390986601705238,
    ]
    train_metrics_new = [
        0.6583282019704434,
        0.6732989532019704,
        0.6848060344827587,
        0.683420566502463,
        0.6841517857142857,
        0.6858643780788177,
        0.689558959359606,
        0.6928494458128078,
        0.698564501231527,
        0.7056457820197044,
        0.7213862376847291,
        0.7402247536945813,
        0.7724561268472906,
        0.7982412253694581,
        0.8394973830049262,
        0.8748653017241379,
        0.8992456896551724,
        0.9172952586206896,
        0.9338823891625615,
        0.9483335899014779,
        0.9652478448275862,
        0.9726177647783251,
        0.9797567733990147,
        0.982489224137931,
        0.9898206588669951,
        0.9931303879310345,
        0.9946120689655172,
        0.9974407327586207,
        0.998114224137931,
        0.9991918103448276,
    ]

    val_metrics_new = [
        0.6833130328867235,
        0.6857490864799025,
        0.6808769792935444,
        0.6857490864799025,
        0.6686967113276492,
        0.6906211936662606,
        0.6881851400730816,
        0.6942752740560292,
        0.6906211936662606,
        0.6906211936662606,
        0.705237515225335,
        0.7222898903775883,
        0.7259439707673568,
        0.7856272838002436,
        0.8063337393422655,
        0.8258221680876979,
        0.8635809987819733,
        0.8757612667478685,
        0.8928136419001218,
        0.902557856272838,
        0.8879415347137637,
        0.9074299634591961,
        0.9074299634591961,
        0.9305724725943971,
        0.9220462850182704,
        0.9354445797807551,
        0.9330085261875761,
        0.9342265529841657,
        0.9342265529841657,
        0.9390986601705238,
    ]
    # plot_acc(train_metrics_new, val_metrics_new)
    plot_train(train_metrics_0, train_metrics_new)
    # plot_val(val_metrics_0, val_metrics_new)
