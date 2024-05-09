'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-05-09 10:36:21
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-05-09 11:23:43
FilePath: /HPV_test/plot.py
Description: 
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''

import matplotlib.pyplot as plt


def plot_acc_old(train_metric, val_metric):
    plt.figure()
    plt.plot(train_metric, label='Train Accuracy')
    plt.plot(val_metric, label='Validation Accuracy')
    # plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, 30)
    plt.ylim(0.6, 1)
    plt.grid(True)
    plt.savefig('acc_old.pdf')


def plot_acc_new(train_metric, val_metric):
    plt.figure()
    plt.plot(train_metric, label='Train Accuracy')
    plt.plot(val_metric, label='Validation Accuracy')
    # plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, 30)
    plt.ylim(0.6, 1)
    plt.grid(True)
    plt.savefig('acc_new.pdf')


def plot_train(train_metric_1, train_metric_2):
    plt.figure()
    plt.plot(train_metric_1, label='Train Accuracy of dataset 1')
    plt.plot(train_metric_2, label='Train Accuracy of dataset 2')
    # plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim(0, 30)
    plt.ylim(0.6, 1)
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
    plt.ylim(0.6, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig('acc_val.pdf')


if __name__ == '__main__':

    train_metrics_old = [
        0.7576809210526316,
        0.7645723684210526,
        0.7646052631578948,
        0.7688322368421052,
        0.7672779605263158,
        0.7672697368421052,
        0.7709621710526317,
        0.7693503289473683,
        0.7750657894736842,
        0.777845394736842,
        0.7851809210526316,
        0.797220394736842,
        0.8126891447368421,
        0.8342023026315789,
        0.8581085526315789,
        0.8869654605263158,
        0.9032648026315789,
        0.9226973684210527,
        0.9358799342105263,
        0.9488980263157895,
        0.9612993421052631,
        0.9649424342105263,
        0.97390625,
        0.9820805921052631,
        0.9841118421052631,
        0.9890625,
        0.9921875,
        0.99375,
        0.9971875,
        0.9984375,
    ]
    val_metrics_old = [
        0.7729196050775741,
        0.7757404795486601,
        0.765867418899859,
        0.7729196050775741,
        0.770098730606488,
        0.7757404795486601,
        0.7743300423131171,
        0.7771509167842031,
        0.7771509167842031,
        0.7757404795486601,
        0.7743300423131171,
        0.7827926657263752,
        0.7898448519040903,
        0.8152327221438646,
        0.8222849083215797,
        0.842031029619182,
        0.8758815232722144,
        0.8787023977433004,
        0.8998589562764457,
        0.9026798307475318,
        0.9012693935119888,
        0.9026798307475318,
        0.92524682651622,
        0.919605077574048,
        0.9167842031029619,
        0.92524682651622,
        0.919605077574048,
        0.9351198871650211,
        0.9337094499294781,
        0.9365303244005642,
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
    plot_acc_new(train_metrics_new, val_metrics_new)
    plot_acc_old(train_metrics_old, val_metrics_old)
    plot_train(train_metrics_old, train_metrics_new)
    plot_val(val_metrics_old, val_metrics_new)
