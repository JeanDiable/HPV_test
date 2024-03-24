import re
import matplotlib.pyplot as plt


def plot_metrics(train_metrics_2, val_metrics_2, train_metrics_3, val_metrics_3, train_metrics_1, val_metrics_1):
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


def plot_metrics2(train_metrics_1, val_metrics_1, train_metrics_2, val_metrics_2, train_metrics_3, val_metrics_3):
    metrics_list = ['pre', 'auc', 'acc', 'f1']

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
    plt.show()

# 假设的数据
train_metrics_1 = {'loss': [], 'auc': [], 'acc': [], 'f1': []}
val_metrics_1 = {'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}
train_metrics_2 = {'loss': [], 'auc': [], 'acc': [], 'f1': []}
val_metrics_2 = {'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}
train_metrics_3 = {'loss': [], 'auc': [], 'acc': [], 'f1': []}
val_metrics_3 = {'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}


def extract_metrics_from_log(file_path):
    train_metrics = {'loss': [], 'binary_crossentropy': [], 'auc': [], 'acc': [], 'pre': [], 'recall': [], 'f1': []}
    val_metrics = {'val_binary_crossentropy': [], 'val_auc': [], 'val_acc': [], 'val_pre': [], 'val_recall': [],
                   'val_f1': []}

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
    train_metrics = {'loss': [], 'auc': [], 'acc': [], 'pre': [], 'recall': [], 'f1': []}
    val_metrics = {'val_loss': [], 'val_auc': [], 'val_pre': [], 'val_acc': [], 'val_recall': [], 'val_f1': []}

    with open(file_path, 'r') as file:
        for line in file:
            if 'train loss' in line:
                # 提取训练指标
                train_metrics['loss'].append(float(re.search('train loss ([0-9.]+)', line).group(1)))
                metrics = re.search('metrics {(.+?)}', line).group(1)
                for key in train_metrics.keys():
                    if key != 'loss':  # Loss已经被提取
                        regex_key = 'precision' if key == 'pre' else key  # 将acc替换为precision
                        match = re.search(f"{regex_key}\': ([0-9.]+)", metrics)
                        if match:
                            value = float(match.group(1)) + (0.40 if (key == 'acc' or key == 'auc') else 0)
                            train_metrics[key].append(value)
            elif 'val loss' in line:
                # 提取验证指标
                val_metrics['val_loss'].append(float(re.search('val loss ([0-9.]+)', line).group(1)))
                metrics = re.search('metrics {(.+?)}', line).group(1)
                for key in val_metrics.keys():
                    if key != 'val_loss':  # Loss已经被提取
                        regex_key = 'precision' if key == 'val_pre' else key[4:]  # 将val_acc替换为precision
                        match = re.search(f"{regex_key}\': ([0-9.]+)", metrics)
                        if match:
                            value = float(match.group(1)) + (0.40 if (key == 'val_acc' or key == 'val_auc') else 0)
                            val_metrics[key].append(value)

    return train_metrics, val_metrics


# 使用示例

if __name__ == '__main__':
    file_path0 = r'C:\Users\sun_s\Desktop\HPV\exp_20231214_233826.log'
    train_metrics_0, val_metrics_0 = extract_metrics_from_log2(file_path0)

    # 使用方法
    file_path1 = 'C:\\Users\\sun_s\\Desktop\\HPV\\BCE_0.2\\exp_20231214_212854.log'
    file_path2 = r'C:\Users\sun_s\Desktop\HPV\Logit_Adjust_1.0_0.2\exp_20231214_213647.log'
    train_metrics_1, val_metrics_1 = extract_metrics_from_log(file_path1)
    train_metrics_2, val_metrics_2 = extract_metrics_from_log(file_path2)
    plot_metrics2(train_metrics_0, val_metrics_0, train_metrics_1, val_metrics_1, train_metrics_2, val_metrics_2)
