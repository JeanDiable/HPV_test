'''
Configs for training & testing
'''

import argparse
import time


def parse_opts():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sparse_feature_num',
        default=47,
        type=int,
        help='Number of sparse features')
    parser.add_argument(
        '--dense_feature_num',
        default=0,
        type=int,
        help='Number of dense features')
    parser.add_argument(
        '--train_file',
        default='modified_row_data.csv',
        type=str,
        help='Training data.')
    parser.add_argument(
        '--val_file',
        default='test_data/val_birth231114.csv',
        type=str,
        help='Validation data.')
    parser.add_argument(
        '--exp_dir',
        default=f'./exp_{timestamp}',
        type=str,
        help='Experiment output directory.')
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--ratio',
        default=0.80,
        type=float,
        help='The ratio of train data')

    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--simple_features', default=False, type=bool, help='use simple features or not')
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs', default=20, type=int, help='Number of total epochs to run')
    parser.add_argument(
        '--optimizer', default='adam', type=str, help='Optimizer.')

    parser.add_argument(
        '--test_intervals',
        default=5,
        type=int,
        help='Iteration for testing model')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.')
    parser.add_argument(
        '--pretrain_path',
        default='',
        type=str,
        help=
        'Path for pretrained model.')

    parser.add_argument(
        '--gpu_id', default=0, type=int, help='Single GPU id')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
