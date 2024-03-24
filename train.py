import os

from deepctr_torch.models import DeepFM

from map import sparse_feature_list, dense_feature_list
from settings import parse_opts
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from log import get_logger
from utils import binary_logit_adjust_loss1, binary_focal_loss, FocalLoss


def train():
    args = parse_opts()
    data = pd.read_csv(args.train_file)
    # data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    is_simple_features = args.simple_features

    sparse_features = ['C' + str(i + 1) for i in range(args.sparse_feature_num)] if not is_simple_features else sparse_feature_list
    dense_features = ['I' + str(i + 1) for i in range(args.dense_feature_num)] if not is_simple_features else dense_feature_list

    data[sparse_features] = data[sparse_features].fillna(-1, )
    data[dense_features] = data[dense_features].fillna(0, )
    data.apply(pd.to_numeric, errors="ignore")
    data = data.sample(frac=1)
    target = ['label']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = data[feat].astype(str)
        data[feat] = lbe.fit_transform(data[feat])
        print(f"nunique of {feat} = {data[feat].nunique()}")

    if args.dense_feature_num != 0:
        mms = MinMaxScaler()
        data[dense_features] = mms.fit_transform(data[dense_features])

    # data, test = train_test_split(data, test_size=0.1)
    embedding_dim = 3
    dnn_hidden_units = (128, 64)
    use_focal_loss = True
    fixlen_feature_columns = ([SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=embedding_dim) for feat in sparse_features] +
                              [DenseFeat(feat, 1, ) for feat in dense_features])

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train_model_input = {name: data[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary', dnn_dropout=0.85, l2_reg_dnn=1e-4,
    #                l2_reg_embedding=1e-5, seed=args.manual_seed, device=device, dnn_hidden_units=dnn_hidden_units)
    model = torch.load("./models/all_features/tmp_DeepFM.h5")
    model.load_state_dict(torch.load("./models/all_features/tmp_DeepFM_weights.h5"))
    model.compile(args.optimizer, FocalLoss() if use_focal_loss else "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc", "acc", "pre", "recall", "f1"],)
    os.makedirs(args.exp_dir, exist_ok=True)
    logger = get_logger(args.exp_dir)
    logger.info('Start training ...')
    logger.info(f"Parameters: Embedding dim: {embedding_dim}, Dnn hidden units: {dnn_hidden_units}, "
                f"Optimizer: {args.optimizer}, Use focal loss: {use_focal_loss}")
    model.fit(logger, train_model_input, data[target].values, batch_size=args.batch_size, epochs=args.n_epochs,
              verbose=2, validation_split=0.15)
    # torch.save(model.state_dict(), f'models/{"simple_features" if is_simple_features else "all_features"}/new_DeepFM_weights.h5')
    # torch.save(model, f'models/{"simple_features" if is_simple_features else "all_features"}/new_DeepFM.h5')
    # pred_ans = model.predict(test_model_input, batch_size=256)
    # ret = eval_metrics(test['label'], pred_ans)
    # logger.info('auc:  %s - acc:  %s - pre:  %s - recall:  %s - f1:  %s',
    #             ret['auc'], ret['accuracy'], ret['precision'], ret['recall'], ret['f1'])

    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # filename = os.path.join('./models', f'/exp_{timestamp}_')
    # model.load_state_dict(torch.load('./models/DeepFM_weights.h5'))
    # model = torch.load('./models/DeepFM.h5')


if __name__ == '__main__':
    train()
