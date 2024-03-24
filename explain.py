import torch
import shap
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def PrintWeight(m):
    for name, param in m.state_dict().items():
        print(name, param)


# 输出模型每一层layer的名字和权重
if __name__ == '__main__':
    model = torch.load("./models/all_features/new_DeepFM.h5")
    model.load_state_dict(torch.load("./models/all_features/new_DeepFM_weights.h5"))
    PrintWeight(model)

# 使用SHAP解释模型 有问题 issue: https://github.com/shap/shap/issues/3466
# if __name__ == '__main__':
#     model = torch.load("./models/all_features/new_DeepFM.h5")
#     model.load_state_dict(torch.load("./models/all_features/new_DeepFM_weights.h5"))
#
#     # torch.save(model.state_dict(), 'models/lowversion_all_features/new_DeepFM_weights.h5', _use_new_zipfile_serialization=False)
#     # torch.save(model, 'models/lowversion_all_features/new_DeepFM.h5', _use_new_zipfile_serialization=False)
#     model.to("cuda:0" if torch.cuda.is_available() else "cpu")
#     data = pd.read_csv("modified_row_data.csv")
#     sparse_features = ['C' + str(i + 1) for i in range(47)]
#     data[sparse_features] = data[sparse_features].fillna(-1, )
#     data.apply(pd.to_numeric, errors="ignore")
#
#     for feat in sparse_features:
#         lbe = LabelEncoder()
#         data[feat] = data[feat].astype(str)
#         data[feat] = lbe.fit_transform(data[feat])
#
#     sample_data = data.sample(frac=0.01)
#     sample_data = sample_data[sparse_features]
#     tensor = torch.tensor(sample_data.to_numpy(), dtype=torch.float32).to("cuda:0" if torch.cuda.is_available() else "cpu")
#     # output = model(tensor)
#     # print(output)
#     explainer = shap.DeepExplainer(model, tensor)
#     shap_values = explainer.shap_values(tensor)
#
#     # 可视化第一个样本的 SHAP 值
#     shap.force_plot(explainer.expected_value, shap_values[0], sample_data.iloc[0])
#
#     # 总结图，展示所有特征的重要性
#     shap.summary_plot(shap_values, sample_data)
