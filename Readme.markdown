# HPV类型判断

### 环境依赖
1. pytorch
2. deepctr-torch@latest
3. <i>SHAP</i>

### 目录结构描述
```
|   birth231206.csv
|   dealt_train_data.csv
|   explain.py      // 用于解释模型的脚本
|   extract.py      // 处理log输出的脚本
|   hpv231229.csv
|   hpv231229.xlsx
|   log.py
|   map.py          // 每一列的映射关系
|   metrics.py
|   modified_row_data.csv
|   Readme.markdown
|   row_data.csv
|   settings.py     // 模型设置
|   shuffled_hpv231229.csv
|   shuffle_dealt_train_data.csv
|   train.py        // 模型训练入口
|   utils.py        // 工具类存放文件
|   数据说明-HPV流行病学分析用.docx
|           
+---extra
|       basemodel.py  //deepctr-torch的basemodel.py有bug 若运行失败 用此文件替换
|       
+---models
|   +---all_features
|   |       DeepFM.h5   // 旧数据下的模型
|   |       DeepFM_weights.h5
|   |       new_DeepFM.h5   // 新数据下的模型
|   |       new_DeepFM_weights.h5
|   |       tmp_DeepFM.h5   // 临时文件 无用
|   |       tmp_DeepFM_weights.h5
|   |       
|   +---lowversion_all_features
|   |       new_DeepFM.h5   // pytorch1.2版本下的存储模型 无用
|   |       new_DeepFM_weights.h5
|   |       
|   \---simple_features
|           DeepFM.h5       // 使用少特征种类下的模型 无用
|           DeepFM_weights.h5
```

### 使用
1. train.py下执行模型的训练
2. explain.py下执行模型的权重输出