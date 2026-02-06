# Nested Privacy Entity Recognition System

## 项目概述

本项目实现了一个基于深度学习的嵌套隐私实体识别系统，专门用于社交网络文本中的隐私信息检测。系统采用了多项创新技术，能够有效识别复杂的嵌套实体结构。

## 核心创新点

### 1. **Span-based + Biaffine Attention 架构**
- 采用span-based方法处理嵌套实体问题
- 使用双仿射注意力机制捕获实体边界关系
- 相比传统序列标注方法，能更好地处理实体嵌套

### 2. **多尺度CNN特征融合**
- 集成3、5、7三种卷积核捕获不同粒度的局部上下文
- 与BERT特征融合，增强模型表征能力

### 3. **对抗训练 (Adversarial Training)**
- **FGM (Fast Gradient Method)**: 快速生成对抗样本
- **PGD (Projected Gradient Descent)**: 多步对抗训练，提升鲁棒性
- 提高模型对输入扰动的抵抗能力

### 4. **模型平滑技术**
- **R-Drop**: 通过KL散度约束不同dropout的输出一致性
- **EMA (Exponential Moving Average)**: 指数移动平均平滑参数
- **Label Smoothing**: 标签平滑防止过拟合

### 5. **高效推理优化**
- 针对RTX 4090优化的批处理策略
- 非极大值抑制(NMS)处理重叠实体
- 支持实时推理

## 文件结构

```
cover/
├── config.py           # 配置文件
├── model.py            # 模型架构
├── data_loader.py      # 数据加载器
├── adversarial.py      # 对抗训练和模型平滑
├── train.py            # 训练脚本
├── inference.py        # 推理和评估
├── demo.py             # 演示脚本
├── requirements.txt    # 依赖包
└── data/               # 数据集
    ├── 409_data_train.txt
    ├── 409_train_lable.txt
    ├── 409_data_test.txt
    └── 409_test_lable.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程会自动：
- 加载数据集
- 应用对抗训练(FGM/PGD)
- 使用R-Drop和EMA进行模型平滑
- 保存最佳模型到 `best_model.pt`

### 2. 评估模型

```bash
python inference.py
```

输出详细的实体级别评估指标，包括每个实体类型的Precision、Recall和F1分数。

### 3. 演示

```bash
python demo.py
```

在示例文本上测试模型效果。

## 模型配置

在 `config.py` 中可以调整以下参数：

- **模型参数**:
  - `model_name`: 预训练模型 (默认: hfl/chinese-roberta-wwm-ext)
  - `max_span_length`: 最大实体长度 (默认: 50)
  - `biaffine_size`: 双仿射注意力维度 (默认: 256)

- **训练参数**:
  - `batch_size`: 批次大小 (默认: 8，适配RTX 4090)
  - `learning_rate`: 学习率 (默认: 2e-5)
  - `num_epochs`: 训练轮数 (默认: 50)

- **对抗训练**:
  - `use_fgm`: 是否使用FGM (默认: True)
  - `use_pgd`: 是否使用PGD (默认: False)
  - `adv_epsilon`: 对抗扰动大小 (默认: 1.0)

- **模型平滑**:
  - `use_rdrop`: 是否使用R-Drop (默认: True)
  - `rdrop_alpha`: R-Drop权重 (默认: 4.0)
  - `label_smoothing`: 标签平滑系数 (默认: 0.1)

## 实体类型

系统识别以下5种隐私实体类型：

1. **BI**: 人名
2. **LOC**: 地点
3. **EDU**: 教育背景
4. **JOB**: 职位
5. **JOB_ADDS**: 工作单位（可能包含嵌套的JOB实体）

## 性能优化

### 针对RTX 4090的优化：
- 批次大小设置为8，充分利用24GB显存
- 梯度累积步数为2，等效批次大小16
- 混合精度训练支持（可选）
- 高效的span生成和过滤策略

### 速度优化：
- 使用span-based方法减少计算复杂度
- 批量推理支持
- 非极大值抑制快速去重

## 预期性能

根据模型设计和训练策略，预期性能指标：

- **F1 Score**: ≥ 95%
- **推理速度**: ~100 samples/sec (RTX 4090)
- **嵌套实体识别准确率**: 显著优于传统序列标注方法

## 技术亮点

1. **创新架构**: Span-based + Biaffine Attention处理嵌套实体
2. **鲁棒训练**: FGM/PGD对抗训练提升泛化能力
3. **模型平滑**: R-Drop + EMA + Label Smoothing多重正则化
4. **高效推理**: 针对4090优化，支持实时应用
5. **嵌套处理**: 专门设计用于识别复杂的嵌套隐私实体

## 论文相关

本代码可用于以下研究方向：
- 嵌套命名实体识别
- 隐私信息检测
- 对抗训练在NER中的应用
- 社交网络文本分析

## 注意事项

1. 首次运行会自动下载预训练模型（约400MB）
2. 建议使用CUDA 11.8+和PyTorch 2.0+
3. 训练时间约2-3小时（RTX 4090）
4. 可根据显存大小调整batch_size

## 引用

如果使用本代码，请引用相关论文和技术。

## License

本项目仅供学术研究使用。
