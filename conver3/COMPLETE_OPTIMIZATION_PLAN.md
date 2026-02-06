# 🎯 从F1=76%提升到95%的完整方案

## 📊 当前状态
- **当前F1**: 76%
- **目标F1**: 95%
- **差距**: 19%

## 🚀 已实施的优化

### 1. Focal Loss (losses.py)
**预期提升**: +3-5%
- 自动关注难分类样本
- 更好地处理类别不平衡
- 配合类别权重使用

### 2. 数据增强 (data_augmentation.py)
**预期提升**: +2-4%
- 随机mask
- 字符交换
- 实体替换

### 3. CRF层 (crf_model.py, enhanced_model.py)
**预期提升**: +5-8%
- BiLSTM-CRF架构
- Viterbi解码
- 更好的序列建模

### 4. 增强模型架构 (enhanced_model.py)
**预期提升**: +3-5%
- Multi-head attention
- 残差连接
- Layer normalization

## 📝 三阶段训练策略

### 阶段1: 基础优化 (Epoch 1-30)
**目标**: F1 76% → 82%

```bash
# 1. 修改config.py
use_focal_loss = True
use_data_augmentation = False
use_crf_model = False
learning_rate = 3e-5
num_epochs = 30

# 2. 运行训练
python train_optimized.py
```

**预期结果**:
- Epoch 10: F1 ~78%
- Epoch 20: F1 ~80%
- Epoch 30: F1 ~82%

### 阶段2: 数据增强 (Epoch 31-60)
**目标**: F1 82% → 87%

```bash
# 1. 修改config.py
use_focal_loss = True
use_data_augmentation = True  # 开启数据增强
aug_prob = 0.15
learning_rate = 2e-5  # 降低学习率
num_epochs = 60

# 2. 从最佳模型继续训练
python resume_training_advanced.py --checkpoint best_model.pt --epochs 60 --lr 2e-5
```

**预期结果**:
- Epoch 40: F1 ~84%
- Epoch 50: F1 ~86%
- Epoch 60: F1 ~87%

### 阶段3: CRF模型 (Epoch 61-100)
**目标**: F1 87% → 95%+

```bash
# 1. 安装CRF依赖
pip install torchcrf

# 2. 修改config.py
use_focal_loss = True
use_data_augmentation = True
use_crf_model = True  # 开启CRF
freeze_bert_layers = 6  # 冻结BERT底层加速训练
learning_rate = 1e-5  # 进一步降低学习率
num_epochs = 100

# 3. 从最佳模型继续训练
python resume_training_advanced.py --checkpoint best_model.pt --epochs 100 --lr 1e-5
```

**预期结果**:
- Epoch 70: F1 ~89%
- Epoch 80: F1 ~92%
- Epoch 90: F1 ~94%
- Epoch 100: F1 ~95%+

## 🔧 快速开始（推荐）

### 方案A: 一步到位（需要更长训练时间）

```bash
# 1. 安装依赖
pip install torchcrf

# 2. 修改config.py，启用所有优化
use_focal_loss = True
use_data_augmentation = True
use_crf_model = True
num_epochs = 100

# 3. 运行训练
python train_optimized.py
```

### 方案B: 渐进式优化（推荐，更稳定）

```bash
# 阶段1: Focal Loss (30 epochs)
python train_optimized.py  # use_focal_loss=True, others=False

# 阶段2: + 数据增强 (继续30 epochs)
# 修改config: use_data_augmentation=True
python resume_training.py

# 阶段3: + CRF (继续40 epochs)
# 修改config: use_crf_model=True
pip install torchcrf
python resume_training.py
```

## 📈 监控指标

训练时关注以下指标：

### 正常训练的标志：
- ✅ Loss持续下降
- ✅ F1每5个epoch提升1-2%
- ✅ 预测分布多样化（不只预测'O'）
- ✅ Recall和Precision同步提升

### 异常情况：
- ❌ Loss不降或震荡
- ❌ F1连续5个epoch不变
- ❌ 95%+预测'O'类
- ❌ Recall很低(<0.5)

## 🎓 进阶技巧

### 技巧1: 学习率调度
```python
# 如果F1卡住
learning_rate = learning_rate * 0.5  # 减半

# 如果训练太慢
learning_rate = learning_rate * 1.5  # 增加
```

### 技巧2: 批量大小调整
```python
# 如果显存不足
batch_size = 8
gradient_accumulation_steps = 2

# 如果显存充足
batch_size = 32
gradient_accumulation_steps = 1
```

### 技巧3: 冻结BERT层
```python
# 加速训练
freeze_bert_layers = 6  # 冻结前6层

# 更好的性能
freeze_bert_layers = 0  # 训练所有层
```

## 🆘 常见问题

### Q1: 安装torchcrf失败？
```bash
pip install torch-crf
# 或
pip install git+https://github.com/kmkurn/pytorch-crf
```

### Q2: CUDA out of memory?
```python
batch_size = 8
freeze_bert_layers = 9
```

### Q3: F1提升缓慢？
- 检查是否启用了Focal Loss
- 确认类别权重是否正确计算
- 尝试降低学习率

### Q4: 训练速度太慢？
- 冻结BERT底层: `freeze_bert_layers = 6`
- 减少数据增强: `aug_prob = 0.1`
- 暂时禁用CRF: `use_crf_model = False`

## 📚 相关资源

- [Focal Loss论文](https://arxiv.org/abs/1708.02002)
- [CRF教程](https://arxiv.org/abs/1603.01360)
- [中文NER数据集](https://github.com/CLUEbenchmark/CLUEDatasetSearch)

## 🎯 预期时间线

| 阶段 | Epochs | 时间 (RTX 4090) | F1目标 |
|------|--------|----------------|--------|
| 阶段1 | 30 | ~1.5小时 | 82% |
| 阶段2 | 30 | ~1.5小时 | 87% |
| 阶段3 | 40 | ~2小时 | 95%+ |
| **总计** | **100** | **~5小时** | **95%+** |

## ✅ 检查清单

开始训练前确认：
- [ ] 已安装torchcrf: `pip install torchcrf`
- [ ] 已修改config.py启用优化功能
- [ ] 已备份当前最佳模型
- [ ] 显存充足（至少8GB）
- [ ] 磁盘空间充足（保存checkpoint）

## 🚀 立即开始

```bash
# 最简单的方式
python train_optimized.py
```

祝训练顺利！🎉
