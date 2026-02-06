# 🚀 高级优化功能使用指南

## 已添加的优化功能

### 1. **Focal Loss** (losses.py)
- 更好地处理类别不平衡
- 自动关注难分类样本
- 配合类别权重使用效果更佳

### 2. **数据增强** (data_augmentation.py)
- 随机mask非实体字符
- 随机交换相邻字符
- 实体替换增强
- 可扩展到更多策略

### 3. **CRF层** (crf_model.py + enhanced_model.py)
- BiLSTM-CRF架构
- Viterbi解码
- 显著提升序列标注准确率

### 4. **增强模型** (enhanced_model.py)
- Multi-head self-attention
- 残差连接
- Layer normalization
- 可选的BERT层冻结

## 🎯 快速开始

### 方法1：使用增强模型（推荐）

```bash
# 1. 安装依赖
pip install torchcrf

# 2. 运行增强版训练
python train_enhanced.py
```

### 方法2：渐进式优化

**Step 1: 先使用Focal Loss**
```python
# 在config.py中设置
use_focal_loss = True
focal_gamma = 2.0
```

**Step 2: 添加数据增强**
```python
# 在config.py中设置
use_data_augmentation = True
aug_prob = 0.15
```

**Step 3: 使用CRF模型**
```python
# 在config.py中设置
use_crf_model = True
```

## 📊 预期性能提升

| 优化方法 | 预期F1提升 | 说明 |
|---------|-----------|------|
| Focal Loss | +3-5% | 更好的类别平衡 |
| 数据增强 | +2-4% | 增加训练样本多样性 |
| CRF层 | +5-8% | 更好的序列建模 |
| 组合使用 | +10-15% | 协同效果 |

**当前F1: 76%**
**预期F1: 86-91%** (使用所有优化)

## ⚙️ 配置说明

在`config.py`中添加以下配置：

```python
# Advanced features
use_focal_loss = True          # 使用Focal Loss
focal_gamma = 2.0              # Focal Loss参数
focal_alpha = None             # 自动使用类别权重

use_data_augmentation = True   # 使用数据增强
aug_prob = 0.15                # 增强概率

use_crf_model = True           # 使用CRF模型
freeze_bert_layers = 6         # 冻结BERT前6层（加速训练）

# BiLSTM-CRF settings
lstm_hidden_size = 512
lstm_num_layers = 2
lstm_dropout = 0.1
```

## 🔧 故障排除

### 问题1: ImportError: No module named 'torchcrf'
```bash
pip install torchcrf
```

### 问题2: CUDA out of memory
```python
# 减小batch size
batch_size = 8

# 或冻结更多BERT层
freeze_bert_layers = 9
```

### 问题3: 训练速度慢
```python
# 冻结BERT底层
freeze_bert_layers = 6

# 减少数据增强
use_data_augmentation = False
```

## 📈 训练建议

1. **第一阶段（Epoch 1-20）**
   - 使用Focal Loss + 类别权重
   - 学习率: 3e-5
   - 预期F1: 0.4 → 0.7

2. **第二阶段（Epoch 21-40）**
   - 添加数据增强
   - 学习率: 2e-5
   - 预期F1: 0.7 → 0.8

3. **第三阶段（Epoch 41-60）**
   - 切换到CRF模型
   - 学习率: 1e-5
   - 预期F1: 0.8 → 0.9+

## 🎓 进阶技巧

### 技巧1: 两阶段训练
```bash
# 阶段1: 快速训练基础模型
python train.py --epochs 30

# 阶段2: 使用CRF fine-tune
python train_enhanced.py --checkpoint best_model.pt --epochs 60
```

### 技巧2: 学习率调整
```python
# 如果F1卡住不动
learning_rate = 1e-5  # 降低学习率

# 如果训练太慢
learning_rate = 5e-5  # 提高学习率
```

### 技巧3: 数据增强强度
```python
# 保守增强（稳定）
aug_prob = 0.1

# 激进增强（多样性）
aug_prob = 0.3
```

## 下一步

创建`train_enhanced.py`来使用所有这些优化功能。
