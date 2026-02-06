# 断点续训使用指南

## 📋 概述

训练过程中会自动保存最佳模型：
- **best_model.pt** - 最佳F1分数的模型（自动保存）

## 🚀 使用方法

### 方法1：简单恢复（推荐）

从最佳模型继续训练：

```bash
python resume_training.py
```

### 方法2：高级恢复（灵活配置）

从最佳模型恢复，并可以修改参数：

```bash
# 从最佳模型恢复
python resume_training_advanced.py --checkpoint best_model.pt

# 恢复并修改训练参数
python resume_training_advanced.py --checkpoint best_model.pt --epochs 100 --lr 2e-5

# 查看所有可用选项
python resume_training_advanced.py --help
```

### 方法3：修改config.py

直接在config.py中设置：

```python
# 在config.py中修改
resume_from_checkpoint = 'best_model.pt'
```

然后正常运行：
```bash
python train.py
```

## 📊 Checkpoint内容

checkpoint包含：
- `epoch`: 训练到的epoch数
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 学习率调度器状态
- `ema_state_dict`: EMA模型状态
- `f1`: 当前F1分数
- `precision`: 精确率
- `recall`: 召回率

## 🔄 恢复训练流程

1. **自动检测checkpoint**
   - 程序会检查指定的checkpoint文件是否存在
   - 如果存在，自动加载所有训练状态

2. **恢复训练状态**
   - 模型权重
   - 优化器状态（包括momentum等）
   - 学习率调度器状态
   - 当前epoch和最佳F1

3. **继续训练**
   - 从下一个epoch开始
   - 保持之前的最佳F1记录
   - 继续保存更好的模型

## 📝 使用场景

### 场景1：训练中断后恢复

```bash
# 训练意外中断，从最佳模型恢复
python resume_training.py
```

### 场景2：从最佳模型继续训练更多epoch

```bash
# 当前训练了50个epoch，想继续训练到100个epoch
python resume_training_advanced.py --checkpoint best_model.pt --epochs 100
```

### 场景3：调整学习率继续训练

```bash
# F1卡在0.9，降低学习率继续fine-tune
python resume_training_advanced.py --checkpoint best_model.pt --lr 1e-5 --epochs 80
```

## ⚠️ 注意事项

### 1. Checkpoint兼容性
- 确保使用相同的模型架构
- 如果修改了`entity_types`，需要重新训练
- 如果修改了模型结构（如hidden_size），无法恢复

### 2. 学习率调整
- 恢复训练时会继续使用原来的学习率调度
- 如果想改变学习率，使用`--lr`参数
- 建议恢复训练时使用较小的学习率

### 3. Epoch计数
- 恢复后epoch会从checkpoint的下一个开始
- 例如：从epoch 25恢复，会显示"Epoch 26/50"

### 4. 最佳模型追踪
- 程序会记住历史最佳F1
- 只有超过历史最佳F1才会保存新的best_model.pt

## 🎯 实际示例

### 示例1：50个epoch后继续训练

```bash
# 初始训练
python train.py  # 训练50个epoch，最佳F1=0.92

# 继续训练到100个epoch
python resume_training_advanced.py --checkpoint best_model.pt --epochs 100
```

输出：
```
Loading checkpoint from best_model.pt...
Resumed from epoch 49, best F1: 0.9200
Will continue training from epoch 50

Epoch 50/100
Train Loss: 0.1234
F1: 0.9250, Precision: 0.9300, Recall: 0.9200
✓ Saved best model with F1: 0.9250
```

### 示例2：Fine-tuning

```bash
# 第一阶段：正常训练
python train.py  # 50 epochs, F1=0.93

# 第二阶段：降低学习率fine-tune
python resume_training_advanced.py \
    --checkpoint best_model.pt \
    --epochs 80 \
    --lr 1e-5
```

## 🔍 检查Checkpoint信息

创建一个简单脚本查看checkpoint内容：

```python
import torch

checkpoint = torch.load('best_model.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"F1: {checkpoint['f1']:.4f}")
print(f"Precision: {checkpoint['precision']:.4f}")
print(f"Recall: {checkpoint['recall']:.4f}")
```

## 💡 最佳实践

1. **定期备份最佳模型**
   - 复制best_model.pt到安全位置
   - 避免意外覆盖

2. **监控训练曲线**
   - 如果F1不再提升，考虑降低学习率继续训练
   - 如果出现过拟合，停止训练

3. **实验不同配置**
   - 可以从同一个checkpoint尝试不同的学习率
   - 比较哪个配置效果更好

## 🆘 常见问题

### Q: 恢复训练后F1反而下降？
A: 可能是学习率过高，尝试使用`--lr 1e-5`降低学习率

### Q: 找不到checkpoint文件？
A: 运行`ls *.pt`查看所有checkpoint，确保文件名正确

### Q: 想从头开始训练？
A: 删除或重命名best_model.pt，或者不设置`resume_from_checkpoint`

### Q: 可以修改batch_size吗？
A: 可以，使用`--batch_size`参数，但可能影响训练稳定性

### Q: 恢复训练会覆盖原checkpoint吗？
A: 不会，只会在F1更好时保存新的best_model.pt
