# 训练优化总结

## 已修复的问题

### 1. NaN Loss 问题
**原因分析：**
- R-Drop的alpha值过大（4.0）导致KL散度爆炸
- Label smoothing在某些情况下导致数值不稳定
- 学习率过高（2e-5）
- 缺少梯度和损失的NaN检测

**解决方案：**
- ✅ 降低R-Drop alpha: 4.0 → 0.5
- ✅ 禁用Label smoothing: 0.1 → 0.0
- ✅ 降低学习率: 2e-5 → 1e-5
- ✅ 添加NaN检测和跳过机制
- ✅ 改进KL loss计算的数值稳定性
- ✅ 添加权重初始化（xavier uniform with gain=0.1）

### 2. 针对RTX 4090的优化
**配置调整：**
- ✅ 增加batch size: 8 → 16
- ✅ 减少gradient accumulation: 2 → 1
- ✅ 保持dropout率合理（0.2-0.3）

### 3. 实体类型问题
- ✅ 添加所有9个实体类型：BI, CNU, EDU, JOB, JOB_ADDS, LOC, MS, PC, PI

### 4. DataLoader问题
- ✅ 添加自定义collate_fn处理变长数据

### 5. Adversarial Training
- ⚠️ 暂时禁用FGM/PGD（存在梯度计算图问题）
- 可以在模型稳定后重新启用

## 当前配置（config.py）

```python
# 训练设置
batch_size = 16              # 针对RTX 4090优化
learning_rate = 1e-5         # 降低以提高稳定性
gradient_accumulation_steps = 1
max_grad_norm = 1.0

# R-Drop
use_rdrop = True
rdrop_alpha = 0.5            # 从4.0降低

# Label smoothing
label_smoothing = 0.0        # 禁用以防止NaN

# Adversarial training
use_fgm = False              # 暂时禁用
use_pgd = False
```

## 训练监控

代码现在会：
1. 检测并跳过产生NaN的batch
2. 检测并跳过有NaN梯度的更新
3. 在控制台输出警告信息
4. 使用梯度裁剪防止梯度爆炸

## 预期性能

**第一个epoch的预期指标：**
- Loss: 应该从2-3开始，逐渐下降
- F1: 初始可能较低（0.1-0.3），会逐步提升
- 训练速度: ~4-5 it/s（RTX 4090）

**如果仍然出现NaN：**
1. 进一步降低学习率到 5e-6
2. 暂时禁用R-Drop（设置 use_rdrop = False）
3. 减小batch size到8

## 下一步优化建议

1. **模型稳定后可以尝试：**
   - 逐步增加R-Drop alpha（0.5 → 1.0 → 2.0）
   - 重新启用Label smoothing（0.05）
   - 修复并启用FGM

2. **性能优化：**
   - 使用混合精度训练（torch.cuda.amp）
   - 调整max_span_length减少计算量
   - 使用更大的batch size

3. **模型改进：**
   - 尝试不同的BERT模型
   - 调整CNN kernel sizes
   - 实验不同的span representation方法

## 运行命令

```bash
# 开始训练
python train.py

# 如果需要调试
python test_fgm.py  # 测试FGM问题
python debug_embeddings.py  # 检查embedding参数
```
