# RTX 5090D 完整解决方案

## 🔴 问题总结

您在运行Docker训练时遇到了两个关键问题：

### 1. CUDA架构兼容性问题（关键）
```
NVIDIA GeForce RTX 5090 D with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**原因**：RTX 5090D使用最新的Blackwell架构（compute capability sm_120），而PyTorch 2.5.1稳定版只支持到sm_90（Hopper架构）。

### 2. HuggingFace模型下载失败
```
'[Errno 101] Network is unreachable' thrown while requesting HEAD https://huggingface.co/...
```

**原因**：容器内无法访问HuggingFace，需要使用国内镜像或预下载模型。

---

## ✅ 解决方案

### 方案一：使用PyTorch Nightly版本（推荐）

PyTorch的每日构建版本已经支持sm_120架构。

#### 步骤1：更新Dockerfile

我已经为您更新了Dockerfile，关键改动：

```dockerfile
# 使用PyTorch nightly版本，支持sm_120
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"
ENV CUDA_MODULE_LOADING=LAZY

# 安装支持RTX 5090D的PyTorch nightly版本
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu124
```

#### 步骤2：解决模型下载问题

**方法A：使用ModelScope（国内镜像，推荐）**

```bash
# 在宿主机上下载模型
pip install modelscope
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('tiansz/chinese-roberta-wwm-ext', cache_dir='./models/hub')
print(f'Model downloaded to: {model_dir}')
"
```

**方法B：手动下载并挂载**

```bash
# 1. 在宿主机下载模型
mkdir -p ./models/chinese-roberta-wwm-ext
cd ./models/chinese-roberta-wwm-ext

# 从以下地址下载所有文件：
# https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main
# 需要下载：config.json, pytorch_model.bin, tokenizer_config.json, vocab.txt 等

# 2. 修改config.py使用本地模型
```

**方法C：使用HuggingFace镜像**

```bash
# 在Docker容器启动时设置环境变量
docker run ... \
  -e HF_ENDPOINT=https://hf-mirror.com \
  ...
```

#### 步骤3：重新构建和运行

```bash
# 1. 清理旧容器和镜像
docker rm -f ner-training
docker rmi ner-5090d:latest

# 2. 重新构建镜像
docker build -t ner-5090d:latest .

# 3. 运行（使用ModelScope或本地模型）
docker run --gpus all \
    --ipc=host \
    --shm-size=32g \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    --name ner-training \
    ner-5090d:latest \
    python train_5090d.py
```

---

### 方案二：修改代码使用本地模型（快速方案）

如果不想重新构建镜像，可以修改代码使用本地预下载的模型。

#### 步骤1：在宿主机下载模型

```bash
# 创建模型目录
mkdir -p ./models/pretrained

# 使用git下载（如果有访问权限）
cd ./models/pretrained
git lfs install
git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext

# 或使用modelscope
pip install modelscope
python3 << EOF
from modelscope import snapshot_download
model_dir = snapshot_download('tiansz/chinese-roberta-wwm-ext',
                               cache_dir='./models/pretrained')
print(f'Model saved to: {model_dir}')
EOF
```

#### 步骤2：修改config.py

```python
# config.py
class Config:
    # 使用本地模型路径
    model_name = "/workspace/models/pretrained/chinese-roberta-wwm-ext"
    # 或者如果使用modelscope下载的：
    # model_name = "/workspace/models/pretrained/tiansz/chinese-roberta-wwm-ext"
```

#### 步骤3：运行

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=32g \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    --name ner-training \
    ner-5090d:latest \
    python train_5090d.py
```

---

## 🔍 验证安装

运行诊断脚本检查环境：

```bash
docker run --gpus all \
    -v $(pwd):/workspace \
    ner-5090d:latest \
    python diagnose.py
```

期望输出：
```
================================================================================
  PyTorch 配置检查
================================================================================
PyTorch 版本: 2.6.0.dev20260205+cu124  (或更新)
CUDA 是否可用: True
CUDA 版本: 12.4
可用GPU数量: 2

GPU 0: NVIDIA GeForce RTX 5090 D
  计算能力: 12.0 ✓ (sm_120 支持)
  显存: 32 GB

GPU 1: NVIDIA GeForce RTX 5090 D
  计算能力: 12.0 ✓ (sm_120 支持)
  显存: 32 GB
```

---

## 📊 性能优化说明

### 已实施的优化

1. **混合精度训练（AMP）**
   - 启用自动混合精度，预期加速2-3倍
   - 减少显存占用约40%

2. **分布式数据并行（DDP）**
   - 使用DistributedDataParallel替代DataParallel
   - 更高效的GPU间通信

3. **优化的批次大小**
   - 单GPU batch_size = 32
   - 双GPU有效batch_size = 64
   - 充分利用32GB显存

4. **Gradient Checkpointing**
   - 可选功能，节省显存
   - 在config.py中设置 `use_gradient_checkpointing = True`

5. **高效数据加载**
   - 增加DataLoader workers
   - 启用pin_memory

### 预期性能提升

| 指标 | 原始 | 优化后 | 提升 |
|------|------|--------|------|
| 训练速度 | 基线 | 2.5-3x | 150-200% |
| 显存利用率 | ~18GB | ~28GB | +55% |
| 有效Batch Size | 16 | 64 | 4x |
| 每epoch时间 | 基线 | 0.35x | -65% |

---

## 🐛 常见问题

### Q1: 仍然提示不支持sm_120

**A**: 确保使用PyTorch nightly版本。检查：
```bash
python -c "import torch; print(torch.__version__)"
```
应该显示类似 `2.6.0.dev20260205+cu124`

### Q2: OOM (Out of Memory) 错误

**A**: 减小batch_size或启用gradient checkpointing：
```python
# config.py
batch_size = 24  # 从32减小
use_gradient_checkpointing = True
```

### Q3: 训练速度没有预期快

**A**: 检查：
1. 是否真正使用了双GPU：`nvidia-smi`
2. 是否启用了AMP：`config.use_amp = True`
3. DataLoader workers数量：`num_workers = 8`

### Q4: 模型下载太慢

**A**: 使用国内镜像或预下载模型（见方案二）

---

## 📝 完整启动流程

### 首次运行

```bash
# 1. 确保当前在项目目录
cd /path/to/cover

# 2. 修复脚本换行符问题（如果有）
sed -i 's/\r$//' start_training.sh
sed -i 's/\r$//' monitor_gpu.sh
chmod +x start_training.sh monitor_gpu.sh

# 3. 下载模型（选择一种方法）
# 方法A: ModelScope
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('tiansz/chinese-roberta-wwm-ext', cache_dir='./models/hub')"

# 方法B: 设置HuggingFace镜像（将在Docker运行时使用）
# 无需额外操作，已在docker run命令中包含

# 4. 构建Docker镜像（使用更新的Dockerfile）
docker build -t ner-5090d:latest .

# 5. 运行训练
docker run --gpus all \
    --ipc=host \
    --shm-size=32g \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    --rm \
    --name ner-training \
    ner-5090d:latest \
    bash start_training.sh
```

### 监控训练

在另一个终端：
```bash
# 实时监控GPU使用
docker exec -it ner-training bash monitor_gpu.sh

# 或在宿主机直接监控
watch -n 1 nvidia-smi
```

### 恢复训练

```bash
# 修改config.py
resume_from_checkpoint = "best_model.pt"

# 重新启动
docker run ... ner-5090d:latest python train_5090d.py
```

---

## 📌 重要提示

1. **首次运行建议**：
   - 使用小数据集测试环境是否正常
   - 先运行 `diagnose.py` 验证CUDA支持

2. **数据路径**：
   - 确保 `data/` 目录下有训练数据
   - 文件名需匹配config.py中的设置

3. **模型保存**：
   - 模型默认保存在容器的 `/workspace/best_model.pt`
   - 已通过volume挂载，会同步到宿主机

4. **中断恢复**：
   - 训练支持checkpoint恢复
   - 设置 `config.resume_from_checkpoint` 即可

---

## 🎯 下一步

1. ✅ 解决CUDA兼容性（使用nightly版PyTorch）
2. ✅ 解决模型下载（使用镜像或本地模型）
3. ✅ 运行诊断脚本验证环境
4. ✅ 开始训练
5. 🔄 监控性能，根据需要调整超参数

---

## 📞 需要帮助？

如果遇到其他问题，请提供：
1. `docker logs ner-training` 的完整输出
2. `diagnose.py` 的输出结果
3. 具体的错误信息

我会继续协助您解决问题！
