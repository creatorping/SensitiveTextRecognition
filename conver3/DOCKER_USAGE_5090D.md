# Docker使用教程 - 双RTX 5090D优化版

本文档详细说明如何在双RTX 5090D显卡上使用Docker运行训练代码。

## 📋 目录

1. [环境要求](#环境要求)
2. [快速开始](#快速开始)
3. [详细步骤](#详细步骤)
4. [性能优化说明](#性能优化说明)
5. [常见问题](#常见问题)
6. [监控和调试](#监控和调试)

---

## 环境要求

### 硬件要求
- **GPU**: 双NVIDIA RTX 5090D (32GB显存 × 2)
- **内存**: 建议64GB以上
- **存储**: 至少50GB可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04/22.04 或其他Linux发行版
- **NVIDIA驱动**: >= 550.x (支持CUDA 12.4)
- **Docker**: >= 20.10
- **NVIDIA Container Toolkit**: 最新版本

---

## 快速开始

### 1. 安装NVIDIA驱动和Docker环境

```bash
# 检查NVIDIA驱动版本
nvidia-smi

# 应该看到两张RTX 5090D显卡，驱动版本 >= 550.x

# 安装Docker（如果未安装）
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证安装
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 2. 构建Docker镜像

```bash
# 进入项目目录
cd /path/to/cover

# 构建镜像（大约需要10-15分钟）
docker build -t ner-5090d:latest .

# 或使用docker-compose构建
docker-compose build
```

### 3. 启动训练

**方式一：使用docker-compose（推荐）**

```bash
# 启动训练
docker-compose up

# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止训练
docker-compose down
```

**方式二：使用docker run**

```bash
docker run --rm --gpus all \
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
    bash start_training.sh
```

---

## 详细步骤

### 步骤1: 准备数据

确保数据文件在正确位置：

```bash
data/
├── 409_data_train.txt
├── 409_train_lable.txt
├── 409_data_test.txt
└── 409_test_lable.txt
```

### 步骤2: 配置检查

编辑 `config.py` 调整训练参数（已针对5090D优化）：

```python
# 关键配置项
batch_size = 32              # 每个GPU的batch size
use_amp = True               # 混合精度训练（提速2-3倍）
num_epochs = 500             # 训练轮数
learning_rate = 3e-5         # 学习率
```

### 步骤3: 构建镜像

```bash
# 查看Dockerfile确认配置
cat Dockerfile

# 构建镜像
docker build -t ner-5090d:latest . --no-cache

# 查看镜像
docker images | grep ner-5090d
```

### 步骤4: 运行训练

```bash
# 使用docker-compose（推荐）
docker-compose up

# 训练将自动开始，输出类似：
# ==========================================
# 🚀 Distributed Training Configuration (Optimized for RTX 5090D)
# ==========================================
# World Size: 2
# Rank: 0
# Local Rank: 0
#   GPU 0: NVIDIA GeForce RTX 5090D
#   GPU 1: NVIDIA GeForce RTX 5090D
# Mixed Precision (AMP): Enabled
# Batch Size per GPU: 32
# Effective Batch Size: 64
# ==========================================
```

### 步骤5: 监控训练

**在容器内监控GPU使用情况：**

```bash
# 进入运行中的容器
docker exec -it ner-training-dual-5090d bash

# 监控GPU
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/training.log
```

**在宿主机监控：**

```bash
# 监控GPU
nvidia-smi -l 1

# 查看容器日志
docker logs -f ner-training-dual-5090d
```

---

## 性能优化说明

### 已实现的优化

#### 1. **DistributedDataParallel (DDP)**
- 使用DDP替代DataParallel，通信效率提升约30%
- 每个GPU独立运行，减少同步开销

#### 2. **自动混合精度训练 (AMP)**
- 使用FP16进行前向和反向传播
- 训练速度提升2-3倍
- 显存占用减少约40%

#### 3. **优化的批次大小**
- 每GPU batch_size=32，充分利用32GB显存
- 有效batch_size=64（32×2），加速收敛

#### 4. **CUDA优化**
- 使用CUDA 12.4和cuDNN 9
- 针对RTX 5090D的计算能力9.0优化
- 启用TensorCore加速

#### 5. **内存优化**
- IPC模式：host（提升多进程通信）
- 共享内存：32GB
- 非阻塞数据传输（non_blocking=True）

### 性能预期

| 配置 | 训练速度 | 显存占用 | 备注 |
|------|---------|---------|------|
| 单GPU (无AMP) | ~100 samples/s | ~28GB | 基准 |
| 单GPU (AMP) | ~250 samples/s | ~18GB | 2.5x加速 |
| 双GPU (DDP+AMP) | ~480 samples/s | ~18GB/GPU | 4.8x加速 |

### 进一步优化建议

如果显存充足，可以增加batch_size：

```python
# config.py
batch_size = 48  # 或更大
```

如果显存不足，可以启用gradient checkpointing：

```python
# config.py
use_gradient_checkpointing = True
```

---

## 常见问题

### Q1: 构建镜像时出现网络错误

**问题**: pip安装PyTorch超时

**解决方案**:
```bash
# 使用国内镜像源
docker build -t ner-5090d:latest . \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

或修改Dockerfile添加：
```dockerfile
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 运行时报错 "CUDA out of memory"

**解决方案**:
1. 减小batch_size：
```python
# config.py
batch_size = 24  # 或16
```

2. 启用gradient checkpointing：
```python
use_gradient_checkpointing = True
```

3. 检查是否有其他程序占用GPU：
```bash
nvidia-smi
# 如果有其他进程，先关闭
```

### Q3: 训练速度慢

**检查清单**:
1. 确认AMP已启用：
```python
# config.py
use_amp = True
```

2. 确认使用DDP而非DataParallel：
```bash
# 应该使用 start_training.sh 或 torchrun
bash start_training.sh
```

3. 检查数据加载是否成为瓶颈：
```python
# data_loader.py
num_workers = 4  # 增加worker数量
pin_memory = True  # 启用内存锁定
```

### Q4: 分布式训练初始化失败

**错误信息**: "RuntimeError: Address already in use"

**解决方案**:
```bash
# 更改master_port
torchrun --nproc_per_node=2 --master_port=29501 train_5090d.py
```

### Q5: Docker容器无法访问GPU

**检查步骤**:
```bash
# 1. 检查NVIDIA驱动
nvidia-smi

# 2. 检查nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 3. 重启Docker服务
sudo systemctl restart docker

# 4. 检查docker-compose配置
cat docker-compose.yml | grep -A 5 "deploy:"
```

### Q6: 模型保存失败

**问题**: 权限错误

**解决方案**:
```bash
# 修改挂载目录权限
chmod -R 777 models/
chmod -R 777 logs/

# 或在docker-compose.yml中添加用户映射
user: "${UID}:${GID}"
```

---

## 监控和调试

### 实时监控GPU使用

```bash
# 方法1: nvidia-smi
watch -n 1 nvidia-smi

# 方法2: nvtop（更友好的界面）
sudo apt install nvtop
nvtop

# 方法3: 使用Python脚本
python -c "
import torch
print(f'GPU 0: {torch.cuda.memory_allocated(0)/1e9:.2f}GB / {torch.cuda.max_memory_allocated(0)/1e9:.2f}GB')
print(f'GPU 1: {torch.cuda.memory_allocated(1)/1e9:.2f}GB / {torch.cuda.max_memory_allocated(1)/1e9:.2f}GB')
"
```

### 查看训练日志

```bash
# 实时查看
docker logs -f ner-training-dual-5090d

# 保存日志到文件
docker logs ner-training-dual-5090d > training.log 2>&1

# 搜索特定信息
docker logs ner-training-dual-5090d | grep "F1:"
```

### 进入容器调试

```bash
# 进入运行中的容器
docker exec -it ner-training-dual-5090d bash

# 在容器内
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
python -c "import torch; print(torch.__version__)"

# 测试模型加载
python -c "from model import NestedPrivacyNER; from config import Config; model = NestedPrivacyNER(Config())"
```

### 性能分析

```bash
# 使用PyTorch Profiler
# 在train_5090d.py中添加：
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # 训练代码
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 高级用法

### 恢复训练

```python
# config.py
resume_from_checkpoint = "best_model_5090d.pt"
```

```bash
# 重新启动训练
docker-compose up
```

### 自定义训练脚本

```bash
# 运行自定义命令
docker-compose run --rm ner-training python your_script.py

# 或
docker run --gpus all --rm -v $(pwd):/workspace ner-5090d:latest python your_script.py
```

### 导出模型

```bash
# 进入容器
docker exec -it ner-training-dual-5090d bash

# 导出为ONNX
python -c "
import torch
from model import NestedPrivacyNER
from config import Config

config = Config()
model = NestedPrivacyNER(config)
checkpoint = torch.load('best_model_5090d.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 导出
dummy_input = (
    torch.randint(0, 1000, (1, 128)),
    torch.ones(1, 128),
    torch.randint(0, 128, (1, 10, 2))
)
torch.onnx.export(model, dummy_input, 'model.onnx')
"
```

---

## 清理和维护

### 清理Docker资源

```bash
# 停止并删除容器
docker-compose down

# 删除镜像
docker rmi ner-5090d:latest

# 清理未使用的资源
docker system prune -a

# 清理卷
docker volume prune
```

### 备份模型

```bash
# 备份最佳模型
cp best_model_5090d.pt backups/best_model_$(date +%Y%m%d_%H%M%S).pt

# 压缩备份
tar -czf model_backup_$(date +%Y%m%d).tar.gz best_model_5090d.pt config.py
```

---

## 技术支持

如遇到问题：

1. 查看本文档的[常见问题](#常见问题)部分
2. 检查Docker和NVIDIA驱动版本
3. 查看详细错误日志：`docker logs ner-training-dual-5090d`
4. 在容器内运行诊断脚本

---

## 更新日志

- **2025-02-05**: 初始版本，针对双RTX 5090D优化
  - 使用CUDA 12.4 + PyTorch 2.5.1
  - 实现DDP + AMP
  - 优化batch size和内存配置
