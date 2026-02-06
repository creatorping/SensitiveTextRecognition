# 离线训练环境设置指南

## 问题说明

您遇到的两个主要问题：

1. **模型下载失败**: 无法连接到 HuggingFace 下载 `hfl/chinese-roberta-wwm-ext` 模型
2. **PyTorch CUDA 兼容性警告**: RTX 5090D 的计算能力 (sm_120) 不被当前 PyTorch 完全支持

## 解决方案

### 第一步：下载模型（在有网络的机器上）

如果您有另一台可以访问外网的机器：

```bash
# 在有网络的机器上运行
python3 download_model.py
```

这将下载模型到 `./models/chinese-roberta-wwm-ext/` 目录。

**或者手动下载：**

访问 HuggingFace 镜像站（国内可访问）：
- https://hf-mirror.com/hfl/chinese-roberta-wwm-ext

下载以下文件到 `./models/chinese-roberta-wwm-ext/`：
- `config.json`
- `pytorch_model.bin`
- `tokenizer_config.json`
- `vocab.txt`
- `tokenizer.json`

### 第二步：传输模型文件

将整个 `models` 文件夹复制到训练机器的项目目录：

```bash
# 使用 scp 或其他方式传输
scp -r models/ user@training-machine:/path/to/conver2/
```

### 第三步：设置离线训练环境

在训练机器上运行：

```bash
chmod +x setup_offline_training.sh
./setup_offline_training.sh
```

这个脚本会：
- 检查模型文件是否完整
- 检查数据文件是否存在
- 设置必要的环境变量
- 生成训练启动脚本

### 第四步：开始训练

```bash
./start_training.sh
```

或使用 Docker：

```bash
docker run --rm --gpus all \
    --ipc=host \
    --shm-size=32g \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    --name ner-training \
    ner-5090d:latest \
    bash start_training.sh
```

## PyTorch CUDA 兼容性说明

RTX 5090D 使用 CUDA 计算能力 12.0 (sm_120)，但当前 PyTorch 版本只支持到 sm_90。

**影响：**
- 会显示警告信息
- **不影响训练功能**，可以正常使用
- 可能无法使用某些最新的 CUDA 12.0 特性

**解决方案（可选）：**

1. **临时方案（推荐）**: 忽略警告，继续训练
   ```bash
   export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
   ```

2. **升级 PyTorch**: 安装支持 CUDA 13.0 的 PyTorch nightly 版本
   ```bash
   pip3 uninstall torch torchvision torchaudio -y
   pip3 install --pre torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/nightly/cu130
   ```

3. **从源码编译**: 获得最佳性能（需要较长时间）
   ```bash
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   export TORCH_CUDA_ARCH_LIST="12.0"
   python setup.py install
   ```

## 已修改的文件

1. **config.py**:
   - 模型路径改为本地路径 `/workspace/models/chinese-roberta-wwm-ext`

2. **data_loader.py**:
   - 添加 `local_files_only=True` 参数

3. **model.py**:
   - 添加 `local_files_only=True` 参数

## 环境变量说明

- `TRANSFORMERS_OFFLINE=1`: 强制 Transformers 使用本地文件
- `HF_DATASETS_OFFLINE=1`: 强制 Datasets 使用本地文件
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`: 允许在不完全支持的 GPU 上使用 TF32
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: 优化 CUDA 内存分配

## 验证设置

运行以下命令验证环境：

```bash
python3 -c "
import torch
from transformers import AutoTokenizer, AutoModel

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f'GPU {i}:', torch.cuda.get_device_name(i))

# 测试加载模型
tokenizer = AutoTokenizer.from_pretrained(
    '/workspace/models/chinese-roberta-wwm-ext',
    local_files_only=True
)
print('✓ Tokenizer loaded successfully')

model = AutoModel.from_pretrained(
    '/workspace/models/chinese-roberta-wwm-ext',
    local_files_only=True
)
print('✓ Model loaded successfully')
"
```

## 故障排除

### 问题 1: 仍然提示无法连接网络

**解决方案**: 确保设置了环境变量
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 问题 2: 找不到模型文件

**解决方案**: 检查模型路径和文件
```bash
ls -la models/chinese-roberta-wwm-ext/
```

应该看到：
- config.json
- pytorch_model.bin
- tokenizer_config.json
- vocab.txt

### 问题 3: CUDA out of memory

**解决方案**: 减小 batch size
在 `config.py` 中修改：
```python
batch_size = 16  # 从 32 减小到 16
```

### 问题 4: 训练速度慢

**解决方案**: 确保启用了混合精度训练
在 `config.py` 中确认：
```python
use_amp = True  # 应该是 True
```

## 性能优化建议

1. **使用混合精度训练 (AMP)**: 已启用，可提速 2-3 倍
2. **增大 batch size**: RTX 5090D 有 32GB 显存，可以使用更大的 batch size
3. **使用梯度累积**: 如果显存不足，增加 `gradient_accumulation_steps`
4. **启用 TF32**: 已通过环境变量启用

## 联系支持

如果仍有问题，请提供：
1. 完整的错误日志
2. `nvidia-smi` 输出
3. `python3 -c "import torch; print(torch.__version__)"` 输出
