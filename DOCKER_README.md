# Docker 部署指南 - PyTorch 5090

## 镜像信息

基础镜像：`crpi-ut0apwr713irl7gy.cn-hangzhou.personal.cr.aliyuncs.com/my_english/pytorch-5090:25.01`

该镜像已预装：
- PyTorch（支持RTX 5090D）
- CUDA 工具链
- cuDNN

## 快速开始

### 方法1：使用便捷脚本（推荐）

```bash
# 构建并运行
./docker_build_and_run.sh
```

### 方法2：使用docker-compose

```bash
# 构建镜像
docker-compose build

# 启动容器（后台运行）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 进入容器
docker-compose exec ner-training bash

# 停止容器
docker-compose down
```

### 方法3：使用docker命令

```bash
# 构建镜像
docker build -f Dockerfile.pytorch5090 -t ner-training:pytorch5090 .

# 运行容器
docker run -d \
  --name ner-training-5090d \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  ner-training:pytorch5090 \
  python train_multi_gpu.py
```

## 配置说明

### GPU配置

docker-compose.yml中配置了双GPU支持：
- 使用GPU 0和1
- 共享内存：32GB
- IPC模式：host（提高多进程性能）

### 挂载目录

- `./` → `/workspace`：代码目录
- `./data` → `/workspace/data`：数据目录
- `./models` → `/workspace/models`：模型保存目录
- Hugging Face缓存：自动管理

### 环境变量

- `NVIDIA_VISIBLE_DEVICES=0,1`：指定使用的GPU
- `CUDA_VISIBLE_DEVICES=0,1`：CUDA可见设备
- `PYTHONUNBUFFERED=1`：Python输出不缓冲
- `TOKENIZERS_PARALLELISM=false`：禁用tokenizers并行警告

## 常用命令

```bash
# 查看容器状态
docker-compose ps

# 查看GPU使用情况（在容器内）
docker-compose exec ner-training nvidia-smi

# 查看实时日志
docker-compose logs -f

# 重启容器
docker-compose restart

# 停止并删除容器
docker-compose down

# 停止并删除容器及卷
docker-compose down -v
```

## 故障排查

### 1. 镜像拉取失败

确保已经登录到阿里云容器镜像服务：
```bash
docker login crpi-ut0apwr713irl7gy.cn-hangzhou.personal.cr.aliyuncs.com
```

### 2. GPU不可用

检查NVIDIA Docker运行时：
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 3. 权限问题

确保脚本有执行权限：
```bash
chmod +x docker_build_and_run.sh
```

## 项目结构

```
.
├── Dockerfile.pytorch5090      # 使用PyTorch 5090镜像的Dockerfile
├── docker-compose.yml          # Docker Compose配置
├── docker_build_and_run.sh    # 便捷构建运行脚本
├── requirements.txt            # Python依赖
├── train_multi_gpu.py         # 多GPU训练脚本
└── data/                      # 数据目录
```
