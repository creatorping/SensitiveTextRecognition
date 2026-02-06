#!/bin/bash
# RTX 5090D 快速修复和部署脚本
# 自动解决CUDA兼容性和网络问题

set -e

echo "=========================================="
echo "RTX 5090D 自动修复和部署脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤1: 检查环境
echo -e "${YELLOW}步骤 1/5: 检查环境${NC}"
echo "检查Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker已安装${NC}"

echo "检查NVIDIA驱动..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: NVIDIA驱动未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA驱动已安装${NC}"

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}✓ 检测到 ${GPU_COUNT} 个GPU${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""

# 步骤2: 修复脚本换行符问题
echo -e "${YELLOW}步骤 2/5: 修复脚本格式${NC}"
for script in start_training.sh monitor_gpu.sh fix_5090d_issues.sh; do
    if [ -f "$script" ]; then
        sed -i 's/\r$//' "$script"
        chmod +x "$script"
        echo -e "${GREEN}✓ 修复 $script${NC}"
    fi
done

echo ""

# 步骤3: 处理模型下载
echo -e "${YELLOW}步骤 3/5: 处理预训练模型${NC}"
echo "选择模型获取方式："
echo "  1) 使用HuggingFace镜像（推荐，自动下载）"
echo "  2) 使用ModelScope下载到本地"
echo "  3) 我已经手动下载到 ./models/chinese-roberta-wwm-ext/"
echo ""
read -p "请选择 (1/2/3): " model_choice

case $model_choice in
    1)
        echo -e "${GREEN}将使用HuggingFace镜像，无需预下载${NC}"
        USE_HF_MIRROR=true
        ;;
    2)
        echo "使用ModelScope下载模型..."
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}错误: Python3未安装${NC}"
            exit 1
        fi

        pip install -q modelscope

        python3 << 'EOF'
from modelscope import snapshot_download
import os

model_dir = snapshot_download(
    'tiansz/chinese-roberta-wwm-ext',
    cache_dir='./models/hub'
)
print(f"模型已下载到: {model_dir}")

# 创建符号链接
os.makedirs('./models/chinese-roberta-wwm-ext', exist_ok=True)
print("请手动复制模型文件到 ./models/chinese-roberta-wwm-ext/")
EOF

        echo -e "${GREEN}✓ ModelScope下载完成${NC}"
        USE_LOCAL_MODEL=true
        ;;
    3)
        if [ ! -d "./models/chinese-roberta-wwm-ext" ]; then
            echo -e "${RED}错误: 目录 ./models/chinese-roberta-wwm-ext 不存在${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 使用本地模型${NC}"
        USE_LOCAL_MODEL=true
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""

# 步骤4: 选择Dockerfile
echo -e "${YELLOW}步骤 4/5: 构建Docker镜像${NC}"
echo "选择Dockerfile版本："
echo "  1) Dockerfile.5090d-fixed (使用PyTorch Nightly，完全支持sm_120)"
echo "  2) Dockerfile (使用PyTorch 2.5.1稳定版，可能有兼容性警告)"
echo ""
read -p "请选择 (1/2，推荐1): " dockerfile_choice

case $dockerfile_choice in
    1)
        DOCKERFILE="Dockerfile.5090d-fixed"
        ;;
    2)
        DOCKERFILE="Dockerfile"
        ;;
    *)
        echo -e "${RED}无效选择，使用默认: Dockerfile.5090d-fixed${NC}"
        DOCKERFILE="Dockerfile.5090d-fixed"
        ;;
esac

if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}错误: $DOCKERFILE 不存在${NC}"
    exit 1
fi

echo "开始构建镜像（这可能需要10-20分钟）..."
docker build -f "$DOCKERFILE" -t ner-5090d:latest . || {
    echo -e "${RED}镜像构建失败${NC}"
    exit 1
}

echo -e "${GREEN}✓ 镜像构建成功${NC}"
echo ""

# 步骤5: 启动训练
echo -e "${YELLOW}步骤 5/5: 启动训练${NC}"

# 清理旧容器
if docker ps -a | grep -q ner-training; then
    echo "清理旧容器..."
    docker rm -f ner-training 2>/dev/null || true
fi

# 构建docker run命令
DOCKER_CMD="docker run --gpus all \
    --ipc=host \
    --shm-size=32g \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1"

# 添加HuggingFace镜像环境变量
if [ "$USE_HF_MIRROR" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -e HF_ENDPOINT=https://hf-mirror.com"
fi

DOCKER_CMD="$DOCKER_CMD \
    --name ner-training \
    ner-5090d:latest"

echo ""
echo "是否立即开始训练？"
echo "  1) 是，开始训练"
echo "  2) 否，先运行诊断脚本"
echo "  3) 否，进入容器shell"
echo ""
read -p "请选择 (1/2/3): " start_choice

case $start_choice in
    1)
        echo -e "${GREEN}开始训练...${NC}"
        echo ""
        echo "提示: 可以在另一个终端运行以下命令监控GPU："
        echo "  watch -n 1 nvidia-smi"
        echo ""
        $DOCKER_CMD bash start_training.sh
        ;;
    2)
        echo -e "${GREEN}运行诊断脚本...${NC}"
        $DOCKER_CMD python diagnose.py
        ;;
    3)
        echo -e "${GREEN}进入容器shell...${NC}"
        $DOCKER_CMD bash
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo -e "${GREEN}部署完成！${NC}"
echo "=========================================="
echo ""
echo "常用命令："
echo "  查看日志: docker logs -f ner-training"
echo "  监控GPU: watch -n 1 nvidia-smi"
echo "  进入容器: docker exec -it ner-training bash"
echo "  停止训练: docker stop ner-training"
echo ""
