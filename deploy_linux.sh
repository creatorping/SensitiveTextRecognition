#!/bin/bash
# Linux系统一键部署脚本 - 双RTX 5090D
# 自动修复所有兼容性问题

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "双RTX 5090D Linux部署脚本"
echo -e "==========================================${NC}"
echo ""

# 步骤1: 修复所有脚本的换行符
echo -e "${YELLOW}[1/6] 修复脚本换行符格式...${NC}"
for file in *.sh; do
    if [ -f "$file" ]; then
        dos2unix "$file" 2>/dev/null || sed -i 's/\r$//' "$file"
        chmod +x "$file"
        echo -e "${GREEN}  ✓ $file${NC}"
    fi
done
echo ""

# 步骤2: 检查环境
echo -e "${YELLOW}[2/6] 检查系统环境...${NC}"

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}  ✗ Docker未安装${NC}"
    echo "  安装命令: curl -fsSL https://get.docker.com | bash"
    exit 1
fi
echo -e "${GREEN}  ✓ Docker已安装: $(docker --version)${NC}"

# 检查NVIDIA驱动
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}  ✗ NVIDIA驱动未安装${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ NVIDIA驱动已安装${NC}"

# 检查GPU
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo -e "${YELLOW}  ⚠ 检测到 $GPU_COUNT 个GPU（建议2个）${NC}"
else
    echo -e "${GREEN}  ✓ 检测到 $GPU_COUNT 个GPU${NC}"
fi

# 显示GPU信息
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    echo -e "${GREEN}    GPU $line${NC}"
done
echo ""

# 步骤3: 检查数据文件
echo -e "${YELLOW}[3/6] 检查数据文件...${NC}"
DATA_FILES=(
    "data/409_data_train.txt"
    "data/409_train_lable.txt"
    "data/409_data_test.txt"
    "data/409_test_lable.txt"
)

ALL_DATA_EXISTS=true
for file in "${DATA_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo -e "${GREEN}  ✓ $file (${size} bytes)${NC}"
    else
        echo -e "${RED}  ✗ $file 不存在${NC}"
        ALL_DATA_EXISTS=false
    fi
done

if [ "$ALL_DATA_EXISTS" = false ]; then
    echo -e "${RED}错误: 缺少数据文件${NC}"
    exit 1
fi
echo ""

# 步骤4: 处理预训练模型
echo -e "${YELLOW}[4/6] 配置预训练模型...${NC}"
echo "选择模型获取方式："
echo "  1) 使用HuggingFace镜像（容器内自动下载）"
echo "  2) 使用ModelScope下载到本地（推荐国内用户）"
echo "  3) 已手动下载到 ./models/chinese-roberta-wwm-ext/"
echo ""
read -p "请选择 [1-3]: " model_choice

USE_HF_MIRROR=false
USE_LOCAL_MODEL=false

case $model_choice in
    1)
        echo -e "${GREEN}  ✓ 将使用HuggingFace镜像${NC}"
        USE_HF_MIRROR=true
        ;;
    2)
        echo -e "${BLUE}  正在使用ModelScope下载模型...${NC}"

        # 检查Python
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}  ✗ Python3未安装${NC}"
            exit 1
        fi

        # 安装modelscope
        pip3 install -q modelscope

        # 下载模型
        python3 << 'PYEOF'
import os
from modelscope import snapshot_download

print("  下载中，请稍候...")
model_dir = snapshot_download(
    'tiansz/chinese-roberta-wwm-ext',
    cache_dir='./models/modelscope'
)
print(f"  模型已下载到: {model_dir}")

# 创建符号链接
target_dir = './models/chinese-roberta-wwm-ext'
os.makedirs(target_dir, exist_ok=True)
print(f"  请将模型文件复制到: {target_dir}")
PYEOF

        echo -e "${GREEN}  ✓ ModelScope下载完成${NC}"
        USE_LOCAL_MODEL=true
        ;;
    3)
        if [ ! -d "./models/chinese-roberta-wwm-ext" ]; then
            echo -e "${RED}  ✗ 目录不存在: ./models/chinese-roberta-wwm-ext${NC}"
            exit 1
        fi

        # 检查必要文件
        required_files=("config.json" "pytorch_model.bin" "vocab.txt")
        for file in "${required_files[@]}"; do
            if [ ! -f "./models/chinese-roberta-wwm-ext/$file" ]; then
                echo -e "${RED}  ✗ 缺少文件: $file${NC}"
                exit 1
            fi
        done

        echo -e "${GREEN}  ✓ 本地模型文件完整${NC}"
        USE_LOCAL_MODEL=true
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac
echo ""

# 步骤5: 选择并构建Docker镜像
echo -e "${YELLOW}[5/6] 构建Docker镜像...${NC}"
echo "选择PyTorch版本："
echo "  1) PyTorch Nightly (推荐，完全支持RTX 5090D sm_120)"
echo "  2) PyTorch 2.5.1 稳定版 (可能有兼容性警告)"
echo ""
read -p "请选择 [1-2]: " pytorch_choice

case $pytorch_choice in
    1)
        DOCKERFILE="Dockerfile.5090d-fixed"
        IMAGE_TAG="ner-5090d:nightly"
        echo -e "${GREEN}  ✓ 使用PyTorch Nightly版本${NC}"
        ;;
    2)
        DOCKERFILE="Dockerfile"
        IMAGE_TAG="ner-5090d:stable"
        echo -e "${YELLOW}  ⚠ 使用稳定版，可能有sm_120警告${NC}"
        ;;
    *)
        echo -e "${RED}无效选择，使用默认: Nightly${NC}"
        DOCKERFILE="Dockerfile.5090d-fixed"
        IMAGE_TAG="ner-5090d:nightly"
        ;;
esac

if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}  ✗ $DOCKERFILE 不存在${NC}"
    exit 1
fi

echo -e "${BLUE}  开始构建镜像（预计10-20分钟）...${NC}"
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" . 2>&1 | tee build.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}  ✗ 镜像构建失败，请查看 build.log${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ 镜像构建成功: $IMAGE_TAG${NC}"
echo ""

# 步骤6: 启动训练
echo -e "${YELLOW}[6/6] 准备启动训练...${NC}"

# 清理旧容器
if docker ps -a --format '{{.Names}}' | grep -q '^ner-training$'; then
    echo -e "${BLUE}  清理旧容器...${NC}"
    docker rm -f ner-training 2>/dev/null || true
fi

# 构建docker run命令
DOCKER_RUN_CMD="docker run --gpus all \
    --ipc=host \
    --shm-size=32g \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e PYTHONUNBUFFERED=1"

# 添加HuggingFace镜像
if [ "$USE_HF_MIRROR" = true ]; then
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD -e HF_ENDPOINT=https://hf-mirror.com"
fi

DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
    --name ner-training \
    $IMAGE_TAG"

echo ""
echo -e "${BLUE}=========================================="
echo "准备就绪！"
echo -e "==========================================${NC}"
echo ""
echo "选择操作："
echo "  1) 立即开始训练"
echo "  2) 运行环境诊断"
echo "  3) 进入容器Shell"
echo "  4) 退出"
echo ""
read -p "请选择 [1-4]: " action_choice

case $action_choice in
    1)
        echo -e "${GREEN}开始训练...${NC}"
        echo ""
        echo -e "${BLUE}提示: 可在另一终端运行以下命令监控GPU:${NC}"
        echo "  watch -n 1 nvidia-smi"
        echo ""
        echo -e "${BLUE}查看训练日志:${NC}"
        echo "  docker logs -f ner-training"
        echo ""
        sleep 2

        $DOCKER_RUN_CMD bash start_training.sh
        ;;
    2)
        echo -e "${GREEN}运行诊断脚本...${NC}"
        echo ""
        $DOCKER_RUN_CMD python diagnose.py
        ;;
    3)
        echo -e "${GREEN}进入容器Shell...${NC}"
        echo ""
        $DOCKER_RUN_CMD bash
        ;;
    4)
        echo -e "${BLUE}退出部署脚本${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=========================================="
echo "部署完成！"
echo -e "==========================================${NC}"
echo ""
echo "常用命令："
echo "  查看日志: docker logs -f ner-training"
echo "  监控GPU: watch -n 1 nvidia-smi"
echo "  进入容器: docker exec -it ner-training bash"
echo "  停止训练: docker stop ner-training"
echo "  删除容器: docker rm -f ner-training"
echo ""
