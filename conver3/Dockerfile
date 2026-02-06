# 1. 升级基础镜像：RTX 50 系列建议使用 CUDA 12.6 或 12.8 的开发镜像
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# --- 关键修正：针对 Blackwell (sm_120) 的环境变量 ---
# 强制 JIT 编译以支持新架构，或者在找不到 kernel 时尝试实时编译
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 如果后续涉及源码编译扩展，启用 12.0 算力支持
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" 

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    git wget vim curl build-essential \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- 修正 1：分步安装依赖，排除 requirements 中的旧版本 torch ---
COPY requirements.txt .
# 使用 grep 过滤掉可能冲突的 torch 库后再安装
RUN grep -ivE "torch|torchvision|torchaudio" requirements.txt > requirements_no_torch.txt \
    && pip install --no-cache-dir -r requirements_no_torch.txt

# --- 修正 2：安装最新的 PyTorch Nightly (cu126 或等待 cu128) ---
# 注意：RTX 5090 D 至少需要 torch >= 2.6.0 (Nightly) 才能尝试匹配 sm_120
RUN pip install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# --- 修正 3：解决 Hugging Face 403 问题 (禁用在线讨论区检测) ---
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
# 强制要求从本地加载模型，或者使用镜像站时禁用元数据自动更新
ENV HF_ENDPOINT=https://hf-mirror.com

COPY . .

RUN mkdir -p /workspace/data /workspace/models /workspace/logs

CMD ["bash", "start_training.sh"]