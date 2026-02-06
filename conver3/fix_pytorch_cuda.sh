#!/bin/bash
# 修复 PyTorch CUDA 兼容性问题
# Fix PyTorch CUDA compatibility for RTX 5090D

echo "=========================================="
echo "修复 RTX 5090D PyTorch 兼容性"
echo "Fix RTX 5090D PyTorch Compatibility"
echo "=========================================="

echo ""
echo "检测到的问题:"
echo "1. PyTorch 不完全支持 RTX 5090D (sm_120)"
echo "2. 需要 PyTorch 2.6+ 或从源码编译"
echo ""

echo "解决方案选项:"
echo ""
echo "选项 1: 安装 PyTorch Nightly (推荐)"
echo "----------------------------------------"
echo "pip3 uninstall torch torchvision torchaudio -y"
echo "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130"
echo ""

echo "选项 2: 使用 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE (临时方案)"
echo "----------------------------------------"
echo "export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1"
echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

echo "选项 3: 从源码编译 PyTorch (最佳性能)"
echo "----------------------------------------"
echo "git clone --recursive https://github.com/pytorch/pytorch"
echo "cd pytorch"
echo "export CMAKE_PREFIX_PATH=\${CONDA_PREFIX:-\$(dirname \$(which conda))/../}"
echo "export TORCH_CUDA_ARCH_LIST=\"12.0\""
echo "python setup.py install"
echo ""

echo "=========================================="
echo "当前建议: 使用选项 2 (临时方案)"
echo "虽然有警告，但不影响训练功能"
echo "=========================================="
