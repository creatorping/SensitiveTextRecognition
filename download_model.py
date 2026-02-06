#!/usr/bin/env python3
"""
下载模型脚本 - 在有网络的环境中运行此脚本下载模型
Download model script - Run this in an environment with internet access
"""
import os
from transformers import AutoModel, AutoTokenizer

def download_model():
    """下载中文RoBERTa模型"""
    model_name = "hfl/chinese-roberta-wwm-ext"
    save_path = "./models/chinese-roberta-wwm-ext"

    print(f"正在下载模型: {model_name}")
    print(f"保存路径: {save_path}")

    # 创建目录
    os.makedirs(save_path, exist_ok=True)

    # 下载tokenizer
    print("\n下载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    print("✓ Tokenizer 下载完成")

    # 下载模型
    print("\n下载模型...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_path)
    print("✓ 模型下载完成")

    print(f"\n✓ 所有文件已保存到: {save_path}")
    print("\n使用方法:")
    print("1. 将整个 models 文件夹复制到训练环境")
    print("2. 在 Docker 中挂载: -v $(pwd)/models:/workspace/models")
    print("3. 训练脚本会自动使用本地模型")

if __name__ == "__main__":
    download_model()
