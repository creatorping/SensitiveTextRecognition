#!/bin/bash

# Docker构建和运行脚本 - 使用PyTorch 5090镜像

echo "=========================================="
echo "构建Docker镜像..."
echo "=========================================="

# 使用docker-compose构建镜像
docker-compose build

if [ $? -ne 0 ]; then
    echo "❌ Docker镜像构建失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Docker镜像构建成功"
echo "=========================================="
echo ""
echo "启动容器..."
echo "=========================================="

# 启动容器
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ 容器启动失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 容器启动成功"
echo "=========================================="
echo ""
echo "查看容器状态："
docker-compose ps

echo ""
echo "查看容器日志："
echo "docker-compose logs -f"
echo ""
echo "进入容器："
echo "docker-compose exec ner-training bash"
echo ""
echo "停止容器："
echo "docker-compose down"
