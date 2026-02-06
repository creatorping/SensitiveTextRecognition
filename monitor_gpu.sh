#!/bin/bash
# GPU实时监控脚本 - 双RTX 5090D

echo "=========================================="
echo "双RTX 5090D实时监控"
echo "按 Ctrl+C 退出"
echo "=========================================="

watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits | column -t -s ","'
