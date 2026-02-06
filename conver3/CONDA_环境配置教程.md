# Conda环境配置与运行教程

## 项目简介

本项目是一个基于深度学习的嵌套隐私实体识别系统，用于社交网络文本中的隐私信息检测。

## 环境要求

- Linux系统
- Conda包管理器
- CUDA 11.8或更高版本（用于GPU加速）
- 至少16GB内存
- 建议使用GPU（RTX 4090/5090D或更高）

## 第一步：创建Conda环境

使用以下命令创建一个新的conda环境，命名为`privacy_ner`，使用Python 3.9：

```bash
conda create -n privacy_ner python=3.9 -y
```

**说明**：
- `-n privacy_ner`：指定环境名称
- `python=3.9`：指定Python版本为3.9
- `-y`：自动确认安装

## 第二步：激活Conda环境

```bash
conda activate privacy_ner
```

激活后，命令行提示符前会显示`(privacy_ner)`，表示环境已激活。

## 第三步：安装PyTorch及CUDA支持

安装PyTorch 2.7.1及CUDA 11.8支持：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**说明**：
- 这将安装PyTorch及其相关组件
- `--index-url`指定使用CUDA 11.8版本
- 下载大小约2GB，需要一定时间

**安装时间**：约3-5分钟（取决于网络速度）

## 第四步：安装项目依赖包

安装transformers、tqdm、scikit-learn和torchcrf：

```bash
pip install transformers>=4.30.0 tqdm>=4.65.0 scikit-learn>=1.3.0 torchcrf>=1.1.0
```

**说明**：
- `transformers`：用于加载预训练的BERT模型
- `tqdm`：显示训练进度条
- `scikit-learn`：用于评估指标计算
- `torchcrf`：条件随机场（CRF）层支持

## 第五步：验证环境安装

运行以下命令验证所有包是否正确安装：

```bash
python -c "import torch; import transformers; from TorchCRF import CRF; print('✓ 所有包导入成功'); print(f'PyTorch版本: {torch.__version__}'); print(f'Transformers版本: {transformers.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

**预期输出**：
```
✓ 所有包导入成功
PyTorch版本: 2.7.1+cu118
Transformers版本: 4.57.6
CUDA可用: True
CUDA版本: 11.8
```

## 第六步：配置模型路径

### 方法1：使用本地模型（推荐，离线环境）

如果您有本地的中文BERT模型，编辑`config.py`文件：

```python
model_name = "/path/to/your/chinese-roberta-wwm-ext-large"
hidden_size = 1024  # large模型使用1024，base模型使用768
```

### 方法2：使用HuggingFace在线模型

如果有网络连接，可以使用在线模型：

```python
model_name = "hfl/chinese-roberta-wwm-ext"
hidden_size = 768  # base模型
```

**注意**：首次运行会自动下载模型（约400MB）

## 第七步：运行训练脚本

确保在项目根目录下，运行：

```bash
python train.py
```

**训练过程说明**：
- 训练将自动加载数据集（`data/`目录下的文件）
- 显示训练进度条和损失值
- 每个epoch约需5-10分钟（取决于GPU性能）
- 最佳模型会保存为`best_model.pt`

**预期输出示例**：
```
Loading data...
Creating model...
Starting training from scratch...

Computing class weights...
Class weights computed:
  O (ID=0): weight=0.0051, count=640
  B_BI (ID=1): weight=6.4892, count=1
  ...

Starting training...

Epoch 1/200
Training:  24%|██▍       | 50/207 [01:18<04:08,  1.58s/it, loss=2.3]
```

## 第八步：运行推理测试

训练完成后，可以运行推理脚本测试模型：

```bash
python inference.py
```

这将在测试集上评估模型性能，输出各实体类型的Precision、Recall和F1分数。

## 第九步：运行演示

使用演示脚本测试模型效果：

```bash
python demo.py
```

这将在示例文本上展示实体识别结果。

## 常见问题解决

### 问题1：CUDA不可用

**症状**：`CUDA可用: False`

**解决方案**：
1. 检查NVIDIA驱动是否正确安装：`nvidia-smi`
2. 确认安装了正确的CUDA版本
3. 重新安装PyTorch with CUDA支持

### 问题2：模型维度不匹配

**症状**：`RuntimeError: expected input[32, 1024, 512] to have 768 channels`

**解决方案**：
- 如果使用`chinese-roberta-wwm-ext-large`，设置`hidden_size = 1024`
- 如果使用`chinese-roberta-wwm-ext`（base版本），设置`hidden_size = 768`

### 问题3：内存不足

**症状**：`CUDA out of memory`

**解决方案**：
在`config.py`中减小batch_size：
```python
batch_size = 16  # 从32减小到16
```

### 问题4：无法下载模型

**症状**：`OSError: We couldn't connect to 'https://huggingface.co'`

**解决方案**：
使用本地模型路径，或配置HuggingFace镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 环境管理命令

### 退出环境
```bash
conda deactivate
```

### 删除环境（如需重新安装）
```bash
conda env remove -n privacy_ner
```

### 查看已安装的包
```bash
conda list
```

### 导出环境配置
```bash
conda env export > environment.yml
```

### 从配置文件创建环境
```bash
conda env create -f environment.yml
```

## 性能优化建议

1. **使用混合精度训练**：在`config.py`中设置`use_amp = True`
2. **调整batch size**：根据GPU显存大小调整
3. **使用梯度累积**：如果显存不足，增加`gradient_accumulation_steps`
4. **启用梯度检查点**：设置`use_gradient_checkpointing = True`节省显存

## 项目文件说明

- `config.py`：配置文件，包含所有超参数
- `model.py`：模型架构定义
- `data_loader.py`：数据加载器
- `train.py`：训练脚本
- `inference.py`：推理和评估脚本
- `demo.py`：演示脚本
- `data/`：数据集目录
  - `409_data_train.txt`：训练数据
  - `409_train_lable.txt`：训练标签
  - `409_data_test.txt`：测试数据
  - `409_test_lable.txt`：测试标签

## 技术支持

如遇到问题，请检查：
1. Python版本是否为3.9
2. CUDA版本是否与PyTorch匹配
3. 所有依赖包是否正确安装
4. 数据文件是否存在于`data/`目录

## 总结

通过以上步骤，您已经成功：
1. ✓ 创建了独立的conda环境
2. ✓ 安装了PyTorch及CUDA支持
3. ✓ 安装了所有项目依赖
4. ✓ 配置了模型路径
5. ✓ 成功运行了训练脚本

现在您可以开始训练自己的嵌套隐私实体识别模型了！

---

**创建日期**：2026-02-05
**适用版本**：Python 3.9, PyTorch 2.7.1, CUDA 11.8
