"""
Configuration for Nested Privacy Entity Recognition
针对H800 80GB显存深度优化，目标F1 > 95%
"""
import torch

class Config:
    # ========== H800 80GB 深度优化配置 ==========

    # Model settings - 使用更强大的模型
    # 选项1: 本地RoBERTa-large (如果没有下载大模型)
    # model_name = "/root/nvme4n1/docker-data/overlay2/c9f7b6741dfcc1013d590cdaba3b9686650ef2eb2c2efe9b218b7f83b9208a37/diff/workspace/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    # hidden_size = 1024

    # 选项2: Erlangshen-MegatronBert-1.3B (13亿参数，推荐)
    model_name = "/root/cw_ws/weishenfuyong/models/Erlangshen-MegatronBert-1.3B"
    hidden_size = 2048

    # 选项3: 如果上面的模型下载失败，使用MacBERT-Large
    # model_name = "hfl/chinese-macbert-large"
    # hidden_size = 1024

    max_length = 512
    num_labels = 28  # 9 entity types × 3 prefixes (B/I/E) + O

    # Span-based settings - 增大以提升性能
    max_span_length = 50
    span_hidden_size = 512      # 从256增加到512
    biaffine_size = 512         # 从256增加到512

    # ========== H800 80GB 训练参数优化 ==========
    batch_size = 16             # 1.3B模型需要较小batch_size
    learning_rate = 2e-5        # 稍微提高学习率
    weight_decay = 0.01
    num_epochs = 100            # 大模型收敛更快
    warmup_ratio = 0.1
    gradient_accumulation_steps = 16  # 有效batch_size = 256
    max_grad_norm = 1.0

    # Early stopping
    early_stopping_patience = 15  # 增加耐心，大模型需要更多时间

    # 混合精度训练 - H800支持BF16，更稳定
    use_amp = True
    amp_dtype = "bfloat16"      # H800推荐使用bfloat16

    # Gradient Checkpointing - 大模型必须启用
    use_gradient_checkpointing = True  # 1.3B模型需要启用以节省显存

    # Class imbalance handling
    reduce_o_weight = True

    # ========== 对抗训练 - 提升鲁棒性 ==========
    use_fgm = True
    use_pgd = False             # 1.3B模型只用FGM，避免OOM
    adv_epsilon = 1.0           # 增大扰动
    adv_alpha = 0.3
    adv_k = 3

    # ========== 正则化技术 ==========
    use_rdrop = True
    rdrop_alpha = 0.7           # 增大R-Drop强度

    # Label smoothing
    label_smoothing = 0.1

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    train_data_path = "data/409_data_train.txt"
    train_label_path = "data/409_train_lable.txt"
    test_data_path = "data/409_data_test.txt"
    test_label_path = "data/409_test_lable.txt"

    # Entity types
    entity_types = ['BI', 'CNU', 'EDU', 'JOB', 'JOB_ADDS', 'LOC', 'MS', 'PC', 'PI']

    # Label mapping
    label2id = {'O': 0}
    id2label = {0: 'O'}
    idx = 1
    for entity_type in entity_types:
        for prefix in ['B', 'I', 'E']:
            label = f"{prefix}_{entity_type}"
            label2id[label] = idx
            id2label[idx] = label
            idx += 1

    # Seed
    seed = 42

    # Resume training
    resume_from_checkpoint = None

    # ========== 高级特性 ==========

    # Focal Loss - 处理类别不平衡
    use_focal_loss = True
    focal_gamma = 2.0
    focal_alpha = None

    # Data Augmentation - 启用数据增强
    use_data_augmentation = True   # 启用数据增强
    aug_prob = 0.2                 # 增强概率

    # CRF Model
    use_crf_model = False
    freeze_bert_layers = 0
    lstm_hidden_size = 512
    lstm_num_layers = 2
    lstm_dropout = 0.1

    # ========== 新增：模型集成与高级优化 ==========

    # 多头注意力增强
    use_multi_head_attention = True
    num_attention_heads = 8

    # 层次化特征融合
    use_layer_fusion = True        # 融合BERT多层特征
    fusion_layers = [-1, -2, -3, -4]  # 使用最后4层

    # Dropout设置
    hidden_dropout = 0.1
    attention_dropout = 0.1
    classifier_dropout = 0.3

    # 对比学习（可选）
    use_contrastive_loss = False
    contrastive_temperature = 0.07

    # 知识蒸馏（可选，用于后续优化）
    use_knowledge_distillation = False
    teacher_model_path = None
    distill_temperature = 4.0
    distill_alpha = 0.5
