"""
Configuration for H800 GPUs (7x 80GB)
针对7个H800 GPU优化的配置
"""
import torch

class Config:
    # ========== Model Settings ==========
    # 使用本地模型路径（需要先下载模型到此目录）
    model_name = "/workspace/models/chinese-roberta-wwm-ext"  # Chinese BERT model
    max_length = 512
    hidden_size = 768
    num_labels = 28  # 9 entity types × 3 prefixes (B/I/E) + O

    # Span-based settings
    max_span_length = 50  # Maximum entity span length
    span_hidden_size = 256
    biaffine_size = 256

    # ========== Training Settings - H800 Optimized ==========
    # H800有80GB显存，可以使用非常大的batch size
    batch_size = 64  # 每个GPU的batch size，7个GPU总共448
    learning_rate = 5e-5  # 大batch size需要更高的学习率
    weight_decay = 0.01
    num_epochs = 200
    warmup_ratio = 0.1
    gradient_accumulation_steps = 2  # 有效batch size = 64*7*2 = 896
    max_grad_norm = 1.0

    # ========== Performance Optimization ==========
    # 混合精度训练（AMP）- 显著提升训练速度
    use_amp = True  # 启用自动混合精度训练，可提速2-3倍

    # Gradient Checkpointing - H800显存充足，不需要启用
    use_gradient_checkpointing = False

    # 数据加载优化
    num_workers = 8  # 每个GPU的数据加载线程数
    prefetch_factor = 4  # 预取批次数，提高数据加载效率
    pin_memory = True  # 使用锁页内存加速数据传输

    # ========== Class Imbalance Handling ==========
    reduce_o_weight = True  # Reduce weight for 'O' class to focus on entities

    # ========== Adversarial Training ==========
    use_fgm = True  # Fast Gradient Method
    use_pgd = False  # PGD较慢，H800可以启用但建议先用FGM
    adv_epsilon = 1.0
    adv_alpha = 0.3
    adv_k = 3  # PGD steps

    # ========== Model Smoothing (R-Drop) ==========
    use_rdrop = True
    rdrop_alpha = 0.5  # KL散度权重

    # ========== Label Smoothing ==========
    label_smoothing = 0.0  # 如果训练稳定可以设置为0.1

    # ========== Device ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== Data Paths ==========
    train_data_path = "data/409_data_train.txt"
    train_label_path = "data/409_train_lable.txt"
    test_data_path = "data/409_data_test.txt"
    test_label_path = "data/409_test_lable.txt"

    # ========== Entity Types ==========
    entity_types = ['BI', 'CNU', 'EDU', 'JOB', 'JOB_ADDS', 'LOC', 'MS', 'PC', 'PI']

    # ========== Label Mapping ==========
    label2id = {'O': 0}
    id2label = {0: 'O'}
    idx = 1
    for entity_type in entity_types:
        for prefix in ['B', 'I', 'E']:
            label = f"{prefix}_{entity_type}"
            label2id[label] = idx
            id2label[idx] = label
            idx += 1

    # ========== Seed ==========
    seed = 42

    # ========== Resume Training ==========
    resume_from_checkpoint = None  # Set to 'best_model_h800.pt' to resume training
    resume_epoch = 0  # Will be automatically set when loading checkpoint

    # ========== Advanced Features ==========
    # Focal Loss
    use_focal_loss = False  # 可以尝试启用
    focal_gamma = 2.0
    focal_alpha = None

    # Data Augmentation
    use_data_augmentation = False  # 可以在后期启用
    aug_prob = 0.15

    # CRF Model
    use_crf_model = False
    freeze_bert_layers = 0
    lstm_hidden_size = 512
    lstm_num_layers = 2
    lstm_dropout = 0.1

    # ========== Optimization for China Network ==========
    # 确保所有模型和数据都在本地，避免网络下载
    local_files_only = True
    trust_remote_code = True

    # ========== Logging ==========
    log_interval = 10  # 每10个batch打印一次
    save_interval = 1  # 每个epoch保存一次

    # ========== Distributed Training Settings ==========
    # DDP backend
    dist_backend = 'nccl'  # NCCL是NVIDIA GPU的最佳选择
    dist_url = 'env://'  # 使用环境变量初始化

    # Gradient communication optimization
    bucket_cap_mb = 25  # DDP bucket size (MB)
    find_unused_parameters = False  # 设置为False可以提升性能
    broadcast_buffers = False  # 减少通信开销
    gradient_as_bucket_view = True  # 内存优化
