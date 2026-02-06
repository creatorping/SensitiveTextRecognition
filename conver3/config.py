"""
Configuration for Nested Privacy Entity Recognition
"""
import torch

class Config:
    # Model settings
    # 使用本地模型路径
    model_name = "/root/nvme4n1/docker-data/overlay2/c9f7b6741dfcc1013d590cdaba3b9686650ef2eb2c2efe9b218b7f83b9208a37/diff/workspace/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"  # Chinese BERT model
    max_length = 512
    hidden_size = 1024  # 使用large模型，hidden_size为1024
    num_labels = 28  # 9 entity types (BI, CNU, EDU, JOB, JOB_ADDS, LOC, MS, PC, PI) × 3 prefixes (B/I/E) + O

    # Span-based settings
    max_span_length = 50  # Maximum entity span length
    span_hidden_size = 256
    biaffine_size = 256

    # Training settings - 针对双H800优化
    batch_size = 64  # Optimized for H800 80GB memory
    learning_rate = 3e-5  # Increased from 1e-5 for better convergence
    weight_decay = 0.01
    num_epochs = 200
    warmup_ratio = 0.1
    gradient_accumulation_steps = 1  # Effective batch size = 32*2*2 = 128 (with 2 GPUs)
    max_grad_norm = 1.0

    # 混合精度训练（AMP）- 显著提升训练速度
    use_amp = True  # 启用自动混合精度训练，可提速2-3倍

    # Gradient Checkpointing - 节省显存，可训练更大模型
    use_gradient_checkpointing = False  # Enable to save memory

    # Class imbalance handling
    reduce_o_weight = True  # Reduce weight for 'O' class to focus on entities

    # Adversarial training
    use_fgm = False  # Temporarily disabled due to gradient issues
    use_pgd = False
    adv_epsilon = 1.0
    adv_alpha = 0.3
    adv_k = 3  # PGD steps

    # Model smoothing (R-Drop)
    use_rdrop = False  # Disabled to reduce memory usage (doubles forward passes)
    rdrop_alpha = 0.5  # Reduced from 4.0 to prevent instability

    # Label smoothing
    label_smoothing = 0.0  # Disabled to prevent NaN

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
    resume_from_checkpoint = None  # Set to 'best_model.pt' to resume training
    resume_epoch = 0  # Will be automatically set when loading checkpoint

    # ========== Advanced Features (New) ==========

    # Focal Loss
    use_focal_loss = True          # Use Focal Loss instead of CE Loss
    focal_gamma = 2.0              # Focusing parameter (higher = focus more on hard examples)
    focal_alpha = None             # Will use class_weights if None

    # Data Augmentation
    use_data_augmentation = False  # Enable data augmentation (set True after initial training)
    aug_prob = 0.15                # Probability of applying augmentation

    # CRF Model
    use_crf_model = False          # Use BiLSTM-CRF model (set True for better performance)
    freeze_bert_layers = 0         # Number of BERT layers to freeze (0 = train all)
    lstm_hidden_size = 512         # BiLSTM hidden size
    lstm_num_layers = 2            # Number of BiLSTM layers
    lstm_dropout = 0.1             # BiLSTM dropout
