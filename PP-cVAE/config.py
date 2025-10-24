"""配置文件"""

import torch

class Config:
    """模型和训练配置"""
    
    # 数据配置
    DATA_DIR = '.'
    METADATA_PATH = 'filtered_metadata.csv'
    
    # 模型配置
    LATENT_DIM = 128
    NUM_PLANES = 9
    INPUT_CHANNELS = 2
    IMAGE_SIZE = 64
    
    # 训练配置
    BATCH_SIZE = 256  
    NUM_EPOCHS = 800
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # VAE损失权重
    VAE_BETA = 0.1
    VAE_WEIGHT = 0.01
    
    # 学习率调度
    LR_PATIENCE = 10
    LR_FACTOR = 0.5
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    SEED = 42
    
    # 保存路径
    MODEL_SAVE_PATH = 'models'
    RESULTS_SAVE_PATH = 'results'
    
    # 数据加载器配置
    NUM_WORKERS = 4  # GPU worker数量  根据设备改一下
    PIN_MEMORY = True 
    PREFETCH_FACTOR = 2  # 预取因子
    
    # Dropout率
    DROPOUT_RATE = 0.3
    
    # 混合精度训练
    USE_MIXED_PRECISION = True
    
    # GPU优化配置
    COMPILE_MODEL = False  # 暂时禁用torch.compile（与channels_last存在兼容性问题）
    CHANNELS_LAST = True  # 使用channels_last内存格式优化卷积操作
    
    # 快速调优模式（减少epoch数用于超参数搜索）
    FAST_TUNING_MODE = False
    FAST_TUNING_EPOCHS = 10
    
    # 评估指标配置
    EVALUATION_METRICS = ['r2', 'mse', 'rmse', 'mae']
    METRICS_DECIMAL_PLACES = 4  # 指标显示的小数位数

class ExperimentConfig(Config):
    """实验配置 - 继承基础配置并允许覆盖"""
    
    def __init__(self, **kwargs):
        # 首先初始化父类的所有属性
        super().__init__()
        
        # 然后通过关键字参数覆盖配置
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                setattr(self, key.upper(), value)

# 预定义的实验配置
EXPERIMENT_CONFIGS = {
    'baseline': {
        'latent_dim': 128,
        'learning_rate': 1e-3,
        'batch_size': 16,
        'vae_beta': 0.1,
        'vae_weight': 0.01
    },
    
    'high_lr': {
        'latent_dim': 128,
        'learning_rate': 5e-3,
        'batch_size': 16,
        'vae_beta': 0.1,
        'vae_weight': 0.01
    },
    
    'large_latent': {
        'latent_dim': 256,
        'learning_rate': 1e-3,
        'batch_size': 16,
        'vae_beta': 0.1,
        'vae_weight': 0.01
    },
    
    'small_batch': {
        'latent_dim': 128,
        'learning_rate': 1e-3,
        'batch_size': 8,
        'vae_beta': 0.1,
        'vae_weight': 0.01
    },
    
    'strong_vae': {
        'latent_dim': 128,
        'learning_rate': 1e-3,
        'batch_size': 16,
        'vae_beta': 1.0,
        'vae_weight': 0.1
    }
}
