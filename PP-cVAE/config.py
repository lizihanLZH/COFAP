import torch
class Config:
    DATA_DIR = '.'
    METADATA_PATH = 'filtered_metadata.csv'
    LATENT_DIM = 128
    NUM_PLANES = 9
    INPUT_CHANNELS = 2
    IMAGE_SIZE = 64
    BATCH_SIZE = 256  
    NUM_EPOCHS = 800
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    VAE_BETA = 0.1
    VAE_WEIGHT = 0.01
    LR_PATIENCE = 10
    LR_FACTOR = 0.5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    MODEL_SAVE_PATH = 'models'
    RESULTS_SAVE_PATH = 'results'
    NUM_WORKERS = 4  
    PIN_MEMORY = True 
    PREFETCH_FACTOR = 2  
    DROPOUT_RATE = 0.3
    USE_MIXED_PRECISION = True
    COMPILE_MODEL = False  
    CHANNELS_LAST = True  
    FAST_TUNING_MODE = False
    FAST_TUNING_EPOCHS = 10
    EVALUATION_METRICS = ['r2', 'mse', 'rmse', 'mae']
    METRICS_DECIMAL_PLACES = 4  
class ExperimentConfig(Config):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                setattr(self, key.upper(), value)
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