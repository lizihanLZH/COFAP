class ExactFusionConfig:
    def __init__(self):
        self.vae_config = {
            'latent_dim': 128,
            'num_planes': 9,
            'dropout_rate': 0.3
        }
        self.fusion_dim = 128
        self.use_descriptors = False
        self.descriptor_dim = 0
        self.use_batch_norm = True
        self.use_layer_norm = True
        self.use_residual = True
        self.feature_regularization = 0.01
        self.offset_correction = True
        self.main_weight = 0.8
        self.fusion_weight = 0.2
        self.reg_weight = 0.01
        self.offset_weight = 0.1
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.dropout_rate = 0.3
        self.early_stopping_patience = 15
        self.max_samples = 1000
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.vae_model_path = 'PP-cVAE/models/best_model.pth'
        self.gcn_cc_model_path = 'BiG-CAE/models/contrastive_model_cc.pth'
        self.gcn_noncc_model_path = 'BiG-CAE/models/contrastive_model_noncc.pth'
        self.lzhnn_model_path = 'PP-NN/final_model_42_RPSA.pth'
        self.output_dir = 'test_exact_output_RPSA'
    def get_model_config(self):
        return {
            'vae_config': self.vae_config,
            'use_descriptors': self.use_descriptors,
            'descriptor_dim': self.descriptor_dim,
            'fusion_dim': self.fusion_dim,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'use_residual': self.use_residual,
            'feature_regularization': self.feature_regularization,
            'offset_correction': self.offset_correction
        }
    def get_loss_config(self):
        return {
            'main_weight': self.main_weight,
            'fusion_weight': self.fusion_weight,
            'reg_weight': self.reg_weight,
            'offset_weight': self.offset_weight
        }
    def get_training_config(self):
        return {
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'early_stopping_patience': self.early_stopping_patience
        }
    def get_data_config(self):
        return {
            'max_samples': self.max_samples,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio
        }
    def get_model_paths(self):
        return {
            'vae': self.vae_model_path,
            'gcn_cc': self.gcn_cc_model_path,
            'gcn_noncc': self.gcn_noncc_model_path,
            'lzhnn': self.lzhnn_model_path
        }
    def update_from_args(self, args):
        if hasattr(args, 'vae_model_path'):
            self.vae_model_path = args.vae_model_path
        if hasattr(args, 'gcn_cc_model_path'):
            self.gcn_cc_model_path = args.gcn_cc_model_path
        if hasattr(args, 'gcn_noncc_model_path'):
            self.gcn_noncc_model_path = args.gcn_noncc_model_path
        if hasattr(args, 'lzhnn_model_path'):
            self.lzhnn_model_path = args.lzhnn_model_path
        if hasattr(args, 'batch_size'):
            self.batch_size = args.batch_size
        if hasattr(args, 'num_epochs'):
            self.num_epochs = args.num_epochs
        if hasattr(args, 'learning_rate'):
            self.learning_rate = args.learning_rate
        if hasattr(args, 'max_samples'):
            self.max_samples = args.max_samples
        if hasattr(args, 'output_dir'):
            self.output_dir = args.output_dir
        if hasattr(args, 'weight_decay'):
            self.weight_decay = args.weight_decay
        if hasattr(args, 'dropout_rate'):
            self.dropout_rate = args.dropout_rate
        if hasattr(args, 'early_stopping_patience'):
            self.early_stopping_patience = args.early_stopping_patience
    def print_config(self):
        print("=" * 60)
        print("融合模型配置")
        print("=" * 60)
        print("模型配置:")
        model_config = self.get_model_config()
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        print("\n损失函数配置:")
        loss_config = self.get_loss_config()
        for key, value in loss_config.items():
            print(f"  {key}: {value}")
        print("\n训练配置:")
        training_config = self.get_training_config()
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        print("\n数据配置:")
        data_config = self.get_data_config()
        for key, value in data_config.items():
            print(f"  {key}: {value}")
        print("\n模型路径:")
        model_paths = self.get_model_paths()
        for key, value in model_paths.items():
            print(f"  {key}: {value}")
        print("=" * 60)
def get_default_config():
    return ExactFusionConfig()
def get_high_regularization_config():
    config = ExactFusionConfig()
    config.feature_regularization = 0.05
    config.reg_weight = 0.05
    config.offset_weight = 0.2
    config.weight_decay = 1e-3
    config.dropout_rate = 0.5
    return config
def get_low_regularization_config():
    config = ExactFusionConfig()
    config.feature_regularization = 0.001
    config.reg_weight = 0.001
    config.offset_weight = 0.05
    config.weight_decay = 1e-5
    config.dropout_rate = 0.1
    return config
def get_balanced_config():
    config = ExactFusionConfig()
    config.feature_regularization = 0.02
    config.reg_weight = 0.02
    config.offset_weight = 0.15
    config.weight_decay = 5e-4
    config.dropout_rate = 0.3
    return config 