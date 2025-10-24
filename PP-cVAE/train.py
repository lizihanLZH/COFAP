"""主训练脚本 - 简化版本的主要入口点"""

import argparse
import os
from datetime import datetime

from config import Config, ExperimentConfig
from dataset import create_data_loaders, get_dataset_stats
from models import create_model, count_parameters
from trainer import Trainer, Evaluator
from visualization import create_comprehensive_report, show_plots
from utils import set_seed, print_device_info, print_model_info, create_experiment_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train VAE-CNN model')
    
    # 基本参数
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--experiment_name', type=str, default='train_run', 
                       help='Experiment name')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--metadata_path', type=str, help='Metadata file path')
    parser.add_argument('--normalize_labels', action='store_true', 
                       help='Normalize labels')
    
    # 模型参数
    parser.add_argument('--dropout_rate', type=float, help='Dropout rate')
    parser.add_argument('--vae_beta', type=float, help='VAE beta parameter')
    parser.add_argument('--vae_weight', type=float, help='VAE loss weight')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='experiments',
                       help='Save directory')
    parser.add_argument('--no_save', action='store_true', 
                       help='Do not save results')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots after training')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    config = Config()
    
    # 从命令行参数覆盖配置
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.latent_dim:
        config.LATENT_DIM = args.latent_dim
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.metadata_path:
        config.METADATA_PATH = args.metadata_path
    if args.dropout_rate:
        config.DROPOUT_RATE = args.dropout_rate
    if args.vae_beta:
        config.VAE_BETA = args.vae_beta
    if args.vae_weight:
        config.VAE_WEIGHT = args.vae_weight
    if args.device:
        import torch
        config.DEVICE = torch.device(args.device)
    
    print(f"\n=== VAE-CNN Training ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 打印设备信息
    print_device_info()
    
    # 打印配置
    print(f"\nConfiguration:")
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    # 创建实验目录
    if not args.no_save:
        experiment_dir = create_experiment_dir(args.experiment_name, args.save_dir)
        print(f"\nExperiment directory: {experiment_dir}")
    else:
        experiment_dir = None
    
    try:
        # 数据集统计
        print(f"\nDataset Statistics:")
        stats = get_dataset_stats(config.METADATA_PATH)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 创建数据加载器
        print(f"\nCreating data loaders...")
        train_loader, val_loader, test_loader, label_scaler = create_data_loaders(
            config, normalize_labels=args.normalize_labels
        )
        
        # 创建模型
        print(f"\nCreating model...")
        model = create_model(config)
        
        # 打印模型信息
        print_model_info(model)
        
        # 创建训练器
        trainer = Trainer(model, config, save_dir=experiment_dir or 'temp')
        
        # 训练模型
        print(f"\nStarting training...")
        print(f"Training for {config.NUM_EPOCHS} epochs...")
        
        trained_model = trainer.train(train_loader, val_loader)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
        # 保存模型
        if not args.no_save:
            model_path, history_path = trainer.save_model(
                filename=f'{args.experiment_name}_model',
                experiment_name=args.experiment_name
            )
        
        # 评估模型
        print(f"\nEvaluating model...")
        evaluator = Evaluator(trained_model, config.DEVICE)
        
        # 训练集评估
        print(f"\nEvaluating on training set...")
        train_results = evaluator.evaluate(train_loader, label_scaler)
        print(f"Training Results:")
        print(f"  R² Score: {train_results['r2']:.4f}")
        print(f"  MSE: {train_results['mse']:.4f}")
        print(f"  RMSE: {train_results['rmse']:.4f}")
        print(f"  MAE: {train_results['mae']:.4f}")

        
        # 验证集评估
        print(f"\nEvaluating on validation set...")
        val_results = evaluator.evaluate(val_loader, label_scaler)
        print(f"Validation Results:")
        print(f"  R² Score: {val_results['r2']:.4f}")
        print(f"  MSE: {val_results['mse']:.4f}")
        print(f"  RMSE: {val_results['rmse']:.4f}")
        print(f"  MAE: {val_results['mae']:.4f}")

        
        # 测试集评估
        print(f"\nEvaluating on test set...")
        test_results = evaluator.evaluate(test_loader, label_scaler)
        print(f"Test Results:")
        print(f"  R² Score: {test_results['r2']:.4f}")
        print(f"  MSE: {test_results['mse']:.4f}")
        print(f"  RMSE: {test_results['rmse']:.4f}")
        print(f"  MAE: {test_results['mae']:.4f}")

        
        # 保存评估结果
        if not args.no_save:
            evaluator.save_results(train_results, f'{args.experiment_name}_train', experiment_dir)
            evaluator.save_results(val_results, f'{args.experiment_name}_val', experiment_dir)
            evaluator.save_results(test_results, f'{args.experiment_name}_test', experiment_dir)
        
        # 创建可视化报告
        if not args.no_save:
            print(f"\nCreating visualization report...")
            create_comprehensive_report(
                trainer.train_losses,
                trainer.val_losses,
                trainer.val_r2_scores,
                test_results['predictions'],
                test_results['targets'],
                experiment_name=args.experiment_name,
                save_dir=experiment_dir
            )
        
        # 显示图表
        if args.show_plots:
            show_plots()
        
        print(f"\n=== Training Completed Successfully ===")
        print(f"Final Test R² Score: {test_results['r2']:.4f}")
        print(f"Final Test RMSE: {test_results['rmse']:.4f}")
        
        if experiment_dir:
            print(f"Results saved to: {experiment_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
