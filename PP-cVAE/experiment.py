"""实验脚本 - 运行单个实验"""

import os
import random
import numpy as np
import torch
import argparse
from datetime import datetime

from config import Config, ExperimentConfig, EXPERIMENT_CONFIGS
from dataset import create_data_loaders
from models import create_model, count_parameters
from trainer import Trainer, Evaluator
from visualization import create_comprehensive_report
from utils import set_seed, create_experiment_dir

def run_experiment(experiment_name='baseline', config_overrides=None, save_results=True, base_config=None):
    """运行单个实验"""
    
    # 设置随机种子
    set_seed(42)
    
    # 创建配置
    if base_config is not None:
        # 从超参数调优传来的base_config，需要继承其属性
        config = ExperimentConfig(**(config_overrides or {}))
        # 复制base_config的重要属性
        for attr in ['FAST_TUNING_MODE', 'FAST_TUNING_EPOCHS', 'USE_MIXED_PRECISION']:
            if hasattr(base_config, attr):
                setattr(config, attr, getattr(base_config, attr))
    elif experiment_name in EXPERIMENT_CONFIGS:
        experiment_config = EXPERIMENT_CONFIGS[experiment_name].copy()
        if config_overrides:
            experiment_config.update(config_overrides)
        config = ExperimentConfig(**experiment_config)
    else:
        # 如果没有预定义配置，直接使用config_overrides
        config = ExperimentConfig(**(config_overrides or {}))
    
    print(f"\n=== Running Experiment: {experiment_name} ===")
    print(f"Device: {config.DEVICE}")
    print(f"Configuration:")
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    # 创建实验目录
    if save_results:
        experiment_dir = create_experiment_dir(experiment_name)
        print(f"Experiment directory: {experiment_dir}")
    else:
        experiment_dir = None
    
    try:
        # 创建数据加载器
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader, label_scaler = create_data_loaders(config)
        
        # 创建模型
        print("\nCreating model...")
        model = create_model(config)
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count:,}")
        
        # 创建训练器
        trainer = Trainer(model, config, save_dir=experiment_dir or 'temp')
        
        # 训练模型
        print("\nStarting training...")
        
        # 检查快速调优模式
        num_epochs = config.NUM_EPOCHS
        if getattr(config, 'FAST_TUNING_MODE', False):
            num_epochs = getattr(config, 'FAST_TUNING_EPOCHS', 10)
            print(f"Fast tuning mode: using {num_epochs} epochs instead of {config.NUM_EPOCHS}")
        
        trained_model = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
        
        # 保存模型
        if save_results:
            model_path, history_path = trainer.save_model(
                filename=f'{experiment_name}_model',
                experiment_name=experiment_name
            )
        
        # 评估模型
        print("\nEvaluating model...")
        evaluator = Evaluator(trained_model, config.DEVICE)
        
        # 训练集评估
        train_results = evaluator.evaluate(train_loader, label_scaler)
        print(f"\nTraining Results:")
        print(f"  R² Score: {train_results['r2']:.4f}")
        print(f"  MSE: {train_results['mse']:.4f}")
        print(f"  RMSE: {train_results['rmse']:.4f}")
        print(f"  MAE: {train_results['mae']:.4f}")

        
        # 验证集评估
        val_results = evaluator.evaluate(val_loader, label_scaler)
        print(f"\nValidation Results:")
        print(f"  R² Score: {val_results['r2']:.4f}")
        print(f"  MSE: {val_results['mse']:.4f}")
        print(f"  RMSE: {val_results['rmse']:.4f}")
        print(f"  MAE: {val_results['mae']:.4f}")

        
        # 测试集评估
        test_results = evaluator.evaluate(test_loader, label_scaler)
        print(f"\nTest Results:")
        print(f"  R² Score: {test_results['r2']:.4f}")
        print(f"  MSE: {test_results['mse']:.4f}")
        print(f"  RMSE: {test_results['rmse']:.4f}")
        print(f"  MAE: {test_results['mae']:.4f}")

        
        # 保存结果
        if save_results:
            # 保存评估结果
            evaluator.save_results(train_results, f'{experiment_name}_train', experiment_dir)
            evaluator.save_results(val_results, f'{experiment_name}_val', experiment_dir)
            evaluator.save_results(test_results, f'{experiment_name}_test', experiment_dir)
            
            # 创建可视化报告
            print("\nCreating visualization report...")
            create_comprehensive_report(
                trainer.train_losses,
                trainer.val_losses,
                trainer.val_r2_scores,
                test_results['predictions'],
                test_results['targets'],
                experiment_name=experiment_name,
                save_dir=experiment_dir
            )
        
        # 返回结果摘要
        results_summary = {
            'experiment_name': experiment_name,
            'config': config.__dict__,
            'model_parameters': param_count,
            'train_r2': train_results['r2'],
            'val_r2': val_results['r2'],
            'test_r2': test_results['r2'],
            'train_rmse': train_results['rmse'],
            'val_rmse': val_results['rmse'],
            'test_rmse': test_results['rmse'],
            'best_val_loss': trainer.best_val_loss,
            'experiment_dir': experiment_dir
        }
        
        print(f"\n=== Experiment {experiment_name} Completed Successfully ===")
        return results_summary
        
    except Exception as e:
        print(f"\nError in experiment {experiment_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='Run VAE-CNN experiment')
    parser.add_argument('--experiment', '-e', type=str, default='baseline',
                       help='Experiment name or config')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension')
    parser.add_argument('--no_save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # 构建配置覆盖
    config_overrides = {}
    if args.epochs:
        config_overrides['num_epochs'] = args.epochs
    if args.lr:
        config_overrides['learning_rate'] = args.lr
    if args.batch_size:
        config_overrides['batch_size'] = args.batch_size
    if args.latent_dim:
        config_overrides['latent_dim'] = args.latent_dim
    
    # 运行实验
    results = run_experiment(
        experiment_name=args.experiment,
        config_overrides=config_overrides,
        save_results=not args.no_save
    )
    
    if results:
        print(f"\nFinal Results Summary:")
        print(f"Test R² Score: {results['test_r2']:.4f}")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
        if results['experiment_dir']:
            print(f"Results saved to: {results['experiment_dir']}")
    else:
        print("Experiment failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
