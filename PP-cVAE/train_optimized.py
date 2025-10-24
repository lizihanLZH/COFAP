"""优化的训练脚本 - 提高GPU利用率"""

import torch
import os
import warnings
warnings.filterwarnings('ignore')

from config import Config
from dataset import create_data_loaders
from models import create_model, count_parameters
from trainer import Trainer
from utils import set_seed

def main():
    """主训练函数"""
    
    # 配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.SEED)
    
    print("=== 优化训练配置 ===")
    print(f"设备: {config.DEVICE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"工作进程数: {config.NUM_WORKERS}")
    print(f"混合精度训练: {config.USE_MIXED_PRECISION}")
    print(f"模型编译: {getattr(config, 'COMPILE_MODEL', False)}")
    print(f"Channels Last: {getattr(config, 'CHANNELS_LAST', False)}")
    print(f"Pin Memory: {getattr(config, 'PIN_MEMORY', False)}")
    
    # GPU信息
    if torch.cuda.is_available():
        print(f"\nGPU信息:")
        print(f"  GPU名称: {torch.cuda.get_device_name()}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  当前内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 创建数据加载器
    print("\n=== 加载数据 ===")
    try:
        train_loader, val_loader, test_loader, label_scaler = create_data_loaders(
            config, 
            normalize_labels=True,
            use_descriptors=True,  # 启用多模态
            descriptor_path='descriptor.csv'
        )
        
        # 检查数据格式
        sample_batch = next(iter(train_loader))
        if len(sample_batch) == 3:
            planes, descriptors, labels = sample_batch
            descriptor_dim = descriptors.shape[1]
            use_multimodal = True
            print(f"✓ 多模态数据加载成功")
            print(f"  投影平面形状: {planes.shape}")
            print(f"  描述符形状: {descriptors.shape}")
            print(f"  描述符维度: {descriptor_dim}")
        else:
            planes, labels = sample_batch
            descriptor_dim = 0
            use_multimodal = False
            print(f"✓ 单模态数据加载成功")
            print(f"  投影平面形状: {planes.shape}")
        
        print(f"  标签形状: {labels.shape}")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print(f"  测试批次数: {len(test_loader)}")
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
    
    # 创建模型
    print("\n=== 创建模型 ===")
    try:
        model = create_model(
            config, 
            descriptor_dim=descriptor_dim,
            use_multimodal=use_multimodal
        )
        model = model.to(config.DEVICE)
        
        print(f"✓ 模型创建成功")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  参数数量: {count_parameters(model):,}")
        print(f"  使用多模态: {use_multimodal}")
        
        # 显示模型内存使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"  模型内存使用: {model_memory:.2f} GB")
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return
    
    # 创建训练器
    print("\n=== 初始化训练器 ===")
    try:
        trainer = Trainer(model, config, save_dir='optimized_experiments')
        print(f"✓ 训练器初始化成功")
        print(f"  优化器: {type(trainer.optimizer).__name__}")
        print(f"  学习率调度器: {type(trainer.scheduler).__name__}")
        print(f"  混合精度: {config.USE_MIXED_PRECISION}")
        
    except Exception as e:
        print(f"✗ 训练器初始化失败: {e}")
        return
    
    # 开始训练
    print("\n=== 开始训练 ===")
    print(f"训练轮数: {config.NUM_EPOCHS}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"权重衰减: {config.WEIGHT_DECAY}")
    print(f"VAE权重: {config.VAE_WEIGHT}")
    print(f"VAE Beta: {config.VAE_BETA}")
    
    try:
        # 训练前的GPU内存状态
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"\n训练前GPU内存: {initial_memory:.2f} GB")
        
        # 开始训练
        trained_model = trainer.train(train_loader, val_loader, num_epochs=config.NUM_EPOCHS)
        
        print("\n=== 训练完成 ===")
        print(f"✓ 训练成功完成")
        print(f"  最终训练损失: {trainer.train_losses[-1]:.6f}")
        print(f"  最终验证损失: {trainer.val_losses[-1]:.6f}")
        print(f"  最终验证R2: {trainer.val_r2_scores[-1]:.6f}")
        print(f"  最佳验证损失: {min(trainer.val_losses):.6f}")
        print(f"  最佳验证R2: {max(trainer.val_r2_scores):.6f}")
        
        # 训练后的GPU内存状态
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\nGPU内存使用:")
            print(f"  当前使用: {final_memory:.2f} GB")
            print(f"  峰值使用: {max_memory:.2f} GB")
            print(f"  内存效率: {(max_memory / torch.cuda.get_device_properties(0).total_memory * 1024**3) * 100:.1f}%")
        
        # 保存模型
        model_path = trainer.save_model(experiment_name='optimized_multimodal')
        print(f"\n✓ 模型已保存: {model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠ 训练被用户中断")
        # 保存当前模型
        model_path = trainer.save_model(experiment_name='optimized_multimodal_interrupted')
        print(f"✓ 中断时模型已保存: {model_path}")
        
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 优化训练完成！")
    print("\n性能优化总结:")
    print(f"  ✓ 批次大小增加到 {config.BATCH_SIZE}")
    print(f"  ✓ 数据加载器使用 {config.NUM_WORKERS} 个工作进程")
    print(f"  ✓ 启用了 pin_memory 和 persistent_workers")
    print(f"  ✓ 混合精度训练: {config.USE_MIXED_PRECISION}")
    if getattr(config, 'COMPILE_MODEL', False):
        print(f"  ✓ 模型编译优化已启用")
    if getattr(config, 'CHANNELS_LAST', False):
        print(f"  ✓ Channels Last 内存格式优化已启用")
    if use_multimodal:
        print(f"  ✓ 多模态架构集成了 {descriptor_dim} 维描述符")

def check_gpu_optimization():
    """检查GPU优化配置"""
    print("=== GPU优化检查 ===")
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        # 检查编译支持
        try:
            torch.compile
            print("✓ torch.compile 支持")
        except AttributeError:
            print("✗ torch.compile 不支持 (需要 PyTorch 2.0+)")
        
        # 检查混合精度支持
        if torch.cuda.is_bf16_supported():
            print("✓ BF16 混合精度支持")
        else:
            print("⚠ BF16 不支持，使用 FP16")
        
        # 检查内存格式支持
        print("✓ Channels Last 内存格式支持")
        
    else:
        print("✗ CUDA 不可用")
    
    print()

if __name__ == "__main__":
    # 检查优化配置
    check_gpu_optimization()
    
    # 开始训练
    main()