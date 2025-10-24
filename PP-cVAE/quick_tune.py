"""快速超参数调优示例脚本"""

import os
import subprocess
import sys
from datetime import datetime

def run_quick_hyperparameter_tuning():
    """运行快速超参数调优"""
    
    print("=== 快速超参数调优 ===")
    print("使用混合精度训练和减少的epoch数来加速调优过程")
    print("")
    
    # 确保在torch环境中
    print("激活conda torch环境...")
    
    # 快速网格搜索
    print("\n1. 运行快速网格搜索 (每个试验10个epoch)...")
    grid_cmd = [
        "python", "tune_model.py",
        "--method", "grid",
        "--trials", "12",  # 限制试验数量
        "--fast_mode",
        "--fast_epochs", "10",
        "--metric", "test_r2",
        "--save_dir", "quick_tuning_results"
    ]
    
    try:
        result = subprocess.run(grid_cmd, capture_output=True, text=True, check=True)
        print("网格搜索完成!")
        print(result.stdout[-500:])  # 显示最后500个字符
    except subprocess.CalledProcessError as e:
        print(f"网格搜索失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    # 快速随机搜索
    print("\n2. 运行快速随机搜索 (每个试验10个epoch)...")
    random_cmd = [
        "python", "tune_model.py",
        "--method", "random",
        "--trials", "15",
        "--fast_mode",
        "--fast_epochs", "10",
        "--metric", "test_r2",
        "--save_dir", "quick_tuning_results"
    ]
    
    try:
        result = subprocess.run(random_cmd, capture_output=True, text=True, check=True)
        print("随机搜索完成!")
        print(result.stdout[-500:])  # 显示最后500个字符
    except subprocess.CalledProcessError as e:
        print(f"随机搜索失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    print("\n=== 快速调优完成 ===")
    print("结果保存在 'quick_tuning_results' 目录中")
    print("查看 'best_config.json' 文件获取最佳配置")
    
    return True

def run_full_training_with_best_config():
    """使用最佳配置运行完整训练"""
    
    print("\n=== 使用最佳配置进行完整训练 ===")
    
    # 检查是否存在最佳配置文件
    best_config_path = "quick_tuning_results/best_config.json"
    if not os.path.exists(best_config_path):
        print(f"未找到最佳配置文件: {best_config_path}")
        print("请先运行快速调优")
        return False
    
    # 读取最佳配置
    import json
    with open(best_config_path, 'r') as f:
        best_config = json.load(f)
    
    best_params = best_config['best_params']
    print(f"最佳参数: {best_params}")
    
    # 构建训练命令
    train_cmd = ["python", "train.py"]
    
    # 添加参数
    if 'epochs' in best_params:
        train_cmd.extend(["--epochs", str(best_params['epochs'])])
    else:
        train_cmd.extend(["--epochs", "100"])  # 默认完整训练
    
    if 'learning_rate' in best_params:
        train_cmd.extend(["--lr", str(best_params['learning_rate'])])
    
    if 'batch_size' in best_params:
        train_cmd.extend(["--batch_size", str(best_params['batch_size'])])
    
    if 'latent_dim' in best_params:
        train_cmd.extend(["--latent_dim", str(best_params['latent_dim'])])
    
    if 'dropout_rate' in best_params:
        train_cmd.extend(["--dropout_rate", str(best_params['dropout_rate'])])
    
    if 'vae_beta' in best_params:
        train_cmd.extend(["--vae_beta", str(best_params['vae_beta'])])
    
    # 添加设备和实验名称
    train_cmd.extend([
        "--device", "cuda",
        "--experiment_name", f"best_config_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ])
    
    print(f"运行命令: {' '.join(train_cmd)}")
    
    try:
        result = subprocess.run(train_cmd, check=True)
        print("完整训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False

def main():
    """主函数"""
    
    print("快速超参数调优和训练脚本")
    print("这个脚本将:")
    print("1. 运行快速超参数调优 (减少epoch数)")
    print("2. 使用最佳配置进行完整训练")
    print("3. 使用混合精度训练加速")
    print("")
    
    # 运行快速调优
    if run_quick_hyperparameter_tuning():
        # 询问是否进行完整训练
        response = input("\n是否使用最佳配置进行完整训练? (y/n): ")
        if response.lower() in ['y', 'yes', '是']:
            run_full_training_with_best_config()
    
    print("\n脚本执行完成!")

if __name__ == '__main__':
    main()