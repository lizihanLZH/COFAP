"""工具函数模块"""

import os
import random
import numpy as np
import torch
import json
from datetime import datetime
import shutil

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_experiment_dir(experiment_name, base_dir='experiments'):
    """创建实验目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'{experiment_name}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    
    return experiment_dir

def save_config(config, filepath):
    """保存配置到文件"""
    config_dict = {}
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            # 转换不可序列化的对象
            if hasattr(value, '__dict__'):
                config_dict[key] = str(value)
            elif isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config(filepath):
    """从文件加载配置"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_device_info():
    """获取设备信息"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }
    return device_info

def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    print("Device Information:")
    print(f"  CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  Device Count: {info['device_count']}")
        print(f"  Current Device: {info['current_device']}")
        print(f"  Device Name: {info['device_name']}")
    else:
        print("  Using CPU")

def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def get_model_size(model):
    """获取模型大小信息"""
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    buffer_count = 0
    
    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_count': param_count,
        'param_size_mb': param_size / 1024**2,
        'buffer_count': buffer_count,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }

def print_model_info(model):
    """打印模型信息"""
    info = get_model_size(model)
    print("Model Information:")
    print(f"  Parameters: {info['param_count']:,}")
    print(f"  Parameter Size: {info['param_size_mb']:.2f} MB")
    print(f"  Buffer Size: {info['buffer_size_mb']:.2f} MB")
    print(f"  Total Size: {info['total_size_mb']:.2f} MB")

def cleanup_experiments(base_dir='experiments', keep_latest=5):
    """清理旧的实验目录"""
    if not os.path.exists(base_dir):
        return
    
    # 获取所有实验目录
    exp_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            exp_dirs.append((item_path, os.path.getctime(item_path)))
    
    # 按创建时间排序
    exp_dirs.sort(key=lambda x: x[1], reverse=True)
    
    # 删除旧的目录
    if len(exp_dirs) > keep_latest:
        for dir_path, _ in exp_dirs[keep_latest:]:
            print(f"Removing old experiment directory: {dir_path}")
            shutil.rmtree(dir_path)

def check_data_integrity(metadata_path, data_dir):
    """检查数据完整性"""
    import pandas as pd
    import h5py
    
    print("Checking data integrity...")
    
    # 读取元数据
    metadata = pd.read_csv(metadata_path)
    
    missing_files = []
    corrupted_files = []
    total_files = len(metadata.groupby('h5_file'))
    
    for i, (h5_file, group) in enumerate(metadata.groupby('h5_file')):
        if i % 100 == 0:
            print(f"Checked {i}/{total_files} files...")
        
        # 构建文件路径
        if 'images1/train/' in h5_file:
            file_path = h5_file.replace('images1/train/', 'train/')
        elif 'images1/val/' in h5_file:
            file_path = h5_file.replace('images1/val/', 'val/')
        elif 'images1/test/' in h5_file:
            file_path = h5_file.replace('images1/test/', 'test/')
        else:
            file_path = h5_file
        
        full_path = os.path.join(data_dir, file_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            missing_files.append(h5_file)
            continue
        
        # 检查文件是否可读
        try:
            with h5py.File(full_path, 'r') as f:
                # 检查是否包含所有9个平面
                for plane_idx in range(9):
                    if f'plane_{plane_idx}' not in f:
                        corrupted_files.append((h5_file, f'Missing plane_{plane_idx}'))
                        break
                    
                    # 检查数据形状
                    plane_data = f[f'plane_{plane_idx}']
                    if plane_data.shape != (2, 64, 64):
                        corrupted_files.append((h5_file, f'Wrong shape for plane_{plane_idx}: {plane_data.shape}'))
                        break
        except Exception as e:
            corrupted_files.append((h5_file, f'Read error: {str(e)}'))
    
    print(f"\nData integrity check completed:")
    print(f"  Total files: {total_files}")
    print(f"  Missing files: {len(missing_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if missing_files:
        print(f"\nFirst 10 missing files:")
        for file in missing_files[:10]:
            print(f"  {file}")
    
    if corrupted_files:
        print(f"\nFirst 10 corrupted files:")
        for file, error in corrupted_files[:10]:
            print(f"  {file}: {error}")
    
    return {
        'total_files': total_files,
        'missing_files': missing_files,
        'corrupted_files': corrupted_files,
        'integrity_ok': len(missing_files) == 0 and len(corrupted_files) == 0
    }

def create_project_structure():
    """创建项目目录结构"""
    directories = [
        'experiments',
        'results',
        'models',
        'logs',
        'data',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def get_git_info():
    """获取Git信息（如果可用）"""
    try:
        import subprocess
        
        # 获取当前commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # 获取分支名
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # 检查是否有未提交的更改
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        return {
            'commit_hash': commit_hash,
            'branch': branch,
            'has_uncommitted_changes': len(status) > 0,
            'status': status
        }
    except:
        return None