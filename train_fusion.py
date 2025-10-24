import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from fusion_model import ExactCrossAttentionFusion, create_exact_fusion_model
from data_loader import ExactDataLoader, create_exact_data_loader


sys.path.append('PP-cVAE')

try:

    original_path = sys.path.copy()
    sys.path.insert(0, 'PP-cVAE')
    
    from config import Config as VAEConfig
    

    sys.path = original_path
    
except ImportError:
    print("使用默认配置")
    class VAEConfig:
        LATENT_DIM = 128
        NUM_PLANES = 9
        DROPOUT_RATE = 0.3

class ExactFusionDataset(Dataset):
    """
    基于原始数据格式的融合数据集
    """
    
    def __init__(self, data_dict):
        self.vae_data = data_dict['vae_data']
        self.descriptors = data_dict['descriptors']
        self.graph_cc = data_dict['graph_cc']
        self.graph_noncc = data_dict['graph_noncc']
        self.lzhnn_data = data_dict['lzhnn_data']
        self.cc_mask = data_dict['cc_mask']
        self.targets = data_dict['targets']
        self.identifiers = data_dict['identifiers']
        
    def __len__(self):
        return len(self.vae_data)
    
    def __getitem__(self, idx):
        sample = {
            'vae_data': self.vae_data[idx],
            'graph_cc': self.graph_cc[idx] if self.graph_cc else None,
            'graph_noncc': self.graph_noncc[idx] if self.graph_noncc else None,
            'lzhnn_data': {
                'topo': self.lzhnn_data['topo'][idx],
                'struct': self.lzhnn_data['struct'][idx]
            },
            'cc_mask': self.cc_mask[idx],
            'targets': self.targets[idx],
            'identifier': self.identifiers[idx]
        }
        
        if self.descriptors is not None:
            sample['descriptors'] = self.descriptors[idx]
            
        return sample

def exact_collate_fn(batch):
    """
    基于原始数据格式的collate函数
    """
    # 处理tensor数据
    vae_data = torch.stack([item['vae_data'] for item in batch])
    cc_mask = torch.stack([item['cc_mask'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    
    # 处理lzhnn数据
    lzhnn_data = {
        'topo': torch.stack([item['lzhnn_data']['topo'] for item in batch]),
        'struct': torch.stack([item['lzhnn_data']['struct'] for item in batch])
    }
    
    # 处理图数据
    graph_cc = [item['graph_cc'] for item in batch]
    graph_noncc = [item['graph_noncc'] for item in batch]
    
    # 处理描述符
    descriptors = None
    if batch[0].get('descriptors') is not None:
        descriptors = torch.stack([item['descriptors'] for item in batch])
    
    # 处理标识符
    identifiers = [item['identifier'] for item in batch]
    
    return {
        'vae_data': vae_data,
        'descriptors': descriptors,
        'graph_cc': graph_cc,
        'graph_noncc': graph_noncc,
        'lzhnn_data': lzhnn_data,
        'cc_mask': cc_mask,
        'targets': targets,
        'identifiers': identifiers
    }

class ExactFusionLoss(nn.Module):
    """
    基于原始损失函数的融合损失
    """
    
    def __init__(self, main_weight=0.8, fusion_weight=0.2):
        super(ExactFusionLoss, self).__init__()
        self.main_weight = main_weight
        self.fusion_weight = fusion_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        # 主要损失：融合预测 vs 目标
        fusion_loss = self.mse_loss(outputs['prediction'], targets)
        
        # VAE损失：VAE预测 vs 目标
        vae_loss = self.mse_loss(outputs['vae_prediction'], targets)
        
        # 总损失
        total_loss = self.main_weight * fusion_loss + (1 - self.main_weight) * vae_loss
        
        return {
            'total_loss': total_loss,
            'fusion_loss': fusion_loss,
            'vae_loss': vae_loss
        }

class ExactEarlyStopping:
    """
    基于原始early stopping的实现
    """
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # 不停止
        else:
            self.counter += 1
            return self.counter >= self.patience  # 是否停止

def exact_train_epoch(model, dataloader, criterion, optimizer, device):
    """
    基于原始训练逻辑的训练epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动数据到设备
        for key in ['vae_data', 'cc_mask', 'targets']:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key].to(device)
        
        if batch['descriptors'] is not None:
            batch['descriptors'] = batch['descriptors'].to(device)
            
        # 修复：只移动实际存在的lzhnn数据键
        for key in ['topo', 'struct']:
            if key in batch['lzhnn_data'] and batch['lzhnn_data'][key] is not None:
                batch['lzhnn_data'][key] = batch['lzhnn_data'][key].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch)
        
        # 计算损失
        loss_dict = criterion(outputs, batch['targets'])
        loss = loss_dict['total_loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

def exact_validate_epoch(model, dataloader, criterion, device):
    """
    基于原始验证逻辑的验证epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            for key in ['vae_data', 'cc_mask', 'targets']:
                if key in batch and batch[key] is not None:
                    batch[key] = batch[key].to(device)
            
            if batch['descriptors'] is not None:
                batch['descriptors'] = batch['descriptors'].to(device)
                
            for key in ['topo', 'struct']:
                batch['lzhnn_data'][key] = batch['lzhnn_data'][key].to(device)
            
            # 前向传播
            outputs = model(batch)
            
            # 计算损失
            loss_dict = criterion(outputs, batch['targets'])
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
    
    return total_loss / num_batches

def calculate_regression_metrics(model, dataloader, criterion, device, label_scaler=None):
    """
    计算回归指标（原值空间）：R²、RMSE、MAE、皮尔森、斯皮尔曼
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            for key in ['vae_data', 'cc_mask', 'targets']:
                if key in batch and batch[key] is not None:
                    batch[key] = batch[key].to(device)
            
            if batch['descriptors'] is not None:
                batch['descriptors'] = batch['descriptors'].to(device)
                
            for key in ['topo', 'struct']:
                if key in batch['lzhnn_data'] and batch['lzhnn_data'][key] is not None:
                    batch['lzhnn_data'][key] = batch['lzhnn_data'][key].to(device)
            
            # 前向传播
            outputs = model(batch)
            
            # 收集预测和目标（标准化空间）
            predictions = outputs['prediction'].cpu().numpy().flatten()
            targets = batch['targets'].cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # 计算损失
            loss_dict = criterion(outputs, batch['targets'])
            total_loss += loss_dict['total_loss'].item()
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 反标准化到原值空间
    if label_scaler is not None:
        # 使用StandardScaler进行反标准化
        predictions_original = label_scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        targets_original = label_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
        
        print(f"  标准化空间 - 预测范围: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
        print(f"  标准化空间 - 目标范围: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
        print(f"  原值空间 - 预测范围: [{predictions_original.min():.4f}, {predictions_original.max():.4f}]")
        print(f"  原值空间 - 目标范围: [{targets_original.min():.4f}, {targets_original.max():.4f}]")
    else:
        # 如果没有标准化器，直接使用原值
        predictions_original = all_predictions
        targets_original = all_targets
        print("  注意: 未提供标签标准化器，在标准化空间计算指标")
    
    # 在原值空间计算回归指标
    mse = mean_squared_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_original, predictions_original)
    r2 = r2_score(targets_original, predictions_original)
    # 相关性指标
    try:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, _ = pearsonr(targets_original, predictions_original)
        spearman_rho, _ = spearmanr(targets_original, predictions_original)
    except Exception as e:
        print(f"  计算相关性失败: {e}")
        pearson_r, spearman_rho = float('nan'), float('nan')
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    
    return {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_r,
        'spearman': spearman_rho,
        'predictions': predictions_original,
        'targets': targets_original
    }

def train_single_fold(model, train_data, val_data, test_data, args, device, fold_idx, output_dir, label_scaler=None):
    """
    训练单个折的模型
    """
    print(f"\n{'='*60}")
    print(f"开始训练第{fold_idx + 1}折")
    print(f"{'='*60}")
    
    # 创建数据加载器
    train_dataset = ExactFusionDataset(train_data)
    val_dataset = ExactFusionDataset(val_data)
    test_dataset = ExactFusionDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=exact_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=exact_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=exact_collate_fn
    )
    
    # 训练配置
    criterion = ExactFusionLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.7, min_lr=1e-6
    )
    early_stopping = ExactEarlyStopping(patience=args.early_stopping_patience)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # 训练
        train_loss = exact_train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = exact_validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = output_dir / f'best_model_fold_{fold_idx + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'fold': fold_idx + 1
            }, model_save_path)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"第{fold_idx + 1}折 Early stopping at epoch {epoch+1}")
            break
    
    # 测试评估 - 计算详细的回归指标（在原值空间）
    test_metrics = calculate_regression_metrics(model, test_loader, criterion, device, label_scaler)
    
    # 验证集评估 - 计算详细的回归指标（在原值空间）
    val_metrics = calculate_regression_metrics(model, val_loader, criterion, device, label_scaler)
    
    return {
        'fold': fold_idx + 1,
        'best_val_loss': best_val_loss,
        'test_loss': test_metrics['loss'],
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_path': str(output_dir / f'best_model_fold_{fold_idx + 1}.pth'),  # 转换为字符串避免JSON序列化问题
        # 验证集指标
        'val_r2': val_metrics['r2'],
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'val_mse': val_metrics['mse'],
        'val_pearson': val_metrics['pearson'],
        'val_spearman': val_metrics['spearman'],
        # 测试集指标
        'test_r2': test_metrics['r2'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_mse': test_metrics['mse'],
        'test_pearson': test_metrics['pearson'],
        'test_spearman': test_metrics['spearman'],
        # 预测结果（用于后续分析）
        'test_predictions': test_metrics['predictions'].tolist(),  # 转换为列表避免JSON序列化问题
        'test_targets': test_metrics['targets'].tolist(),
        'val_predictions': val_metrics['predictions'].tolist(),
        'val_targets': val_metrics['targets'].tolist()
    }

def create_data_splits(data_dict, train_ratio=0.7, val_ratio=0.15):
    """
    创建数据分割 - 使用分层采样确保数据分布均衡
    """
    print("创建数据分割...")
    
    # 获取数据
    targets = data_dict['targets']
    identifiers = data_dict['identifiers']
    
    # 计算分割点
    total_samples = len(targets)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    print(f"  总样本数: {total_samples}")
    print(f"  训练集: {train_size} ({train_ratio*100:.1f}%)")
    print(f"  验证集: {val_size} ({val_ratio*100:.1f}%)")
    print(f"  测试集: {test_size} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    # 使用分层采样 - 基于目标值范围分层
    target_values = targets.squeeze().numpy()
    
    # 创建分层索引
    # 将目标值分为几个区间，确保每个区间在训练、验证、测试集中都有代表
    num_bins = min(10, len(targets) // 100)  # 根据数据量确定bin数量
    if num_bins < 3:
        num_bins = 3
    
    # 创建分层索引
    bin_indices = np.linspace(target_values.min(), target_values.max(), num_bins + 1)
    bin_labels = np.digitize(target_values, bin_indices)
    
    # 分层采样
    train_indices = []
    val_indices = []
    test_indices = []
    
    for bin_id in range(1, num_bins + 1):
        bin_mask = bin_labels == bin_id
        bin_indices_list = np.where(bin_mask)[0]
        
        if len(bin_indices_list) > 0:
            # 随机打乱
            np.random.shuffle(bin_indices_list)
            
            # 按比例分配
            bin_train_size = max(1, int(len(bin_indices_list) * train_ratio))
            bin_val_size = max(1, int(len(bin_indices_list) * val_ratio))
            
            train_indices.extend(bin_indices_list[:bin_train_size])
            val_indices.extend(bin_indices_list[bin_train_size:bin_train_size + bin_val_size])
            test_indices.extend(bin_indices_list[bin_train_size + bin_val_size:])
    
    # 确保索引数量正确
    train_indices = train_indices[:train_size]
    val_indices = val_indices[:val_size]
    test_indices = test_indices[:test_size]
    
    # 如果某个集合样本不足，从其他集合补充
    while len(train_indices) < train_size:
        remaining = list(set(range(total_samples)) - set(train_indices) - set(val_indices) - set(test_indices))
        if remaining:
            train_indices.append(remaining[0])
        else:
            break
    
    while len(val_indices) < val_size:
        remaining = list(set(range(total_samples)) - set(train_indices) - set(val_indices) - set(test_indices))
        if remaining:
            val_indices.append(remaining[0])
        else:
            break
    
    while len(test_indices) < test_size:
        remaining = list(set(range(total_samples)) - set(train_indices) - set(val_indices) - set(test_indices))
        if remaining:
            test_indices.append(remaining[0])
        else:
            break
    
    print(f"  实际分割: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
    
    # 验证分割质量
    train_targets = targets[train_indices]
    val_targets = targets[val_indices]
    test_targets = targets[test_indices]
    
    print(f"  目标值分布:")
    print(f"    训练集: [{train_targets.min():.4f}, {train_targets.max():.4f}], 均值={train_targets.mean():.4f}")
    print(f"    验证集: [{val_targets.min():.4f}, {val_targets.max():.4f}], 均值={val_targets.mean():.4f}")
    print(f"    测试集: [{test_targets.min():.4f}, {test_targets.max():.4f}], 均值={test_targets.mean():.4f}")
    
    # 检查分布相似性
    train_std = train_targets.std()
    val_std = val_targets.std()
    test_std = test_targets.std()
    
    if abs(train_std - val_std) / train_std > 0.3 or abs(train_std - test_std) / train_std > 0.3:
        print("  ⚠️ 警告: 数据集间标准差差异较大，可能影响模型泛化")
    
    def split_data(data_dict, indices):
        """分割数据字典"""
        split_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                split_dict[key] = value[indices]
            elif isinstance(value, list):
                split_dict[key] = [value[i] for i in indices]
            else:
                split_dict[key] = value
        return split_dict
    
    # 创建分割后的数据
    train_data = split_data(data_dict, train_indices)
    val_data = split_data(data_dict, val_indices)
    test_data = split_data(data_dict, test_indices)
    
    return train_data, val_data, test_data

def create_cross_validation_splits(data_dict, n_folds=5, random_state=42):
    """
    创建五折交叉验证的数据分割 - 使用分层K折确保每折数据分布均衡
    """
    print(f"创建{n_folds}折交叉验证数据分割...")
    
    # 获取数据
    targets = data_dict['targets']
    identifiers = data_dict['identifiers']
    target_values = targets.squeeze().numpy()
    
    # 设置随机种子确保可重复性
    np.random.seed(random_state)
    
    # 使用分层K折分割
    from sklearn.model_selection import StratifiedKFold
    
    # 创建分层标签 - 基于目标值分位数
    n_bins = min(10, len(targets) // 50)  # 根据数据量确定bin数量
    if n_bins < 5:
        n_bins = 5
    
    # 创建分层标签
    bin_edges = np.quantile(target_values, np.linspace(0, 1, n_bins + 1))
    bin_labels = np.digitize(target_values, bin_edges) - 1
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)
    
    # 创建分层K折分割器
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_splits = []
    
    for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(range(len(targets)), bin_labels)):
        print(f"  第{fold_idx + 1}折:")
        
        # 将训练+验证集进一步分为训练集和验证集 (8:2)
        train_val_targets = target_values[train_val_indices]
        train_val_bin_labels = bin_labels[train_val_indices]
        
        # 对训练+验证集再次分层分割
        train_val_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + fold_idx)
        train_indices_fold, val_indices_fold = next(iter(train_val_skf.split(train_val_indices, train_val_bin_labels)))
        
        # 转换为原始索引
        train_indices = train_val_indices[train_indices_fold]
        val_indices = train_val_indices[val_indices_fold]
        
        # 统计信息
        train_targets = targets[train_indices]
        val_targets = targets[val_indices]
        test_targets = targets[test_indices]
        
        print(f"    训练集: {len(train_indices)} 样本")
        print(f"    验证集: {len(val_indices)} 样本")
        print(f"    测试集: {len(test_indices)} 样本")
        print(f"    目标值范围 - 训练:[{train_targets.min():.4f}, {train_targets.max():.4f}], "
              f"验证:[{val_targets.min():.4f}, {val_targets.max():.4f}], "
              f"测试:[{test_targets.min():.4f}, {test_targets.max():.4f}]")
        
        # 检查分布相似性
        train_std = train_targets.std()
        val_std = val_targets.std()
        test_std = test_targets.std()
        
        if abs(train_std - val_std) / train_std > 0.4 or abs(train_std - test_std) / train_std > 0.4:
            print(f"    ⚠️ 警告: 第{fold_idx + 1}折数据分布差异较大")
        
        cv_splits.append({
            'fold': fold_idx + 1,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        })
    
    return cv_splits

def split_data_by_indices(data_dict, indices):
    """根据索引分割数据字典"""
    split_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            split_dict[key] = value[indices]
        elif isinstance(value, list):
            split_dict[key] = [value[i] for i in indices]
        else:
            split_dict[key] = value
    return split_dict

def run_cross_validation(data_dict, args, device, output_dir):
    """
    运行五折交叉验证
    """
    print("="*80)
    print("开始五折交叉验证训练")
    print("="*80)
    
    # 创建交叉验证分割
    cv_splits = create_cross_validation_splits(data_dict, n_folds=5, random_state=42)
    
    # 存储每折的结果
    fold_results = []
    
    # 检查是否有描述符
    use_descriptors = data_dict['descriptors'] is not None
    descriptor_dim = data_dict['descriptors'].shape[1] if use_descriptors else 0
    
    # 获取标签标准化器
    label_scaler = data_dict.get('label_scaler', None)
    if label_scaler is not None:
        print(f"✅ 已获取标签标准化器，将在原值空间计算性能指标")
        print(f"  原始标签均值: {label_scaler.mean_[0]:.4f}")
        print(f"  原始标签标准差: {label_scaler.scale_[0]:.4f}")
    else:
        # 不再基于已标准化的targets二次拟合标准化器，直接在标准化空间计算并给出明确提示
        print("⚠️ 未获取到标签标准化器，将在标准化空间计算指标（不进行二次拟合）")
    
    # VAE配置
    vae_config = {
        'latent_dim': 128,
        'num_planes': 9,
        'dropout_rate': args.dropout_rate
    }
    
    # 预训练模型路径
    model_paths = {
        'vae': args.vae_model_path,
        'gcn_cc': args.gcn_cc_model_path,
        'gcn_noncc': args.gcn_noncc_model_path,
        'lzhnn': args.lzhnn_model_path
    }
    
    for fold_idx, split in enumerate(cv_splits):
        print(f"\n🔄 处理第{fold_idx + 1}/5折...")
        
        # 创建该折的数据
        train_data = split_data_by_indices(data_dict, split['train_indices'])
        val_data = split_data_by_indices(data_dict, split['val_indices'])
        test_data = split_data_by_indices(data_dict, split['test_indices'])
        
        # 创建新模型（每折使用独立的模型实例）
        model = create_exact_fusion_model(
            vae_config=vae_config,
            use_descriptors=use_descriptors,
            descriptor_dim=descriptor_dim
        ).to(device)
        
        # 加载预训练权重
        model.load_pretrained_weights(model_paths)
        
        # 训练该折
        fold_result = train_single_fold(
            model, train_data, val_data, test_data, 
            args, device, fold_idx, output_dir, label_scaler
        )
        
        fold_results.append(fold_result)
        
        # 打印该折结果
        print(f"第{fold_idx + 1}折结果:")
        print(f"  验证集 - 损失: {fold_result['best_val_loss']:.4f}, R²: {fold_result['val_r2']:.4f}, RMSE: {fold_result['val_rmse']:.4f}, MAE: {fold_result['val_mae']:.4f}, Pearson: {fold_result['val_pearson']:.4f}, Spearman: {fold_result['val_spearman']:.4f}")
        print(f"  测试集 - 损失: {fold_result['test_loss']:.4f}, R²: {fold_result['test_r2']:.4f}, RMSE: {fold_result['test_rmse']:.4f}, MAE: {fold_result['test_mae']:.4f}, Pearson: {fold_result['test_pearson']:.4f}, Spearman: {fold_result['test_spearman']:.4f}")
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return fold_results

def aggregate_cv_results(fold_results, output_dir):
    """
    聚合交叉验证结果并生成统计报告
    """
    print("\n" + "="*80)
    print("交叉验证结果聚合 (原值空间)")
    print("="*80)
    
    # 提取关键指标
    val_losses = [result['best_val_loss'] for result in fold_results]
    test_losses = [result['test_loss'] for result in fold_results]
    final_train_losses = [result['final_train_loss'] for result in fold_results]
    final_val_losses = [result['final_val_loss'] for result in fold_results]
    
    # 提取回归指标
    val_r2_scores = [result['val_r2'] for result in fold_results]
    test_r2_scores = [result['test_r2'] for result in fold_results]
    val_rmse_scores = [result['val_rmse'] for result in fold_results]
    test_rmse_scores = [result['test_rmse'] for result in fold_results]
    val_mae_scores = [result['val_mae'] for result in fold_results]
    test_mae_scores = [result['test_mae'] for result in fold_results]
    val_mse_scores = [result['val_mse'] for result in fold_results]
    test_mse_scores = [result['test_mse'] for result in fold_results]
    # 新增相关性指标
    val_pearson_scores = [result.get('val_pearson', float('nan')) for result in fold_results]
    test_pearson_scores = [result.get('test_pearson', float('nan')) for result in fold_results]
    val_spearman_scores = [result.get('val_spearman', float('nan')) for result in fold_results]
    test_spearman_scores = [result.get('test_spearman', float('nan')) for result in fold_results]
    
    # 计算统计量
    import statistics
    
    def calculate_stats(values):
        return {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'values': values
        }
    
    stats = {
        'val_loss': calculate_stats(val_losses),
        'test_loss': calculate_stats(test_losses),
        'final_train_loss': calculate_stats(final_train_losses),
        'final_val_loss': calculate_stats(final_val_losses),
        # 验证集回归指标
        'val_r2': calculate_stats(val_r2_scores),
        'val_rmse': calculate_stats(val_rmse_scores),
        'val_mae': calculate_stats(val_mae_scores),
        'val_mse': calculate_stats(val_mse_scores),
        'val_pearson': calculate_stats(val_pearson_scores),
        'val_spearman': calculate_stats(val_spearman_scores),
        # 测试集回归指标
        'test_r2': calculate_stats(test_r2_scores),
        'test_rmse': calculate_stats(test_rmse_scores),
        'test_mae': calculate_stats(test_mae_scores),
        'test_mse': calculate_stats(test_mse_scores),
        'test_pearson': calculate_stats(test_pearson_scores),
        'test_spearman': calculate_stats(test_spearman_scores)
    }
    
    # 打印统计结果
    print("📊 交叉验证统计结果:")
    print("-" * 70)
    
    # 损失统计
    print("🔸 损失指标:")
    print(f"  验证损失: {stats['val_loss']['mean']:.4f} ± {stats['val_loss']['std']:.4f}")
    print(f"    范围: [{stats['val_loss']['min']:.4f}, {stats['val_loss']['max']:.4f}]")
    print(f"  测试损失: {stats['test_loss']['mean']:.4f} ± {stats['test_loss']['std']:.4f}")
    print(f"    范围: [{stats['test_loss']['min']:.4f}, {stats['test_loss']['max']:.4f}]")
    
    # 验证集回归指标
    print("\n🔸 验证集回归指标:")
    print(f"  R²: {stats['val_r2']['mean']:.4f} ± {stats['val_r2']['std']:.4f}")
    print(f"    范围: [{stats['val_r2']['min']:.4f}, {stats['val_r2']['max']:.4f}]")
    print(f"  RMSE: {stats['val_rmse']['mean']:.4f} ± {stats['val_rmse']['std']:.4f}")
    print(f"    范围: [{stats['val_rmse']['min']:.4f}, {stats['val_rmse']['max']:.4f}]")
    print(f"  MAE: {stats['val_mae']['mean']:.4f} ± {stats['val_mae']['std']:.4f}")
    print(f"    范围: [{stats['val_mae']['min']:.4f}, {stats['val_mae']['max']:.4f}]")
    print(f"  Pearson: {stats['val_pearson']['mean']:.4f} ± {stats['val_pearson']['std']:.4f}")
    print(f"  Spearman: {stats['val_spearman']['mean']:.4f} ± {stats['val_spearman']['std']:.4f}")
    
    # 测试集回归指标
    print("\n🔸 测试集回归指标:")
    print(f"  R²: {stats['test_r2']['mean']:.4f} ± {stats['test_r2']['std']:.4f}")
    print(f"    范围: [{stats['test_r2']['min']:.4f}, {stats['test_r2']['max']:.4f}]")
    print(f"  RMSE: {stats['test_rmse']['mean']:.4f} ± {stats['test_rmse']['std']:.4f}")
    print(f"    范围: [{stats['test_rmse']['min']:.4f}, {stats['test_rmse']['max']:.4f}]")
    print(f"  MAE: {stats['test_mae']['mean']:.4f} ± {stats['test_mae']['std']:.4f}")
    print(f"    范围: [{stats['test_mae']['min']:.4f}, {stats['test_mae']['max']:.4f}]")
    print(f"  Pearson: {stats['test_pearson']['mean']:.4f} ± {stats['test_pearson']['std']:.4f}")
    print(f"  Spearman: {stats['test_spearman']['mean']:.4f} ± {stats['test_spearman']['std']:.4f}")
    
    # 计算稳定性指标
    val_loss_cv = stats['val_loss']['std'] / stats['val_loss']['mean'] if stats['val_loss']['mean'] > 0 else 0
    test_loss_cv = stats['test_loss']['std'] / stats['test_loss']['mean'] if stats['test_loss']['mean'] > 0 else 0
    val_r2_cv = stats['val_r2']['std'] / abs(stats['val_r2']['mean']) if stats['val_r2']['mean'] != 0 else 0
    test_r2_cv = stats['test_r2']['std'] / abs(stats['test_r2']['mean']) if stats['test_r2']['mean'] != 0 else 0
    
    print(f"\n🎯 模型稳定性:")
    print(f"验证损失变异系数: {val_loss_cv:.4f}")
    print(f"测试损失变异系数: {test_loss_cv:.4f}")
    print(f"验证R²变异系数: {val_r2_cv:.4f}")
    print(f"测试R²变异系数: {test_r2_cv:.4f}")
    
    # 综合稳定性评估（基于R²的稳定性）
    if val_r2_cv < 0.1:
        stability_level = "🌟 非常稳定"
    elif val_r2_cv < 0.2:
        stability_level = "✅ 稳定"
    elif val_r2_cv < 0.3:
        stability_level = "⚠️ 一般稳定"
    else:
        stability_level = "❌ 不稳定"
    
    print(f"整体稳定性: {stability_level}")
    
    # 保存详细结果
    import json
    import pandas as pd
    
    # 保存JSON格式的完整结果
    results_file = output_dir / 'cross_validation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'fold_results': fold_results,
            'statistics': stats,
            'stability': {
                'val_loss_cv': val_loss_cv,
                'test_loss_cv': test_loss_cv,
                'val_r2_cv': val_r2_cv,
                'test_r2_cv': test_r2_cv,
                'stability_level': stability_level
            }
        }, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式的摘要
    summary_data = []
    for i, result in enumerate(fold_results):
        summary_data.append({
            'fold': result['fold'],
            'best_val_loss': result['best_val_loss'],
            'test_loss': result['test_loss'],
            'val_r2': result['val_r2'],
            'test_r2': result['test_r2'],
            'val_rmse': result['val_rmse'],
            'test_rmse': result['test_rmse'],
            'val_mae': result['val_mae'],
            'test_mae': result['test_mae'],
            'val_pearson': result.get('val_pearson', float('nan')),
            'test_pearson': result.get('test_pearson', float('nan')),
            'val_spearman': result.get('val_spearman', float('nan')),
            'test_spearman': result.get('test_spearman', float('nan')),
            'final_train_loss': result['final_train_loss'],
            'final_val_loss': result['final_val_loss']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / 'cross_validation_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    
    # 添加统计行
    stats_row = pd.DataFrame([{
        'fold': 'Mean',
        'best_val_loss': stats['val_loss']['mean'],
        'test_loss': stats['test_loss']['mean'],
        'val_r2': stats['val_r2']['mean'],
        'test_r2': stats['test_r2']['mean'],
        'val_rmse': stats['val_rmse']['mean'],
        'test_rmse': stats['test_rmse']['mean'],
        'val_mae': stats['val_mae']['mean'],
        'test_mae': stats['test_mae']['mean'],
        'val_pearson': stats['val_pearson']['mean'],
        'test_pearson': stats['test_pearson']['mean'],
        'val_spearman': stats['val_spearman']['mean'],
        'test_spearman': stats['test_spearman']['mean'],
        'final_train_loss': stats['final_train_loss']['mean'],
        'final_val_loss': stats['final_val_loss']['mean']
    }])
    
    std_row = pd.DataFrame([{
        'fold': 'Std',
        'best_val_loss': stats['val_loss']['std'],
        'test_loss': stats['test_loss']['std'],
        'val_r2': stats['val_r2']['std'],
        'test_r2': stats['test_r2']['std'],
        'val_rmse': stats['val_rmse']['std'],
        'test_rmse': stats['test_rmse']['std'],
        'val_mae': stats['val_mae']['std'],
        'test_mae': stats['test_mae']['std'],
        'val_pearson': stats['val_pearson']['std'],
        'test_pearson': stats['test_pearson']['std'],
        'val_spearman': stats['val_spearman']['std'],
        'test_spearman': stats['test_spearman']['std'],
        'final_train_loss': stats['final_train_loss']['std'],
        'final_val_loss': stats['final_val_loss']['std']
    }])
    
    final_summary = pd.concat([summary_df, stats_row, std_row], ignore_index=True)
    final_summary.to_csv(summary_file, index=False, encoding='utf-8')
    
    print(f"\n📄 结果已保存:")
    print(f"  详细结果: {results_file}")
    print(f"  摘要表格: {summary_file}")
    
    return stats, fold_results

def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='交叉注意力融合训练')
    parser.add_argument('--vae_model_path', type=str, default='PP-cVAE/models/best_model_CH401.pth')
    parser.add_argument('--gcn_cc_model_path', type=str, default='BiG-CAE/models/contrastive_model_cc_CH401.pth')
    parser.add_argument('--gcn_noncc_model_path', type=str, default='BiG-CAE/models/contrastive_model_noncc_CH401.pth')
    parser.add_argument('--lzhnn_model_path', type=str, default='PP-NN/final_model_42_CH401.pth')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='exact_fusion_output')
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 增加权重衰减
    parser.add_argument('--dropout_rate', type=float, default=0.3)   # 增加dropout
    parser.add_argument('--early_stopping_patience', type=int, default=15)  # 增加早停耐心值
    parser.add_argument('--use_cross_validation', action='store_true', help='使用五折交叉验证训练')
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # === 1. 基于原始代码加载数据 ===
    print("基于原始代码加载数据...")
    data_loader = create_exact_data_loader(max_samples=args.max_samples)
    data_dict = data_loader.load_all_data()
    
    # 数据验证 - 检查数据质量
    print("\n🔍 数据质量检查...")
    targets = data_dict['targets']
    print(f"  目标值统计:")
    print(f"    范围: [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"    均值: {targets.mean():.4f}")
    print(f"    标准差: {targets.std():.4f}")
    print(f"    样本数: {len(targets)}")
    
    # 检查数据分布
    if targets.std() < 0.1:
        print("  ⚠️ 警告: 目标值标准差过小，可能存在数据问题")
    
    # 根据参数选择训练模式
    if args.use_cross_validation:
        print(f"\n🔄 使用{args.n_folds}折交叉验证模式")
        
        # 运行交叉验证
        fold_results = run_cross_validation(data_dict, args, device, output_dir)
        
        # 聚合结果
        stats, _ = aggregate_cv_results(fold_results, output_dir)
        
    # 打印最终总结
    print("\n" + "="*80)
    print("🎉 五折交叉验证训练完成!")
    print("="*80)
    
    print("📊 最终性能总结 (原值空间):")
    print(f"  验证集 R²: {stats['val_r2']['mean']:.4f} ± {stats['val_r2']['std']:.4f}")
    print(f"  测试集 R²: {stats['test_r2']['mean']:.4f} ± {stats['test_r2']['std']:.4f}")
    print(f"  验证集 RMSE: {stats['val_rmse']['mean']:.4f} ± {stats['val_rmse']['std']:.4f}")
    print(f"  测试集 RMSE: {stats['test_rmse']['mean']:.4f} ± {stats['test_rmse']['std']:.4f}")
    print(f"  验证集 MAE: {stats['val_mae']['mean']:.4f} ± {stats['val_mae']['std']:.4f}")
    print(f"  测试集 MAE: {stats['test_mae']['mean']:.4f} ± {stats['test_mae']['std']:.4f}")
    
    print(f"\n📈 损失指标:")
    print(f"  平均验证损失: {stats['val_loss']['mean']:.4f} ± {stats['val_loss']['std']:.4f}")
    print(f"  平均测试损失: {stats['test_loss']['mean']:.4f} ± {stats['test_loss']['std']:.4f}")
    
    # 计算稳定性（基于R²）
    val_r2_cv = stats['val_r2']['std'] / abs(stats['val_r2']['mean']) if stats['val_r2']['mean'] != 0 else 0
    if val_r2_cv < 0.1:
        stability_level = "🌟 非常稳定"
    elif val_r2_cv < 0.2:
        stability_level = "✅ 稳定"
    elif val_r2_cv < 0.3:
        stability_level = "⚠️ 一般稳定"
    else:
        stability_level = "❌ 不稳定"
    
    print(f"\n🎯 模型稳定性: {stability_level} (R²变异系数: {val_r2_cv:.4f})")
    print(f"📁 结果保存路径: {output_dir}")
    return
    
    # 常规单次训练模式
    print(f"\n🔄 使用常规单次训练模式")
    
    # 数据分割
    train_data, val_data, test_data = create_data_splits(data_dict)
    
    # 验证分割质量
    train_targets = train_data['targets']
    val_targets = val_data['targets']
    test_targets = test_data['targets']
    
    print(f"  数据分割统计:")
    print(f"    训练集: {len(train_targets)} 样本, R²范围: [{train_targets.min():.4f}, {train_targets.max():.4f}]")
    print(f"    验证集: {len(val_targets)} 样本, R²范围: [{val_targets.min():.4f}, {val_targets.max():.4f}]")
    print(f"    测试集: {len(test_targets)} 样本, R²范围: [{test_targets.min():.4f}, {test_targets.max():.4f}]")
    
    # === 2. 基于原始VAE配置创建模型 ===
    print("\n创建融合模型...")
    vae_config = {
        'latent_dim': 128,
        'num_planes': 9,
        'dropout_rate': args.dropout_rate  # 使用参数化的dropout
    }
    
    # 检查是否有描述符
    use_descriptors = data_dict['descriptors'] is not None
    descriptor_dim = data_dict['descriptors'].shape[1] if use_descriptors else 0
    
    model = create_exact_fusion_model(
        vae_config=vae_config,
        use_descriptors=use_descriptors,
        descriptor_dim=descriptor_dim
    ).to(device)
    
    # === 3. 基于原始保存格式加载预训练权重 ===
    print("加载预训练权重...")
    model_paths = {
        'vae': args.vae_model_path,
        'gcn_cc': args.gcn_cc_model_path,
        'gcn_noncc': args.gcn_noncc_model_path,
        'lzhnn': args.lzhnn_model_path
    }
    model.load_pretrained_weights(model_paths)
    
    # === 4. 创建数据加载器 ===
    print("创建数据加载器...")
    train_dataset = ExactFusionDataset(train_data)
    val_dataset = ExactFusionDataset(val_data)
    test_dataset = ExactFusionDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=exact_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=exact_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=exact_collate_fn
    )
    
    # === 5. 训练配置 ===
    criterion = ExactFusionLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay  # 使用参数化的权重衰减
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.7, min_lr=1e-6  # 调整学习率调度
    )
    early_stopping = ExactEarlyStopping(patience=args.early_stopping_patience)
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("融合模型信息")
    print("=" * 60)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
    
    # 获取归一化后的权重信息
    main_weight, fusion_weight = model.get_normalized_weights()
    print(f"主模型权重: {main_weight.item():.2f}")
    print(f"融合模型权重: {fusion_weight.item():.2f}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"Dropout率: {args.dropout_rate}")
    print("=" * 60)
    
    # === 6. 训练循环 ===
    print("开始训练...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = exact_train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = exact_validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"验证损失: {val_loss:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.2e}")
        
        # 检查过拟合
        if len(train_losses) > 5:
            train_trend = sum(train_losses[-5:]) / 5
            val_trend = sum(val_losses[-5:]) / 5
            if val_trend > train_trend * 1.2:  # 验证损失比训练损失高20%
                print("  ⚠️ 检测到过拟合趋势，考虑调整正则化参数")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = output_dir / 'best_fusion_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': {
                    'vae_config': vae_config,
                    'use_descriptors': use_descriptors,
                    'descriptor_dim': descriptor_dim,
                    'weight_decay': args.weight_decay,
                    'dropout_rate': args.dropout_rate
                }
            }, model_save_path)
            print(f"保存最佳模型: {model_save_path}")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # === 7. 测试评估 ===
    print("\n" + "=" * 60)
    print("测试评估")
    print("=" * 60)
    
    test_loss = exact_validate_epoch(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}")
    
    # 训练历史分析
    print(f"\n📊 训练历史分析:")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  最终验证损失: {val_losses[-1]:.4f}")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    
    if len(train_losses) > 1:
        train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
        print(f"  训练损失改善: {train_improvement:.2f}%")
        print(f"  验证损失改善: {val_improvement:.2f}%")
    
    # ===== 详细性能评估 =====
    print("\n" + "="*70)
    print("🔍 开始详细性能评估...")
    print("="*70)
    
    try:
        from fusion_model_evaluator import create_comprehensive_evaluation_pipeline
        
        # 创建评估结果目录
        eval_output_dir = output_dir / "evaluation_results"
        eval_output_dir.mkdir(exist_ok=True)
        
        # 执行全面评估
        evaluator, eval_results = create_comprehensive_evaluation_pipeline(
            model=model,
            test_loader=test_loader,
            val_loader=val_loader,
            individual_models=None,  # 可以后续添加单独模型对比
            save_dir=str(eval_output_dir)
        )
        
        print("\n📈 评估结果摘要:")
        if 'test' in eval_results:
            test_metrics = eval_results['test']['regression_metrics']
            print(f"  • 测试集 R²: {test_metrics['R²']:.4f}")
            print(f"  • 测试集 RMSE: {test_metrics['RMSE']:.4f}")
            print(f"  • 测试集 MAE: {test_metrics['MAE']:.4f}")
            print(f"  • 测试集 MAPE: {test_metrics['MAPE']:.2f}%")
            
            # 推理性能
            if 'inference_stats' in eval_results['test']:
                inf_stats = eval_results['test']['inference_stats']
                print(f"  • 推理速度: {inf_stats['samples_per_second']:.1f} samples/s")
                print(f"  • 平均推理时间: {inf_stats['avg_inference_time']*1000:.2f} ms")
        
        if 'validation' in eval_results:
            val_metrics = eval_results['validation']['regression_metrics']
            print(f"  • 验证集 R²: {val_metrics['R²']:.4f}")
            print(f"  • 验证集 RMSE: {val_metrics['RMSE']:.4f}")
            
            # 泛化性能分析
            test_r2 = eval_results['test']['regression_metrics']['R²']
            val_r2 = val_metrics['R²']
            generalization_gap = abs(test_r2 - val_r2)
            if generalization_gap < 0.05:
                print(f"  • 泛化性能: ✅ 良好 (差异: {generalization_gap:.4f})")
            else:
                print(f"  • 泛化性能: ⚠️ 需要关注 (差异: {generalization_gap:.4f})")
        
        print(f"\n📋 详细评估报告已生成: {eval_output_dir}")
        print("  📄 evaluation_report.md - 完整评估报告")
        print("  📊 performance_summary.csv - 性能摘要表")
        print("  🎯 predictions_vs_targets.png - 预测散点图")
        print("  📈 error_distribution.png - 误差分布图")
        print("  🕸️ performance_radar.png - 性能雷达图")
        print("  📋 test_predictions.csv - 测试集预测结果")
        
        # 性能等级评估
        test_r2 = eval_results['test']['regression_metrics']['R²']
        if test_r2 > 0.9:
            performance_level = "🌟 优秀"
        elif test_r2 > 0.8:
            performance_level = "✅ 良好"
        elif test_r2 > 0.7:
            performance_level = "⚠️ 一般"
        else:
            performance_level = "❌ 需要改进"
        
        print(f"\n🏆 模型性能等级: {performance_level} (R² = {test_r2:.4f})")
        
    except Exception as e:
        print(f"⚠️ 性能评估失败: {e}")
        print("训练已完成，但跳过详细评估")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 基于原始代码的交叉注意力融合训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存路径: {output_dir}")

if __name__ == "__main__":
    main() 