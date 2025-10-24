"""训练和评估模块"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models import vae_loss_function
import json
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, config, save_dir='experiments'):
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.device = config.DEVICE
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化优化器和调度器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.LR_FACTOR, 
            patience=config.LR_PATIENCE
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_r2_scores = []
        self.val_mae_scores = []
        self.val_rmse_scores = []
        self.best_val_loss = float('inf')
        self.best_val_r2 = 0.0
        self.best_val_mae = float('inf')
        self.best_val_rmse = float('inf')
        self.best_model_state = None
        
        # 混合精度训练
        self.use_mixed_precision = getattr(config, 'USE_MIXED_PRECISION', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        if self.use_mixed_precision:
            print("Using mixed precision training for faster performance")
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        
        # 添加进度条
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            # 处理不同的数据格式（支持多模态）
            if len(batch_data) == 3:  # 包含描述符
                data, descriptors, target = batch_data
                data = data.to(self.device)
                descriptors = descriptors.to(self.device)
                target = target.to(self.device)
            else:  # 只有投影平面数据
                data, target = batch_data
                data = data.to(self.device)
                target = target.to(self.device)
                descriptors = None
            
            # GPU优化：转换为channels_last内存格式（仅对4D张量）
            if hasattr(self.config, 'CHANNELS_LAST') and self.config.CHANNELS_LAST:
                # 检查张量维度，channels_last只支持4D张量
                if data.dim() == 4:
                    data = data.to(memory_format=torch.channels_last)
                elif data.dim() == 5:
                    # 对于5D张量（batch, planes, channels, height, width），
                    # 我们需要重新整形为4D来应用channels_last
                    batch_size, num_planes, channels, height, width = data.shape
                    data = data.view(batch_size * num_planes, channels, height, width)
                    data = data.to(memory_format=torch.channels_last)
                    data = data.view(batch_size, num_planes, channels, height, width)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with autocast():
                    if hasattr(self.model, 'use_descriptors') and self.model.use_descriptors:
                        prediction, recon_loss, kl_loss = self.model(data, descriptors)
                    else:
                        prediction, recon_loss, kl_loss = self.model(data)
                    
                    # 总损失
                    vae_loss = vae_loss_function(recon_loss, kl_loss, beta=self.config.VAE_BETA)
                    mse_loss = F.mse_loss(prediction, target)
                    total_loss_batch = mse_loss + self.config.VAE_WEIGHT * vae_loss
                
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if hasattr(self.model, 'use_descriptors') and self.model.use_descriptors:
                    prediction, recon_loss, kl_loss = self.model(data, descriptors)
                else:
                    prediction, recon_loss, kl_loss = self.model(data)
                
                # 总损失
                vae_loss = vae_loss_function(recon_loss, kl_loss, beta=self.config.VAE_BETA)
                mse_loss = F.mse_loss(prediction, target)
                total_loss_batch = mse_loss + self.config.VAE_WEIGHT * vae_loss
                
                total_loss_batch.backward()
                self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            
            # 更新进度条显示当前损失
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        return total_loss / len(train_loader), total_mse / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            # 添加验证进度条
            pbar = tqdm(val_loader, desc='Validating', leave=False)
            for batch_data in pbar:
                # 处理不同的数据格式（支持多模态）
                if len(batch_data) == 3:  # 包含描述符
                    data, descriptors, target = batch_data
                    data = data.to(self.device)
                    descriptors = descriptors.to(self.device)
                    target = target.to(self.device)
                else:  # 只有投影平面数据
                    data, target = batch_data
                    data = data.to(self.device)
                    target = target.to(self.device)
                    descriptors = None
                
                # GPU优化：转换为channels_last内存格式（仅对4D张量）
                if hasattr(self.config, 'CHANNELS_LAST') and self.config.CHANNELS_LAST:
                    # 检查张量维度，channels_last只支持4D张量
                    if data.dim() == 4:
                        data = data.to(memory_format=torch.channels_last)
                    elif data.dim() == 5:
                        # 对于5D张量（batch, planes, channels, height, width），
                        # 我们需要重新整形为4D来应用channels_last
                        batch_size, num_planes, channels, height, width = data.shape
                        data = data.view(batch_size * num_planes, channels, height, width)
                        data = data.to(memory_format=torch.channels_last)
                        data = data.view(batch_size, num_planes, channels, height, width)
                
                if hasattr(self.model, 'use_descriptors') and self.model.use_descriptors:
                    prediction, recon_loss, kl_loss = self.model(data, descriptors)
                else:
                    prediction, recon_loss, kl_loss = self.model(data)
                
                vae_loss = vae_loss_function(recon_loss, kl_loss, beta=self.config.VAE_BETA)
                mse_loss = F.mse_loss(prediction, target)
                total_loss_batch = mse_loss + self.config.VAE_WEIGHT * vae_loss
                
                total_loss += total_loss_batch.item()
                predictions.extend(prediction.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_r2 = r2_score(targets, predictions)
        
        # 计算MAE和RMSE
        val_mae = mean_absolute_error(targets, predictions)
        val_rmse = np.sqrt(mean_squared_error(targets, predictions))
        
        return val_loss, val_r2, val_mae, val_rmse, predictions, targets
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """完整训练过程"""
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
            
        self.model = self.model.to(self.device)
        
        print(f'Training on {self.device}')
        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        
        # 添加epoch级别的进度条
        epoch_pbar = tqdm(range(num_epochs), desc='Training Progress')
        for epoch in epoch_pbar:
            # 训练
            train_loss, train_mse = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_r2, val_mae, val_rmse, val_predictions, val_targets = self.validate_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
                    # 记录最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_r2 = val_r2
            self.best_val_mae = val_mae
            self.best_val_rmse = val_rmse
            self.best_model_state = self.model.state_dict().copy()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_r2_scores.append(val_r2)
            self.val_mae_scores.append(val_mae)
            self.val_rmse_scores.append(val_rmse)
            
            # 更新epoch进度条显示当前指标
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss:.4f}',
                'Val_Loss': f'{val_loss:.4f}',
                'Val_R2': f'{val_r2:.4f}',
                'Val_MAE': f'{val_mae:.4f}',
                'Val_RMSE': f'{val_rmse:.4f}',
                'Best_Val': f'{self.best_val_loss:.4f}'
            })
            
            # 打印详细进度（减少频率）
            if epoch % 20 == 0 or epoch == num_epochs - 1:
                print(f'\nEpoch {epoch:3d}: Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}, '
                      f'Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}')
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # 输出最佳验证结果
        print(f'\n{"="*60}')
        print(f'训练完成！最佳验证结果：')
        print(f'最佳验证损失: {self.best_val_loss:.6f}')
        print(f'对应的R²分数: {self.best_val_r2:.6f}')
        print(f'对应的MAE: {self.best_val_mae:.6f}')
        print(f'对应的RMSE: {self.best_val_rmse:.6f}')
        print(f'{"="*60}')
        
        return self.model
    
    def save_model(self, filename=None, experiment_name=None):
        """保存模型和训练历史"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if experiment_name:
                filename = f'{experiment_name}_{timestamp}'
            else:
                filename = f'model_{timestamp}'
        
        # 保存模型
        model_path = os.path.join(self.save_dir, f'{filename}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_r2_scores': self.val_r2_scores,
            'val_mae_scores': self.val_mae_scores,
            'val_rmse_scores': self.val_rmse_scores,
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2,
            'best_val_mae': self.best_val_mae,
            'best_val_rmse': self.best_val_rmse
        }, model_path)
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, f'{filename}_history.json')
        
        # 转换numpy类型为Python原生类型，确保JSON可序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            else:
                return obj
        
        history = {
            'train_losses': convert_to_serializable(self.train_losses),
            'val_losses': convert_to_serializable(self.val_losses),
            'val_r2_scores': convert_to_serializable(self.val_r2_scores),
            'val_mae_scores': convert_to_serializable(self.val_mae_scores),
            'val_rmse_scores': convert_to_serializable(self.val_rmse_scores),
            'best_val_loss': convert_to_serializable(self.best_val_loss),
            'best_val_r2': convert_to_serializable(self.best_val_r2),
            'best_val_mae': convert_to_serializable(self.best_val_mae),
            'best_val_rmse': convert_to_serializable(self.best_val_rmse),
            'config': {k: str(v) for k, v in self.config.__dict__.items()}
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f'Model saved to {model_path}')
        print(f'History saved to {history_path}')
        
        return model_path, history_path
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.val_r2_scores = checkpoint['val_r2_scores']
            self.val_mae_scores = checkpoint.get('val_mae_scores', [])
            self.val_rmse_scores = checkpoint.get('val_rmse_scores', [])
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_r2 = checkpoint.get('best_val_r2', 0.0)
            self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
            self.best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        
        print(f'Model loaded from {model_path}')
        return self.model

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader, label_scaler=None):
        """评估模型"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                prediction, _, _ = self.model(data)
                
                predictions.extend(prediction.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # 如果使用了标签标准化，需要反标准化
        if label_scaler is not None:
            predictions = label_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            targets = label_scaler.inverse_transform(np.array(targets).reshape(-1, 1)).flatten()
        
        # 计算指标
        r2 = r2_score(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        results = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
        
        return results
    
    def save_results(self, results, filename, save_dir='results'):
        """保存评估结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存数值结果
        metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in results.items() if k not in ['predictions', 'targets']}
        metrics_path = os.path.join(save_dir, f'{filename}_metrics.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存预测结果
        predictions_path = os.path.join(save_dir, f'{filename}_predictions.npz')
        np.savez(predictions_path, 
                predictions=results['predictions'], 
                targets=results['targets'])
        
        print(f'Results saved to {metrics_path} and {predictions_path}')
        return metrics_path, predictions_path
