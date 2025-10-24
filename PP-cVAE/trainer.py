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
    def __init__(self, model, config, save_dir='experiments'):
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.device = config.DEVICE
        os.makedirs(save_dir, exist_ok=True)
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
        self.use_mixed_precision = getattr(config, 'USE_MIXED_PRECISION', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        if self.use_mixed_precision:
            print("Using mixed precision training for faster performance")
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_mse = 0
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:  
                data, descriptors, target = batch_data
                data = data.to(self.device)
                descriptors = descriptors.to(self.device)
                target = target.to(self.device)
            else:  
                data, target = batch_data
                data = data.to(self.device)
                target = target.to(self.device)
                descriptors = None
            if hasattr(self.config, 'CHANNELS_LAST') and self.config.CHANNELS_LAST:
                if data.dim() == 4:
                    data = data.to(memory_format=torch.channels_last)
                elif data.dim() == 5:
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
                vae_loss = vae_loss_function(recon_loss, kl_loss, beta=self.config.VAE_BETA)
                mse_loss = F.mse_loss(prediction, target)
                total_loss_batch = mse_loss + self.config.VAE_WEIGHT * vae_loss
                total_loss_batch.backward()
                self.optimizer.step()
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        return total_loss / len(train_loader), total_mse / len(train_loader)
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating', leave=False)
            for batch_data in pbar:
                if len(batch_data) == 3:  
                    data, descriptors, target = batch_data
                    data = data.to(self.device)
                    descriptors = descriptors.to(self.device)
                    target = target.to(self.device)
                else:  
                    data, target = batch_data
                    data = data.to(self.device)
                    target = target.to(self.device)
                    descriptors = None
                if hasattr(self.config, 'CHANNELS_LAST') and self.config.CHANNELS_LAST:
                    if data.dim() == 4:
                        data = data.to(memory_format=torch.channels_last)
                    elif data.dim() == 5:
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
        val_mae = mean_absolute_error(targets, predictions)
        val_rmse = np.sqrt(mean_squared_error(targets, predictions))
        return val_loss, val_r2, val_mae, val_rmse, predictions, targets
    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        self.model = self.model.to(self.device)
        print(f'Training on {self.device}')
        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        epoch_pbar = tqdm(range(num_epochs), desc='Training Progress')
        for epoch in epoch_pbar:
            train_loss, train_mse = self.train_epoch(train_loader)
            val_loss, val_r2, val_mae, val_rmse, val_predictions, val_targets = self.validate_epoch(val_loader)
            self.scheduler.step(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_r2 = val_r2
            self.best_val_mae = val_mae
            self.best_val_rmse = val_rmse
            self.best_model_state = self.model.state_dict().copy()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_r2_scores.append(val_r2)
            self.val_mae_scores.append(val_mae)
            self.val_rmse_scores.append(val_rmse)
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss:.4f}',
                'Val_Loss': f'{val_loss:.4f}',
                'Val_R2': f'{val_r2:.4f}',
                'Val_MAE': f'{val_mae:.4f}',
                'Val_RMSE': f'{val_rmse:.4f}',
                'Best_Val': f'{self.best_val_loss:.4f}'
            })
            if epoch % 20 == 0 or epoch == num_epochs - 1:
                print(f'\nEpoch {epoch:3d}: Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}, '
                      f'Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}')
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        print(f'\n{"="*60}')
        print(f'训练完成！最佳验证结果：')
        print(f'最佳验证损失: {self.best_val_loss:.6f}')
        print(f'对应的R²分数: {self.best_val_r2:.6f}')
        print(f'对应的MAE: {self.best_val_mae:.6f}')
        print(f'对应的RMSE: {self.best_val_rmse:.6f}')
        print(f'{"="*60}')
        return self.model
    def save_model(self, filename=None, experiment_name=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if experiment_name:
                filename = f'{experiment_name}_{timestamp}'
            else:
                filename = f'model_{timestamp}'
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
        history_path = os.path.join(self.save_dir, f'{filename}_history.json')
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
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def evaluate(self, test_loader, label_scaler=None):
        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                prediction, _, _ = self.model(data)
                predictions.extend(prediction.cpu().numpy())
                targets.extend(target.cpu().numpy())
        if label_scaler is not None:
            predictions = label_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            targets = label_scaler.inverse_transform(np.array(targets).reshape(-1, 1)).flatten()
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
        os.makedirs(save_dir, exist_ok=True)
        metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in results.items() if k not in ['predictions', 'targets']}
        metrics_path = os.path.join(save_dir, f'{filename}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        predictions_path = os.path.join(save_dir, f'{filename}_predictions.npz')
        np.savez(predictions_path, 
                predictions=results['predictions'], 
                targets=results['targets'])
        print(f'Results saved to {metrics_path} and {predictions_path}')
        return metrics_path, predictions_path