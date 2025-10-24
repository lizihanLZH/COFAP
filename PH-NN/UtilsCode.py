import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class EarlyStopping:
    def __init__(self, prefix, patience=30):
        dt = datetime.datetime.now()
        Path("callbacks").mkdir(parents=True, exist_ok=True)
        
        filename = "callbacks/{}_early_stopping_{}_{:02d}-{:02d}-{:02d}.pth".format(
            prefix, datetime.datetime.now().date(), dt.hour, dt.minute, dt.second
        )
        
        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
    
    def step(self, score, model):
        self.timestep += 1
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, model):
        torch.save(
            {"model_state_dict": model.state_dict(), "timestep": self.timestep},
            self.filename,
        )
    
    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename)["model_state_dict"])

def get_stratified_folds(y, bins=10):
    return np.searchsorted(np.percentile(y, np.linspace(100 / bins, 100, bins)), y)

class COFDataset(Dataset):
    """Dataset for COF multi-modal features"""
    
    def __init__(self, topo_features, struct_features, targets):
        self.topo_features = torch.FloatTensor(topo_features)
        self.struct_features = torch.FloatTensor(struct_features)
        self.targets = torch.FloatTensor(targets)
        
        # Ensure all have same length
        assert len(self.topo_features) == len(self.struct_features) == len(self.targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.topo_features[idx],
            self.struct_features[idx],
            self.targets[idx]
        )

def get_samples(topo_features, struct_features, targets):
    """Get samples in the format expected by the dataset"""
    return list(zip(topo_features, struct_features, targets))

def collate_fn(samples):
    """Collate function for multi-modal features"""
    topo_feats, struct_feats, targets = zip(*samples)
    
    return (
        torch.stack(topo_feats),
        torch.stack(struct_feats),
        torch.stack(targets)
    )

def create_dataloader(topo_features, struct_features, targets, 
                     batch_size=32, shuffle=True, num_workers=8):
    """Create dataloader for multi-modal COF data"""
    dataset = COFDataset(topo_features, struct_features, targets)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

