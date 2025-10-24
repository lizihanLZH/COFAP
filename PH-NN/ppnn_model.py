import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiModalSAGEModel(nn.Module):
    """
    Multi-modal SAGE model with separate MLPs for topological and structural features
    """
    
    def __init__(
        self,
        topo_dim: int = 18,  # topological fingerprint dimension (fixed)
        struct_dim: int = 5,  # structural features dimension
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation=F.relu,
    ):
        super(MultiModalSAGEModel, self).__init__()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        # Separate MLPs for each modality
        self.topo_mlp = self._build_mlp(topo_dim, hidden_dim, num_layers)
        self.struct_mlp = self._build_mlp(struct_dim, hidden_dim, num_layers)
        
        # Final MLP for combined features
        self.final_mlp = self._build_mlp(hidden_dim * 2, hidden_dim, num_layers)
        
        # Output layer
        self.out = nn.Linear(hidden_dim, 1)
    
    def _build_mlp(self, input_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
        """Build MLP with specified dimensions"""
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(self.dropout)
        
        # Additional layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(self.dropout)
        
        return nn.Sequential(*layers)
    
    def forward(self, topo_feat, struct_feat):
        """
        Forward pass with separate feature inputs
        """
        # Process each modality separately
        h_topo = self.topo_mlp(topo_feat)
        h_struct = self.struct_mlp(struct_feat)
        
        # Concatenate processed features
        h_combined = torch.cat([h_topo, h_struct], dim=1)
        
        # Final processing
        h_final = self.final_mlp(h_combined)
        
        # Output
        return self.out(h_final)

def train_step(dataloader, model, loss_fn, optimizer, device):
    """Training step for multi-modal model"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    
    total_loss = 0.0
    
    for batch, (topo_feat, struct_feat, y) in enumerate(dataloader):
        topo_feat = topo_feat.to(device)
        struct_feat = struct_feat.to(device)
        y = y.to(device)
        
        pred = model(topo_feat, struct_feat)
        loss = loss_fn(pred, y.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch % 50 == 0:
            loss_val, current = loss.item(), batch * len(y)
            print(f"train loss: {loss_val:>7f} [{current:>5d}/{size:>5d}]")
    
    return total_loss / num_batches

def valid_step(dataloader, model, loss_fn, device):
    """Validation step for multi-modal model"""
    num_batches = len(dataloader)
    model.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        for topo_feat, struct_feat, y in dataloader:
            topo_feat = topo_feat.to(device)
            struct_feat = struct_feat.to(device)
            y = y.to(device)
            
            pred = model(topo_feat, struct_feat)
            total_loss += loss_fn(pred, y.unsqueeze(1)).item()
    
    total_loss /= num_batches
    print(f"valid loss: {total_loss:>8f} \n")
    
    return total_loss

def predict_dataloader(model, dataloader, device):
    """Prediction for multi-modal model"""
    model.eval()
    
    y_true = torch.tensor([], dtype=torch.float32, device=device)
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for topo_feat, struct_feat, y in dataloader:
            topo_feat = topo_feat.to(device)
            struct_feat = struct_feat.to(device)
            y = y.to(device)
            
            pred = model(topo_feat, struct_feat)
            
            y_true = torch.cat((y_true, y), 0)
            y_pred = torch.cat((y_pred, pred.squeeze(1)), 0)
    
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    return y_true, y_pred

