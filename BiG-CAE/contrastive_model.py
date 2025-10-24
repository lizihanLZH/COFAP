#!/usr/bin/env python3
"""
基于对比学习的自编码器用于COF性质预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import mean_nodes
from dgl.nn.pytorch import HeteroGraphConv, GraphConv
import numpy as np

EMBEDDINGS = {
    "l2n": (300, 200),  # linker到node的嵌入维度
    "n2l": (200, 300),  # node到linker的嵌入维度
}

class ContrastiveAutoencoder(nn.Module):
    """对比学习的自编码器"""
    
    def __init__(
        self,
        encoder_dim=128,
        latent_dim=64,
        decoder_dim=128,
        temperature=0.1,
        alpha=0.1,  # 重构损失权重
        beta=1.0,   # 对比损失权重
    ):
        super(ContrastiveAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # 编码器 - 图卷积网络
        self.encoder = GraphEncoder(
            conv_dim=encoder_dim,
            latent_dim=latent_dim
        )
        
        # 解码器 - 重构图结构
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            decoder_dim=decoder_dim
        )
        
        # 性质预测头
        self.property_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 投影头 - 用于对比学习
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def encode(self, g):
        """编码图结构"""
        return self.encoder(g)
    
    def decode(self, z):
        """解码重构图结构"""
        return self.decoder(z)
    
    def project(self, z):
        """投影到对比学习空间"""
        return F.normalize(self.projector(z), dim=1)
    
    def forward(self, g, mode='autoencoder'):
        """前向传播"""
        if mode == 'autoencoder':
            # 自编码器模式
            z = self.encode(g)
            reconstructed = self.decode(z)
            property_pred = self.property_head(z)
            return reconstructed, property_pred, z
        
        elif mode == 'contrastive':
            # 对比学习模式
            z = self.encode(g)
            projected = self.project(z)
            return projected, z
        
        elif mode == 'property':
            # 性质预测模式
            z = self.encode(g)
            property_pred = self.property_head(z)
            return property_pred

class GraphEncoder(nn.Module):
    """图编码器"""
    
    def __init__(self, conv_dim=128, latent_dim=64, num_layers=3):
        super(GraphEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 第一层卷积
        self.conv_layers.append(
            HeteroGraphConv(
                {
                    k: GraphConv(v[0], conv_dim, norm='both')
                    for k, v in EMBEDDINGS.items()
                },
                aggregate="mean",
            )
        )
        self.norms.append(nn.ModuleDict({k: nn.LayerNorm(conv_dim) for k in ("n", "l")}))
        
        # 后续卷积层
        for i in range(1, num_layers):
            self.conv_layers.append(
                HeteroGraphConv(
                    {
                        k: GraphConv(conv_dim, conv_dim, norm='both')
                        for k, v in EMBEDDINGS.items()
                    },
                    aggregate="mean",
                )
            )
            self.norms.append(nn.ModuleDict({k: nn.LayerNorm(conv_dim) for k in ("n", "l")}))
        
        # 潜在空间映射
        self.latent_mapping = nn.Sequential(
            nn.Linear(conv_dim, conv_dim // 2),
            nn.ReLU(),
            nn.Linear(conv_dim // 2, latent_dim)
        )
        
    def forward(self, g):
        with g.local_scope():
            if all(g.num_nodes(ntype) == 0 for ntype in g.ntypes):
                return torch.zeros(1, self.latent_mapping[-1].out_features, device=next(self.parameters()).device)
            
            h = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}
            
            # 图卷积层
            for i in range(self.num_layers):
                h = self.conv_layers[i](g, (h, h))
                h = {k: self.norms[i][k](v) for k, v in h.items()}
                h = {k: F.elu(v) for k, v in h.items()}
            
            g.ndata["h"] = h
            
            # 图级别池化
            hg = 0
            for ntype in g.ntypes:
                if g.num_nodes(ntype) > 0:
                    hg = hg + mean_nodes(g, "h", ntype=ntype)
            
            # 映射到潜在空间
            z = self.latent_mapping(hg)
            return z

class GraphDecoder(nn.Module):
    """图解码器"""
    
    def __init__(self, latent_dim=64, decoder_dim=128):
        super(GraphDecoder, self).__init__()
        
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim // 2)
        )
        
    def forward(self, z):
        """解码潜在表示"""
        return self.decoder_net(z)

class ContrastiveLoss(nn.Module):
    """对比学习损失"""
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2, labels=None):
        """
        计算对比学习损失
        z1, z2: 两个增强视图的表示
        labels: 可选，用于有监督对比学习
        """
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.mm(z1, z2.T) / self.temperature
        
        # 对角线元素是正样本对
        labels = torch.arange(z1.size(0), device=z1.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss

class ReconstructionLoss(nn.Module):
    """重构损失"""
    
    def __init__(self, loss_type='mse'):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
    def forward(self, reconstructed, original):
        if self.loss_type == 'mse':
            return F.mse_loss(reconstructed, original)
        elif self.loss_type == 'mae':
            return F.l1_loss(reconstructed, original)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(reconstructed, original)

def create_augmented_views(g, augmentation_strength=0.1):
    """创建图的数据增强视图"""
    # 这里可以实现各种图增强方法
    # 例如：节点特征噪声、边删除、子图采样等
    
    # 简单的节点特征噪声增强
    g_aug1 = g.clone()
    g_aug2 = g.clone()
    
    for ntype in g_aug1.ntypes:
        if g_aug1.num_nodes(ntype) > 0:
            noise1 = torch.randn_like(g_aug1.nodes[ntype].data["feat"]) * augmentation_strength
            noise2 = torch.randn_like(g_aug2.nodes[ntype].data["feat"]) * augmentation_strength
            g_aug1.nodes[ntype].data["feat"] += noise1
            g_aug2.nodes[ntype].data["feat"] += noise2
    
    return g_aug1, g_aug2

def train_contrastive_autoencoder(
    model, 
    train_loader, 
    val_loader, 
    epochs=100, 
    lr=0.001,
    device='cuda'
):
    """训练对比学习自编码器"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    contrastive_loss_fn = ContrastiveLoss(temperature=0.1)
    reconstruction_loss_fn = ReconstructionLoss(loss_type='huber')
    property_loss_fn = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (graphs, labels) in enumerate(train_loader):
            graphs = graphs.to(device)
            labels = labels.to(device)
            
            # 创建增强视图
            aug_graphs1, aug_graphs2 = [], []
            for g in graphs:
                g1, g2 = create_augmented_views(g, augmentation_strength=0.1)
                aug_graphs1.append(g1)
                aug_graphs2.append(g2)
            
            # 对比学习损失
            z1 = model.encode(aug_graphs1)
            z2 = model.encode(aug_graphs2)
            proj1 = model.project(z1)
            proj2 = model.project(z2)
            contrastive_loss = contrastive_loss_fn(proj1, proj2)
            
            # 重构损失
            reconstructed, property_pred, z = model(graphs, mode='autoencoder')
            recon_loss = reconstruction_loss_fn(reconstructed, z)
            
            # 性质预测损失
            prop_loss = property_loss_fn(property_pred.squeeze(), labels)
            
            # 总损失
            total_batch_loss = (
                model.beta * contrastive_loss + 
                model.alpha * recon_loss + 
                prop_loss
            )
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                      f"Loss: {total_batch_loss.item():.6f}, "
                      f"Contrastive: {contrastive_loss.item():.6f}, "
                      f"Recon: {recon_loss.item():.6f}, "
                      f"Property: {prop_loss.item():.6f}")
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graphs, labels in val_loader:
                graphs = graphs.to(device)
                labels = labels.to(device)
                
                property_pred = model(graphs, mode='property')
                val_loss += property_loss_fn(property_pred.squeeze(), labels).item()
        
        val_loss /= len(val_loader)
        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_losses[-1]:.6f}, "
              f"Val Loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = ContrastiveAutoencoder(
        encoder_dim=128,
        latent_dim=64,
        decoder_dim=128,
        temperature=0.1,
        alpha=0.1,
        beta=1.0
    )
    
    print("对比学习自编码器模型创建成功！")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 这里需要你的数据加载器
    # train_loader, val_loader = create_data_loaders(...)
    # train_contrastive_autoencoder(model, train_loader, val_loader) 