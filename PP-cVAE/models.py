import torch
import torch.nn as nn
import torch.nn.functional as F

class PPCVAE(nn.Module):
    
    def __init__(self, input_channels=2, latent_dim=128):
        super(PPCVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class DescriptorMLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.3):
        super(DescriptorMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x):
        return self.mlp(x)

class MultiModalVAECNN(nn.Module):
    
    def __init__(self, latent_dim=128, num_planes=9, descriptor_dim=0, dropout_rate=0.3):
        super(MultiModalVAECNN, self).__init__()
        self.num_planes = num_planes
        self.latent_dim = latent_dim
        self.descriptor_dim = descriptor_dim
        self.use_descriptors = descriptor_dim > 0
        
        self.vae = PPCVAE(input_channels=2, latent_dim=latent_dim)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(num_planes, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        if self.use_descriptors:
            self.descriptor_mlp = DescriptorMLP(
                input_dim=descriptor_dim,
                hidden_dims=[64, 32],
                dropout_rate=dropout_rate
            )
            descriptor_feature_dim = 32
        else:
            self.descriptor_mlp = None
            descriptor_feature_dim = 0
        
        combined_feature_dim = 32 + latent_dim + descriptor_feature_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, descriptors=None):
        batch_size = x.size(0)
        
        plane_features = []
        total_recon_loss = 0
        total_kl_loss = 0
        
        for i in range(self.num_planes):
            plane_data = x[:, i]
            recon, mu, logvar, z = self.vae(plane_data)
            recon_loss = F.mse_loss(recon, plane_data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss
            
            plane_features.append(z)
        
        stacked_features = torch.stack(plane_features, dim=1)
        
        fused = self.fusion_conv(stacked_features)
        fused = fused.squeeze(-1)
        
        avg_features = torch.mean(torch.stack(plane_features, dim=1), dim=1)
        
        features_to_combine = [fused, avg_features]
        
        if self.use_descriptors and descriptors is not None:
            descriptor_features = self.descriptor_mlp(descriptors)
            features_to_combine.append(descriptor_features)
        
        combined_features = torch.cat(features_to_combine, dim=1)
        
        prediction = self.regressor(combined_features)
        
        return prediction.squeeze(-1), total_recon_loss, total_kl_loss
    
    def get_features(self, x, descriptors=None):
        batch_size = x.size(0)
        plane_features = []
        
        with torch.no_grad():
            for i in range(self.num_planes):
                plane_data = x[:, i]
                _, _, _, z = self.vae(plane_data)
                plane_features.append(z)
        
        stacked_features = torch.stack(plane_features, dim=1)
        fused = self.fusion_conv(stacked_features)
        fused = fused.squeeze(-1)
        avg_features = torch.mean(torch.stack(plane_features, dim=1), dim=1)
        
        features_to_combine = [fused, avg_features]
        
        if self.use_descriptors and descriptors is not None:
            descriptor_features = self.descriptor_mlp(descriptors)
            features_to_combine.append(descriptor_features)
        
        combined_features = torch.cat(features_to_combine, dim=1)
        
        return combined_features

class MultiPlaneVAECNN(nn.Module):
    
    def __init__(self, latent_dim=128, num_planes=9, dropout_rate=0.3):
        super(MultiPlaneVAECNN, self).__init__()
        self.num_planes = num_planes
        self.latent_dim = latent_dim
        
        self.vae = PPCVAE(input_channels=2, latent_dim=latent_dim)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(num_planes, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(32 + latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        plane_features = []
        total_recon_loss = 0
        total_kl_loss = 0
        
        for i in range(self.num_planes):
            plane_data = x[:, i]
            recon, mu, logvar, z = self.vae(plane_data)
            recon_loss = F.mse_loss(recon, plane_data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss
            
            plane_features.append(z)
        
        stacked_features = torch.stack(plane_features, dim=1)
        
        fused = self.fusion_conv(stacked_features)
        fused = fused.squeeze(-1)
        
        avg_features = torch.mean(torch.stack(plane_features, dim=1), dim=1)
        
        combined_features = torch.cat([fused, avg_features], dim=1)
        
        prediction = self.regressor(combined_features)
        
        return prediction.squeeze(-1), total_recon_loss, total_kl_loss
    
    def get_features(self, x):
        batch_size = x.size(0)
        plane_features = []
        
        with torch.no_grad():
            for i in range(self.num_planes):
                plane_data = x[:, i]
                _, _, _, z = self.vae(plane_data)
                plane_features.append(z)
        
        stacked_features = torch.stack(plane_features, dim=1)
        fused = self.fusion_conv(stacked_features)
        fused = fused.squeeze(-1)
        avg_features = torch.mean(torch.stack(plane_features, dim=1), dim=1)
        combined_features = torch.cat([fused, avg_features], dim=1)
        
        return combined_features

def create_model(config, descriptor_dim=0, use_multimodal=False):
    if use_multimodal and descriptor_dim > 0:
        model = MultiModalVAECNN(
            latent_dim=config.LATENT_DIM,
            num_planes=config.NUM_PLANES,
            descriptor_dim=descriptor_dim,
            dropout_rate=config.DROPOUT_RATE
        )
    else:
        model = MultiPlaneVAECNN(
            latent_dim=config.LATENT_DIM,
            num_planes=config.NUM_PLANES,
            dropout_rate=config.DROPOUT_RATE
        )
    
    if hasattr(config, 'CHANNELS_LAST') and config.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    
    if hasattr(config, 'COMPILE_MODEL') and config.COMPILE_MODEL:
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception as e:
            print(f"{e}")
    
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vae_loss_function(recon_loss, kl_loss, beta=1.0):
    return recon_loss + beta * kl_loss