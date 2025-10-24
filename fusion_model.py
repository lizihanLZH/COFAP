import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append('PP-cVAE')
sys.path.append('BiG-CAE') 
sys.path.append('PP-NN')
try:
    original_path = sys.path.copy()
    sys.path.insert(0, 'PP-cVAE')
    from models import DescriptorMLP, MultiModalVAECNN, MultiPlaneVAECNN
    sys.path = original_path
    VAE_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"{e}")
    VAE_MODELS_AVAILABLE = False
try:
    original_path = sys.path.copy()  
    sys.path.insert(0, 'BiG-CAE')
    from contrastive_autoencoder import ContrastiveAutoencoder
    sys.path = original_path
    GCN_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"{e}")
    GCN_MODELS_AVAILABLE = False
try:
    original_path = sys.path.copy()
    sys.path.insert(0, 'PP-NN')
    from Net import MultiModalSAGEModel
    sys.path = original_path
    LZHNN_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"{e}")
    LZHNN_MODELS_AVAILABLE = False
class ExactCrossAttentionFusion(nn.Module):
    def __init__(self, 
                 vae_config,
                 use_descriptors=False,
                 descriptor_dim=0,
                 fusion_dim=128):
        super(ExactCrossAttentionFusion, self).__init__()
        if not VAE_MODELS_AVAILABLE:
            raise ImportError("PP-cVAE不可用")
        if not GCN_MODELS_AVAILABLE:
            raise ImportError("BiG-CAE不可用")  
        if not LZHNN_MODELS_AVAILABLE:
            raise ImportError("PP-NN不可用")
        if use_descriptors and descriptor_dim > 0:
            self.vae_model = MultiModalVAECNN(
                latent_dim=vae_config.get('latent_dim', 128),
                num_planes=vae_config.get('num_planes', 9),
                descriptor_dim=descriptor_dim,
                dropout_rate=vae_config.get('dropout_rate', 0.3)
            )
        else:
            self.vae_model = MultiPlaneVAECNN(
                latent_dim=vae_config.get('latent_dim', 128),
                num_planes=vae_config.get('num_planes', 9),
                dropout_rate=vae_config.get('dropout_rate', 0.3)
            )
        self.gcn_cc_model = ContrastiveAutoencoder(
            encoder_dim=128,
            latent_dim=64,
            decoder_dim=128,
            temperature=0.1,
            alpha=0.1,
            beta=1.0
        )
        self.gcn_noncc_model = ContrastiveAutoencoder(
            encoder_dim=128,
            latent_dim=64,
            decoder_dim=128,
            temperature=0.1,
            alpha=0.1,
            beta=1.0
        )
        self.lzhnn_model = MultiModalSAGEModel(
            topo_dim=18,
            struct_dim=5,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
            activation=F.relu
        )
        self._freeze_pretrained_models()
        vae_feature_dim = self._calculate_vae_feature_dim(vae_config, use_descriptors, descriptor_dim)
        self.vae_proj = nn.Linear(vae_feature_dim, fusion_dim)
        self.gcn_proj = nn.Linear(64, fusion_dim)
        self.lzhnn_proj = nn.Linear(128, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 1)
        )
        self.main_weight_raw = nn.Parameter(torch.tensor(0.8))
        self.fusion_weight_raw = nn.Parameter(torch.tensor(0.2))
    def get_normalized_weights(self):
        weights = torch.softmax(torch.stack([self.main_weight_raw, self.fusion_weight_raw]), dim=0)
        return weights[0], weights[1]  
    def _calculate_vae_feature_dim(self, vae_config, use_descriptors, descriptor_dim):
        dim = 32 + vae_config.get('latent_dim', 128)
        if use_descriptors and descriptor_dim > 0:
            dim += 32
        return dim
    def _freeze_pretrained_models(self):
        for model in [self.vae_model, self.gcn_cc_model, self.gcn_noncc_model, self.lzhnn_model]:
            for param in model.parameters():
                param.requires_grad = False
    def to(self, device):
        super().to(device)
        self.vae_model = self.vae_model.to(device)
        self.gcn_cc_model = self.gcn_cc_model.to(device)
        self.gcn_noncc_model = self.gcn_noncc_model.to(device)
        self.lzhnn_model = self.lzhnn_model.to(device)
        self.vae_proj = self.vae_proj.to(device)
        self.gcn_proj = self.gcn_proj.to(device)
        self.lzhnn_proj = self.lzhnn_proj.to(device)
        self.cross_attention = self.cross_attention.to(device)
        self.fusion_net = self.fusion_net.to(device)
        print(f"{device}")
        return self
    def _update_vae_feature_projection(self, descriptor_dim):
        new_vae_feature_dim = 32 + 128  
        if descriptor_dim > 0:
            new_vae_feature_dim += 32  
        print(f"  更新VAE特征投影层: {self.vae_proj.in_features} -> {new_vae_feature_dim}")
        fusion_dim = self.vae_proj.out_features  
        device = next(self.parameters()).device
        self.vae_proj = nn.Linear(new_vae_feature_dim, fusion_dim).to(device)
        print(f"  VAE投影层移动到设备: {device}")
    def load_pretrained_weights(self, model_paths):
        if 'vae' in model_paths and model_paths['vae']:
            try:
                print(f"加载VAE权重: {model_paths['vae']}")
                checkpoint = torch.load(model_paths['vae'], map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  检测到标准格式，state_dict键数量: {len(state_dict)}")
                else:
                    state_dict = checkpoint
                    print(f"  检测到直接state_dict格式，键数量: {len(state_dict)}")
                has_descriptor = any(key.startswith('descriptor_mlp') for key in state_dict.keys())
                has_vae_prefix = any(key.startswith('vae.') for key in state_dict.keys())
                print(f"  权重文件分析: 包含描述符={has_descriptor}, VAE前缀={has_vae_prefix}")
                if has_descriptor:
                    descriptor_dim = state_dict['descriptor_mlp.mlp.0.weight'].shape[1]
                    regressor_input_dim = state_dict['regressor.0.weight'].shape[1]
                    print(f"  检测到描述符维度: {descriptor_dim}")
                    print(f"  检测到回归器输入维度: {regressor_input_dim}")
                    expected_regressor_dim = 32 + 128 + 32  
                    if regressor_input_dim != expected_regressor_dim:
                        print(f"  ⚠ 回归器维度不匹配: 期望{expected_regressor_dim}, 实际{regressor_input_dim}")
                    print(f"  重新创建MultiModalVAECNN模型...")
                    new_vae = MultiModalVAECNN(
                        latent_dim=128,
                        num_planes=9,
                        descriptor_dim=descriptor_dim,
                        dropout_rate=0.3
                    )
                    missing_keys, unexpected_keys = new_vae.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f"  缺失的键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                    if unexpected_keys:
                        print(f"  意外的键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                    device = next(self.parameters()).device
                    new_vae = new_vae.to(device)
                    print(f"  VAE模型移动到设备: {device}")
                    for param in new_vae.parameters():
                        param.requires_grad = False
                    self.vae_model = new_vae
                    print("✓ VAE模型（带描述符）权重加载成功")
                    self._update_vae_feature_projection(descriptor_dim)
                else:
                    print(f"  使用MultiPlaneVAECNN模型...")
                    missing_keys, unexpected_keys = self.vae_model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f"  缺失的键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                    if unexpected_keys:
                        print(f"  意外的键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                    print("✓ VAE模型（无描述符）权重加载成功")
            except Exception as e:
                print(f"⚠ VAE权重加载失败: {e}")
                import traceback
                traceback.print_exc()
        if 'gcn_cc' in model_paths and model_paths['gcn_cc']:
            try:
                print(f"加载GCN CC权重: {model_paths['gcn_cc']}")
                checkpoint = torch.load(model_paths['gcn_cc'], map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  检测到标准格式，state_dict键数量: {len(state_dict)}")
                else:
                    state_dict = checkpoint
                    print(f"  检测到直接state_dict格式，键数量: {len(state_dict)}")
                encoder_keys = [key for key in state_dict.keys() if key.startswith('encoder')]
                decoder_keys = [key for key in state_dict.keys() if key.startswith('decoder')]
                property_keys = [key for key in state_dict.keys() if key.startswith('property_head')]
                projector_keys = [key for key in state_dict.keys() if key.startswith('projector')]
                print(f"  权重文件分析:")
                print(f"    编码器层: {len(encoder_keys)} 个")
                print(f"    解码器层: {len(decoder_keys)} 个")
                print(f"    性质预测层: {len(property_keys)} 个")
                print(f"    投影层: {len(projector_keys)} 个")
                encoder_state_dict = {key: value for key, value in state_dict.items() if key.startswith('encoder')}
                if encoder_state_dict:
                    print(f"  提取编码器权重: {len(encoder_state_dict)} 个")
                    cleaned_encoder_state_dict = {}
                    for key, value in encoder_state_dict.items():
                        if key.startswith('encoder.'):
                            cleaned_key = key[8:]  
                            cleaned_encoder_state_dict[cleaned_key] = value
                        else:
                            cleaned_encoder_state_dict[key] = value
                    print(f"  清理后的编码器权重: {len(cleaned_encoder_state_dict)} 个")
                    missing_keys, unexpected_keys = self.gcn_cc_model.encoder.load_state_dict(cleaned_encoder_state_dict, strict=True)
                    if missing_keys:
                        print(f"  ❌ 编码器缺失的键: {missing_keys}")
                    if unexpected_keys:
                        print(f"  ⚠ 编码器意外的键: {unexpected_keys}")
                    if not missing_keys:
                        print("✓ GCN CC编码器权重加载成功")
                    else:
                        print("❌ GCN CC编码器权重加载失败")
                else:
                    print("❌ 权重文件中没有找到编码器权重")
            except Exception as e:
                print(f"⚠ GCN CC权重加载失败: {e}")
                import traceback
                traceback.print_exc()
        if 'gcn_noncc' in model_paths and model_paths['gcn_noncc']:
            try:
                print(f"加载GCN Non-CC权重: {model_paths['gcn_noncc']}")
                checkpoint = torch.load(model_paths['gcn_noncc'], map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  检测到标准格式，state_dict键数量: {len(state_dict)}")
                else:
                    state_dict = checkpoint
                    print(f"  检测到直接state_dict格式，键数量: {len(state_dict)}")
                encoder_keys = [key for key in state_dict.keys() if key.startswith('encoder')]
                decoder_keys = [key for key in state_dict.keys() if key.startswith('decoder')]
                property_keys = [key for key in state_dict.keys() if key.startswith('property_head')]
                projector_keys = [key for key in state_dict.keys() if key.startswith('projector')]
                print(f"  权重文件分析:")
                print(f"    编码器层: {len(encoder_keys)} 个")
                print(f"    解码器层: {len(decoder_keys)} 个")
                print(f"    性质预测层: {len(property_keys)} 个")
                print(f"    投影层: {len(projector_keys)} 个")
                encoder_state_dict = {key: value for key, value in state_dict.items() if key.startswith('encoder')}
                if encoder_state_dict:
                    print(f"  提取编码器权重: {len(encoder_state_dict)} 个")
                    cleaned_encoder_state_dict = {}
                    for key, value in encoder_state_dict.items():
                        if key.startswith('encoder.'):
                            cleaned_key = key[8:]  
                            cleaned_encoder_state_dict[cleaned_key] = value
                        else:
                            cleaned_encoder_state_dict[key] = value
                    print(f"  清理后的编码器权重: {len(cleaned_encoder_state_dict)} 个")
                    missing_keys, unexpected_keys = self.gcn_noncc_model.encoder.load_state_dict(cleaned_encoder_state_dict, strict=True)
                    if missing_keys:
                        print(f"  ❌ 编码器缺失的键: {missing_keys}")
                    if unexpected_keys:
                        print(f"  ⚠ 编码器意外的键: {unexpected_keys}")
                    if not missing_keys:
                        print("✓ GCN Non-CC编码器权重加载成功")
                    else:
                        print("❌ GCN Non-CC编码器权重加载失败")
                else:
                    print("❌ 权重文件中没有找到编码器权重")
            except Exception as e:
                print(f"⚠ GCN Non-CC权重加载失败: {e}")
                import traceback
                traceback.print_exc()
        lzhnn_path = model_paths.get('lzhnn') or model_paths.get('lzhnn_cc')
        if lzhnn_path:
            try:
                print(f"加载lzhnn权重: {lzhnn_path}")
                checkpoint = torch.load(lzhnn_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  检测到标准格式，state_dict键数量: {len(state_dict)}")
                else:
                    state_dict = checkpoint
                    print(f"  检测到直接state_dict格式，键数量: {len(state_dict)}")
                topo_keys = [key for key in state_dict.keys() if key.startswith('topo_mlp')]
                chem_keys = [key for key in state_dict.keys() if key.startswith('chem_mlp')]
                struct_keys = [key for key in state_dict.keys() if key.startswith('struct_mlp')]
                final_keys = [key for key in state_dict.keys() if key.startswith('final_mlp')]
                out_keys = [key for key in state_dict.keys() if key.startswith('out')]
                print(f"  权重文件分析:")
                print(f"    拓扑MLP层: {len(topo_keys)} 个")
                print(f"    化学MLP层: {len(chem_keys)} 个")
                print(f"    结构MLP层: {len(struct_keys)} 个")
                print(f"    最终MLP层: {len(final_keys)} 个")
                print(f"    输出层: {len(out_keys)} 个")
                feature_keys = topo_keys + chem_keys + struct_keys + final_keys
                feature_state_dict = {key: value for key, value in state_dict.items() if key in feature_keys}
                if feature_state_dict:
                    print(f"  提取特征提取权重: {len(feature_state_dict)} 个")
                    missing_keys, unexpected_keys = self.lzhnn_model.load_state_dict(feature_state_dict, strict=False)
                    if missing_keys:
                        print(f"  ⚠ 特征提取缺失的键: {missing_keys}")
                    if unexpected_keys:
                        print(f"  ⚠ 特征提取意外的键: {unexpected_keys}")
                    key_layers_loaded = all([
                        len(topo_keys) > 0,
                        len(chem_keys) > 0,
                        len(struct_keys) > 0,
                        len(final_keys) > 0
                    ])
                    if key_layers_loaded:
                        print("✓ lzhnn特征提取权重加载成功")
                    else:
                        print("⚠ lzhnn特征提取权重部分加载")
                else:
                    print("❌ 权重文件中没有找到特征提取权重")
            except Exception as e:
                print(f"⚠ lzhnn权重加载失败: {e}")
                import traceback
                traceback.print_exc()
    def extract_vae_features(self, vae_input, descriptors=None):
        with torch.no_grad():
            features = self.vae_model.get_features(vae_input, descriptors)
            return features
    def extract_gcn_features(self, graph_data, model_type='cc'):
        if graph_data is None:
            device = next(self.gcn_cc_model.parameters()).device
            return torch.zeros(1, 64, device=device)
        model = self.gcn_cc_model if model_type == 'cc' else self.gcn_noncc_model
        device = next(model.parameters()).device
        with torch.no_grad():
            try:
                if isinstance(graph_data, list):
                    batch_features = []
                    for graph in graph_data:
                        if graph is not None:
                            if graph.device != device:
                                graph = graph.to(device)
                            latent_features = model.encoder(graph)
                            batch_features.append(latent_features)
                        else:
                            batch_features.append(torch.zeros(1, 64, device=device))
                    if batch_features:
                        return torch.cat(batch_features, dim=0)
                    else:
                        return torch.zeros(1, 64, device=device)
                else:
                    if graph_data.device != device:
                        graph_data = graph_data.to(device)
                latent_features = model.encoder(graph_data)
                return latent_features
            except Exception as e:
                print(f"GCN特征提取失败: {e}")
                if isinstance(graph_data, list):
                    batch_size = len(graph_data)
                else:
                    batch_size = 1
                return torch.zeros(batch_size, 64, device=device)
    def extract_lzhnn_features(self, topo_feat, struct_feat):
        with torch.no_grad():
            try:
                device = next(self.lzhnn_model.parameters()).device
                topo_feat = topo_feat.to(device)
                struct_feat = struct_feat.to(device)
                h_topo = self.lzhnn_model.topo_mlp(topo_feat)
                h_struct = self.lzhnn_model.struct_mlp(struct_feat)
                h_combined = torch.cat([h_topo, h_struct], dim=1)
                features = self.lzhnn_model.final_mlp(h_combined)
                return features
            except Exception as e:
                device = next(self.lzhnn_model.parameters()).device
                batch_size = topo_feat.size(0) if topo_feat is not None else 1
                return torch.zeros(batch_size, 128, device=device)
    def forward(self, inputs):
        device = next(self.parameters()).device
        batch_size = inputs['vae_data'].size(0)
        vae_features = self.extract_vae_features(
            inputs['vae_data'], 
            inputs.get('descriptors')
        )
        with torch.no_grad():
            if hasattr(self.vae_model, 'descriptor_mlp'):
                vae_pred, _, _ = self.vae_model(inputs['vae_data'], inputs.get('descriptors'))
            else:
                vae_pred, _, _ = self.vae_model(inputs['vae_data'])
        gcn_cc_features = None
        gcn_noncc_features = None
        if inputs.get('graph_cc') is not None:
            gcn_cc_features = self.extract_gcn_features(inputs['graph_cc'], 'cc')
        if inputs.get('graph_noncc') is not None:
            gcn_noncc_features = self.extract_gcn_features(inputs['graph_noncc'], 'noncc')
        lzhnn_features = None
        if inputs.get('lzhnn_data') is not None:
            lzhnn_data = inputs['lzhnn_data']
            lzhnn_features = self.extract_lzhnn_features(
                lzhnn_data['topo'], 
                lzhnn_data['struct']
            )
        vae_proj = self.vae_proj(vae_features)  
        if vae_proj.dim() > 2:
            vae_proj = vae_proj.squeeze()  
        if vae_proj.dim() == 1:
            vae_proj = vae_proj.unsqueeze(0)  
        auxiliary_features = []
        if gcn_cc_features is not None or gcn_noncc_features is not None:
            cc_mask = inputs.get('cc_mask', torch.ones(batch_size, 1, device=device))
            if gcn_cc_features is not None and gcn_noncc_features is not None:
                if gcn_cc_features.size(0) != batch_size:
                    print(f"警告：GCN CC特征batch_size不匹配: {gcn_cc_features.size(0)} vs {batch_size}")
                    if gcn_cc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_cc_features.size(0), 64, device=device)
                        gcn_cc_features = torch.cat([gcn_cc_features, padding], dim=0)
                    else:
                        gcn_cc_features = gcn_cc_features[:batch_size]
                if gcn_noncc_features.size(0) != batch_size:
                    print(f"警告：GCN Non-CC特征batch_size不匹配: {gcn_noncc_features.size(0)} vs {batch_size}")
                    if gcn_noncc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_noncc_features.size(0), 64, device=device)
                        gcn_noncc_features = torch.cat([gcn_noncc_features, padding], dim=0)
                    else:
                        gcn_noncc_features = gcn_noncc_features[:batch_size]
                gcn_features = cc_mask * gcn_cc_features + (1 - cc_mask) * gcn_noncc_features
            elif gcn_cc_features is not None:
                if gcn_cc_features.size(0) != batch_size:
                    if gcn_cc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_cc_features.size(0), 64, device=device)
                        gcn_cc_features = torch.cat([gcn_cc_features, padding], dim=0)
                    else:
                        gcn_cc_features = gcn_cc_features[:batch_size]
                gcn_features = gcn_cc_features
            else:
                if gcn_noncc_features.size(0) != batch_size:
                    if gcn_noncc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_noncc_features.size(0), 64, device=device)
                        gcn_noncc_features = torch.cat([gcn_noncc_features, padding], dim=0)
                    else:
                        gcn_noncc_features = gcn_noncc_features[:batch_size]
                gcn_features = gcn_noncc_features
            gcn_proj = self.gcn_proj(gcn_features)
            if gcn_proj.dim() > 2:
                gcn_proj = gcn_proj.squeeze()
            if gcn_proj.dim() == 1:
                gcn_proj = gcn_proj.unsqueeze(0)
            auxiliary_features.append(gcn_proj)
        if lzhnn_features is not None:
            if lzhnn_features.size(0) != batch_size:
                lzhnn_features = lzhnn_features.expand(batch_size, -1)
            lzhnn_proj = self.lzhnn_proj(lzhnn_features)
            if lzhnn_proj.dim() > 2:
                lzhnn_proj = lzhnn_proj.squeeze()
            if lzhnn_proj.dim() == 1:
                lzhnn_proj = lzhnn_proj.unsqueeze(0)
            auxiliary_features.append(lzhnn_proj)
        if auxiliary_features:
            query = vae_proj.unsqueeze(1)  
            auxiliary_stack = torch.stack(auxiliary_features, dim=1)  
            attended_features, attention_weights = self.cross_attention(
                query, auxiliary_stack, auxiliary_stack
            )
            attended_features = attended_features.squeeze(1)  
            all_features = torch.cat([vae_proj, attended_features], dim=1)  
            fusion_pred = self.fusion_net(all_features).squeeze(-1)
            main_weight, fusion_weight = self.get_normalized_weights()
            final_pred = main_weight * vae_pred + fusion_weight * fusion_pred
        else:
            final_pred = vae_pred
            attention_weights = None
        return {
            'prediction': final_pred,
            'vae_prediction': vae_pred,
            'attention_weights': attention_weights,
            'vae_features': vae_features
        }
    def get_attention_analysis(self, inputs):
        device = next(self.parameters()).device
        batch_size = inputs['vae_data'].size(0)
        vae_features = self.extract_vae_features(
            inputs['vae_data'], 
            inputs.get('descriptors')
        )
        with torch.no_grad():
            if hasattr(self.vae_model, 'descriptor_mlp'):
                vae_pred, _, _ = self.vae_model(inputs['vae_data'], inputs.get('descriptors'))
            else:
                vae_pred, _, _ = self.vae_model(inputs['vae_data'])
        gcn_cc_features = None
        gcn_noncc_features = None
        if inputs.get('graph_cc') is not None:
            gcn_cc_features = self.extract_gcn_features(inputs['graph_cc'], 'cc')
        if inputs.get('graph_noncc') is not None:
            gcn_noncc_features = self.extract_gcn_features(inputs['graph_noncc'], 'noncc')
        lzhnn_features = None
        if inputs.get('lzhnn_data') is not None:
            lzhnn_data = inputs['lzhnn_data']
            lzhnn_features = self.extract_lzhnn_features(
                lzhnn_data['topo'], 
                lzhnn_data['struct']
            )
        vae_proj = self.vae_proj(vae_features)
        auxiliary_features = []
        feature_sources = ['vae']  
        gcn_features = None
        if gcn_cc_features is not None or gcn_noncc_features is not None:
            cc_mask = inputs.get('cc_mask', torch.ones(batch_size, 1, device=device))
            if gcn_cc_features is not None and gcn_noncc_features is not None:
                if gcn_cc_features.size(0) != batch_size:
                    if gcn_cc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_cc_features.size(0), 64, device=device)
                        gcn_cc_features = torch.cat([gcn_cc_features, padding], dim=0)
                    else:
                        gcn_cc_features = gcn_cc_features[:batch_size]
                if gcn_noncc_features.size(0) != batch_size:
                    if gcn_noncc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_noncc_features.size(0), 64, device=device)
                        gcn_noncc_features = torch.cat([gcn_noncc_features, padding], dim=0)
                    else:
                        gcn_noncc_features = gcn_noncc_features[:batch_size]
                gcn_features = cc_mask * gcn_cc_features + (1 - cc_mask) * gcn_noncc_features
            elif gcn_cc_features is not None:
                if gcn_cc_features.size(0) != batch_size:
                    if gcn_cc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_cc_features.size(0), 64, device=device)
                        gcn_cc_features = torch.cat([gcn_cc_features, padding], dim=0)
                    else:
                        gcn_cc_features = gcn_cc_features[:batch_size]
                gcn_features = gcn_cc_features
            else:
                if gcn_noncc_features.size(0) != batch_size:
                    if gcn_noncc_features.size(0) < batch_size:
                        padding = torch.zeros(batch_size - gcn_noncc_features.size(0), 64, device=device)
                        gcn_noncc_features = torch.cat([gcn_noncc_features, padding], dim=0)
                    else:
                        gcn_noncc_features = gcn_noncc_features[:batch_size]
                gcn_features = gcn_noncc_features
            gcn_proj = self.gcn_proj(gcn_features)
            auxiliary_features.append(gcn_proj)
            feature_sources.append('gcn')
        if lzhnn_features is not None:
            if lzhnn_features.size(0) != batch_size:
                lzhnn_features = lzhnn_features.expand(batch_size, -1)
            lzhnn_proj = self.lzhnn_proj(lzhnn_features)
            auxiliary_features.append(lzhnn_proj)
            feature_sources.append('lzhnn')
        attention_info = {
            'feature_sources': feature_sources,
            'feature_norms': {
                'vae': torch.norm(vae_proj, dim=1).cpu().numpy()
            },
            'vae_weight': self.get_normalized_weights()[0].item(),  
            'fusion_weight': self.get_normalized_weights()[1].item()  
        }
        if auxiliary_features:
            query = vae_proj.unsqueeze(1)
            auxiliary_stack = torch.stack(auxiliary_features, dim=1)
            attended_features, attention_weights = self.cross_attention(
                query, auxiliary_stack, auxiliary_stack
            )
            attended_features = attended_features.squeeze(1)
            attention_weights = attention_weights.squeeze(1)  
            for i, source in enumerate(feature_sources[1:]):  
                feature_norm = torch.norm(auxiliary_features[i], dim=1).cpu().numpy()
                attention_info['feature_norms'][source] = feature_norm
            attention_info['attention_weights'] = attention_weights.cpu().numpy()  
            attention_info['attention_mean'] = attention_weights.mean(dim=0).cpu().numpy()  
            attention_info['attention_std'] = attention_weights.std(dim=0).cpu().numpy()   
            if len(auxiliary_features) > 0:
                aux_effective_weight = fusion_weight * attention_weights
                vae_effective_weight = torch.full((batch_size,), main_weight, device=device)
                attention_info['effective_weights'] = {
                    'vae': vae_effective_weight.cpu().numpy()
                }
                for i, source in enumerate(feature_sources[1:]):
                    attention_info['effective_weights'][source] = aux_effective_weight[:, i].cpu().numpy()
            all_features = torch.cat([vae_proj, attended_features], dim=1)  
            fusion_pred = self.fusion_net(all_features).squeeze(-1)
            main_weight, fusion_weight = self.get_normalized_weights()
            final_pred = main_weight * vae_pred + fusion_weight * fusion_pred
            attention_info['predictions'] = {
                'vae_prediction': vae_pred.cpu().numpy(),
                'fusion_prediction': fusion_pred.cpu().numpy(),
                'final_prediction': final_pred.cpu().numpy()
            }
        else:
            final_pred = vae_pred
            attention_info['attention_weights'] = None
            attention_info['effective_weights'] = {'vae': np.ones(batch_size)}
            attention_info['predictions'] = {
                'vae_prediction': vae_pred.cpu().numpy(),
                'fusion_prediction': vae_pred.cpu().numpy(),
                'final_prediction': final_pred.cpu().numpy()
            }
        return final_pred, attention_info
def create_exact_fusion_model(vae_config, use_descriptors=False, descriptor_dim=0):
    return ExactCrossAttentionFusion(
        vae_config=vae_config,
        use_descriptors=use_descriptors,
        descriptor_dim=descriptor_dim,
        fusion_dim=128
    ) 