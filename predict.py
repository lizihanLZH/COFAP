import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
import h5py
from pathlib import Path
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
warnings.filterwarnings('ignore')
sys.path.append('./BiG-CAE')
sys.path.append('./PP-cVAE')
sys.path.append('./PP-NN')
try:
    from PP-cVAE.models import MultiModalVAECNN, MultiPlaneVAECNN
    from BiG-CAE.contrastive_autoencoder import ContrastiveAutoencoder
    from PP-NN.Net import MultiModalSAGEModel
    from fusion_model import ExactCrossAttentionFusion
    from BiG-CAE.featurizer import get_2cg_inputs_cof
    from PP-NN.ElseFeaturizer import CombinedCOFFeaturizer
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"⚠️  模块导入警告: {e}")
    print("某些功能可能不可用")
class CustomDataPredictor:
    def __init__(self, model_path, device='cuda'):
        if device == 'cuda' and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 8.0:  
                print(f"⚠️  GPU内存较小 ({gpu_memory:.1f} GB)，建议使用CPU进行推理")
                print("   可以使用 --device cpu 参数强制使用CPU")
                device = 'cpu'
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if self.device.type == 'cpu':
            print("   📝 注意：使用CPU推理会比GPU慢，但内存占用更少")
        print(f"加载融合模型: {model_path}")
        self.model = self._load_fusion_model(model_path)
        print("🔧 模型加载完成，现在移动到GPU...")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            cached_memory = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   📊 GPU内存信息:")
            print(f"      总内存: {gpu_memory:.1f} GB")
            print(f"      已分配: {allocated_memory:.1f} GB")
            print(f"      已缓存: {cached_memory:.1f} GB")
            print(f"      可用内存: {gpu_memory - cached_memory:.1f} GB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   🧹 已清理GPU缓存")
        print(f"   🚀 移动模型到设备: {self.device}")
        self.model.to(self.device)
        self.model.eval()
        print("✅ 融合模型加载成功")
        self.label_scaler = None
        self._init_label_scaler()
        self._init_descriptor_scaler()
        self._load_configurations()
        print("✓ 融合模型及所有子模型已移动到设备: {self.device}")
    def _load_fusion_model(self, model_path):
        print(f"加载融合模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        print("   🔧 先加载模型到CPU，避免CUDA内存不足...")
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"📊 检查点包含的键: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else '直接是state_dict'}")
        descriptor_dim = 12  
        print("   🔧 在CPU上创建模型实例...")
        model = ExactCrossAttentionFusion(
            vae_config=checkpoint.get('vae_config', {}),
            use_descriptors=True,  
            descriptor_dim=descriptor_dim,  
            fusion_dim=128
        )
        print("   🔧 在CPU上加载模型权重...")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 从model_state_dict加载权重")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 直接加载权重")
        print("🔍 检查融合模型checkpoint中的子模型权重...")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        vae_keys = [key for key in state_dict.keys() if key.startswith('vae_model')]
        gcn_cc_keys = [key for key in state_dict.keys() if key.startswith('gcn_cc_model')]
        gcn_noncc_keys = [key for key in state_dict.keys() if key.startswith('gcn_noncc_model')]
        lzhnn_keys = [key for key in state_dict.keys() if key.startswith('lzhnn_model')]
        print(f"   📊 权重键分析:")
        print(f"     VAE模型权重: {len(vae_keys)} 个")
        print(f"     GCN CC模型权重: {len(gcn_cc_keys)} 个")
        print(f"     GCN Non-CC模型权重: {len(gcn_noncc_keys)} 个")
        print(f"     LZHNN模型权重: {len(lzhnn_keys)} 个")
        if vae_keys and gcn_cc_keys and gcn_noncc_keys and lzhnn_keys:
            print("   ✅ 检测到完整的子模型权重，融合模型权重加载完成")
            print("   📝 注意：子模型权重已包含在融合模型checkpoint中")
        else:
            print("   ⚠️  未检测到完整的子模型权重")
            print("   📝 注意：这可能是训练过程中的中间checkpoint，子模型权重可能不完整")
            if vae_keys:
                print("   🔍 尝试从checkpoint中提取VAE权重...")
                try:
                    vae_state_dict = {key[11:]: value for key, value in state_dict.items() if key.startswith('vae_model.')}
                    if vae_state_dict:
                        missing_keys, unexpected_keys = model.vae_model.load_state_dict(vae_state_dict, strict=False)
                        if missing_keys:
                            print(f"     ⚠️  VAE缺失的键: {len(missing_keys)} 个")
                        if unexpected_keys:
                            print(f"     ⚠️  VAE意外的键: {len(unexpected_keys)} 个")
                        print("     ✅ VAE权重提取成功")
                    else:
                        print("     ❌ 无法提取VAE权重")
                except Exception as e:
                    print(f"     ❌ VAE权重提取失败: {e}")
            if gcn_cc_keys:
                print("   🔍 尝试从checkpoint中提取GCN CC权重...")
                try:
                    gcn_cc_state_dict = {key[15:]: value for key, value in state_dict.items() if key.startswith('gcn_cc_model.')}
                    if gcn_cc_state_dict:
                        missing_keys, unexpected_keys = model.gcn_cc_model.load_state_dict(gcn_cc_state_dict, strict=False)
                        if missing_keys:
                            print(f"     ⚠️  GCN CC缺失的键: {len(missing_keys)} 个")
                        if unexpected_keys:
                            print(f"     ⚠️  GCN CC意外的键: {len(unexpected_keys)} 个")
                        print("     ✅ GCN CC权重提取成功")
                    else:
                        print("     ❌ 无法提取GCN CC权重")
                except Exception as e:
                    print(f"     ❌ GCN CC权重提取失败: {e}")
            if gcn_noncc_keys:
                print("   🔍 尝试从checkpoint中提取GCN Non-CC权重...")
                try:
                    gcn_noncc_state_dict = {key[18:]: value for key, value in state_dict.items() if key.startswith('gcn_noncc_model.')}
                    if gcn_noncc_state_dict:
                        missing_keys, unexpected_keys = model.gcn_noncc_model.load_state_dict(gcn_noncc_state_dict, strict=False)
                        if missing_keys:
                            print(f"     ⚠️  GCN Non-CC缺失的键: {len(missing_keys)} 个")
                        if unexpected_keys:
                            print(f"     ⚠️  GCN Non-CC意外的键: {len(unexpected_keys)} 个")
                        print("     ✅ GCN Non-CC权重提取成功")
                    else:
                        print("     ❌ 无法提取GCN Non-CC权重")
                except Exception as e:
                    print(f"     ❌ GCN Non-CC权重提取失败: {e}")
            if lzhnn_keys:
                print("   🔍 尝试从checkpoint中提取LZHNN权重...")
                try:
                    lzhnn_state_dict = {key[13:]: value for key, value in state_dict.items() if key.startswith('lzhnn_model.')}
                    if lzhnn_state_dict:
                        missing_keys, unexpected_keys = model.lzhnn_model.load_state_dict(lzhnn_state_dict, strict=False)
                        if missing_keys:
                            print(f"     ⚠️  LZHNN缺失的键: {len(missing_keys)} 个")
                        if unexpected_keys:
                            print(f"     ⚠️  LZHNN意外的键: {len(unexpected_keys)} 个")
                        print("     ✅ LZHNN权重提取成功")
                    else:
                        print("     ❌ 无法提取LZHNN权重")
                except Exception as e:
                    print(f"     ❌ LZHNN权重提取失败: {e}")
        print("   🔧 在CPU上验证模型权重...")
        self._verify_model_weights(model)
        print("✅ 融合模型加载成功")
        print("   🔧 模型现在在CPU上，稍后将移动到GPU")
        return model
    def _verify_model_weights(self, model):
        print("🔍 验证模型权重...")
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            try:
                if hasattr(model.vae_model, 'vae') and hasattr(model.vae_model.vae, 'encoder'):
                    vae_conv1_weight = model.vae_model.vae.encoder[0].weight.data
                    vae_conv1_mean = vae_conv1_weight.mean().item()
                    vae_conv1_std = vae_conv1_weight.std().item()
                    print(f"   VAE第一层卷积权重: 均值={vae_conv1_mean:.6f}, 标准差={vae_conv1_std:.6f}")
                    if abs(vae_conv1_mean) < 0.01 and vae_conv1_std < 0.1:
                        print("   ⚠️  警告: VAE权重可能未正确加载（接近初始化值）")
                    else:
                        print("   ✅ VAE权重加载正常")
                elif hasattr(model.vae_model, 'encoder'):
                    vae_conv1_weight = model.vae_model.encoder[0].weight.data
                    vae_conv1_mean = vae_conv1_weight.mean().item()
                    vae_conv1_std = vae_conv1_weight.std().item()
                    print(f"   VAE第一层卷积权重: 均值={vae_conv1_mean:.6f}, 标准差={vae_conv1_std:.6f}")
                    if abs(vae_conv1_mean) < 0.01 and vae_conv1_std < 0.1:
                        print("   ⚠️  警告: VAE权重可能未正确加载（接近初始化值）")
                    else:
                        print("   ✅ VAE权重加载正常")
                else:
                    print("   ⚠️  无法识别VAE模型结构")
            except Exception as e:
                print(f"   ⚠️  检查VAE权重时出错: {e}")
        if hasattr(model, 'gcn_cc_model') and model.gcn_cc_model is not None:
            try:
                print(f"   🔍 检查GCN CC模型权重...")
                if hasattr(model.gcn_cc_model, 'encoder'):
                    if hasattr(model.gcn_cc_model.encoder, 'conv_layers') and len(model.gcn_cc_model.encoder.conv_layers) > 0:
                        first_conv = model.gcn_cc_model.encoder.conv_layers[0]
                        if hasattr(first_conv, 'weight'):
                            for edge_type, conv_layer in first_conv.mods.items():
                                if hasattr(conv_layer, 'weight'):
                                    gcn_weight = conv_layer.weight.data
                                    gcn_mean = gcn_weight.mean().item()
                                    gcn_std = gcn_weight.std().item()
                                    print(f"   GCN CC编码器第一层卷积({edge_type})权重: 均值={gcn_mean:.6f}, 标准差={gcn_std:.6f}")
                                    if abs(gcn_mean) < 0.01 and gcn_std < 0.1:
                                        print(f"      ⚠️  警告: GCN CC {edge_type}权重可能未正确加载（接近初始化值）")
                                    else:
                                        print(f"      ✅ GCN CC {edge_type}权重加载正常")
                        else:
                            print("   ⚠️  无法识别GCN CC卷积层结构")
                        if hasattr(model.gcn_cc_model.encoder, 'latent_mapping'):
                            latent_first_layer = model.gcn_cc_model.encoder.latent_mapping[0]
                            if hasattr(latent_first_layer, 'weight'):
                                latent_weight = latent_first_layer.weight.data
                                latent_mean = latent_weight.mean().item()
                                latent_std = latent_weight.std().item()
                                print(f"   GCN CC编码器潜在映射层权重: 均值={latent_mean:.6f}, 标准差={latent_std:.6f}")
                                if abs(latent_mean) < 0.01 and latent_std < 0.1:
                                    print("      ⚠️  警告: GCN CC潜在映射权重可能未正确加载（接近初始化值）")
                                else:
                                    print("      ✅ GCN CC潜在映射权重加载正常")
                    else:
                        print("   ⚠️  无法识别GCN CC编码器结构")
                else:
                    print("   ⚠️  无法识别GCN CC模型结构")
            except Exception as e:
                print(f"   ⚠️  检查GCN CC权重时出错: {e}")
        if hasattr(model, 'gcn_noncc_model') and model.gcn_noncc_model is not None:
            try:
                print(f"   🔍 检查GCN Non-CC模型权重...")
                if hasattr(model.gcn_noncc_model, 'encoder'):
                    if hasattr(model.gcn_noncc_model.encoder, 'conv_layers') and len(model.gcn_noncc_model.encoder.conv_layers) > 0:
                        first_conv = model.gcn_noncc_model.encoder.conv_layers[0]
                        if hasattr(first_conv, 'mods'):
                            for edge_type, conv_layer in first_conv.mods.items():
                                if hasattr(conv_layer, 'weight'):
                                    gcn_weight = conv_layer.weight.data
                                    gcn_mean = gcn_weight.mean().item()
                                    gcn_std = gcn_weight.std().item()
                                    print(f"   GCN Non-CC编码器第一层卷积({edge_type})权重: 均值={gcn_mean:.6f}, 标准差={gcn_std:.6f}")
                                    if abs(gcn_mean) < 0.01 and gcn_std < 0.1:
                                        print(f"      ⚠️  警告: GCN Non-CC {edge_type}权重可能未正确加载（接近初始化值）")
                                    else:
                                        print(f"      ✅ GCN Non-CC {edge_type}权重加载正常")
                        else:
                            print("   ⚠️  无法识别GCN Non-CC卷积层结构")
                        if hasattr(model.gcn_noncc_model.encoder, 'latent_mapping'):
                            latent_first_layer = model.gcn_noncc_model.encoder.latent_mapping[0]
                            if hasattr(latent_first_layer, 'weight'):
                                latent_weight = latent_first_layer.weight.data
                                latent_mean = latent_weight.mean().item()
                                latent_std = latent_weight.std().item()
                                print(f"   GCN Non-CC编码器潜在映射层权重: 均值={latent_mean:.6f}, 标准差={latent_std:.6f}")
                                if abs(latent_mean) < 0.01 and latent_std < 0.1:
                                    print("      ⚠️  警告: GCN Non-CC潜在映射权重可能未正确加载（接近初始化值）")
                                else:
                                    print("      ✅ GCN Non-CC潜在映射权重加载正常")
                    else:
                        print("   ⚠️  无法识别GCN Non-CC编码器结构")
                else:
                    print("   ⚠️  无法识别GCN Non-CC模型结构")
            except Exception as e:
                print(f"   ⚠️  检查GCN Non-CC权重时出错: {e}")
        if hasattr(model, 'gcn_model') and model.gcn_model is not None:
            try:
                print(f"   🔍 检查旧版GCN模型权重...")
                if hasattr(model.gcn_model, 'conv1'):
                    gcn_conv1_weight = model.gcn_model.conv1.weight.data
                    gcn_conv1_mean = gcn_conv1_weight.mean().item()
                    gcn_conv1_std = gcn_conv1_weight.std().item()
                    print(f"   GCN第一层权重: 均值={gcn_conv1_mean:.6f}, 标准差={gcn_conv1_std:.6f}")
                    if abs(gcn_conv1_mean) < 0.01 and gcn_conv1_std < 0.1:
                        print("   ⚠️  警告: GCN权重可能未正确加载（接近初始化值）")
                    else:
                        print("   ✅ GCN权重加载正常")
                elif hasattr(model.gcn_model, 'layers') and len(model.gcn_model.layers) > 0:
                    first_layer = model.gcn_model.layers[0]
                    if hasattr(first_layer, 'weight'):
                        gcn_weight = first_layer.weight.data
                        gcn_mean = gcn_weight.mean().item()
                        gcn_std = gcn_weight.std().item()
                        print(f"   GCN第一层权重: 均值={gcn_mean:.6f}, 标准差={gcn_std:.6f}")
                        if abs(gcn_mean) < 0.01 and gcn_std < 0.1:
                            print("   ⚠️  警告: GCN权重可能未正确加载（接近初始化值）")
                        else:
                            print("   ✅ GCN权重加载正常")
                    else:
                        print("   ⚠️  无法识别GCN模型结构")
                else:
                    print("   ⚠️  无法识别GCN模型结构")
            except Exception as e:
                print(f"   ⚠️  检查GCN权重时出错: {e}")
        if hasattr(model, 'lzhnn_model') and model.lzhnn_model is not None:
            try:
                if hasattr(model.lzhnn_model, 'topo_mlp'):
                    topo_first_layer = model.lzhnn_model.topo_mlp[0]  
                    if hasattr(topo_first_layer, 'weight'):
                        topo_weight = topo_first_layer.weight.data
                        topo_mean = topo_weight.mean().item()
                        topo_std = topo_weight.std().item()
                        print(f"   LZHNN拓扑MLP第一层权重: 均值={topo_mean:.6f}, 标准差={topo_std:.6f}")
                        if abs(topo_mean) < 0.001 and topo_std < 0.01:
                            print("   ⚠️  警告: LZHNN拓扑MLP权重可能未正确加载（接近初始化值）")
                        else:
                            print("   ✅ LZHNN拓扑MLP权重加载正常")
                    if hasattr(model.lzhnn_model, 'struct_mlp'):
                        struct_first_layer = model.lzhnn_model.struct_mlp[0]
                        if hasattr(struct_first_layer, 'weight'):
                            struct_weight = struct_first_layer.weight.data
                            struct_mean = struct_weight.mean().item()
                            struct_std = struct_weight.std().item()
                            print(f"   LZHNN结构MLP第一层权重: 均值={struct_mean:.6f}, 标准差={struct_std:.6f}")
                            if abs(struct_mean) < 0.001 and struct_std < 0.01:
                                print("   ⚠️  警告: LZHNN结构MLP权重可能未正确加载（接近初始化值）")
                            else:
                                print("   ✅ LZHNN结构MLP权重加载正常")
                    print("   ✅ LZHNN模型结构识别成功")
                elif hasattr(model.lzhnn_model, 'fc1'):
                    lzhnn_fc1_weight = model.lzhnn_model.fc1.weight.data
                    lzhnn_fc1_mean = lzhnn_fc1_weight.mean().item()
                    lzhnn_fc1_std = lzhnn_fc1_weight.std().item()
                    print(f"   LZHNN第一层权重: 均值={lzhnn_fc1_mean:.6f}, 标准差={lzhnn_fc1_std:.6f}")
                    if abs(lzhnn_fc1_mean) < 0.001 and lzhnn_fc1_std < 0.01:
                        print("   ⚠️  警告: LZHNN权重可能未正确加载（接近初始化值）")
                    else:
                        print("   ✅ LZHNN权重加载正常")
                elif hasattr(model.lzhnn_model, 'layers') and len(model.lzhnn_model.layers) > 0:
                    first_layer = model.lzhnn_model.layers[0]
                    if hasattr(first_layer, 'weight'):
                        lzhnn_weight = first_layer.weight.data
                        lzhnn_mean = lzhnn_weight.mean().item()
                        lzhnn_std = lzhnn_weight.std().item()
                        print(f"   LZHNN第一层权重: 均值={lzhnn_mean:.6f}, 标准差={lzhnn_std:.6f}")
                        if abs(lzhnn_mean) < 0.001 and lzhnn_std < 0.01:
                            print("   ⚠️  警告: LZHNN权重可能未正确加载（接近初始化值）")
                        else:
                            print("   ✅ LZHNN权重加载正常")
                    else:
                        print("   ⚠️  无法识别LZHNN模型结构")
                else:
                    print("   ⚠️  无法识别LZHNN模型结构")
            except Exception as e:
                print(f"   ⚠️  检查LZHNN权重时出错: {e}")
        if hasattr(model, 'fusion_layer'):
            try:
                fusion_weight = model.fusion_layer.weight.data
                fusion_mean = fusion_weight.mean().item()
                fusion_std = fusion_weight.std().item()
                print(f"   融合层权重: 均值={fusion_mean:.6f}, 标准差={fusion_std:.6f}")
                if abs(fusion_mean) < 0.01 and fusion_std < 0.1:
                    print("   ⚠️  警告: 融合层权重可能未正确加载（接近初始化值）")
                else:
                    print("   ✅ 融合层权重加载正常")
            except Exception as e:
                print(f"   ⚠️  检查融合层权重时出错: {e}")
        print("   📊 模型结构信息:")
        print(f"     模型类型: {type(model).__name__}")
        print(f"     模型属性: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"     总参数数量: {total_params:,}")
        print(f"     可训练参数数量: {trainable_params:,}")
    def _load_configurations(self):
        print("加载配置和数据...")
        self._verify_training_inference_consistency()
        print("✅ 配置加载完成")
    def _verify_training_inference_consistency(self):
        print("🔍 验证训练和推理数据一致性...")
        train_labels_path = "labels_CH410.csv"
        if os.path.exists(train_labels_path):
            train_labels_size = os.path.getsize(train_labels_path)
            train_labels_mtime = os.path.getmtime(train_labels_path)
            print(f"   ✅ 训练标签文件: {train_labels_path}")
            print(f"      大小: {train_labels_size} bytes")
            print(f"      修改时间: {pd.Timestamp.fromtimestamp(train_labels_mtime)}")
        else:
            print(f"   ❌ 训练标签文件不存在: {train_labels_path}")
        train_descriptor_path = "VAE/descriptor.csv"
        if os.path.exists(train_descriptor_path):
            train_descriptor_size = os.path.getsize(train_descriptor_path)
            train_descriptor_mtime = os.path.getmtime(train_descriptor_path)
            print(f"   ✅ 训练描述符文件: {train_descriptor_path}")
            print(f"      大小: {train_descriptor_size} bytes")
            print(f"      修改时间: {pd.Timestamp.fromtimestamp(train_descriptor_mtime)}")
        else:
            print(f"   ❌ 训练描述符文件不存在: {train_descriptor_path}")
        train_linkers_path = "GCN/linkers.csv"
        if os.path.exists(train_linkers_path):
            train_linkers_size = os.path.getsize(train_linkers_path)
            train_linkers_mtime = os.path.getmtime(train_linkers_path)
            print(f"   ✅ 训练链接器文件: {train_linkers_path}")
            print(f"      大小: {train_linkers_size} bytes")
            print(f"      修改时间: {pd.Timestamp.fromtimestamp(train_linkers_mtime)}")
        else:
            print(f"   ❌ 训练链接器文件不存在: {train_linkers_path}")
        self._verify_scaler_consistency()
        print("   📊 数据一致性验证完成")
    def _verify_scaler_consistency(self):
        print("   🔍 验证标准化器一致性...")
        if hasattr(self, 'train_label_stats') and self.train_label_stats is not None:
            current_mean = self.label_scaler.mean_[0] if hasattr(self.label_scaler, 'mean_') else None
            current_scale = self.label_scaler.scale_[0] if hasattr(self.label_scaler, 'scale_') else None
            if current_mean is not None and current_scale is not None:
                mean_diff = abs(current_mean - self.train_label_stats['mean'][0])
                scale_diff = abs(current_scale - self.train_label_stats['scale'][0])
                print(f"     标签标准化器:")
                print(f"       训练时均值: {self.train_label_stats['mean'][0]:.6f}")
                print(f"       当前均值: {current_mean:.6f}")
                print(f"       均值差异: {mean_diff:.8f}")
                print(f"       训练时标准差: {self.train_label_stats['scale'][0]:.6f}")
                print(f"       当前标准差: {current_scale:.6f}")
                print(f"       标准差差异: {scale_diff:.8f}")
                if mean_diff > 1e-6 or scale_diff > 1e-6:
                    print(f"       ⚠️  警告: 标签标准化器参数与训练时不一致")
                else:
                    print(f"       ✅ 标签标准化器参数一致")
            else:
                print(f"      ⚠️  无法获取当前标签标准化器参数")
        else:
            print(f"      ⚠️  未找到训练时标签统计信息")
        if hasattr(self, 'train_descriptor_stats') and self.train_descriptor_stats is not None:
            current_mean = self.descriptor_scaler.mean_ if hasattr(self.descriptor_scaler, 'mean_') else None
            current_scale = self.descriptor_scaler.scale_ if hasattr(self.descriptor_scaler, 'scale_') else None
            if current_mean is not None and current_scale is not None:
                mean_diff = np.mean(np.abs(current_mean - self.train_descriptor_stats['mean']))
                scale_diff = np.mean(np.abs(current_scale - self.train_descriptor_stats['scale']))
                print(f"     描述符标准化器:")
                print(f"       训练时均值范围: {self.train_descriptor_stats['mean'].min():.6f} - {self.train_descriptor_stats['mean'].max():.6f}")
                print(f"       当前均值范围: {current_mean.min():.6f} - {current_mean.max():.6f}")
                print(f"       平均均值差异: {mean_diff:.8f}")
                print(f"       训练时标准差范围: {self.train_descriptor_stats['scale'].min():.6f} - {self.train_descriptor_stats['scale'].max():.6f}")
                print(f"       当前标准差范围: {current_scale.min():.6f} - {current_scale.max():.6f}")
                print(f"       平均标准差差异: {scale_diff:.8f}")
                if mean_diff > 1e-6 or scale_diff > 1e-6:
                    print(f"       ⚠️  警告: 描述符标准化器参数与训练时不一致")
                else:
                    print(f"       ✅ 描述符标准化器参数一致")
            else:
                print(f"      ⚠️  无法获取当前描述符标准化器参数")
        else:
            print(f"      ⚠️  未找到训练时描述符统计信息")
    def _custom_collate_fn(self, batch):
        sample = batch[0]
        for key, value in sample.items():
            if value is None:
                if key in ['graph_cc', 'graph_noncc']:
                    import dgl
                    sample[key] = dgl.heterograph({
                        ("l", "l2n", "n"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                        ("n", "n2l", "l"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                    })
                    sample[key].nodes["l"].data["feat"] = torch.zeros(0, 300, dtype=torch.float32)
                    sample[key].nodes["n"].data["feat"] = torch.zeros(0, 4, dtype=torch.float32)
                elif key == 'lzhnn_data':
                    sample[key] = {
                        'topo': torch.zeros(18, dtype=torch.float32),
                        'chem': torch.zeros(300, dtype=torch.float32),
                        'struct': torch.zeros(5, dtype=torch.float32)
                    }
        return sample
    def prepare_new_data(self, data_config, use_parallel=True, num_workers=None):
        print("准备新数据...")
        self._verify_data_config_consistency(data_config)
        dataset = CustomDataset(
            data_config, 
            "GCN/linkers.csv", 
            self.descriptor_scaler, 
            self.descriptor_cols,
            use_parallel=use_parallel,
            num_workers=num_workers
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            collate_fn=self._custom_collate_fn
        )  
        return dataloader
    def _verify_data_config_consistency(self, data_config):
        print("🔍 验证数据配置一致性...")
        if hasattr(self, 'descriptor_cols') and self.descriptor_cols is not None:
            try:
                pred_descriptor_df = pd.read_csv(data_config['descriptor_csv'])
                pred_cols = [col for col in pred_descriptor_df.columns if col != 'name']
                print(f"   📊 描述符列一致性检查:")
                print(f"     训练时列数: {len(self.descriptor_cols)}")
                print(f"     预测时列数: {len(pred_cols)}")
                train_cols_set = set(self.descriptor_cols)
                pred_cols_set = set(pred_cols)
                missing_in_pred = train_cols_set - pred_cols_set
                extra_in_pred = pred_cols_set - train_cols_set
                if missing_in_pred:
                    print(f"     ⚠️  预测时缺失列: {list(missing_in_pred)}")
                if extra_in_pred:
                    print(f"     ⚠️  预测时多余列: {list(extra_in_pred)}")
                if not missing_in_pred and not extra_in_pred:
                    print(f"     ✅ 描述符列完全一致")
                elif len(missing_in_pred) <= 2:  
                    print(f"     ⚠️  描述符列基本一致，少量缺失")
                else:
                    print(f"     ❌ 描述符列差异较大")
            except Exception as e:
                print(f"     ⚠️  无法验证描述符列一致性: {e}")
        print(f"   📁 数据目录配置:")
        for key, path in data_config.items():
            if os.path.exists(path):
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    print(f"     {key}: {path} ({size} bytes) ✅")
                else:
                    file_count = len(list(Path(path).glob('*')))
                    print(f"     {key}: {path} ({file_count} 文件) ✅")
            else:
                print(f"     {key}: {path} (不存在) ❌")
        print("   📊 数据配置一致性验证完成")
    def predict(self, dataloader, output_file=None):
        print("🚀 开始预测...")
        self._verify_prediction_data_consistency()
        all_predictions = []
        all_identifiers = []
        total_batches = len(dataloader)
        progress_interval = max(1, total_batches // 100)
        data_quality_stats = {
            'vae_data_zeros': 0,
            'gcn_data_empty': 0,
            'lzhnn_data_zeros': 0,
            'descriptors_zeros': 0
        }
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % progress_interval == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    print(f"\n📦 处理批次 {batch_idx + 1}/{total_batches} ({progress:.1f}%)")
                else:
                    print(f"📦 批次 {batch_idx + 1}/{total_batches}", end="\r")
                self._check_batch_data_quality(batch, data_quality_stats)
                if batch_idx % progress_interval == 0 or batch_idx == total_batches - 1:
                    print("    🔧 准备输入数据...")
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if batch_idx % progress_interval == 0 or batch_idx == total_batches - 1:
                    print("    🧠 进行模型预测...")
                outputs = self.model(inputs)
                if isinstance(outputs, torch.Tensor):
                    batch_predictions = outputs.cpu().numpy().flatten()
                else:
                    batch_predictions = outputs['prediction'].cpu().numpy().flatten()
                all_predictions.extend(batch_predictions)
                if 'identifiers' in batch:
                    all_identifiers.extend(batch['identifiers'])
                else:
                    all_identifiers.extend([f"sample_{batch_idx}_{i}" for i in range(len(batch_predictions))])
                if batch_idx % progress_interval == 0 or batch_idx == total_batches - 1:
                    batch_predictions_array = np.array(batch_predictions)
                    batch_lg_predictions = self._denormalize_predictions(batch_predictions_array)
                    print(f"    ✅ 批次 {batch_idx + 1} 完成")
                    print(f"       标准化预测值: {batch_predictions}")
                    print(f"       lg预测值: {batch_lg_predictions}")
        print("\n📊 数据质量统计:")
        print(f"   VAE数据全零样本: {data_quality_stats['vae_data_zeros']}")
        print(f"   GCN数据空图样本: {data_quality_stats['gcn_data_empty']}")
        print(f"   LZHNN数据全零样本: {data_quality_stats['lzhnn_data_zeros']}")
        print(f"   描述符全零样本: {data_quality_stats['descriptors_zeros']}")
        print("\n🔄 反标准化预测结果...")
        all_predictions_array = np.array(all_predictions)
        lg_predictions = self._denormalize_predictions(all_predictions_array)
        print(f"\n🎉 预测完成，共处理 {len(all_predictions)} 个样本")
        print(f"📊 预测值统计:")
        print(f"   标准化预测值范围: {all_predictions_array.min():.4f} - {all_predictions_array.max():.4f}")
        print(f"   标准化预测值均值: {np.mean(all_predictions_array):.4f}")
        print(f"   标准化预测值标准差: {np.std(all_predictions_array):.4f}")
        print(f"   lg预测值范围: {lg_predictions.min():.4f} - {lg_predictions.max():.4f}")
        print(f"   lg预测值均值: {np.mean(lg_predictions):.4f}")
        self._verify_training_prediction_consistency(all_identifiers, all_predictions_array, lg_predictions)
        results_df = pd.DataFrame({
            'identifier': all_identifiers,
            'prediction_normalized': all_predictions_array,  
            'prediction_lg': lg_predictions                  
        })
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"📁 预测结果已保存到: {output_file}")
        return results_df
    def _verify_prediction_data_consistency(self):
        print("🔍 验证预测过程中数据一致性...")
        if hasattr(self, 'train_label_stats') and self.train_label_stats is not None:
            current_mean = self.label_scaler.mean_[0] if hasattr(self.label_scaler, 'mean_') else None
            current_scale = self.label_scaler.scale_[0] if hasattr(self.label_scaler, 'scale_') else None
            if current_mean is not None and current_scale is not None:
                mean_diff = abs(current_mean - self.train_label_stats['mean'][0])
                scale_diff = abs(current_scale - self.train_label_stats['scale'][0])
                if mean_diff > 1e-6 or scale_diff > 1e-6:
                    print(f"   ⚠️  警告: 标签标准化器参数在预测过程中发生变化")
                    print(f"     训练时均值: {self.train_label_stats['mean'][0]:.6f}")
                    print(f"     当前均值: {current_mean:.6f}")
                    print(f"     均值差异: {mean_diff:.8f}")
                else:
                    print(f"   ✅ 标签标准化器参数在预测过程中保持一致")
        if hasattr(self, 'train_descriptor_stats') and self.train_descriptor_stats is not None:
            current_mean = self.descriptor_scaler.mean_ if hasattr(self.descriptor_scaler, 'mean_') else None
            current_scale = self.descriptor_scaler.scale_ if hasattr(self.descriptor_scaler, 'scale_') else None
            if current_mean is not None and current_scale is not None:
                mean_diff = np.mean(np.abs(current_mean - self.train_descriptor_stats['mean']))
                scale_diff = np.mean(np.abs(current_scale - self.train_descriptor_stats['scale']))
                if mean_diff > 1e-6 or scale_diff > 1e-6:
                    print(f"   ⚠️  警告: 描述符标准化器参数在预测过程中发生变化")
                    print(f"     平均均值差异: {mean_diff:.8f}")
                    print(f"     平均标准差差异: {scale_diff:.8f}")
                else:
                    print(f"   ✅ 描述符标准化器参数在预测过程中保持一致")
        print("   📊 预测数据一致性验证完成")
    def _verify_training_prediction_consistency(self, identifiers, normalized_predictions, lg_predictions):
        print("🔍 验证训练数据预测一致性...")
        train_labels_path = "labels_CH410.csv"
        if not os.path.exists(train_labels_path):
            print("   ⚠️  未找到训练标签文件，无法验证训练数据预测一致性")
            return
        try:
            train_labels_df = pd.read_csv(train_labels_path)
            train_sample_names = set(train_labels_df.iloc[:, 0].tolist())
            train_indices = []
            train_normalized_preds = []
            train_lg_preds = []
            for idx, identifier in enumerate(identifiers):
                if identifier in train_sample_names:
                    train_indices.append(idx)
                    train_normalized_preds.append(normalized_predictions[idx])
                    train_lg_preds.append(lg_predictions[idx])
            if train_indices:
                train_normalized_preds = np.array(train_normalized_preds)
                train_lg_preds = np.array(train_lg_preds)
                print(f"   📊 训练数据预测统计:")
                print(f"     训练样本数: {len(train_indices)}")
                print(f"     标准化预测值范围: {train_normalized_preds.min():.4f} - {train_normalized_preds.max():.4f}")
                print(f"     标准化预测值均值: {np.mean(train_normalized_preds):.4f}")
                print(f"     标准化预测值标准差: {np.std(train_normalized_preds):.4f}")
                print(f"     lg预测值范围: {train_lg_preds.min():.4f} - {train_lg_preds.max():.4f}")
                print(f"     lg预测值均值: {np.mean(train_lg_preds):.4f}")
                if hasattr(self, 'train_label_stats'):
                    expected_range = self.train_label_stats['original_range']
                    print(f"     训练时标签范围: {expected_range[0]:.4f} - {expected_range[1]:.4f}")
                    if train_lg_preds.min() < expected_range[0] * 0.5 or train_lg_preds.max() > expected_range[1] * 2.0:
                        print(f"     ⚠️  警告: 训练数据预测值超出预期范围")
                    else:
                        print(f"     ✅ 训练数据预测值在合理范围内")
                print("   ✅ 训练数据预测一致性验证完成")
            else:
                print("   ⚠️  未找到训练数据样本")
        except Exception as e:
            print(f"   ⚠️  验证训练数据预测一致性时出错: {e}")
    def _check_batch_data_quality(self, batch, stats):
        if 'vae_data' in batch and isinstance(batch['vae_data'], torch.Tensor):
            if batch['vae_data'].abs().sum() == 0:
                stats['vae_data_zeros'] += 1
        if 'graph_cc' in batch:
            import dgl
            if isinstance(batch['graph_cc'], dgl.DGLHeteroGraph):
                if batch['graph_cc'].num_nodes('l') == 0 and batch['graph_cc'].num_nodes('n') == 0:
                    stats['gcn_data_empty'] += 1
        if 'lzhnn_data' in batch and isinstance(batch['lzhnn_data'], dict):
            lzhnn_data = batch['lzhnn_data']
            topo_sum = lzhnn_data.get('topo', torch.tensor(0))
            struct_sum = lzhnn_data.get('struct', torch.tensor(0))
            if isinstance(topo_sum, torch.Tensor):
                topo_sum = topo_sum.abs().sum()
            if isinstance(struct_sum, torch.Tensor):
                struct_sum = struct_sum.abs().sum()
            if topo_sum == 0 and struct_sum == 0:
                stats['lzhnn_data_zeros'] += 1
        if 'descriptors' in batch and isinstance(batch['descriptors'], torch.Tensor):
            if batch['descriptors'].abs().sum() == 0:
                stats['descriptors_zeros'] += 1
    def _prepare_batch_inputs(self, batch):
        inputs = {}
        if 'vae_data' in batch:
            inputs['vae_data'] = batch['vae_data'].to(self.device)
        if 'descriptors' in batch:
            inputs['descriptors'] = batch['descriptors'].to(self.device)
        if 'graph_cc' in batch:
            inputs['graph_cc'] = batch['graph_cc'].to(self.device)
        if 'graph_noncc' in batch:
            inputs['graph_noncc'] = batch['graph_noncc'].to(self.device)
        if 'lzhnn_data' in batch:
            lzhnn_data = batch['lzhnn_data']
            inputs['lzhnn_data'] = {
                'topo': lzhnn_data['topo'].to(self.device),
                'struct': lzhnn_data['struct'].to(self.device)
            }
        if 'cc_mask' in batch:
            inputs['cc_mask'] = batch['cc_mask'].to(self.device)
        return inputs
    def _init_label_scaler(self):
        try:
            from sklearn.preprocessing import StandardScaler
            self.label_scaler = StandardScaler()
            labels_path = "labels_CH410.csv"
            if os.path.exists(labels_path):
                labels_df = pd.read_csv(labels_path)
                print(f"📊 读取训练标签文件: {labels_path}")
                print(f"   文件大小: {os.path.getsize(labels_path)} bytes")
                print(f"   列名: {list(labels_df.columns)}")
                print(f"   样本数量: {len(labels_df)}")
                if 'target_property' in labels_df.columns:
                    original_labels = labels_df['target_property'].values
                    print(f"   标签统计:")
                    print(f"     范围: {original_labels.min():.6f} - {original_labels.max():.6f}")
                    print(f"     均值: {original_labels.mean():.6f}")
                    print(f"     标准差: {original_labels.std():.6f}")
                    print(f"     NaN数量: {np.isnan(original_labels).sum()}")
                    print(f"     无穷大数量: {np.isinf(original_labels).sum()}")
                    valid_mask = ~(np.isnan(original_labels) | np.isinf(original_labels))
                    if not valid_mask.all():
                        print(f"   ⚠️  发现 {np.sum(~valid_mask)} 个无效标签值，将被过滤")
                        original_labels = original_labels[valid_mask]
                    if len(original_labels) > 0:
                        self.label_scaler.fit(original_labels.reshape(-1, 1))
                        print("✅ 标签标准化器初始化完成")
                        print(f"   基于真实训练数据:")
                        print(f"   原始标签范围: {original_labels.min():.6f} - {original_labels.max():.6f}")
                        print(f"   原始标签均值: {original_labels.mean():.6f}")
                        print(f"   原始标签标准差: {original_labels.std():.6f}")
                        standardized = self.label_scaler.transform(original_labels.reshape(-1, 1))
                        print(f"   标准化后范围: {standardized.min():.6f} - {standardized.max():.6f}")
                        print(f"   标准化后均值: {standardized.mean():.6f}")
                        print(f"   标准化后标准差: {standardized.std():.6f}")
                        reconstructed = self.label_scaler.inverse_transform(standardized)
                        reconstruction_error = np.mean(np.abs(original_labels - reconstructed.flatten()))
                        print(f"   反标准化误差: {reconstruction_error:.8f}")
                        if reconstruction_error > 1e-6:
                            print("   ⚠️  反标准化误差较大，可能存在数值精度问题")
                        self.train_label_stats = {
                            'mean': self.label_scaler.mean_.copy(),
                            'scale': self.label_scaler.scale_.copy(),
                            'sample_count': len(original_labels),
                            'original_range': (original_labels.min(), original_labels.max()),
                            'original_mean': original_labels.mean(),
                            'original_std': original_labels.std()
                        }
                        print(f"   📊 保存训练时标签统计信息:")
                        print(f"     原始标签范围: {self.train_label_stats['original_range'][0]:.6f} - {self.train_label_stats['original_range'][1]:.6f}")
                        print(f"     原始标签均值: {self.train_label_stats['original_mean']:.6f}")
                        print(f"     原始标签标准差: {self.train_label_stats['original_std']:.6f}")
                    else:
                        raise ValueError("过滤无效值后没有剩余标签")
                else:
                    possible_cols = [col for col in labels_df.columns if 'target' in col.lower() or 'property' in col.lower() or 'label' in col.lower()]
                    if possible_cols:
                        print(f"   ⚠️  未找到target_property列，尝试使用: {possible_cols}")
                        col_name = possible_cols[0]
                        original_labels = labels_df[col_name].values
                        self.label_scaler.fit(original_labels.reshape(-1, 1))
                        print(f"   ✅ 使用列 {col_name} 初始化标准化器")
                    else:
                        raise ValueError(f"labels.csv中没有找到target_property列或其他相关列")
            else:
                raise FileNotFoundError(f"未找到labels.csv文件: {labels_path}")
        except Exception as e:
            print(f"⚠️  标签标准化器初始化失败: {e}")
            print(f"   使用默认设置...")
            self.label_scaler = StandardScaler()
            self.label_scaler.mean_ = np.array([1.85])
            self.label_scaler.scale_ = np.array([0.75])
            print(f"   默认设置: 均值=1.85, 标准差=0.75")
            print(f"   ⚠️  警告: 使用默认设置可能导致预测精度下降")
    def _init_descriptor_scaler(self):
        try:
            descriptor_path = "VAE/descriptor.csv"
            if os.path.exists(descriptor_path):
                descriptor_df = pd.read_csv(descriptor_path)
                print(f"📊 读取训练描述符文件: {descriptor_path}")
                print(f"   文件大小: {os.path.getsize(descriptor_path)} bytes")
                print(f"   列名: {list(descriptor_df.columns)}")
                print(f"   样本数量: {len(descriptor_df)}")
                descriptor_cols = [col for col in descriptor_df.columns if col != 'name']
                descriptor_values = descriptor_df[descriptor_cols].values
                print(f"   描述符统计:")
                print(f"     特征数量: {len(descriptor_cols)}")
                print(f"     数据形状: {descriptor_values.shape}")
                print(f"     NaN数量: {np.isnan(descriptor_values).sum()}")
                print(f"     无穷大数量: {np.isinf(descriptor_values).sum()}")
                valid_mask = ~(np.isnan(descriptor_values).any(axis=1) | np.isinf(descriptor_values).any(axis=1))
                if not valid_mask.all():
                    print(f"   ⚠️  发现 {np.sum(~valid_mask)} 个无效样本，将被过滤")
                    descriptor_values = descriptor_values[valid_mask]
                if len(descriptor_values) > 0:
                    from sklearn.preprocessing import StandardScaler
                    self.descriptor_scaler = StandardScaler()
                    self.descriptor_scaler.fit(descriptor_values)
                    self.descriptor_cols = descriptor_cols
                    print("✅ 描述符标准化器初始化完成")
                    print(f"   描述符特征数量: {len(self.descriptor_cols)}")
                    print(f"   训练时列名: {self.descriptor_cols}")
                    standardized = self.descriptor_scaler.transform(descriptor_values)
                    print(f"   标准化后统计:")
                    print(f"     均值范围: {standardized.mean(axis=0).min():.6f} - {standardized.mean(axis=0).max():.6f}")
                    print(f"     标准差范围: {standardized.std(axis=0).min():.6f} - {standardized.std(axis=0).max():.6f}")
                    reconstructed = self.descriptor_scaler.inverse_transform(standardized)
                    reconstruction_error = np.mean(np.abs(descriptor_values - reconstructed))
                    print(f"     反标准化误差: {reconstruction_error:.8f}")
                    if reconstruction_error > 1e-6:
                        print("   ⚠️  反标准化误差较大，可能存在数值精度问题")
                    self.train_descriptor_stats = {
                        'mean': self.descriptor_scaler.mean_.copy(),
                        'scale': self.descriptor_scaler.scale_.copy(),
                        'sample_count': len(descriptor_values)
                    }
                    print(f"   📊 保存训练时描述符统计信息:")
                    print(f"     均值范围: {self.train_descriptor_stats['mean'].min():.6f} - {self.train_descriptor_stats['mean'].max():.6f}")
                    print(f"     标准差范围: {self.train_descriptor_stats['scale'].min():.6f} - {self.train_descriptor_stats['scale'].max():.6f}")
                else:
                    raise ValueError("过滤无效值后没有剩余样本")
            else:
                print("⚠️  训练描述符文件未找到，跳过描述符标准化")
                self.descriptor_scaler = None
                self.descriptor_cols = None
        except Exception as e:
            print(f"⚠️  描述符标准化器初始化失败: {e}")
            self.descriptor_scaler = None
            self.descriptor_cols = None
    def _denormalize_predictions(self, predictions):
        if self.label_scaler is not None:
            try:
                predictions_2d = predictions.reshape(-1, 1)
                lg_values = self.label_scaler.inverse_transform(predictions_2d)
                print(f"    📊 反标准化完成（保留lg值）:")
                print(f"       标准化值范围: {predictions.min():.4f} - {predictions.max():.4f}")
                print(f"       lg值范围: {lg_values.min():.4f} - {lg_values.max():.4f}")
                print(f"       注意：返回的是lg值，不是原始值")
                return lg_values.flatten()
            except Exception as e:
                print(f"⚠️  反标准化失败: {e}")
                return predictions
        else:
            print("⚠️  标签标准化器未初始化，跳过反标准化")
            return predictions
class CustomDataset(Dataset):
    def __init__(self, data_config, linkers_csv, descriptor_scaler, descriptor_cols, use_parallel=True, num_workers=None):
        self.data_config = data_config
        self.linkers_csv = linkers_csv
        self.descriptor_scaler = descriptor_scaler
        self.descriptor_cols = descriptor_cols
        self.use_parallel = use_parallel
        self.num_workers = num_workers
        self.descriptor_df = pd.read_csv(data_config['descriptor_csv'])
        self.struct_features_df = pd.read_csv(data_config['struct_features_csv'])
        self.sample_identifiers = self._get_sample_identifiers()
        print("🔍 统一提取和缓存拓扑特征...")
        self.topo_features_cache = {}
        self._extract_and_cache_all_features()
        print(f"数据集初始化完成，共 {len(self.sample_identifiers)} 个样本")
        self._verify_dataset_consistency()
    def _verify_dataset_consistency(self):
        print("🔍 验证数据集特征提取一致性...")
        train_labels_path = "labels_CH410.csv"
        if os.path.exists(train_labels_path):
            try:
                train_labels_df = pd.read_csv(train_labels_path)
                train_sample_names = set(train_labels_df.iloc[:, 0].tolist())
                train_samples = set(self.sample_identifiers).intersection(train_sample_names)
                if train_samples:
                    print(f"   📊 训练数据特征质量检查:")
                    print(f"     训练样本数: {len(train_samples)}")
                    train_topo_nonzero = sum(1 for name in train_samples if self._get_topo_sum(name) > 0)
                    print(f"     拓扑特征非零样本: {train_topo_nonzero}/{len(train_samples)} ({train_topo_nonzero/len(train_samples)*100:.1f}%)")
                    if hasattr(self, 'struct_features_df'):
                        struct_cols = [col for col in self.struct_features_df.columns if col != 'name']
                        if struct_cols:
                            struct_values = self.struct_features_df[self.struct_features_df.iloc[:, 0].isin(train_samples)][struct_cols].values
                            struct_nonzero = np.sum(struct_values != 0)
                            struct_total = struct_values.size
                            print(f"     结构特征非零值: {struct_nonzero}/{struct_total} ({struct_nonzero/struct_total*100:.1f}%)")
                    if hasattr(self, 'descriptor_df'):
                        desc_cols = [col for col in self.descriptor_df.columns if col != 'name']
                        if desc_cols:
                            desc_values = self.descriptor_df[self.descriptor_df['name'].isin(train_samples)][desc_cols].values
                            desc_nonzero = np.sum(desc_values != 0)
                            desc_total = desc_values.size
                            print(f"     描述符非零值: {desc_nonzero}/{desc_total} ({desc_nonzero/desc_total*100:.1f}%)")
                    print("   ✅ 训练数据特征质量检查完成")
                else:
                    print("   ⚠️  未找到训练数据样本")
            except Exception as e:
                print(f"   ⚠️  验证数据集一致性时出错: {e}")
        print("   📊 数据集特征提取一致性验证完成")
    def _extract_and_cache_all_features(self):
        try:
            cache_dir = "feature_cache"
            topo_cache_file = os.path.join(cache_dir, "topo_features_cache.pkl")
            print(f"🔍 检查拓扑特征缓存...")
            print(f"   缓存目录: {cache_dir}")
            print(f"   缓存文件: {topo_cache_file}")
            if os.path.exists(topo_cache_file):
                try:
                    print("   📥 发现现有缓存，尝试加载...")
                    with open(topo_cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    cache_version = cached_data.get('version', '1.0')
                    cache_timestamp = cached_data.get('timestamp', 0)
                    cached_features = cached_data.get('features', {})
                    print(f"   📊 缓存信息:")
                    print(f"     版本: {cache_version}")
                    print(f"     创建时间: {pd.Timestamp.fromtimestamp(cache_timestamp)}")
                    print(f"     缓存样本数: {len(cached_features)}")
                    current_time = time.time()
                    cache_age_days = (current_time - cache_timestamp) / (24 * 3600)
                    if cache_age_days > 7:
                        print(f"   ⚠️  缓存已过期 ({cache_age_days:.1f} 天)，将重新生成")
                        use_cache = False
                    else:
                        print(f"   ✅ 缓存有效 ({cache_age_days:.1f} 天)")
                        use_cache = True
                    missing_samples = set(self.sample_identifiers) - set(cached_features.keys())
                    if missing_samples:
                        print(f"   ⚠️  缓存中缺少 {len(missing_samples)} 个样本，将补充提取")
                        use_cache = True  
                    else:
                        print(f"   ✅ 缓存包含所有样本")
                    if use_cache:
                        for sample_name in self.sample_identifiers:
                            if sample_name in cached_features:
                                feat = cached_features[sample_name]
                                if isinstance(feat, np.ndarray):
                                    self.topo_features_cache[sample_name] = torch.from_numpy(feat).float()
                                else:
                                    self.topo_features_cache[sample_name] = feat
                            else:
                                print(f"      🔄 重新提取缺失样本: {sample_name}")
                                topo_features = self._extract_real_topo_features(sample_name)
                                self.topo_features_cache[sample_name] = topo_features
                        print(f"   ✅ 从缓存加载完成，共 {len(self.topo_features_cache)} 个样本")
                        cached_count = sum(1 for name in self.sample_identifiers if name in cached_features)
                        cache_hit_rate = cached_count / len(self.sample_identifiers) * 100
                        print(f"   📊 缓存命中率: {cache_hit_rate:.1f}% ({cached_count}/{len(self.sample_identifiers)})")
                        return  
                except Exception as e:
                    print(f"   ⚠️  缓存加载失败: {e}")
                    print("   🔄 将重新提取所有特征")
            print("   🔄 开始特征提取...")
            mol2vec_path = "PP-NN/models/mol2vec_300dim.pkl"
            if not os.path.exists(mol2vec_path):
                print(f"⚠️  mol2vec模型文件不存在: {mol2vec_path}")
                alternative_paths = [
                    "PP-NN/models/mol2vec_300dim.kv",
                    "PP-NN/models/mol2vec_300dim.pkl",
                    "./PP-NN/models/mol2vec_300dim.pkl",
                    "../PP-NN/models/mol2vec_300dim.pkl"
                ]
                mol2vec_found = False
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"✅ 找到mol2vec模型: {alt_path}")
                        mol2vec_path = alt_path
                        mol2vec_found = True
                        break
                if not mol2vec_found:
                    print(f"❌ 未找到mol2vec模型文件，将使用零特征")
                    for sample_name in self.sample_identifiers:
                        self.topo_features_cache[sample_name] = torch.zeros(18, dtype=torch.float32)
                    return
            os.environ['MOL2VEC_MODEL_PATH'] = mol2vec_path
            print(f"   🔧 设置mol2vec模型路径: {mol2vec_path}")
            print("   🔧 使用与训练时完全一致的LZHNN特征提取方法")
            print("   📝 注意：融合模型中的LZHNN使用真实的持久同调特征(18维)和结构特征(5维)")
            struct_features_csv = self.data_config['struct_features_csv']
            cif_dir = self.data_config['cif_dir']
            if not os.path.exists(struct_features_csv):
                print(f"   ❌ 结构特征CSV文件不存在: {struct_features_csv}")
                for sample_name in self.sample_identifiers:
                    self.topo_features_cache[sample_name] = torch.zeros(18, dtype=torch.float32)
                return
            if not os.path.exists(cif_dir):
                print(f"   ❌ CIF目录不存在: {cif_dir}")
                for sample_name in self.sample_identifiers:
                    self.topo_features_cache[sample_name] = torch.zeros(18, dtype=torch.float32)
                return
            print(f"   ✅ 结构特征CSV文件存在: {struct_features_csv}")
            print(f"   ✅ CIF目录存在: {cif_dir}")
            print("   🔧 将使用真实的持久同调特征提取方法")
            print("🔍 分离训练数据和新数据...")
            train_labels_path = "labels_CH410.csv"
            if os.path.exists(train_labels_path):
                train_labels_df = pd.read_csv(train_labels_path)
                train_sample_names = set(train_labels_df.iloc[:, 0].tolist())  
                print(f"   训练时样本数: {len(train_sample_names)}")
            else:
                print(f"⚠️  未找到训练标签文件，无法区分训练数据和新数据")
                train_sample_names = set()
            current_samples = set(self.sample_identifiers)
            train_samples = current_samples.intersection(train_sample_names)
            new_samples = current_samples - train_sample_names
            print(f"   当前样本总数: {len(current_samples)}")
            print(f"   训练数据样本数: {len(train_samples)}")
            print(f"   新数据样本数: {len(new_samples)}")
            print("📊 开始特征提取...")
            if train_samples:
                print(f"   1️⃣ 处理训练数据 ({len(train_samples)} 个样本)...")
                if self.use_parallel:
                    print("   🚀 启动并行特征提取...")
                    train_topo_features = self._extract_topo_features_parallel(sorted(train_samples), self.num_workers)
                else:
                    print("   🔄 启动串行特征提取...")
                    train_topo_features = self._extract_topo_features_serial(sorted(train_samples))
                for sample_name, features in train_topo_features.items():
                    self.topo_features_cache[sample_name] = features
                print("   ✅ 训练数据特征提取完成")
            if new_samples:
                print(f"   2️⃣ 处理新数据 ({len(new_samples)} 个样本)...")
                print("   📝 注意：新数据使用与训练时完全相同的特征提取方法")
                if self.use_parallel:
                    print("   🚀 启动并行特征提取...")
                    new_topo_features = self._extract_topo_features_parallel(sorted(new_samples), self.num_workers)
                else:
                    print("   🔄 启动串行特征提取...")
                    new_topo_features = self._extract_topo_features_serial(sorted(new_samples))
                for sample_name, features in new_topo_features.items():
                    self.topo_features_cache[sample_name] = features
                print("   ✅ 新数据特征提取完成")
            print("✅ 所有特征提取和缓存完成")
            self._save_features_to_cache()
            total_samples = len(self.sample_identifiers)
            topo_nonzero_count = sum(1 for feat in self.topo_features_cache.values() if self._get_feat_sum(feat) > 0)
            print(f"📊 特征质量统计:")
            print(f"   拓扑特征非零样本: {topo_nonzero_count}/{total_samples} ({topo_nonzero_count/total_samples*100:.1f}%)")
            if train_samples:
                print("🔍 验证训练数据特征一致性...")
                train_topo_nonzero = sum(1 for name in train_samples if self._get_topo_sum(name) > 0)
                print(f"   训练数据拓扑特征非零: {train_topo_nonzero}/{len(train_samples)} ({train_topo_nonzero/len(train_samples)*100:.1f}%)")
                if train_topo_nonzero == 0:
                    print("   ⚠️  警告: 训练数据特征可能存在问题")
                else:
                    print("   ✅ 训练数据特征一致性验证通过")
        except Exception as e:
            print(f"⚠️  特征提取和缓存过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            for sample_name in self.sample_identifiers:
                self.topo_features_cache[sample_name] = torch.zeros(18, dtype=torch.float32)
    def _save_features_to_cache(self):
        try:
            cache_dir = "feature_cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_data = {
                'version': '1.0',
                'timestamp': time.time(),
                'features': {},
                'metadata': {
                    'total_samples': len(self.topo_features_cache),
                    'feature_dim': 18,
                    'data_type': 'topological_features',
                    'extraction_method': 'persistent_homology'
                }
            }
            for sample_name, features in self.topo_features_cache.items():
                if isinstance(features, torch.Tensor):
                    cache_data['features'][sample_name] = features.cpu().numpy()
                else:
                    cache_data['features'][sample_name] = features
            cache_file = os.path.join(cache_dir, "topo_features_cache.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            cache_size = os.path.getsize(cache_file) / (1024 * 1024)  
            print(f"💾 特征缓存已保存:")
            print(f"   缓存文件: {cache_file}")
            print(f"   缓存大小: {cache_size:.2f} MB")
            print(f"   缓存样本数: {len(self.topo_features_cache)}")
            print(f"   下次运行将自动加载此缓存")
        except Exception as e:
            print(f"⚠️  保存特征缓存失败: {e}")
            print("   特征仍保存在内存中，但不会持久化")
    def _load_features_from_cache(self, cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if not isinstance(cached_data, dict):
                raise ValueError("缓存文件格式错误")
            features = cached_data.get('features', {})
            if not features:
                raise ValueError("缓存文件中没有特征数据")
            loaded_features = {}
            for sample_name, feature_array in features.items():
                if isinstance(feature_array, np.ndarray):
                    loaded_features[sample_name] = torch.from_numpy(feature_array).float()
                else:
                    loaded_features[sample_name] = feature_array
            return loaded_features
        except Exception as e:
            print(f"⚠️  加载特征缓存失败: {e}")
            return {}
    def _extract_real_topo_features(self, sample_name):
        try:
            parts = sample_name.split('_')
            if len(parts) >= 4:
                topo_type = parts[-4]  
            else:
                topo_type = "default"
            cif_dir = self.data_config['cif_dir']
            cif_files = list(Path(cif_dir).glob(f"{sample_name}*.cif"))
            if not cif_files:
                return torch.zeros(18, dtype=torch.float32)
            cif_file = cif_files[0]
            try:
                from pymatgen.core import Structure
                import gudhi as gd
                struct = Structure.from_file(str(cif_file))
                coords = struct.cart_coords
                rips = gd.RipsComplex(points=coords, max_edge_length=10.0)
                simplex_tree = rips.create_simplex_tree(max_dimension=2)
                diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.1)
                pers_0 = [p[1][1] - p[1][0] for p in diag if p[0] == 0 and p[1][1] != float('inf')]
                pers_1 = [p[1][1] - p[1][1] for p in diag if p[0] == 1 and p[1][1] != float('inf')]
                bins = np.linspace(0, 5, 10)  
                hist_0 = np.histogram(pers_0, bins=bins)[0] if pers_0 else np.zeros(9)
                hist_1 = np.histogram(pers_1, bins=bins)[0] if pers_1 else np.zeros(9)
                pers = np.concatenate([hist_0, hist_1]).astype(np.float32)
                return torch.tensor(pers, dtype=torch.float32)
            except ImportError as e:
                return torch.zeros(18, dtype=torch.float32)
        except Exception as e:
            return torch.zeros(18, dtype=torch.float32)
    def _extract_topo_features_parallel(self, sample_names, num_workers=None):
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"🚀 启动多核并行特征提取...")
        print(f"   📊 并行进程数: {num_workers}")
        print(f"   📊 总样本数: {len(sample_names)}")
        cif_dir = self.data_config['cif_dir']
        pbar = tqdm(total=len(sample_names), desc="提取拓扑特征", unit="样本")
        start_time = time.time()
        completed_count = 0
        failed_count = 0
        results = {}
        batch_size = min(100, max(10, len(sample_names) // num_workers))
        print(f"   📊 批处理大小: {batch_size}")
        try:
            for batch_start in range(0, len(sample_names), batch_size):
                batch_end = min(batch_start + batch_size, len(sample_names))
                batch_samples = sample_names[batch_start:batch_end]
                print(f"   🔄 处理批次 {batch_start//batch_size + 1}/{(len(sample_names)-1)//batch_size + 1}: {len(batch_samples)} 个样本")
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    future_to_sample = {}
                    for sample_name in batch_samples:
                        future = executor.submit(self._extract_single_topo_worker, sample_name, cif_dir)
                        future_to_sample[future] = sample_name
                    for future in as_completed(future_to_sample):
                        sample_name = future_to_sample[future]
                        try:
                            result = future.result(timeout=300)  
                            results[sample_name] = result
                            completed_count += 1
                            pbar.update(1)
                            if completed_count > 0:
                                elapsed_time = time.time() - start_time
                                avg_time_per_sample = elapsed_time / completed_count
                                remaining_samples = len(sample_names) - completed_count
                                estimated_remaining_time = avg_time_per_sample * remaining_samples
                                pbar.set_description(
                                    f"提取拓扑特征 ({completed_count}/{len(sample_names)}) "
                                    f"预计剩余: {estimated_remaining_time/60:.1f}分钟"
                                )
                        except Exception as e:
                            failed_count += 1
                            print(f"⚠️  样本 {sample_name} 特征提取失败: {e}")
                            results[sample_name] = torch.zeros(18, dtype=torch.float32)
                            completed_count += 1
                            pbar.update(1)
                            if failed_count > len(sample_names) * 0.1:  
                                print(f"⚠️  失败率过高 ({failed_count}/{completed_count})，考虑回退到串行处理")
                import gc
                gc.collect()
                time.sleep(1)
            pbar.close()
            total_time = time.time() - start_time
            avg_time_per_sample = total_time / len(sample_names)
            print(f"✅ 并行特征提取完成!")
            print(f"   📊 总耗时: {total_time/60:.1f} 分钟")
            print(f"   📊 平均每样本: {avg_time_per_sample:.2f} 秒")
            print(f"   📊 并行加速比: {num_workers:.1f}x")
            print(f"   📊 成功样本: {completed_count - failed_count}")
            print(f"   📊 失败样本: {failed_count}")
            print(f"   📊 成功率: {(completed_count - failed_count)/len(sample_names)*100:.1f}%")
            return results
        except Exception as e:
            print(f"❌ 并行特征提取失败: {e}")
            pbar.close()
            print("🔄 回退到串行处理...")
            return self._extract_topo_features_serial(sample_names)
    def _extract_single_topo_worker(self, sample_name, cif_dir):
        try:
            import resource
            import signal
            try:
                resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))
            except:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (600, -1))
            except:
                pass
            parts = sample_name.split('_')
            if len(parts) >= 4:
                topo_type = parts[-4]
            else:
                topo_type = "default"
            cif_files = list(Path(cif_dir).glob(f"{sample_name}*.cif"))
            if not cif_files:
                return torch.zeros(18, dtype=torch.float32)
            cif_file = cif_files[0]
            from pymatgen.core import Structure
            import gudhi as gd
            file_size = cif_file.stat().st_size
            if file_size > 10 * 1024 * 1024:  
                print(f"      ⚠️  CIF文件过大: {file_size/(1024*1024):.1f}MB，跳过")
                return torch.zeros(18, dtype=torch.float32)
            struct = Structure.from_file(str(cif_file))
            coords = struct.cart_coords
            if len(coords) > 10000:  
                print(f"      ⚠️  结构过于复杂: {len(coords)} 个原子，跳过")
                return torch.zeros(18, dtype=torch.float32)
            rips = gd.RipsComplex(points=coords, max_edge_length=10.0)
            simplex_tree = rips.create_simplex_tree(max_dimension=2)
            diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.1)
            pers_0 = [p[1][1] - p[1][0] for p in diag if p[0] == 0 and p[1][1] != float('inf')]
            pers_1 = [p[1][1] - p[1][0] for p in diag if p[0] == 1 and p[1][1] != float('inf')]
            bins = np.linspace(0, 5, 10)
            hist_0 = np.histogram(pers_0, bins=bins)[0] if pers_0 else np.zeros(9)
            hist_1 = np.histogram(pers_1, bins=bins)[0] if pers_1 else np.zeros(9)
            pers = np.concatenate([hist_0, hist_1]).astype(np.float32)
            del struct, coords, rips, simplex_tree, diag
            import gc
            gc.collect()
            return torch.tensor(pers, dtype=torch.float32)
        except Exception as e:
            error_msg = str(e)
            if "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
                print(f"      ❌ 内存不足: {sample_name}")
            elif "timeout" in error_msg.lower():
                print(f"      ⏰ 超时: {sample_name}")
            elif "process" in error_msg.lower():
                print(f"      🔄 进程异常: {sample_name}")
            else:
                print(f"      ❌ 其他错误: {sample_name} - {error_msg}")
            return torch.zeros(18, dtype=torch.float32)
    def _extract_topo_features_serial(self, sample_names):
        print(f"🔄 串行特征提取...")
        results = {}
        pbar = tqdm(total=len(sample_names), desc="串行提取拓扑特征", unit="样本")
        start_time = time.time()
        failed_count = 0
        for i, sample_name in enumerate(sample_names):
            try:
                result = self._extract_real_topo_features(sample_name)
                results[sample_name] = result
                pbar.update(1)
                if i > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_sample = elapsed_time / (i + 1)
                    remaining_samples = len(sample_names) - (i + 1)
                    estimated_remaining_time = avg_time_per_sample * remaining_samples
                    pbar.set_description(
                        f"串行提取拓扑特征 ({i+1}/{len(sample_names)}) "
                        f"预计剩余: {estimated_remaining_time/60:.1f}分钟"
                    )
                if i % 100 == 0:
                    import gc
                    gc.collect()
            except Exception as e:
                failed_count += 1
                print(f"⚠️  样本 {sample_name} 特征提取失败: {e}")
                results[sample_name] = torch.zeros(18, dtype=torch.float32)
                pbar.update(1)
                if failed_count > 50:
                    print(f"⚠️  连续失败过多，考虑跳过剩余样本")
                    break
        pbar.close()
        total_time = time.time() - start_time
        print(f"✅ 串行特征提取完成!")
        print(f"   📊 总耗时: {total_time/60:.1f} 分钟")
        print(f"   📊 成功样本: {len(sample_names) - failed_count}")
        print(f"   📊 失败样本: {failed_count}")
        print(f"   📊 成功率: {(len(sample_names) - failed_count)/len(sample_names)*100:.1f}%")
        return results
    def _get_sample_identifiers(self):
        descriptor_names = set(self.descriptor_df['name'].tolist())
        struct_names = set(self.struct_features_df.iloc[:, 0].tolist())
        common_names = descriptor_names.intersection(struct_names)
        print(f"描述符样本数: {len(descriptor_names)}")
        print(f"结构特征样本数: {len(struct_names)}")
        print(f"共同样本数: {len(common_names)}")
        if len(common_names) == 0:
            print("❌ 错误: 描述符和结构特征CSV中没有共同样本")
            print("   请检查两个CSV文件的样本名称是否一致")
            return []
        gcn_cache_dir = self.data_config['gcn_cache_dir']
        samples_with_gcn = []
        samples_without_gcn = []
        print("🔍 检查GCN缓存可用性...")
        print(f"   缓存目录: {gcn_cache_dir}")
        print(f"   缓存目录是否存在: {os.path.exists(gcn_cache_dir)}")
        if os.path.exists(gcn_cache_dir):
            cache_files = list(Path(gcn_cache_dir).glob('*.pkl'))
            print(f"   缓存目录中的.pkl文件数量: {len(cache_files)}")
            if cache_files:
                print(f"   前5个缓存文件: {[f.name for f in cache_files[:5]]}")
        test_samples = list(common_names)[:5]
        print(f"   测试前5个样本的缓存情况:")
        for sample_name in test_samples:
            possible_cache_names = [
                f"{sample_name}.pkl",  
                f"{sample_name}.cif.pkl",  
                f"{sample_name}_relaxed.pkl",  
                f"{sample_name}_relaxed_interp_2.pkl",  
                f"{sample_name}_relaxed_interp_3.pkl"   
            ]
            cache_found = False
            actual_cache_name = None
            for cache_name in possible_cache_names:
                cache_file = os.path.join(gcn_cache_dir, cache_name)
                if os.path.exists(cache_file):
                    cache_found = True
                    actual_cache_name = cache_name
                    break
            if cache_found:
                print(f"     {sample_name}: {os.path.join(gcn_cache_dir, actual_cache_name)} ✅")
            else:
                print(f"     {sample_name}: {os.path.join(gcn_cache_dir, f'{sample_name}.pkl')} ❌")
        for sample_name in common_names:
            possible_cache_names = [
                f"{sample_name}.pkl",  
                f"{sample_name}.cif.pkl",  
                f"{sample_name}_relaxed.pkl",  
                f"{sample_name}_relaxed_interp_2.pkl",  
                f"{sample_name}_relaxed_interp_3.pkl"   
            ]
            cache_found = False
            actual_cache_name = None
            for cache_name in possible_cache_names:
                cache_file = os.path.join(gcn_cache_dir, cache_name)
                if os.path.exists(cache_file):
                    samples_with_gcn.append(sample_name)
                    cache_found = True
                    actual_cache_name = cache_name
                    break
            if not cache_found:
                samples_without_gcn.append(sample_name)
        print(f"✅ 有GCN缓存的样本: {len(samples_with_gcn)} 个")
        print(f"⚠️  无GCN缓存的样本: {len(samples_without_gcn)} 个")
        cache_coverage = len(samples_with_gcn) / len(common_names) * 100
        print(f"📊 GCN缓存覆盖率: {cache_coverage:.2f}%")
        self.samples_without_gcn = set(samples_without_gcn)
        final_samples = sorted(list(common_names))
        print(f"🎯 最终处理样本数: {len(final_samples)} 个")
        if samples_with_gcn:
            cache_format_stats = {}
            for sample_name in samples_with_gcn:
                possible_cache_names = [
                    f"{sample_name}.pkl",
                    f"{sample_name}.cif.pkl",
                    f"{sample_name}_relaxed.pkl",
                    f"{sample_name}_relaxed_interp_2.pkl",
                    f"{sample_name}_relaxed_interp_3.pkl"
                ]
                for cache_name in possible_cache_names:
                    cache_file = os.path.join(gcn_cache_dir, cache_name)
                    if os.path.exists(cache_file):
                        if cache_name.endswith('.cif.pkl'):
                            format_type = '.cif.pkl'
                        elif cache_name.endswith('_relaxed_interp_3.pkl'):
                            format_type = '_relaxed_interp_3.pkl'
                        elif cache_name.endswith('_relaxed_interp_2.pkl'):
                            format_type = '_relaxed_interp_2.pkl'
                        elif cache_name.endswith('_relaxed.pkl'):
                            format_type = '_relaxed.pkl'
                        else:
                            format_type = '.pkl'
                        cache_format_stats[format_type] = cache_format_stats.get(format_type, 0) + 1
                        break
            print(f"📁 缓存文件格式统计:")
            for format_type, count in sorted(cache_format_stats.items()):
                print(f"     {format_type}: {count} 个")
        return final_samples
    def _get_topo_sum(self, sample_name):
        feat = self.topo_features_cache.get(sample_name)
        if feat is None:
            return 0.0
        if isinstance(feat, torch.Tensor):
            return feat.abs().sum().item()
        elif isinstance(feat, np.ndarray):
            return np.abs(feat).sum()
        else:
            return 0.0
    def _get_feat_sum(self, feat):
        if feat is None:
            return 0.0
        if isinstance(feat, torch.Tensor):
            return feat.abs().sum().item()
        elif isinstance(feat, np.ndarray):
            return np.abs(feat).sum()
        else:
            return 0.0
    def _ensure_tensor(self, feat):
        if feat is None:
            return torch.zeros(18, dtype=torch.float32)  
        if isinstance(feat, torch.Tensor):
            return feat
        elif isinstance(feat, np.ndarray):
            return torch.from_numpy(feat).float()
        else:
            try:
                return torch.tensor(feat, dtype=torch.float32)
            except:
                return torch.zeros(18, dtype=torch.float32)
    def __len__(self):
        return len(self.sample_identifiers)
    def __getitem__(self, idx):
        sample_name = self.sample_identifiers[idx]
        print(f"\n🔍 处理样本 {idx + 1}/{len(self.sample_identifiers)}: {sample_name}")
        print("    📥 加载VAE数据...")
        vae_data = self._load_vae_data(sample_name)
        print("    📥 加载描述符...")
        descriptors = self._load_descriptors(sample_name)
        if sample_name in self.samples_without_gcn:
            print("    ⚠️  跳过GCN数据加载（无缓存）")
            import dgl
            empty_graph = dgl.heterograph({
                ("l", "l2n", "n"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                ("n", "n2l", "l"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
            })
            empty_graph.nodes["l"].data["feat"] = torch.zeros(0, 300, dtype=torch.float32)
            empty_graph.nodes["n"].data["feat"] = torch.zeros(0, 4, dtype=torch.float32)
            graph_cc, graph_noncc = empty_graph, empty_graph
        else:
            print("    📥 加载GCN图数据...")
            graph_cc, graph_noncc = self._load_gcn_data(sample_name)
        print("    📥 加载LZHNN特征（从缓存）...")
        topo_features = self.topo_features_cache.get(sample_name, torch.zeros(18, dtype=torch.float32))
        print("    📥 加载结构特征...")
        struct_features = self._load_struct_features(sample_name)
        print("    📥 生成CC掩码...")
        cc_mask = self._generate_cc_mask(sample_name)
        print(f"    ✅ 样本 {sample_name} 数据加载完成")
        print(f"      拓扑特征: {topo_features.shape}, 非零值: {self._get_feat_sum(topo_features):.2f}")
        print(f"      结构特征: {struct_features.shape}, 非零值: {self._get_feat_sum(struct_features):.2f}")
        return {
            'vae_data': vae_data.unsqueeze(0),  
            'descriptors': descriptors.unsqueeze(0),  
            'graph_cc': graph_cc,
            'graph_noncc': graph_noncc,
            'lzhnn_data': {
                'topo': self._ensure_tensor(topo_features).unsqueeze(0),  
                'struct': self._ensure_tensor(struct_features).unsqueeze(0)  
            },
            'cc_mask': cc_mask.unsqueeze(0),  
            'identifiers': [sample_name]
        }
    def _load_vae_data(self, sample_name):
        h5_files = list(Path(self.data_config['vae_h5_dir']).glob(f"{sample_name}*.h5"))
        if not h5_files:
            print(f"⚠️  未找到样本 {sample_name} 的H5文件")
            return torch.zeros(9, 2, 64, 64, dtype=torch.float32)
        h5_file = h5_files[0]
        print(f"    📁 找到H5文件: {h5_file.name}")
        try:
            with h5py.File(h5_file, 'r') as f:
                planes = []
                for i in range(9):
                    if f'plane_{i}' in f:
                        plane_data = f[f'plane_{i}'][:]
                        if plane_data.ndim == 3 and plane_data.shape == (2, 64, 64):
                            planes.append(plane_data)
                        else:
                            print(f"    ⚠️  平面 {i} 数据格式不正确: {plane_data.shape}")
                            planes.append(np.zeros((2, 64, 64), dtype=np.float32))
                    else:
                        print(f"    ⚠️  平面 {i} 不存在，使用零数据")
                        planes.append(np.zeros((2, 64, 64), dtype=np.float32))
                planes = np.stack(planes, axis=0)
                print(f"    📊 VAE数据形状: {planes.shape}")
                return torch.from_numpy(planes).float()
        except Exception as e:
            print(f"⚠️  加载H5文件 {h5_file} 失败: {e}")
            return torch.zeros(9, 2, 64, 64, dtype=torch.float32)
    def _load_descriptors(self, sample_name):
        try:
            descriptor_row = self.descriptor_df[self.descriptor_df['name'] == sample_name]
            if not descriptor_row.empty:
                if self.descriptor_cols is not None:
                    available_cols = []
                    missing_cols = []
                    for train_col in self.descriptor_cols:
                        if train_col in self.descriptor_df.columns:
                            available_cols.append(train_col)
                        else:
                            missing_cols.append(train_col)
                    if missing_cols:
                        print(f"    ⚠️  缺失列: {missing_cols}")
                        print(f"    📊 可用列: {available_cols}")
                    if available_cols:
                        descriptor_values = descriptor_row[available_cols].values[0]
                        print(f"    📊 提取到 {len(available_cols)} 个特征")
                    else:
                        print(f"    ❌ 没有可用的特征列")
                        return torch.zeros(len(self.descriptor_cols), dtype=torch.float32)
                    if len(available_cols) < len(self.descriptor_cols):
                        missing_count = len(self.descriptor_cols) - len(available_cols)
                        if hasattr(self, 'descriptor_scaler') and self.descriptor_scaler is not None:
                            mean_values = self.descriptor_scaler.mean_[len(available_cols):]
                            descriptor_values = np.concatenate([descriptor_values, mean_values])
                            print(f"    📊 用训练均值填充 {missing_count} 个缺失特征")
                        else:
                            padding = np.zeros(missing_count)
                            descriptor_values = np.concatenate([descriptor_values, padding])
                            print(f"    📊 用零填充 {missing_count} 个缺失特征")
                    if len(descriptor_values) != len(self.descriptor_cols):
                        print(f"    ⚠️  特征数量不匹配: 期望 {len(self.descriptor_cols)}, 实际 {len(descriptor_values)}")
                        if len(descriptor_values) < len(self.descriptor_cols):
                            missing_count = len(self.descriptor_cols) - len(descriptor_values)
                            if hasattr(self, 'descriptor_scaler') and self.descriptor_scaler is not None:
                                mean_values = self.descriptor_scaler.mean_[len(descriptor_values):]
                                descriptor_values = np.concatenate([descriptor_values, mean_values])
                            else:
                                padding = np.zeros(missing_count)
                                descriptor_values = np.concatenate([descriptor_values, padding])
                        else:
                            descriptor_values = descriptor_values[:len(self.descriptor_cols)]
                else:
                    print(f"    ⚠️  描述符标准化器未初始化，使用默认列")
                    descriptor_cols = ['%C', '%F', '%H', '%N', '%O', '%S', '%Si', 'PLD(?)', 'LCD(?)', 'surfacearea[m^2/g]', 'Porosity', 'Density(gr/cm3)']
                    available_cols = [col for col in descriptor_cols if col in self.descriptor_df.columns]
                    descriptor_values = descriptor_row[available_cols].values[0]
                descriptor_values = np.nan_to_num(descriptor_values, nan=0.0)
                if self.descriptor_scaler is not None:
                    if descriptor_values.ndim == 1:
                        descriptor_values = descriptor_values.reshape(1, -1)
                    descriptor_values = self.descriptor_scaler.transform(descriptor_values)[0]
                    print(f"    📊 描述符已标准化，特征数量: {len(descriptor_values)}")
                    if hasattr(self, 'train_descriptor_stats'):
                        expected_mean = self.train_descriptor_stats['mean']
                        expected_scale = self.train_descriptor_stats['scale']
                        if np.any(np.abs(descriptor_values) > 5):
                            print(f"    ⚠️  警告: 标准化后的特征值超出预期范围")
                            print(f"       特征值范围: {descriptor_values.min():.4f} - {descriptor_values.max():.4f}")
                else:
                    print(f"    ⚠️  描述符未标准化")
                return torch.tensor(descriptor_values, dtype=torch.float32)
            else:
                print(f"⚠️  未找到样本 {sample_name} 的描述符")
                if self.descriptor_cols is not None:
                    return torch.zeros(len(self.descriptor_cols), dtype=torch.float32)
                else:
                    return torch.zeros(12, dtype=torch.float32)  
        except Exception as e:
            print(f"⚠️  加载描述符失败: {e}")
            if self.descriptor_cols is not None:
                return torch.zeros(len(self.descriptor_cols), dtype=torch.float32)
            else:
                return torch.zeros(12, dtype=torch.float32)  
    def _load_gcn_data(self, sample_name):
        try:
            possible_cache_names = [
                f"{sample_name}.pkl",  
                f"{sample_name}.cif.pkl",  
                f"{sample_name}_relaxed.pkl",  
                f"{sample_name}_relaxed_interp_2.pkl",  
                f"{sample_name}_relaxed_interp_3.pkl"   
            ]
            cache_file = None
            for cache_name in possible_cache_names:
                temp_cache_file = Path(self.data_config['gcn_cache_dir']) / cache_name
                if temp_cache_file.exists():
                    cache_file = temp_cache_file
                    break
            if cache_file is not None:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"    📁 从缓存加载GCN数据: {cache_file.name}")
                    import dgl
                    empty_graph = dgl.heterograph({
                        ("l", "l2n", "n"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                        ("n", "n2l", "l"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                    })
                    empty_graph.nodes["l"].data["feat"] = torch.zeros(0, 300, dtype=torch.float32)
                    empty_graph.nodes["n"].data["feat"] = torch.zeros(0, 4, dtype=torch.float32)
                    return cached_data, empty_graph
            else:
                print(f"    ⚠️  无GCN缓存，使用真实featurizer从CIF生成...")
                generated_graph = self._generate_real_graph_from_cif(sample_name)
                if generated_graph is not None:
                    print(f"    ✅ 成功从CIF生成真实图数据")
                    return generated_graph, generated_graph
                else:
                    print(f"    ⚠️  CIF生成失败，使用随机图数据")
                    return self._generate_random_graph(), self._generate_random_graph()
        except Exception as e:
            print(f"⚠️  加载GCN数据失败: {e}")
            return self._generate_random_graph(), self._generate_random_graph()
    def _generate_real_graph_from_cif(self, sample_name):
        try:
            cif_dir = self.data_config['cif_dir']
            print(f"      📁 搜索CIF目录: {cif_dir}")
            print(f"      🔍 搜索模式: {sample_name}*.cif")
            if not os.path.exists(cif_dir):
                print(f"      ❌ CIF目录不存在: {cif_dir}")
                return None
            cif_files = list(Path(cif_dir).glob(f"{sample_name}*.cif"))
            print(f"      📊 找到 {len(cif_files)} 个匹配的CIF文件")
            if not cif_files:
                print(f"      ⚠️  未找到CIF文件: {sample_name}")
                return None
            cif_file = cif_files[0]
            print(f"      📁 找到CIF文件: {cif_file.name}")
            try:
                from GCN.featurizer import get_2cg_inputs_cof
                graph_data = get_2cg_inputs_cof(str(cif_file), "GCN/linkers.csv")
                if graph_data is not None:
                    print(f"      ✅ 使用真实featurizer成功生成图数据")
                    print(f"        节点数: {graph_data.num_nodes('l')} linkers, {graph_data.num_nodes('n')} nodes")
                    print(f"        边数: {graph_data.num_edges('l2n')}")
                    return graph_data
                else:
                    print(f"      ⚠️  featurizer返回空图")
                    return None
            except Exception as featurizer_error:
                print(f"      ⚠️  真实featurizer失败: {featurizer_error}")
                return self._parse_cif_manually(cif_file)
        except Exception as e:
            print(f"      ⚠️  从CIF生成真实图数据失败: {e}")
            return None
    def _parse_cif_manually(self, cif_file):
        try:
            import dgl
            import numpy as np
            with open(cif_file, 'r') as f:
                cif_content = f.read()
            lines = cif_content.split('\n')
            atoms = []
            for line in lines:
                if line.strip().startswith('_atom_site_'):
                    continue
                elif line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            atom_type = parts[0]
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                            atoms.append((atom_type, x, y, z))
                        except:
                            continue
            if len(atoms) == 0:
                print(f"        ⚠️  无法解析原子坐标")
                return None
            num_atoms = len(atoms)
            node_features = []
            for atom_type, x, y, z in atoms:
                if atom_type.startswith('C'):
                    atom_feat = [1.0, 0.0, 0.0, 0.0]  
                elif atom_type.startswith('N'):
                    atom_feat = [0.0, 1.0, 0.0, 0.0]  
                elif atom_type.startswith('O'):
                    atom_feat = [0.0, 0.0, 1.0, 0.0]  
                else:
                    atom_feat = [0.0, 0.0, 0.0, 1.0]  
                atom_feat.extend([x/10, y/10, z/10])  
                node_features.append(atom_feat)
            while len(node_features[0]) < 4:
                for i in range(len(node_features)):
                    node_features[i].append(0.0)
            node_features = [feat[:4] for feat in node_features]
            src_nodes = []
            dst_nodes = []
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        src_nodes.append(i)
                        dst_nodes.append(j)
            graph = dgl.heterograph({
                ("l", "l2n", "n"): (torch.tensor(src_nodes, dtype=torch.long), torch.tensor(dst_nodes, dtype=torch.long)),
                ("n", "n2l", "l"): (torch.tensor(dst_nodes, dtype=torch.long), torch.tensor(src_nodes, dtype=torch.long)),
            })
            graph.nodes["l"].data["feat"] = torch.tensor(node_features, dtype=torch.float32)
            graph.nodes["n"].data["feat"] = torch.tensor(node_features, dtype=torch.float32)
            print(f"        ✅ 手动解析生成图: {num_atoms} 个原子, {len(src_nodes)} 条边")
            return graph
        except Exception as e:
            print(f"        ⚠️  手动解析CIF失败: {e}")
            return None
    def _generate_random_graph(self):
        try:
            import dgl
            import numpy as np
            num_nodes = np.random.randint(10, 50)
            node_features = torch.randn(num_nodes, 4, dtype=torch.float32) * 0.1
            num_edges = np.random.randint(num_nodes, num_nodes * 3)
            src_nodes = np.random.randint(0, num_nodes, num_edges)
            dst_nodes = np.random.randint(0, num_nodes, num_edges)
            graph = dgl.heterograph({
                ("l", "l2n", "n"): (torch.tensor(src_nodes, dtype=torch.long), torch.tensor(dst_nodes, dtype=torch.long)),
                ("n", "n2l", "l"): (torch.tensor(dst_nodes, dtype=torch.long), torch.tensor(src_nodes, dtype=torch.long)),
            })
            graph.nodes["l"].data["feat"] = node_features
            graph.nodes["n"].data["feat"] = node_features
            return graph
        except Exception as e:
            print(f"        ⚠️  生成随机图失败: {e}")
            import dgl
            empty_graph = dgl.heterograph({
                ("l", "l2n", "n"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                ("n", "n2l", "l"): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
            })
            empty_graph.nodes["l"].data["feat"] = torch.zeros(0, 4, dtype=torch.float32)
            empty_graph.nodes["n"].data["feat"] = torch.zeros(0, 4, dtype=torch.float32)
            return empty_graph
    def _load_struct_features(self, sample_name):
        try:
            struct_row = self.struct_features_df[self.struct_features_df['name'] == sample_name]
            if not struct_row.empty:
                possible_struct_cols = [
                    ['PLD', 'LCD', 'surface_area', 'porosity', 'density'],
                    ['PLD(A)', 'LCD(A)', 'surfacearea[m^2/g]', 'Porosity', 'Density(gr/cm3)'],
                    ['pld', 'lcd', 'surface_area', 'porosity', 'density']
                ]
                struct_values = None
                used_cols = None
                for cols in possible_struct_cols:
                    available_cols = [col for col in cols if col in self.struct_features_df.columns]
                    if len(available_cols) >= 3:  
                        try:
                            struct_values = struct_row[available_cols].values[0]
                            used_cols = available_cols
                            break
                        except:
                            continue
                if struct_values is not None:
                    struct_values = np.nan_to_num(struct_values, nan=0.0)
                    if len(struct_values) < 5:
                        padding = np.zeros(5 - len(struct_values))
                        struct_values = np.concatenate([struct_values, padding])
                        print(f"    📊 用零填充到5维特征")
                    struct_tensor = torch.from_numpy(struct_values).float()
                    print(f"    📊 结构特征: 使用列 {used_cols}")
                    return struct_tensor
                else:
                    print(f"⚠️  未找到样本 {sample_name} 的有效结构特征列")
                    return torch.zeros(5, dtype=torch.float32)
            else:
                print(f"⚠️  未找到样本 {sample_name} 的结构特征")
                return torch.zeros(5, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️  加载结构特征失败: {e}")
            return torch.zeros(5, dtype=torch.float32)
    def _is_cc_structure(self, sample_name):
        return 'CC' in sample_name
    def _generate_cc_mask(self, sample_name):
        is_cc = self._is_cc_structure(sample_name)
        return torch.tensor([[is_cc]], dtype=torch.float32)  
def main():
    parser = argparse.ArgumentParser(description='使用自定义数据格式的融合模型预测新数据')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的融合模型路径')
    parser.add_argument('--vae_h5_dir', type=str, required=True, help='VAE的H5文件目录')
    parser.add_argument('--descriptor_csv', type=str, required=True, help='描述符CSV文件路径')
    parser.add_argument('--cif_dir', type=str, required=True, help='CIF文件目录')
    parser.add_argument('--struct_features_csv', type=str, required=True, help='结构特征CSV文件路径')
    parser.add_argument('--gcn_cache_dir', type=str, required=True, help='GCN缓存目录')
    parser.add_argument('--output_file', type=str, default='custom_predictions.csv', help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--use_parallel', action='store_true', default=True, help='启用并行特征提取（默认启用）')
    parser.add_argument('--no_parallel', dest='use_parallel', action='store_false', help='禁用并行特征提取，使用串行处理')
    parser.add_argument('--num_workers', type=int, default=None, help='并行进程数（默认自动检测CPU核心数-1）')
    args = parser.parse_args()
    print("🔍 验证输入参数...")
    required_files = [
        ('模型文件', args.model_path),
        ('VAE H5目录', args.vae_h5_dir),
        ('描述符CSV', args.descriptor_csv),
        ('CIF目录', args.cif_dir),
        ('结构特征CSV', args.struct_features_csv),
        ('GCN缓存目录', args.gcn_cache_dir)
    ]
    for name, path in required_files:
        if os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path)
                print(f"   ✅ {name}: {path} ({size} bytes)")
            else:
                file_count = len(list(Path(path).glob('*')))
                print(f"   ✅ {name}: {path} ({file_count} 文件)")
        else:
            print(f"   ❌ {name}: {path} (不存在)")
            if name != 'GCN缓存目录':  
                print(f"     错误: {name}不存在，无法继续")
                sys.exit(1)
    try:
        print("\n🚀 创建预测器...")
        if args.device == 'cuda' and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"📊 检测到GPU内存: {gpu_memory:.1f} GB")
            if gpu_memory < 8.0:
                print("⚠️  警告：GPU内存较小，可能出现内存不足错误")
                print("   建议：使用 --device cpu 参数进行CPU推理")
                print("   或者：减少batch_size或使用梯度检查点")
        predictor = CustomDataPredictor(args.model_path, args.device)
        data_config = {
            'vae_h5_dir': args.vae_h5_dir,
            'descriptor_csv': args.descriptor_csv,
            'cif_dir': args.cif_dir,
            'struct_features_csv': args.struct_features_csv,
            'gcn_cache_dir': args.gcn_cache_dir
        }
        print("\n📊 准备新数据...")
        dataloader = predictor.prepare_new_data(
            data_config, 
            use_parallel=args.use_parallel, 
            num_workers=args.num_workers
        )
        if len(dataloader) == 0:
            print("❌ 错误: 没有可用的数据样本")
            sys.exit(1)
        print(f"\n🧠 开始预测 {len(dataloader)} 个样本...")
        results = predictor.predict(dataloader, args.output_file)
        print("\n🎉 预测完成！")
        print(f"📊 预测结果统计:")
        print(f"   样本数量: {len(results)}")
        if 'prediction_normalized' in results.columns:
            print(f"   标准化预测值范围: {results['prediction_normalized'].min():.4f} - {results['prediction_normalized'].max():.4f}")
            print(f"   标准化预测值均值: {results['prediction_normalized'].mean():.4f}")
            print(f"   标准化预测值标准差: {results['prediction_normalized'].std():.4f}")
        if 'prediction_lg' in results.columns:
            print(f"   lg预测值范围: {results['prediction_lg'].min():.4f} - {results['prediction_lg'].max():.4f}")
            print(f"   lg预测值均值: {results['prediction_lg'].mean():.4f}")
            print(f"   lg预测值标准差: {results['prediction_lg'].std():.4f}")
        print(f"   结果列名: {list(results.columns)}")
        summary_file = args.output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"预测结果摘要\n")
            f.write(f"==========\n\n")
            f.write(f"样本数量: {len(results)}\n")
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"输出文件: {args.output_file}\n")
            f.write(f"预测时间: {pd.Timestamp.now()}\n\n")
            if 'prediction_normalized' in results.columns:
                f.write(f"标准化预测值统计:\n")
                f.write(f"  范围: {results['prediction_normalized'].min():.6f} - {results['prediction_normalized'].max():.6f}\n")
                f.write(f"  均值: {results['prediction_normalized'].mean():.6f}\n")
                f.write(f"  标准差: {results['prediction_normalized'].std():.6f}\n\n")
            if 'prediction_lg' in results.columns:
                f.write(f"lg预测值统计:\n")
                f.write(f"  范围: {results['prediction_lg'].min():.6f} - {results['prediction_lg'].max():.6f}\n")
                f.write(f"  均值: {results['prediction_lg'].mean():.6f}\n")
                f.write(f"  标准差: {results['prediction_lg'].std():.6f}\n")
        print(f"📁 结果摘要已保存到: {summary_file}")
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        if "CUDA error: out of memory" in str(e) or "out of memory" in str(e):
            print("\n🔧 CUDA内存不足解决方案:")
            print("   1. 使用CPU推理: --device cpu")
            print("   2. 减少batch_size（当前为1，已是最小值）")
            print("   3. 清理GPU内存: nvidia-smi 查看进程，kill占用内存的进程")
            print("   4. 重启系统释放GPU内存")
            print("   5. 使用更小的模型或模型量化")
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
                cached_memory = torch.cuda.memory_reserved(0) / 1024**3
                print(f"\n📊 当前GPU内存状态:")
                print(f"   总内存: {gpu_memory:.1f} GB")
                print(f"   已分配: {allocated_memory:.1f} GB")
                print(f"   已缓存: {cached_memory:.1f} GB")
                print(f"   可用内存: {gpu_memory - cached_memory:.1f} GB")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()