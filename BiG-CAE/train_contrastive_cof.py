import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from featurizer import get_2cg_inputs_cof, parse_cof_filename
from utils import EarlyStopping, get_stratified_folds, get_samples, collate_fn, COFDataset, create_cof_dataloader
from contrastive_autoencoder import ContrastiveAutoencoder, ContrastiveLoss, ReconstructionLoss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
def load_data(structures_dir, labels_csv, property_name):
    print("加载数据...")
    cif_files = glob.glob(os.path.join(structures_dir, "*.cif"))
    print(f"找到 {len(cif_files)} 个CIF文件")
    if os.path.exists(labels_csv):
        labels_df = pd.read_csv(labels_csv)
        print(f"从 {labels_csv} 加载标签")
        file_to_label = {}
        for _, row in labels_df.iterrows():
            filename = row.get('filename', '').strip()
            if filename and property_name in row:
                cif_filename = f"{filename}.cif"
                file_to_label[cif_filename] = float(row[property_name])
        valid_files = []
        labels = []
        for cif_file in cif_files:
            filename = os.path.basename(cif_file)
            if filename in file_to_label:
                valid_files.append(cif_file)
                labels.append(file_to_label[filename])
        print(f"找到 {len(valid_files)} 个有标签的文件")
        return valid_files, labels
    else:
        print(f"标签文件 {labels_csv} 未找到，生成随机标签用于测试")
        labels = np.random.randn(len(cif_files)).tolist()
        return cif_files, labels
def normalize_labels(labels):
    scaler = StandardScaler()
    labels_normalized = scaler.fit_transform(labels.reshape(-1, 1)).flatten()
    return labels_normalized, scaler
def split_by_pattern(files, labels):
    cc_files, cc_labels = [], []
    other_files, other_labels = [], []
    for f, l in zip(files, labels):
        filename = os.path.basename(f)
        _, connection_groups, _ = parse_cof_filename(filename)
        if connection_groups and len(connection_groups) >= 3:
            pattern = connection_groups[2]
            if pattern == "CC":
                cc_files.append(f)
                cc_labels.append(l)
            else:
                other_files.append(f)
                other_labels.append(l)
    return (cc_files, cc_labels), (other_files, other_labels)
def create_data_loaders(files, labels, linkers_csv, batch_size, test_size, val_size, random_state):
    print("创建数据加载器...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        files, labels, test_size=test_size, random_state=random_state, stratify=None
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=None
    )
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    train_loader = create_cof_dataloader(X_train, y_train, linkers_csv, batch_size, shuffle=True, num_workers=1)
    val_loader = create_cof_dataloader(X_val, y_val, linkers_csv, batch_size, shuffle=False, num_workers=1)
    test_loader = create_cof_dataloader(X_test, y_test, linkers_csv, batch_size, shuffle=False, num_workers=1)
    return train_loader, val_loader, test_loader
def train_contrastive_model(
    model, 
    train_loader, 
    val_loader, 
    epochs=50, 
    lr=0.0005,
    device='cuda',
    tag="cc"
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5, verbose=True)
    contrastive_loss_fn = ContrastiveLoss(temperature=0.1)
    reconstruction_loss_fn = ReconstructionLoss(loss_type='huber')
    property_loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(prefix=tag, patience=30)
    train_losses = []
    val_losses = []
    print("开始训练对比学习自编码器...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        contrastive_loss_sum = 0.0
        recon_loss_sum = 0.0
        prop_loss_sum = 0.0
        for batch_idx, (graphs, labels) in enumerate(train_loader):
            import dgl
            graphs = graphs.to(device)
            labels = labels.to(device).float()
            graphs_list = dgl.unbatch(graphs)
            aug_graphs1, aug_graphs2 = [], []
            for g in graphs_list:
                g1, g2 = g.clone(), g.clone()
                for ntype in g1.ntypes:
                    if g1.num_nodes(ntype) > 0:
                        noise1 = torch.randn_like(g1.nodes[ntype].data["feat"]) * 0.1
                        noise2 = torch.randn_like(g2.nodes[ntype].data["feat"]) * 0.1
                        g1.nodes[ntype].data["feat"] += noise1
                        g2.nodes[ntype].data["feat"] += noise2
                aug_graphs1.append(g1)
                aug_graphs2.append(g2)
            aug_batch1 = dgl.batch(aug_graphs1)
            aug_batch2 = dgl.batch(aug_graphs2)
            z1 = model.encode(aug_batch1)
            z2 = model.encode(aug_batch2)
            proj1 = model.project(z1)
            proj2 = model.project(z2)
            contrastive_loss = contrastive_loss_fn(proj1, proj2)
            reconstructed, property_pred, z = model(graphs, mode='autoencoder')
            recon_loss = reconstruction_loss_fn(reconstructed, z)
            prop_loss = property_loss_fn(property_pred.squeeze(), labels)
            total_batch_loss = (
                model.beta * contrastive_loss + 
                model.alpha * recon_loss + 
                prop_loss
            )
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
            contrastive_loss_sum += contrastive_loss.item()
            recon_loss_sum += recon_loss.item()
            prop_loss_sum += prop_loss.item()
            if batch_idx % 20 == 0:
                print(f"{tag.upper()} Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                      f"Total Loss: {total_batch_loss.item():.6f}, "
                      f"Contrastive: {contrastive_loss.item():.6f}, "
                      f"Recon: {recon_loss.item():.6f}, "
                      f"Property: {prop_loss.item():.6f}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graphs, labels in val_loader:
                graphs = graphs.to(device)
                labels = labels.to(device).float()
                property_pred = model(graphs, mode='property')
                val_loss += property_loss_fn(property_pred.squeeze(), labels).item()
        val_loss /= len(val_loader)
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if early_stopping.step(val_loss, model):
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
        print(f"{tag.upper()} Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Avg Contrastive: {contrastive_loss_sum/len(train_loader):.6f}, "
              f"Avg Recon: {recon_loss_sum/len(train_loader):.6f}, "
              f"Avg Property: {prop_loss_sum/len(train_loader):.6f}")
    return train_losses, val_losses
def evaluate_model_with_denormalize(model, dataloader, device, scaler):
    model.eval()
    y_true = torch.tensor([], dtype=torch.float32, device=device)
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            pred = model(X, mode='property')
            y_pred = torch.cat((y_pred, pred), 0)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_true_denorm = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    r2 = r2_score(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(np.mean((y_true_denorm - y_pred_denorm) ** 2))
    return mae, r2, rmse, y_true_denorm, y_pred_denorm
def save_contrastive_results(y_true, y_pred, train_losses, val_losses, tag):
    results_df = pd.DataFrame({
        'true_values': y_true,
        'predicted_values': y_pred,
        'absolute_error': np.abs(y_true - y_pred)
    })
    results_df.to_csv(f"results/contrastive_predictions_{tag}.csv", index=False)
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv(f"results/contrastive_history_{tag}.csv", index=False)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title(f'对比学习训练损失曲线 ({tag})')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'对比学习预测vs真实值 ({tag})')
    plt.subplot(1, 3, 3)
    errors = y_true - y_pred
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title(f'预测误差分布 ({tag})')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/contrastive_results_{tag}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比学习结果已保存到 results/contrastive_results_{tag}.png")
def main():
    parser = argparse.ArgumentParser(description='对比学习自编码器COF性质预测')
    parser.add_argument('--structures_dir', type=str, default='structures1',
                        help='包含CIF文件的目录')
    parser.add_argument('--linkers_csv', type=str, default='linkers.csv',
                        help='linkers.csv文件路径')
    parser.add_argument('--labels_csv', type=str, default='labels.csv',
                        help='标签CSV文件路径')
    parser.add_argument('--property_name', type=str, default='target_property',
                        help='标签CSV中性质列的名称')
    parser.add_argument('--encoder_dim', type=int, default=128,
                        help='编码器维度')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='潜在空间维度')
    parser.add_argument('--decoder_dim', type=int, default=128,
                        help='解码器维度')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='对比学习温度参数')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='重构损失权重')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='对比损失权重')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='学习率')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    files, labels = load_data(args.structures_dir, args.labels_csv, args.property_name)
    labels_normalized, scaler = normalize_labels(np.array(labels))
    print(f"标签标准化 - 均值: {np.mean(labels_normalized):.6f}, 标准差: {np.std(labels_normalized):.6f}")
    (cc_files, cc_labels), (other_files, other_labels) = split_by_pattern(files, labels_normalized)
    for tag, sub_files, sub_labels in [("cc", cc_files, cc_labels), ("noncc", other_files, other_labels)]:
        if len(sub_files) == 0:
            print(f"{tag} 类数据为空，跳过")
            continue
        print(f"\n==== 开始处理 {tag} 类数据 ====")
        train_loader, val_loader, test_loader = create_data_loaders(
            sub_files, sub_labels, args.linkers_csv, args.batch_size,
            args.test_size, args.val_size, args.random_state
        )
        model = ContrastiveAutoencoder(
            encoder_dim=args.encoder_dim,
            latent_dim=args.latent_dim,
            decoder_dim=args.decoder_dim,
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta
        )
        print(f"对比学习自编码器模型参数量: {sum(p.numel() for p in model.parameters())}")
        train_losses, val_losses = train_contrastive_model(
            model, train_loader, val_loader, args.epochs, args.lr, device, tag
        )
        print(f"\n{tag.upper()} 最终评估:")
        print("-" * 50)
        val_mae, val_r2, val_rmse, val_true, val_pred = evaluate_model_with_denormalize(
            model, val_loader, device, scaler
        )
        print(f"验证集 - MAE: {val_mae:.4f}, R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
        test_mae, test_r2, test_rmse, test_true, test_pred = evaluate_model_with_denormalize(
            model, test_loader, device, scaler
        )
        print(f"测试集 - MAE: {test_mae:.4f}, R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        model_path = f"models/contrastive_model_{tag}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'scaler': scaler
        }, model_path)
        print(f"对比学习模型已保存到 {model_path}")
        save_contrastive_results(test_true, test_pred, train_losses, val_losses, tag)
    print("\n对比学习自编码器训练完成！")
if __name__ == "__main__":
    main() 