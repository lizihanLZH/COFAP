import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
from pathlib import Path
from ElseFeaturizer import CombinedCOFFeaturizer
from Net import MultiModalSAGEModel, train_step, valid_step, predict_dataloader
from UtilsCode import create_dataloader, EarlyStopping, get_stratified_folds
def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Modal COF Property Prediction Model')
    parser.add_argument('--struct_features_csv', type=str, required=True,
                       help='Path to structural features CSV file')
    parser.add_argument('--target_csv', type=str, required=True,
                       help='Path to target CSV file')
    parser.add_argument('--cif_dir', type=str, required=True,
                       help='Directory containing CIF files')
    parser.add_argument('--cache_dir', type=str, default='feature_cache',
                       help='Directory for feature cache (default: feature_cache)')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Enable feature caching (default: True)')
    parser.add_argument('--clear_cache', action='store_true', default=False,
                       help='Clear existing cache before training (default: False)')
    parser.add_argument('--topo_dim', type=int, default=18,
                       help='Topological fingerprint dimension')
    parser.add_argument('--struct_dim', type=int, default=5,
                       help='Structural features dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of MLP layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=200,
                       help='Early stopping patience')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size (from remaining data)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--num_workers', type=int, default=12,
                       help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    return parser.parse_args()
def load_data(args):
    print("Loading data...")
    target_df = pd.read_csv(args.target_csv)
    cof_names = target_df.iloc[:, 0].tolist()
    print(f"Found {len(cof_names)} COFs in target file")
    featurizer = CombinedCOFFeaturizer(
        struct_features_csv=args.struct_features_csv,
        target_csv=args.target_csv,
        cache_dir=args.cache_dir
    )
    if args.clear_cache:
        print("🗑️ Clearing existing cache...")
        featurizer.clear_cache()
    cache_info = featurizer.get_cache_info()
    print("\n📊 Cache Information:")
    print(f"   Cache directory: {cache_info['cache_dir']}")
    print(f"   Cache loaded: {cache_info['cache_loaded']}")
    if cache_info['cache_files']['topo_features']:
        print(f"   Topo features cache: {cache_info['cache_files']['topo_features']['size_mb']:.2f} MB")
    if cache_info['cache_files']['struct_features']:
        print(f"   Struct features cache: {cache_info['cache_files']['struct_features']['size_mb']:.2f} MB")
    if cache_info['cache_files']['targets']:
        print(f"   Targets cache: {cache_info['cache_files']['targets']['size_mb']:.2f} MB")
    print(f"\n🔄 Starting feature extraction (cache {'enabled' if args.use_cache else 'disabled'})...")
    start_time = pd.Timestamp.now()
    batch_features = featurizer.batch_featurize(
        cof_names, 
        args.cif_dir, 
        args.num_workers,
        use_cache=args.use_cache
    )
    end_time = pd.Timestamp.now()
    extraction_time = (end_time - start_time).total_seconds()
    print(f"✅ Feature extraction completed in {extraction_time:.2f} seconds")
    if args.use_cache:
        updated_cache_info = featurizer.get_cache_info()
        print(f"\n📊 Updated Cache Information:")
        print(f"   Cache loaded: {updated_cache_info['cache_loaded']}")
        if updated_cache_info['cache_files']['topo_features']:
            print(f"   Topo features cache: {updated_cache_info['cache_files']['topo_features']['size_mb']:.2f} MB")
        if updated_cache_info['cache_files']['struct_features']:
            print(f"   Struct features cache: {updated_cache_info['cache_files']['struct_features']['size_mb']:.2f} MB")
        if updated_cache_info['cache_files']['targets']:
            print(f"   Targets cache: {updated_cache_info['cache_files']['targets']['size_mb']:.2f} MB")
    return batch_features
def prepare_data_splits(batch_features, args):
    print("Preparing data splits...")
    topo_features = batch_features['topo']
    struct_features = batch_features['struct']
    targets = batch_features['targets']
    print(f"Feature shapes:")
    print(f"  Topological: {topo_features.shape}")
    print(f"  Structural: {struct_features.shape}")
    print(f"  Targets: {targets.shape}")
    indices = np.arange(len(targets))
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=get_stratified_folds(targets)
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=get_stratified_folds(targets[train_val_idx])
    )
    print(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    splits = {}
    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[split_name] = {
            'topo': topo_features[idx],
            'struct': struct_features[idx],
            'targets': targets[idx]
        }
    return splits
def create_dataloaders(splits, args):
    print("Creating data loaders...")
    dataloaders = {}
    for split_name, data in splits.items():
        shuffle = (split_name == 'train')
        dataloader = create_dataloader(
            topo_features=data['topo'],
            struct_features=data['struct'],
            targets=data['targets'],
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers
        )
        dataloaders[split_name] = dataloader
    return dataloaders
def evaluate_model(model, dataloader, device):
    y_true, y_pred = predict_dataloader(model, dataloader, device)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
def main():
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print("\n" + "="*60)
    print("LZHNN Training Configuration")
    print("="*60)
    print(f"Data paths:")
    print(f"  Structural features: {args.struct_features_csv}")
    print(f"  Target values: {args.target_csv}")
    print(f"  CIF directory: {args.cif_dir}")
    print(f"  Cache directory: {args.cache_dir}")
    print(f"  Use cache: {args.use_cache}")
    print(f"  Clear cache: {args.clear_cache}")
    print(f"\nModel parameters:")
    print(f"  Topological dimension: {args.topo_dim}")
    print(f"  Structural dimension: {args.struct_dim}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Dropout rate: {args.dropout}")
    print(f"\nTraining parameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Number of workers: {args.num_workers}")
    print("="*60)
    batch_features = load_data(args)
    splits = prepare_data_splits(batch_features, args)
    dataloaders = create_dataloaders(splits, args)
    model = MultiModalSAGEModel(
        topo_dim=args.topo_dim,
        struct_dim=args.struct_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(prefix='lzhnn_model', patience=args.patience)
    print("Starting training...")
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        train_loss = train_step(dataloaders['train'], model, criterion, optimizer, device)
        train_losses.append(train_loss)
        val_loss = valid_step(dataloaders['val'], model, criterion, device)
        val_losses.append(val_loss)
        if early_stopping.step(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    early_stopping.load_checkpoint(model)
    print("\nFinal Evaluation:")
    print("=" * 50)
    for split_name, dataloader in dataloaders.items():
        metrics = evaluate_model(model, dataloader, device)
        print(f"{split_name.upper()} - RMSE: {metrics['rmse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    model_path = f"final_model_{args.random_state}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"\nModel saved to {model_path}")
    if args.use_cache:
        print("\n📊 Final Cache Information:")
        from ElseFeaturizer import CombinedCOFFeaturizer
        featurizer = CombinedCOFFeaturizer(
            struct_features_csv=args.struct_features_csv,
            target_csv=args.target_csv,
            cache_dir=args.cache_dir
        )
        final_cache_info = featurizer.get_cache_info()
        print(f"   Cache directory: {final_cache_info['cache_dir']}")
        print(f"   Cache loaded: {final_cache_info['cache_loaded']}")
        if final_cache_info['cache_files']['topo_features']:
            print(f"   Topo features cache: {final_cache_info['cache_files']['topo_features']['size_mb']:.2f} MB")
        if final_cache_info['cache_files']['struct_features']:
            print(f"   Struct features cache: {final_cache_info['cache_files']['struct_features']['size_mb']:.2f} MB")
        if final_cache_info['cache_files']['targets']:
            print(f"   Targets cache: {final_cache_info['cache_files']['targets']['size_mb']:.2f} MB")
if __name__ == "__main__":
    main()