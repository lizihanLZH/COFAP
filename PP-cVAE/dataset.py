import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
class ChemicalDataset(Dataset):
    def __init__(self, metadata_df, data_dir, transform=None, normalize_labels=False, 
                 descriptor_df=None, normalize_descriptors=True):
        self.metadata = metadata_df
        self.data_dir = data_dir
        self.transform = transform
        self.normalize_labels = normalize_labels
        self.descriptor_df = descriptor_df
        self.normalize_descriptors = normalize_descriptors
        self.file_groups = self.metadata.groupby('h5_file')
        self.file_list = list(self.file_groups.groups.keys())
        if self.normalize_labels:
            labels = [self.file_groups.get_group(f)['label'].iloc[0] for f in self.file_list]
            self.label_scaler = StandardScaler()
            self.label_scaler.fit(np.array(labels).reshape(-1, 1))
        else:
            self.label_scaler = None
        if self.descriptor_df is not None and self.normalize_descriptors:
            descriptor_cols = [col for col in self.descriptor_df.columns if col != 'name']
            self.descriptor_scaler = StandardScaler()
            self.descriptor_scaler.fit(self.descriptor_df[descriptor_cols].values)
            self.descriptor_cols = descriptor_cols
        else:
            self.descriptor_scaler = None
            self.descriptor_cols = None
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        h5_file = self.file_list[idx]
        file_data = self.file_groups.get_group(h5_file)
        label = file_data['label'].iloc[0]
        if self.label_scaler is not None:
            label = self.label_scaler.transform([[label]])[0, 0]
        if 'images1/train/' in h5_file:
            file_path = h5_file.replace('images1/train/', 'train/')
        elif 'images1/val/' in h5_file:
            file_path = h5_file.replace('images1/val/', 'val/')
        elif 'images1/test/' in h5_file:
            file_path = h5_file.replace('images1/test/', 'test/')
        else:
            file_path = h5_file
        file_path = os.path.join(self.data_dir, file_path)
        planes = []
        try:
            with h5py.File(file_path, 'r') as f:
                for i in range(9):
                    plane_data = f[f'plane_{i}'][:]
                    planes.append(plane_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            planes = [np.zeros((2, 64, 64), dtype=np.float32) for _ in range(9)]
        planes = np.stack(planes, axis=0)
        planes = torch.from_numpy(planes).float()
        if self.transform:
            planes = self.transform(planes)
        descriptors = None
        if self.descriptor_df is not None:
            structure_name = os.path.basename(h5_file).replace('.h5', '')
            descriptor_row = self.descriptor_df[self.descriptor_df['name'] == structure_name]
            if not descriptor_row.empty:
                descriptor_values = descriptor_row[self.descriptor_cols].values[0]
                if self.descriptor_scaler is not None:
                    descriptor_values = self.descriptor_scaler.transform([descriptor_values])[0]
                descriptors = torch.tensor(descriptor_values, dtype=torch.float32)
            else:
                descriptors = torch.zeros(len(self.descriptor_cols), dtype=torch.float32)
        if descriptors is not None:
            return planes, descriptors, torch.tensor(label, dtype=torch.float32)
        else:
            return planes, torch.tensor(label, dtype=torch.float32)
    def get_label_scaler(self):
        return self.label_scaler
def create_data_loaders(config, normalize_labels=False, use_descriptors=False, descriptor_path=None):
    print('Loading metadata...')
    metadata = pd.read_csv(config.METADATA_PATH)
    descriptor_df = None
    if use_descriptors and descriptor_path:
        print('Loading descriptors...')
        descriptor_df = pd.read_csv(descriptor_path)
        print(f'Loaded {len(descriptor_df)} descriptor entries')
        print(f'Descriptor features: {list(descriptor_df.columns)}')
    train_df = metadata[metadata['dataset'] == 'train'].copy()
    val_df = metadata[metadata['dataset'] == 'val'].copy()
    test_df = metadata[metadata['dataset'] == 'test'].copy()
    print(f'Train samples: {len(train_df.groupby("h5_file"))}')
    print(f'Val samples: {len(val_df.groupby("h5_file"))}')
    print(f'Test samples: {len(test_df.groupby("h5_file"))}')
    train_dataset = ChemicalDataset(train_df, config.DATA_DIR, normalize_labels=normalize_labels, 
                                   descriptor_df=descriptor_df)
    val_dataset = ChemicalDataset(val_df, config.DATA_DIR, normalize_labels=normalize_labels, 
                                 descriptor_df=descriptor_df)
    test_dataset = ChemicalDataset(test_df, config.DATA_DIR, normalize_labels=normalize_labels, 
                                  descriptor_df=descriptor_df)
    dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': getattr(config, 'PIN_MEMORY', True),
        'prefetch_factor': getattr(config, 'PREFETCH_FACTOR', 2) if config.NUM_WORKERS > 0 else None,
        'persistent_workers': config.NUM_WORKERS > 0  
    }
    dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    return train_loader, val_loader, test_loader, train_dataset.get_label_scaler()
def get_dataset_stats(metadata_path):
    metadata = pd.read_csv(metadata_path)
    stats = {
        'total_files': len(metadata.groupby('h5_file')),
        'train_files': len(metadata[metadata['dataset'] == 'train'].groupby('h5_file')),
        'val_files': len(metadata[metadata['dataset'] == 'val'].groupby('h5_file')),
        'test_files': len(metadata[metadata['dataset'] == 'test'].groupby('h5_file')),
        'label_min': metadata['label'].min(),
        'label_max': metadata['label'].max(),
        'label_mean': metadata['label'].mean(),
        'label_std': metadata['label'].std()
    }
    return stats