import datetime
import numpy as np
import dgl
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
def process_one(args):
    i, cif_file, labels, linkers_csv_path = args
    from featurizer import get_2cg_inputs_cof
    try:
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cif_basename = os.path.basename(cif_file)
        cache_file = os.path.join(cache_dir, cif_basename + ".pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                graph = pickle.load(f)
        else:
            graph = get_2cg_inputs_cof(cif_file, linkers_csv_path)
            with open(cache_file, "wb") as f:
                pickle.dump(graph, f)
        if graph.num_nodes() > 0:
            return (i, graph, labels[i])
        else:
            print(f"Warning: Empty graph for {cif_file}")
    except Exception as e:
        print(f"Error processing {cif_file}: {e}")
    return None
class EarlyStopping:
    def __init__(self, prefix, patience=50):
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
def get_samples(graphs, labels):
    return [(graph, label) for graph, label in zip(graphs, labels)]
def collate_fn(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
class COFDataset(Dataset):
    def __init__(self, cif_files, labels, linkers_csv_path="linkers.csv"):
        self.cif_files = cif_files
        self.labels = labels
        self.linkers_csv_path = linkers_csv_path
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    def __len__(self):
        return len(self.cif_files)
    def __getitem__(self, idx):
        import pickle
        from featurizer import get_2cg_inputs_cof
        cif_file = self.cif_files[idx]
        cache_file = os.path.join(self.cache_dir, os.path.basename(cif_file) + ".pkl")
        graph = None
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    graph = pickle.load(f)
            except Exception as e:
                print(f"读取cache失败，尝试现场建图: {cache_file}, 错误: {e}")
        if graph is None:
            graph = get_2cg_inputs_cof(cif_file, self.linkers_csv_path)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(graph, f)
            except Exception as e:
                print(f"写入cache失败: {cache_file}, 错误: {e}")
        return graph, self.labels[idx]
    def get_valid_indices(self):
        return list(range(len(self.cif_files)))
def create_cof_dataloader(cif_files, labels, linkers_csv_path="linkers.csv", batch_size=32, shuffle=True, num_workers=0):
    dataset = COFDataset(cif_files, labels, linkers_csv_path)
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
def split_cof_data(cif_files, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    from sklearn.model_selection import train_test_split
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    X_train, X_temp, y_train, y_temp = train_test_split(
        cif_files, labels, test_size=(1-train_ratio), random_state=random_state, stratify=None
    )
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_test_ratio), random_state=random_state, stratify=None
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)