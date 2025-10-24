# COFAP

A comprehensive deep learning framework for predicting properties of Covalent Organic Frameworks (COFs) using multiple neural network architectures and cross-attention fusion mechanisms.

##  Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/lizihanLZH/COFAP
cd model-code
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Individual Model Training

**VAE Model (PP-cVAE)**
```bash
cd PP-cVAE
python train.py --data_dir /path/to/data --epochs 100
```

**Graph Model (BiG-CAE)**
```bash
cd BiG-CAE
python train_contrastive_cof.py --structures_dir /path/to/structures --labels_csv labels.csv
```

**Topological Model (PH-NN)**
```bash
cd PH-NN
python train.py --struct_features_csv features.csv --target_csv targets.csv --cif_dir /path/to/cif
```

#### 2. Fusion Model Training

**Single Training**
```bash
python train_fusion.py \
    --vae_model_path PP-cVAE/models/best_model.pth \
    --gcn_cc_model_path BiG-CAE/models/contrastive_model_cc.pth \
    --gcn_noncc_model_path BiG-CAE/models/contrastive_model_noncc.pth \
    --lzhnn_model_path PH-NN/final_model_42.pth \
    --batch_size 16 \
    --num_epochs 50
```

**Cross-Validation Training**
```bash
python train_fusion.py \
    --use_cross_validation \
    --n_folds 5 \
    --batch_size 16 \
    --num_epochs 50
```

#### 3. Prediction

```bash
python predict.py \
    --vae_data /path/to/vae_data.h5 \
    --descriptors /path/to/descriptors.csv \
    --cif_files /path/to/cif_files \
    --output_dir predictions/
```



##  Project Structure

```
model-code/
├── PP-cVAE/                 # Variational Autoencoder
│   ├── models.py           # VAE model definitions
│   ├── dataset.py          # Data loading utilities
│   ├── trainer.py          # Training logic
│   └── config.py           # Configuration settings
├── BiG-CAE/                # Graph Contrastive Autoencoder
│   ├── contrastive_model.py # Contrastive learning model
│   ├── featurizer.py       # Graph feature extraction
│   └── utils.py            # Utility functions
├── PH-NN/                  # Persistent Homology Neural Network
│   ├── ppnn_model.py       # PH-NN model definition
│   ├── featurizer_topo.py  # Topological feature extraction
│   └── featurizer_struct.py # Structural feature extraction
├── fusion_model.py         # Cross-attention fusion model
├── train_fusion.py         # Fusion model training
├── predict.py              # Prediction pipeline
└── data_loader.py          # Unified data loading
```


##  Performance Metrics

The framework provides comprehensive evaluation metrics:

- **Regression Metrics**: R², RMSE, MAE, MSE
- **Correlation Metrics**: Pearson, Spearman
- **Model Stability**: Cross-validation analysis
- **Inference Performance**: Speed and memory usage

## 🛠️ Advanced Usage

### Custom Data Format

The framework supports various data formats:

- **VAE Data**: HDF5 files with 2D projections
- **Graph Data**: CIF files with graph representations
- **Topological Data**: CIF files with persistent homology features
- **Descriptors**: CSV files with molecular descriptors

### Caching System

Enable feature caching for faster training:

```python
# In PH-NN training
python train.py --use_cache --cache_dir feature_cache
```


##  License

This project is licensed under the MIT License - see the LICENSE file for details.



**Note**: This framework is designed for research purposes. For production use, additional testing and validation are recommended.
