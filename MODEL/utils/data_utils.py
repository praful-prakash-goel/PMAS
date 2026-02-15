"""Utility functions for data preprocessing"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rul(df):
    """Compute Remaining Useful Life (RUL) for each machine"""
    print("Computing RUL for each machine...")
    df = df.copy()
    df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
    
    df['RUL'] = 0.0
    
    for machine_id in df['machine_id'].unique():
        machine_mask = df['machine_id'] == machine_id
        machine_data = df[machine_mask].copy()
        
        failure_indices = machine_data[machine_data['machine_failure'] == 1].index.tolist()
        
        if len(failure_indices) > 0:
            for idx in machine_data.index:
                next_failures = [f for f in failure_indices if f > idx]
                
                if len(next_failures) > 0:
                    next_failure_idx = next_failures[0]
                    rul_value = next_failure_idx - idx
                else:
                    rul_value = machine_data.index[-1] - idx
                
                df.loc[idx, 'RUL'] = float(rul_value)
        else:
            for idx in machine_data.index:
                rul_value = machine_data.index[-1] - idx
                df.loc[idx, 'RUL'] = float(rul_value)
    
    print(f"RUL computed. Range: [{df['RUL'].min():.2f}, {df['RUL'].max():.2f}]")
    return df


def create_health_labels(df, critical_threshold=50, warning_threshold=150):
    """Create health status labels based on RUL"""
    df = df.copy()
    
    df['health_status'] = 2  # Default: Healthy
    df.loc[df['RUL'] < warning_threshold, 'health_status'] = 1  # Warning
    df.loc[df['RUL'] < critical_threshold, 'health_status'] = 0  # Critical
    
    max_rul = df['RUL'].max()
    df['health_score'] = df['RUL'] / max_rul
    
    print(f"Health labels created:")
    print(f"  Critical (0): {(df['health_status'] == 0).sum()}")
    print(f"  Warning (1): {(df['health_status'] == 1).sum()}")
    print(f"  Healthy (2): {(df['health_status'] == 2).sum()}")
    
    return df


def time_based_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data by time (no random shuffling to prevent leakage)"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\nTime-based split:")
    print(f"  Train: {len(train_df)} samples ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({test_ratio*100:.1f}%)")
    
    return train_df, val_df, test_df


def create_scalers(train_df, continuous_features, categorical_features):
    """Create and fit scalers on training data only"""
    scalers = {}
    
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_features])
    scalers['continuous'] = scaler
    
    for col in categorical_features:
        le = LabelEncoder()
        le.fit(train_df[col])
        scalers[col] = le
    
    machine_le = LabelEncoder()
    machine_le.fit(train_df['machine_id'])
    scalers['machine_id'] = machine_le
    
    return scalers


def save_scalers(scalers, path):
    """Save scalers to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {path}")


def load_scalers(path):
    """Load scalers from disk"""
    with open(path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers


def compute_metrics(y_true, y_pred, task='regression'):
    """Compute evaluation metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score, f1_score
    
    metrics = {}
    
    if task == 'regression':
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics['MAE'] = mae
        metrics['RMSE'] = rmse
        metrics['R2'] = r2
    
    elif task == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        metrics['Accuracy'] = accuracy
        metrics['F1_Macro'] = f1_macro
        metrics['F1_Weighted'] = f1_weighted
    
    return metrics


def print_metrics(metrics, prefix=""):
    """Pretty print metrics"""
    print(f"\n{prefix}Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")