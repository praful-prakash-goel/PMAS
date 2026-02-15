"""
MAIN EXECUTION SCRIPT
Predictive Maintenance Automation System

This script orchestrates the complete end-to-end pipeline:
1. Data loading and preprocessing
2. RUL computation
3. Feature engineering
4. Model training (TFT, Health Status, Maintenance Type)
5. Evaluation
6. Example inference
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

# Import utilities
from utils.data_utils import (
    set_seed, compute_rul, create_health_labels, time_based_split,
    create_scalers, save_scalers, load_scalers, print_metrics
)
from utils.datasets import RULSequenceDataset, HealthStatusDataset, MaintenanceDataset

# Import models
from models.tft_model import TemporalFusionTransformer
from models.other_models import HealthStatusClassifier, MaintenanceTypeClassifier

# Import training functions
from training.train_tft import train_tft_model, evaluate_tft_model
from training.train_other_models import (
    train_health_status_model, train_maintenance_model,
    evaluate_classifier_model
)


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)


def main():
    """Main execution function"""
    
    print_header("PREDICTIVE MAINTENANCE AUTOMATION SYSTEM")
    print(f"\nDevice: {config.DEVICE_NAME}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    
    # Set reproducibility
    set_seed(config.RANDOM_SEED)
    
    # ========================================================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ========================================================================
    print_header("STEP 1: DATA LOADING AND PREPROCESSING")
    
    print(f"\nLoading data from: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Machines: {df['machine_id'].nunique()}")
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # ========================================================================
    # STEP 2: COMPUTE RUL
    # ========================================================================
    print_header("STEP 2: COMPUTE REMAINING USEFUL LIFE (RUL)")
    
    df = compute_rul(df)
    
    # Create health status labels
    df = create_health_labels(
        df,
        critical_threshold=config.HEALTH_CRITICAL_THRESHOLD,
        warning_threshold=config.HEALTH_WARNING_THRESHOLD
    )
    
    # Drop machine_failure to prevent data leakage
    print("\n⚠ Dropping 'machine_failure' column to prevent data leakage")
    df = df.drop('machine_failure', axis=1)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c not in ['timestamp', 'machine_id', 'RUL', 'health_status', 'health_score']]}")
    
    # ========================================================================
    # STEP 3: TIME-BASED SPLIT
    # ========================================================================
    print_header("STEP 3: TIME-BASED TRAIN/VAL/TEST SPLIT")
    
    train_df, val_df, test_df = time_based_split(
        df,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO
    )
    
    # ========================================================================
    # STEP 4: CREATE SCALERS (FIT ON TRAIN ONLY)
    # ========================================================================
    print_header("STEP 4: CREATE AND FIT SCALERS")
    
    scalers = create_scalers(
        train_df,
        config.TIME_VARYING_FEATURES,
        config.CATEGORICAL_FEATURES
    )
    
    # Save scalers
    scaler_path = os.path.join(config.CHECKPOINT_DIR, 'scalers.pkl')
    save_scalers(scalers, scaler_path)
    
    # Get number of machines for embedding
    num_machines = len(scalers['machine_id'].classes_)
    num_maintenance_types = len(scalers['last_maintenance_Type'].classes_)
    print(f"\nNumber of machines: {num_machines}")
    print(f"Number of maintenance types: {num_maintenance_types}")
    
    # ========================================================================
    # STEP 5: CREATE DATASETS
    # ========================================================================
    print_header("STEP 5: CREATE PYTORCH DATASETS")
    
    # Calculate time-varying feature dimension
    time_varying_dim = len(config.TIME_VARYING_FEATURES) + len(config.CATEGORICAL_FEATURES)
    
    print(f"\nCreating RUL Sequence Datasets (sequence_length={config.TFT_CONFIG['sequence_length']})")
    train_rul_dataset = RULSequenceDataset(
        train_df, scalers, config.TFT_CONFIG['sequence_length'],
        config.TIME_VARYING_FEATURES, config.CATEGORICAL_FEATURES, 'RUL'
    )
    val_rul_dataset = RULSequenceDataset(
        val_df, scalers, config.TFT_CONFIG['sequence_length'],
        config.TIME_VARYING_FEATURES, config.CATEGORICAL_FEATURES, 'RUL'
    )
    test_rul_dataset = RULSequenceDataset(
        test_df, scalers, config.TFT_CONFIG['sequence_length'],
        config.TIME_VARYING_FEATURES, config.CATEGORICAL_FEATURES, 'RUL'
    )
    print(f"  Train: {len(train_rul_dataset)} sequences")
    print(f"  Val:   {len(val_rul_dataset)} sequences")
    print(f"  Test:  {len(test_rul_dataset)} sequences")
    
    print(f"\nCreating Health Status Datasets")
    train_health_dataset = HealthStatusDataset(
        train_df, scalers, config.TIME_VARYING_FEATURES,
        config.CATEGORICAL_FEATURES, 'health_status'
    )
    val_health_dataset = HealthStatusDataset(
        val_df, scalers, config.TIME_VARYING_FEATURES,
        config.CATEGORICAL_FEATURES, 'health_status'
    )
    test_health_dataset = HealthStatusDataset(
        test_df, scalers, config.TIME_VARYING_FEATURES,
        config.CATEGORICAL_FEATURES, 'health_status'
    )
    print(f"  Train: {len(train_health_dataset)} samples")
    print(f"  Val:   {len(val_health_dataset)} samples")
    print(f"  Test:  {len(test_health_dataset)} samples")
    
    print(f"\nCreating Maintenance Type Datasets")
    train_maint_dataset = MaintenanceDataset(
        train_df, scalers, config.TIME_VARYING_FEATURES,
        config.CATEGORICAL_FEATURES, 'last_maintenance_Type'
    )
    val_maint_dataset = MaintenanceDataset(
        val_df, scalers, config.TIME_VARYING_FEATURES,
        config.CATEGORICAL_FEATURES, 'last_maintenance_Type'
    )
    test_maint_dataset = MaintenanceDataset(
        test_df, scalers, config.TIME_VARYING_FEATURES,
        config.CATEGORICAL_FEATURES, 'last_maintenance_Type'
    )
    print(f"  Train: {len(train_maint_dataset)} samples")
    print(f"  Val:   {len(val_maint_dataset)} samples")
    print(f"  Test:  {len(test_maint_dataset)} samples")
    
    # ========================================================================
    # STEP 6: CREATE DATALOADERS
    # ========================================================================
    print_header("STEP 6: CREATE DATALOADERS")
    
    # RUL DataLoaders
    train_rul_loader = DataLoader(
        train_rul_dataset, batch_size=config.TFT_TRAINING['batch_size'],
        shuffle=True, num_workers=0
    )
    val_rul_loader = DataLoader(
        val_rul_dataset, batch_size=config.TFT_TRAINING['batch_size'],
        shuffle=False, num_workers=0
    )
    test_rul_loader = DataLoader(
        test_rul_dataset, batch_size=config.TFT_TRAINING['batch_size'],
        shuffle=False, num_workers=0
    )
    
    # Health Status DataLoaders
    train_health_loader = DataLoader(
        train_health_dataset, batch_size=config.HEALTH_TRAINING['batch_size'],
        shuffle=True, num_workers=0
    )
    val_health_loader = DataLoader(
        val_health_dataset, batch_size=config.HEALTH_TRAINING['batch_size'],
        shuffle=False, num_workers=0
    )
    test_health_loader = DataLoader(
        test_health_dataset, batch_size=config.HEALTH_TRAINING['batch_size'],
        shuffle=False, num_workers=0
    )
    
    # Maintenance DataLoaders
    train_maint_loader = DataLoader(
        train_maint_dataset, batch_size=config.MAINTENANCE_TRAINING['batch_size'],
        shuffle=True, num_workers=0
    )
    val_maint_loader = DataLoader(
        val_maint_dataset, batch_size=config.MAINTENANCE_TRAINING['batch_size'],
        shuffle=False, num_workers=0
    )
    test_maint_loader = DataLoader(
        test_maint_dataset, batch_size=config.MAINTENANCE_TRAINING['batch_size'],
        shuffle=False, num_workers=0
    )
    
    print("✓ DataLoaders created successfully")
    
    # ========================================================================
    # STEP 7: TRAIN MODEL 1 - TFT FOR RUL PREDICTION
    # ========================================================================
    print_header("STEP 7: TRAIN MODEL 1 - TEMPORAL FUSION TRANSFORMER")
    
    # Initialize TFT model
    tft_model = TemporalFusionTransformer(
        num_static_vars=num_machines,
        static_embedding_dim=config.TFT_CONFIG['embedding_dim'],
        time_varying_dim=time_varying_dim,
        hidden_size=config.TFT_CONFIG['hidden_size'],
        lstm_layers=config.TFT_CONFIG['lstm_layers'],
        dropout=config.TFT_CONFIG['dropout'],
        attention_heads=config.TFT_CONFIG['attention_heads'],
        output_size=config.TFT_CONFIG['output_size']
    ).to(config.DEVICE)
    
    print(f"\nModel Architecture:")
    print(f"  Static Embedding Dim: {config.TFT_CONFIG['embedding_dim']}")
    print(f"  Time-varying Features: {time_varying_dim}")
    print(f"  Hidden Size: {config.TFT_CONFIG['hidden_size']}")
    print(f"  LSTM Layers: {config.TFT_CONFIG['lstm_layers']}")
    print(f"  Attention Heads: {config.TFT_CONFIG['attention_heads']}")
    print(f"  Total Parameters: {sum(p.numel() for p in tft_model.parameters()):,}")
    
    # Train TFT
    tft_model, tft_history = train_tft_model(
        tft_model, train_rul_loader, val_rul_loader,
        config.TFT_TRAINING, config.DEVICE, config.CHECKPOINT_DIR
    )
    
    # Evaluate TFT
    tft_metrics, tft_predictions, tft_targets = evaluate_tft_model(
        tft_model, test_rul_loader, config.DEVICE
    )
    
    # ========================================================================
    # STEP 8: TRAIN MODEL 2 - HEALTH STATUS CLASSIFIER
    # ========================================================================
    print_header("STEP 8: TRAIN MODEL 2 - HEALTH STATUS CLASSIFIER")
    
    # Get input dimension for health model
    health_input_dim = len(config.TIME_VARYING_FEATURES) + len(config.CATEGORICAL_FEATURES) + 1  # +1 for machine_id
    
    # Initialize Health Status model
    health_model = HealthStatusClassifier(
        input_dim=health_input_dim,
        hidden_sizes=config.HEALTH_CONFIG['hidden_sizes'],
        dropout=config.HEALTH_CONFIG['dropout'],
        num_classes=config.HEALTH_CONFIG['num_classes']
    ).to(config.DEVICE)
    
    print(f"\nModel Architecture:")
    print(f"  Input Dim: {health_input_dim}")
    print(f"  Hidden Layers: {config.HEALTH_CONFIG['hidden_sizes']}")
    print(f"  Output Classes: {config.HEALTH_CONFIG['num_classes']} (Critical, Warning, Healthy)")
    print(f"  Total Parameters: {sum(p.numel() for p in health_model.parameters()):,}")
    
    # Train Health Status model
    health_model, health_history = train_health_status_model(
        health_model, train_health_loader, val_health_loader,
        config.HEALTH_TRAINING, config.DEVICE, config.CHECKPOINT_DIR, 'health'
    )
    
    # Evaluate Health Status model
    health_metrics, health_predictions, health_targets = evaluate_classifier_model(
        health_model, test_health_loader, config.DEVICE, 'Health Status'
    )
    
    # ========================================================================
    # STEP 9: TRAIN MODEL 3 - MAINTENANCE TYPE CLASSIFIER
    # ========================================================================
    print_header("STEP 9: TRAIN MODEL 3 - MAINTENANCE TYPE CLASSIFIER")
    
    # Get input dimension for maintenance model (includes RUL)
    maint_input_dim = len(config.TIME_VARYING_FEATURES) + len([c for c in config.CATEGORICAL_FEATURES if c != 'last_maintenance_Type']) + 1 + 1  # +1 machine_id, +1 RUL
    
    # Initialize Maintenance model
    maint_model = MaintenanceTypeClassifier(
        input_dim=maint_input_dim,
        hidden_sizes=config.MAINTENANCE_CONFIG['hidden_sizes'],
        dropout=config.MAINTENANCE_CONFIG['dropout'],
        num_classes=config.MAINTENANCE_CONFIG['num_classes']
    ).to(config.DEVICE)
    
    print(f"\nModel Architecture:")
    print(f"  Input Dim: {maint_input_dim}")
    print(f"  Hidden Layers: {config.MAINTENANCE_CONFIG['hidden_sizes']}")
    print(f"  Output Classes: {config.MAINTENANCE_CONFIG['num_classes']}")
    print(f"  Total Parameters: {sum(p.numel() for p in maint_model.parameters()):,}")
    
    # Train Maintenance model
    maint_model, maint_history = train_maintenance_model(
        maint_model, train_maint_loader, val_maint_loader,
        config.MAINTENANCE_TRAINING, config.DEVICE, config.CHECKPOINT_DIR
    )
    
    # Evaluate Maintenance model
    maint_metrics, maint_predictions, maint_targets = evaluate_classifier_model(
        maint_model, test_maint_loader, config.DEVICE, 'Maintenance Type'
    )
    
    # ========================================================================
    # STEP 10: FINAL SUMMARY AND EXAMPLE INFERENCE
    # ========================================================================
    print_header("FINAL SUMMARY")
    
    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print("\n1. RUL Prediction (TFT):")
    print(f"   MAE:  {tft_metrics['MAE']:.2f} hours")
    print(f"   RMSE: {tft_metrics['RMSE']:.2f} hours")
    print(f"   R²:   {tft_metrics['R2']:.4f}")
    
    print("\n2. Health Status Classification:")
    print(f"   Accuracy: {health_metrics['Accuracy']:.4f}")
    print(f"   F1-Score: {health_metrics['F1_Macro']:.4f}")
    
    print("\n3. Maintenance Type Prediction:")
    print(f"   Accuracy: {maint_metrics['Accuracy']:.4f}")
    print(f"   F1-Score: {maint_metrics['F1_Macro']:.4f}")
    
    # ========================================================================
    # EXAMPLE INFERENCE
    # ========================================================================
    print_header("EXAMPLE INFERENCE")
    
    # Get a random sample from test set
    sample_idx = np.random.randint(0, len(test_df) - config.TFT_CONFIG['sequence_length'])
    sample_machine = test_df.iloc[sample_idx]['machine_id']
    
    print(f"\n🔍 Sample Machine: {sample_machine}")
    print(f"Sample Index: {sample_idx}")
    
    # Get the sample data
    tft_model.eval()
    health_model.eval()
    maint_model.eval()
    
    with torch.no_grad():
        # Get RUL prediction
        rul_sample = test_rul_dataset[sample_idx]
        static = rul_sample['static'].unsqueeze(0).to(config.DEVICE)
        time_varying = rul_sample['time_varying'].unsqueeze(0).to(config.DEVICE)
        
        rul_pred = tft_model(static, time_varying).item()
        rul_true = rul_sample['target'].item()
        
        # Get Health Status prediction
        health_sample = test_health_dataset[sample_idx + config.TFT_CONFIG['sequence_length'] - 1]
        health_features = health_sample['features'].unsqueeze(0).to(config.DEVICE)
        
        health_logits = health_model(health_features)
        health_pred = torch.argmax(health_logits, dim=1).item()
        health_true = health_sample['target'].item()
        
        health_labels = ['Critical', 'Warning', 'Healthy']
        
        # Get Maintenance Type prediction
        maint_sample = test_maint_dataset[sample_idx + config.TFT_CONFIG['sequence_length'] - 1]
        maint_features = maint_sample['features'].unsqueeze(0).to(config.DEVICE)
        
        maint_logits = maint_model(maint_features)
        maint_pred = torch.argmax(maint_logits, dim=1).item()
        maint_true = maint_sample['target'].item()
        
        maint_labels = scalers['last_maintenance_Type'].classes_
    
    print("\n📋 PREDICTIONS:")
    print(f"\n1. Remaining Useful Life:")
    print(f"   Predicted: {rul_pred:.2f} hours")
    print(f"   Actual:    {rul_true:.2f} hours")
    print(f"   Error:     {abs(rul_pred - rul_true):.2f} hours")
    
    print(f"\n2. Health Status:")
    print(f"   Predicted: {health_labels[health_pred]}")
    print(f"   Actual:    {health_labels[health_true]}")
    print(f"   Match:     {'✓' if health_pred == health_true else '✗'}")
    
    print(f"\n3. Recommended Maintenance:")
    print(f"   Predicted: {maint_labels[maint_pred]}")
    print(f"   Actual:    {maint_labels[maint_true]}")
    print(f"   Match:     {'✓' if maint_pred == maint_true else '✗'}")
    
    print_header("TRAINING COMPLETE")
    
    print("\n✅ All models trained and evaluated successfully!")
    print(f"\n📁 Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print(f"   - tft_best.pth")
    print(f"   - health_best.pth")
    print(f"   - maintenance_best.pth")
    print(f"   - scalers.pkl")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
