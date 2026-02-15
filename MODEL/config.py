"""
Configuration file for Predictive Maintenance Automation System
Contains all hyperparameters, paths, and model settings
"""

import torch
import os

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Predictive_Maintenance_Synthetic_Data.csv')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# DEVICE CONFIGURATION
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    DEVICE_NAME = 'CUDA'
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    DEVICE_NAME = 'MPS (Apple Silicon)'
else:
    DEVICE = torch.device('cpu')
    DEVICE_NAME = 'CPU'

# REPRODUCIBILITY
RANDOM_SEED = 42

# DATA CONFIGURATION
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

STATIC_FEATURES = ['machine_id']
TIME_VARYING_FEATURES = [
    'process_temperature', 'air_temperature', 'vibration', 'torque', 'rpm',
    'current', 'operating_hours', 'time_since_last_maintenance',
    'idle_duration', 'power_consumption'
]
CATEGORICAL_FEATURES = ['last_maintenance_Type']
TARGET_COLUMN = 'RUL'

HEALTH_CRITICAL_THRESHOLD = 50
HEALTH_WARNING_THRESHOLD = 150

# MODEL 1: TFT (RUL PREDICTION)
TFT_CONFIG = {
    'hidden_size': 128,
    'lstm_layers': 2,
    'dropout': 0.1,
    'attention_heads': 4,
    'embedding_dim': 32,
    'sequence_length': 24,
    'output_size': 1,
}

TFT_TRAINING = {
    'batch_size': 256,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 10,
    'grad_clip_value': 1.0,
    'huber_delta': 10.0,
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'scheduler_min_lr': 1e-6,
}

# MODEL 2: HEALTH STATUS
HEALTH_CONFIG = {
    'model_type': 'classification',
    'hidden_sizes': [128, 64, 32],
    'dropout': 0.3,
    'num_classes': 3,
}

HEALTH_TRAINING = {
    'batch_size': 256,
    'epochs': 40,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 8,
    'grad_clip_value': 1.0,
    'scheduler_factor': 0.5,
    'scheduler_patience': 4,
}

# MODEL 3: MAINTENANCE SUGGESTION
MAINTENANCE_CONFIG = {
    'hidden_sizes': [128, 64, 32],
    'dropout': 0.3,
    'num_classes': 3,
}

MAINTENANCE_TRAINING = {
    'batch_size': 256,
    'epochs': 40,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 8,
    'grad_clip_value': 1.0,
    'scheduler_factor': 0.5,
    'scheduler_patience': 4,
}

LOG_INTERVAL = 10
SAVE_BEST_MODEL = True
