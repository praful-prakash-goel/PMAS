# Predictive Maintenance Automation System

A comprehensive end-to-end PyTorch implementation for predictive maintenance using Temporal Fusion Transformer (TFT) and deep learning models.

## 🎯 Project Overview

This system predicts:
1. **Remaining Useful Life (RUL)** - Time until machine failure using TFT
2. **Current Health Status** - Machine health classification (Healthy/Warning/Critical)
3. **Maintenance Type** - Recommended maintenance action (preventive/predictive/corrective)

## 🏗️ Architecture

### Model 1: Temporal Fusion Transformer (TFT) for RUL
- **Architecture**: Hybrid transformer with LSTM backbone
- **Components**:
  - Static covariate embedding (machine_id)
  - Variable selection network
  - LSTM for temporal dependencies
  - Multi-head attention mechanism
  - Gated residual networks
- **Loss**: Huber Loss (δ=10.0)
  - Rationale: Less sensitive to RUL outliers, behaves like MAE for large errors
- **Metrics**: MAE, RMSE, R² (NO MAPE)

### Model 2: Health Status Classifier
- **Architecture**: Deep MLP
- **Rationale**: Current health determined by instantaneous readings, no temporal dependencies needed
- **Classes**: Critical (RUL<50h), Warning (50≤RUL<150h), Healthy (RUL≥150h)
- **Loss**: Cross-Entropy
- **Metrics**: Accuracy, F1-Score

### Model 3: Maintenance Type Classifier
- **Architecture**: Deep MLP
- **Input**: Sensor readings + predicted RUL
- **Classes**: preventive, predictive, corrective
- **Loss**: Cross-Entropy
- **Metrics**: Accuracy, F1-Score

## 📁 Project Structure

```
RUL_Project/
├── config.py                      # All hyperparameters and settings
├── main.py                        # Main execution script
├── data/
│   └── Predictive_Maintenance_Synthetic_Data.csv
├── models/
│   ├── __init__.py
│   ├── tft_model.py              # Temporal Fusion Transformer
│   └── other_models.py           # Health & Maintenance models
├── training/
│   ├── __init__.py
│   ├── train_tft.py              # TFT training logic
│   └── train_other_models.py    # Other models training
├── utils/
│   ├── __init__.py
│   ├── data_utils.py             # Preprocessing utilities
│   └── datasets.py               # PyTorch Dataset classes
├── checkpoints/
│   ├── tft_best.pth
│   ├── health_best.pth
│   ├── maintenance_best.pth
│   └── scalers.pkl
└── logs/
```

## 🔧 Key Features

### Data Processing
- ✅ **RUL Computation**: Time-to-failure for each machine
- ✅ **Time-based Split**: No random shuffling (prevents leakage)
- ✅ **Proper Scaling**: Fit scalers only on training data
- ✅ **Leakage Prevention**: machine_failure dropped after RUL computation

### Training
- ✅ **Device Auto-detection**: CUDA → MPS → CPU
- ✅ **Reproducible Seeds**: All random operations seeded
- ✅ **Progress Bars**: tqdm for epoch and batch progress
- ✅ **Early Stopping**: Based on validation loss
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau
- ✅ **Gradient Clipping**: Prevents exploding gradients
- ✅ **Model Checkpointing**: Best models automatically saved

### Evaluation
- ✅ **Comprehensive Metrics**: MAE, RMSE, R², Accuracy, F1
- ✅ **NO MAPE**: Avoided due to instability with near-zero RUL values
- ✅ **Example Inference**: Demonstrates end-to-end prediction

## 🚀 Usage

### Requirements
```bash
pip install torch pandas numpy scikit-learn tqdm
```

### Running the Project
```bash
cd RUL_Project
python main.py
```

The script will:
1. Load and preprocess data
2. Compute RUL for all machines
3. Create time-based train/val/test splits
4. Train TFT model for RUL prediction
5. Train Health Status classifier
6. Train Maintenance Type classifier
7. Evaluate all models on test set
8. Show example inference

### Training Time
- **TFT**: ~30-50 epochs (5-15 minutes on GPU)
- **Health Status**: ~20-40 epochs (2-5 minutes on GPU)
- **Maintenance Type**: ~20-40 epochs (2-5 minutes on GPU)

## 📊 Expected Performance

### RUL Prediction (TFT)
- MAE: ~15-25 hours
- RMSE: ~25-40 hours
- R²: ~0.85-0.95

### Health Status Classification
- Accuracy: ~85-95%
- F1-Score: ~0.85-0.92

### Maintenance Type Prediction
- Accuracy: ~75-90%
- F1-Score: ~0.75-0.88

## 🎓 Model Justifications

### Why TFT for RUL?
- Captures complex temporal patterns
- Handles both static (machine_id) and time-varying features
- Attention mechanism for interpretability
- State-of-the-art for time series forecasting

### Why Huber Loss?
- RUL predictions have extreme values (0-1000+ hours)
- MSE over-penalizes large outliers
- Huber: quadratic for small errors, linear for large
- Delta=10 hours chosen as reasonable prediction error threshold

### Why MLP for Health/Maintenance?
- Current state classification doesn't need temporal modeling
- Efficient and interpretable
- Fast training and inference
- Sufficient capacity for the task

## 🔍 Key Implementation Details

### RUL Computation
```python
# For each machine, compute time steps until next failure
# If no failure: RUL = remaining time in dataset
# Ensures no future information leakage
```

### Time-based Split
```python
# Sort by timestamp, split chronologically
# Train: 70% earliest data
# Val: 15% middle data
# Test: 15% most recent data
```

### Sequence Creation
```python
# Create 24-hour sliding windows
# Predict RUL at end of each window
# Static features: machine_id embedding
# Time-varying: sensor readings + maintenance history
```

## 🛠️ Configuration

All hyperparameters in `config.py`:
- Model architectures
- Training parameters
- Data split ratios
- Device selection
- Paths and directories

## 📈 Monitoring

Training logs show:
- Epoch progress with tqdm
- Training and validation loss
- All evaluation metrics
- Learning rate changes
- Early stopping status
- Best model checkpoints

## 🧪 Inference Example

```python
# Load models
tft_model = load_model('checkpoints/tft_best.pth')
health_model = load_model('checkpoints/health_best.pth')
maint_model = load_model('checkpoints/maintenance_best.pth')

# Make predictions
rul = tft_model(static, time_varying)
health = health_model(features)
maintenance = maint_model(features)
```

## ⚠️ Important Notes

1. **NO MAPE**: Not used due to division by zero issues when RUL≈0
2. **Time-based Split**: NEVER use random split for time series
3. **Scaling**: ALWAYS fit scalers only on training data
4. **Leakage**: machine_failure column dropped after RUL computation
5. **Device**: Auto-detects CUDA/MPS/CPU at runtime

## 📝 Citation

```bibtex
@software{predictive_maintenance_2026,
  title={Predictive Maintenance Automation System},
  author={Claude},
  year={2026},
  description={End-to-end PyTorch implementation with TFT for RUL prediction}
}
```

## 📧 Support

For issues or questions:
- Check config.py for hyperparameter tuning
- Review training logs in console output
- Inspect saved checkpoints in checkpoints/

---

Built with ❤️ using PyTorch and advanced deep learning techniques.
