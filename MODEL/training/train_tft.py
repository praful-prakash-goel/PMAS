"""Training script for Temporal Fusion Transformer (RUL Prediction)"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tft_model import TemporalFusionTransformer
from utils.data_utils import compute_metrics


class HuberLoss(nn.Module):
    """Huber Loss for RUL prediction"""
    def __init__(self, delta=10.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions, targets):
        error = predictions - targets
        abs_error = torch.abs(error)
        
        quadratic = torch.min(abs_error, torch.tensor(self.delta).to(error.device))
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return torch.mean(loss)


def train_tft_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        static = batch['static'].to(device)
        time_varying = batch['time_varying'].to(device)
        targets = batch['target'].to(device)
        
        predictions = model(static, time_varying)
        loss = criterion(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate_tft(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for batch in pbar:
            static = batch['static'].to(device)
            time_varying = batch['time_varying'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(static, time_varying)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    
    predictions_array = np.concatenate(all_predictions)
    targets_array = np.concatenate(all_targets)
    
    metrics = compute_metrics(targets_array, predictions_array, task='regression')
    
    return avg_loss, metrics, predictions_array, targets_array


def train_tft_model(model, train_loader, val_loader, config, device, checkpoint_dir):
    """Complete training loop for TFT model"""
    print("\n" + "="*60)
    print("TRAINING TEMPORAL FUSION TRANSFORMER (RUL PREDICTION)")
    print("="*60)
    
    criterion = HuberLoss(delta=config['huber_delta'])
    print(f"\nLoss Function: Huber Loss (delta={config['huber_delta']})")
    print("  Rationale: Less sensitive to RUL outliers")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['scheduler_min_lr'],
        verbose=True
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_tft_epoch(
            model, train_loader, criterion, optimizer, device, config['grad_clip_value']
        )
        
        val_loss, val_metrics, _, _ = validate_tft(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['MAE'])
        history['val_rmse'].append(val_metrics['RMSE'])
        history['val_r2'].append(val_metrics['R2'])
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val MAE:    {val_metrics['MAE']:.4f}")
        print(f"  Val RMSE:   {val_metrics['RMSE']:.4f}")
        print(f"  Val R²:     {val_metrics['R2']:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            checkpoint_path = os.path.join(checkpoint_dir, 'tft_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'history': history
            }, checkpoint_path)
            print(f"  ✓ Best model saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    
    print(f"\nTRAINING COMPLETED")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    return model, history


def evaluate_tft_model(model, test_loader, device):
    """Evaluate TFT model on test set"""
    print("\n" + "="*60)
    print("EVALUATING TFT MODEL ON TEST SET")
    print("="*60)
    
    criterion = HuberLoss(delta=10.0)
    test_loss, test_metrics, predictions, targets = validate_tft(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  MAE:       {test_metrics['MAE']:.4f} hours")
    print(f"  RMSE:      {test_metrics['RMSE']:.4f} hours")
    print(f"  R²:        {test_metrics['R2']:.4f}")
    
    return test_metrics, predictions, targets