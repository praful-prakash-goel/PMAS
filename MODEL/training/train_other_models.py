"""
Training scripts for Health Status and Maintenance Type models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import compute_metrics


def train_classifier_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    """Train classifier for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        features = batch['features'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate_classifier(model, dataloader, criterion, device):
    """Validate classifier"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for batch in pbar:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    
    # Compute metrics
    predictions_array = np.concatenate(all_predictions)
    targets_array = np.concatenate(all_targets)
    
    metrics = compute_metrics(targets_array, predictions_array, task='classification')
    
    return avg_loss, metrics, predictions_array, targets_array


def train_health_status_model(model, train_loader, val_loader, config, device, checkpoint_dir, model_name='health'):
    """
    Complete training loop for Health Status Classification
    
    Args:
        model: Health status model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Training configuration dict
        device: torch device
        checkpoint_dir: Directory to save checkpoints
        model_name: Name for saving checkpoint
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*60)
    print("TRAINING HEALTH STATUS CLASSIFIER")
    print("="*60)
    print("\nArchitecture: Deep MLP")
    print("Rationale: Current health status is determined by instantaneous")
    print("           sensor readings without temporal dependencies.")
    print("           MLP is efficient and interpretable for this task.")
    
    # Loss function: Cross Entropy
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_macro': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Gradient Clipping: {config['grad_clip_value']}")
    print(f"  Early Stopping Patience: {config['patience']}")
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_classifier_epoch(
            model, train_loader, criterion, optimizer, device, config['grad_clip_value']
        )
        
        # Validate
        val_loss, val_metrics, _, _ = validate_classifier(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['Accuracy'])
        history['val_f1_macro'].append(val_metrics['F1_Macro'])
        
        # Print epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_metrics['Accuracy']:.4f}")
        print(f"  Val F1:     {val_metrics['F1_Macro']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'history': history
            }, checkpoint_path)
            print(f"  ✓ Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['patience']})")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"{'='*60}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    return model, history


def train_maintenance_model(model, train_loader, val_loader, config, device, checkpoint_dir):
    """
    Complete training loop for Maintenance Type Classifier
    
    Similar structure to health status training
    """
    print("\n" + "="*60)
    print("TRAINING MAINTENANCE TYPE CLASSIFIER")
    print("="*60)
    print("\nArchitecture: Deep MLP")
    print("Rationale: Maintenance decision based on current state + RUL.")
    print("           Sequential processing not required for this task.")
    
    # Loss function: Cross Entropy
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_macro': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_classifier_epoch(
            model, train_loader, criterion, optimizer, device, config['grad_clip_value']
        )
        
        # Validate
        val_loss, val_metrics, _, _ = validate_classifier(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['Accuracy'])
        history['val_f1_macro'].append(val_metrics['F1_Macro'])
        
        # Print epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_metrics['Accuracy']:.4f}")
        print(f"  Val F1:     {val_metrics['F1_Macro']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, 'maintenance_best.pth')
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
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    
    return model, history


def evaluate_classifier_model(model, test_loader, device, model_name='Model'):
    """
    Evaluate classifier on test set
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()} ON TEST SET")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics, predictions, targets = validate_classifier(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Accuracy:  {test_metrics['Accuracy']:.4f}")
    print(f"  F1 (Macro): {test_metrics['F1_Macro']:.4f}")
    print(f"  F1 (Weighted): {test_metrics['F1_Weighted']:.4f}")
    
    # Class distribution in predictions
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\nPrediction Distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(predictions)*100:.1f}%)")
    
    return test_metrics, predictions, targets
