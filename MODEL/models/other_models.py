"""
Models for Health Status and Maintenance Type Prediction
"""

import torch
import torch.nn as nn


class HealthStatusClassifier(nn.Module):
    """
    MLP Classifier for Health Status (Healthy, Warning, Critical)
    
    Uses deep MLP architecture with dropout for regularization.
    Choice of MLP: 
    - Current health status is determined by instantaneous sensor readings
    - No temporal dependencies needed (unlike RUL which needs history)
    - MLP is efficient and interpretable for this task
    """
    
    def __init__(self, input_dim, hidden_sizes=[128, 64, 32], dropout=0.3, num_classes=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build network layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.network(x)


class MaintenanceTypeClassifier(nn.Module):
    """
    MLP Classifier for Maintenance Type Prediction
    
    Predicts next recommended maintenance: preventive, predictive, corrective
    
    Architecture choice:
    - MLP is suitable as maintenance decision is based on current state + RUL
    - Doesn't require sequential processing
    - Can incorporate learned representations from TFT if desired
    """
    
    def __init__(self, input_dim, hidden_sizes=[128, 64, 32], dropout=0.3, num_classes=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build network layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.network(x)
