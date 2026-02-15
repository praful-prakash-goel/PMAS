"""PyTorch Dataset classes for RUL, Health Status, and Maintenance Prediction"""

import torch
from torch.utils.data import Dataset
import numpy as np


class RULSequenceDataset(Dataset):
    """Dataset for Temporal Fusion Transformer (TFT)"""
    
    def __init__(self, df, scalers, sequence_length, continuous_features, 
                 categorical_features, target_col='RUL'):
        self.df = df.copy().sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
        self.scalers = scalers
        self.sequence_length = sequence_length
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        
        self.sequences = []
        self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences per machine"""
        for machine_id in self.df['machine_id'].unique():
            machine_data = self.df[self.df['machine_id'] == machine_id].copy()
            
            machine_encoded = self.scalers['machine_id'].transform([machine_id])[0]
            
            continuous_scaled = self.scalers['continuous'].transform(
                machine_data[self.continuous_features]
            )
            
            categorical_encoded = []
            for col in self.categorical_features:
                encoded = self.scalers[col].transform(machine_data[col])
                categorical_encoded.append(encoded.reshape(-1, 1))
            
            if len(categorical_encoded) > 0:
                categorical_encoded = np.hstack(categorical_encoded)
            else:
                categorical_encoded = np.zeros((len(machine_data), 0))
            
            time_varying = np.hstack([continuous_scaled, categorical_encoded])
            targets = machine_data[self.target_col].values
            
            for i in range(len(machine_data) - self.sequence_length + 1):
                seq_features = time_varying[i:i + self.sequence_length]
                target = targets[i + self.sequence_length - 1]
                
                self.sequences.append({
                    'static': machine_encoded,
                    'time_varying': seq_features,
                    'target': target
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sample = self.sequences[idx]
        
        return {
            'static': torch.tensor(sample['static'], dtype=torch.long),
            'time_varying': torch.tensor(sample['time_varying'], dtype=torch.float32),
            'target': torch.tensor(sample['target'], dtype=torch.float32)
        }


class HealthStatusDataset(Dataset):
    """Dataset for Health Status prediction"""
    
    def __init__(self, df, scalers, continuous_features, categorical_features, 
                 target_col='health_status'):
        self.df = df.copy()
        self.scalers = scalers
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        
        self.features, self.targets = self._prepare_data()
    
    def _prepare_data(self):
        """Prepare scaled features and targets"""
        continuous_scaled = self.scalers['continuous'].transform(
            self.df[self.continuous_features]
        )
        
        categorical_encoded = []
        for col in self.categorical_features:
            encoded = self.scalers[col].transform(self.df[col])
            categorical_encoded.append(encoded.reshape(-1, 1))
        
        if len(categorical_encoded) > 0:
            categorical_encoded = np.hstack(categorical_encoded)
        else:
            categorical_encoded = np.zeros((len(self.df), 0))
        
        machine_encoded = self.scalers['machine_id'].transform(self.df['machine_id'])
        machine_encoded = machine_encoded.reshape(-1, 1)
        
        features = np.hstack([continuous_scaled, categorical_encoded, machine_encoded])
        targets = self.df[self.target_col].values
        
        return features, targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], 
                                 dtype=torch.float32 if 'score' in self.target_col else torch.long)
        }


class MaintenanceDataset(Dataset):
    """Dataset for Maintenance Type prediction"""
    
    def __init__(self, df, scalers, continuous_features, categorical_features,
                 target_col='last_maintenance_Type'):
        self.df = df.copy()
        self.scalers = scalers
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        
        self.features, self.targets = self._prepare_data()
    
    def _prepare_data(self):
        """Prepare scaled features and targets"""
        continuous_scaled = self.scalers['continuous'].transform(
            self.df[self.continuous_features]
        )
        
        categorical_cols = [c for c in self.categorical_features if c != self.target_col]
        categorical_encoded = []
        for col in categorical_cols:
            encoded = self.scalers[col].transform(self.df[col])
            categorical_encoded.append(encoded.reshape(-1, 1))
        
        if len(categorical_encoded) > 0:
            categorical_encoded = np.hstack(categorical_encoded)
        else:
            categorical_encoded = np.zeros((len(self.df), 0))
        
        machine_encoded = self.scalers['machine_id'].transform(self.df['machine_id'])
        machine_encoded = machine_encoded.reshape(-1, 1)
        
        rul_values = self.df['RUL'].values.reshape(-1, 1)
        rul_normalized = rul_values / self.df['RUL'].max()
        
        features = np.hstack([continuous_scaled, categorical_encoded, 
                             machine_encoded, rul_normalized])
        
        targets = self.scalers[self.target_col].transform(self.df[self.target_col])
        
        return features, targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }