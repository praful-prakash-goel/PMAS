"""
Predictive Maintenance System using Temporal Fusion Transformer
Optimized for Mac MPS (Metal Performance Shaders)

This system predicts:
1. Remaining Useful Life (RUL)
2. Current Health Status
3. Next Suggested Maintenance Date
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Check device availability
def get_device():
    """Get the best available device (MPS for Mac, CUDA for GPU, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (consider using a Mac with M1/M2/M3 chip for MPS)")
    return device

DEVICE = get_device()


class DataPreprocessor:
    """Handles all data preprocessing for predictive maintenance"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = None
        
    def preprocess(self, df, is_training=True):
        """
        Preprocess the time series data
        
        Args:
            df: Raw dataframe
            is_training: Whether this is training data (fit scalers) or test data (transform only)
        """
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by machine_id and timestamp
        df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Handle machine_id encoding
        if is_training:
            self.label_encoders['machine_id'] = LabelEncoder()
            df['machine_id_encoded'] = self.label_encoders['machine_id'].fit_transform(df['machine_id'])
        else:
            # Handle unseen machine IDs gracefully
            try:
                df['machine_id_encoded'] = self.label_encoders['machine_id'].transform(df['machine_id'])
            except ValueError as e:
                # If there are unseen machine IDs, map them to the closest known one
                known_machines = self.label_encoders['machine_id'].classes_
                df['machine_id_encoded'] = df['machine_id'].apply(
                    lambda x: self.label_encoders['machine_id'].transform([x])[0] 
                    if x in known_machines else 0
                )
                print(f"⚠️  Warning: Some machine IDs were not seen during training. Mapped to default.")
        
        # Handle last_maintenance_Type (categorical)
        # Replace 'None' string with actual None or handle it
        df['last_maintenance_Type'] = df['last_maintenance_Type'].replace('None', 'NoMaintenance')
        
        if is_training:
            self.label_encoders['maintenance_type'] = LabelEncoder()
            df['maintenance_type_encoded'] = self.label_encoders['maintenance_type'].fit_transform(
                df['last_maintenance_Type'].fillna('NoMaintenance')
            )
        else:
            # Handle unseen maintenance types gracefully
            known_types = self.label_encoders['maintenance_type'].classes_
            df['maintenance_type_encoded'] = df['last_maintenance_Type'].fillna('NoMaintenance').apply(
                lambda x: self.label_encoders['maintenance_type'].transform([x])[0] 
                if x in known_types else 0
            )
        
        # Feature engineering
        df['temp_diff'] = df['process_temperature'] - df['air_temperature']
        df['power_per_rpm'] = df['power_consumption'] / (df['rpm'] + 1e-6)
        df['torque_per_current'] = df['torque'] / (df['current'] + 1e-6)
        df['vibration_rpm_ratio'] = df['vibration'] * df['rpm']
        
        # Rolling statistics per machine (using groupby)
        for window in [3, 6, 12]:
            for col in ['vibration', 'temperature_diff', 'power_consumption', 'torque']:
                if col == 'temperature_diff':
                    source_col = 'temp_diff'
                else:
                    source_col = col
                
                df[f'{col}_rolling_mean_{window}h'] = df.groupby('machine_id')[source_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window}h'] = df.groupby('machine_id')[source_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).fillna(0)
        
        # Calculate RUL (Remaining Useful Life)
        # Assuming machines need maintenance every 168 hours (1 week) or when degradation is detected
        # This is a simplified approach - you may need domain knowledge for actual RUL calculation
        df = self._calculate_rul(df)
        
        # Define feature columns
        if self.feature_columns is None:
            self.feature_columns = [
                'process_temperature', 'air_temperature', 'vibration', 'torque', 
                'rpm', 'current', 'operating_hours', 'time_since_last_maintenance',
                'idle_duration', 'power_consumption', 'temp_diff', 'power_per_rpm',
                'torque_per_current', 'vibration_rpm_ratio', 'hour', 'day_of_week',
                'day_of_month', 'month', 'is_weekend', 'maintenance_type_encoded'
            ]
            
            # Add rolling features
            for window in [3, 6, 12]:
                for col in ['vibration', 'temperature_diff', 'power_consumption', 'torque']:
                    self.feature_columns.extend([
                        f'{col}_rolling_mean_{window}h',
                        f'{col}_rolling_std_{window}h'
                    ])
        
        # Scale numerical features
        if is_training:
            self.scalers['features'] = StandardScaler()
            df[self.feature_columns] = self.scalers['features'].fit_transform(df[self.feature_columns])
        else:
            df[self.feature_columns] = self.scalers['features'].transform(df[self.feature_columns])
        
        return df
    
    def _calculate_rul(self, df):
        """
        Calculate Remaining Useful Life
        
        This is a simplified approach. In practice, you would:
        1. Have historical failure data
        2. Use degradation models
        3. Consider multiple failure modes
        """
        # For demonstration, we'll estimate RUL based on time since last maintenance
        # and equipment degradation indicators
        
        MAINTENANCE_CYCLE = 168  # hours (1 week)
        
        # Basic RUL: time until next scheduled maintenance
        df['rul_basic'] = MAINTENANCE_CYCLE - df['time_since_last_maintenance']
        
        # Adjust RUL based on degradation indicators
        # Higher vibration, temperature, and lower efficiency = lower RUL
        df['degradation_score'] = (
            (df['vibration'] - df['vibration'].min()) / (df['vibration'].max() - df['vibration'].min() + 1e-6) * 0.3 +
            (df['temp_diff'] - df['temp_diff'].min()) / (df['temp_diff'].max() - df['temp_diff'].min() + 1e-6) * 0.3 +
            (df['power_consumption'] - df['power_consumption'].min()) / (df['power_consumption'].max() - df['power_consumption'].min() + 1e-6) * 0.4
        )
        
        # Adjust RUL based on degradation
        df['rul'] = df['rul_basic'] * (1 - df['degradation_score'] * 0.5)
        df['rul'] = df['rul'].clip(lower=0)
        
        # Health status: Good (>120h), Warning (60-120h), Critical (<60h)
        df['health_status'] = pd.cut(df['rul'], 
                                      bins=[-np.inf, 60, 120, np.inf],
                                      labels=['Critical', 'Warning', 'Good'])
        
        return df


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences"""
    
    def __init__(self, data, sequence_length=24, prediction_horizon=12):
        """
        Args:
            data: Preprocessed dataframe
            sequence_length: Number of past timesteps to use
            prediction_horizon: Number of future timesteps to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.machine_ids = data['machine_id'].unique()
        
        # Store data per machine
        self.sequences = []
        self.targets = []
        self.static_features = []
        
        feature_columns = [col for col in data.columns if col not in 
                          ['timestamp', 'machine_id', 'rul', 'health_status', 'rul_basic', 
                           'degradation_score', 'last_maintenance_Type']]
        
        for machine_id in self.machine_ids:
            machine_data = data[data['machine_id'] == machine_id].sort_values('timestamp')
            
            # Extract features and targets
            features = machine_data[feature_columns].values
            rul_values = machine_data['rul'].values
            
            # Create sequences
            for i in range(len(features) - sequence_length - prediction_horizon + 1):
                # Input sequence
                seq = features[i:i + sequence_length]
                
                # Target: RUL at prediction_horizon steps ahead
                target_rul = rul_values[i + sequence_length + prediction_horizon - 1]
                
                # Static features (machine_id encoding)
                static = machine_data.iloc[i]['machine_id_encoded']
                
                self.sequences.append(seq)
                self.targets.append(target_rul)
                self.static_features.append(static)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.static_features = np.array(self.static_features, dtype=np.int64)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'static': torch.LongTensor([self.static_features[idx]]),
            'target': torch.FloatTensor([self.targets[idx]])
        }


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for RUL prediction
    Optimized for Mac MPS
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, 
                 dropout=0.1, num_machines=5):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Static feature embedding (machine_id)
        self.machine_embedding = nn.Embedding(num_machines, hidden_dim // 4)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Variable selection network (feature importance)
        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Static context integration
        self.static_context = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Quantile prediction heads for uncertainty estimation
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(3)  # 10th, 50th, 90th percentiles
        ])
        
    def forward(self, x, static_features):
        """
        Args:
            x: (batch_size, sequence_length, input_dim)
            static_features: (batch_size, 1) - machine_id
        """
        batch_size, seq_len, _ = x.shape
        
        # Variable selection (feature importance)
        # Average across time for selection weights
        x_avg = x.mean(dim=1)  # (batch_size, input_dim)
        projected = self.input_projection(x_avg)  # (batch_size, hidden_dim)
        selection_weights = self.variable_selection(projected)  # (batch_size, input_dim)
        
        # Apply feature selection
        x_selected = x * selection_weights.unsqueeze(1)  # (batch_size, seq_len, input_dim)
        
        # Project input
        x_proj = self.input_projection(x_selected)  # (batch_size, seq_len, hidden_dim)
        
        # Embed static features
        static_emb = self.machine_embedding(static_features.squeeze(1))  # (batch_size, hidden_dim//4)
        static_context = self.static_context(static_emb)  # (batch_size, hidden_dim)
        
        # Add static context to each timestep
        x_proj = x_proj + static_context.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_proj)  # (batch_size, seq_len, hidden_dim)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm1(lstm_out + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(lstm_out)
        lstm_out = self.norm2(lstm_out + ffn_out)
        
        # Use last timestep for prediction
        final_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Main output
        output = self.output_layer(final_hidden)  # (batch_size, 1)
        
        # Quantile predictions for uncertainty
        quantiles = [q_layer(final_hidden) for q_layer in self.quantile_outputs]
        
        return output, quantiles, selection_weights


class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty estimation"""
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of 3 tensors (batch_size, 1) for each quantile
            targets: (batch_size, 1)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.cat(losses))


def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """Train the TFT model"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    mse_loss = nn.MSELoss()
    quantile_loss = QuantileLoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            sequences = batch['sequence'].to(DEVICE)
            static = batch['static'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            optimizer.zero_grad()
            
            predictions, quantiles, _ = model(sequences, static)
            
            # Combined loss: MSE for main prediction + quantile loss
            loss_mse = mse_loss(predictions, targets)
            loss_quantile = quantile_loss(quantiles, targets)
            loss = loss_mse + 0.1 * loss_quantile
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(DEVICE)
                static = batch['static'].to(DEVICE)
                targets = batch['target'].to(DEVICE)
                
                predictions, quantiles, _ = model(sequences, static)
                loss = mse_loss(predictions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_tft_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*60)
    print(f"Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print("="*60 + "\n")
    
    return train_losses, val_losses


def predict_with_uncertainty(model, data_loader):
    """Make predictions with uncertainty bounds"""
    model.eval()
    predictions = []
    uncertainties = []
    targets_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(DEVICE)
            static = batch['static'].to(DEVICE)
            targets = batch['target']
            
            pred, quantiles, _ = model(sequences, static)
            
            # Calculate uncertainty (90th - 10th percentile)
            uncertainty = (quantiles[2] - quantiles[0]).cpu().numpy()
            
            predictions.extend(pred.cpu().numpy())
            uncertainties.extend(uncertainty)
            targets_list.extend(targets.numpy())
    
    return np.array(predictions), np.array(uncertainties), np.array(targets_list)


def calculate_health_status(rul):
    """Convert RUL to health status"""
    if rul > 120:
        return "Good"
    elif rul > 60:
        return "Warning"
    else:
        return "Critical"


def calculate_next_maintenance(current_timestamp, rul):
    """Calculate next maintenance date"""
    next_maintenance = current_timestamp + pd.Timedelta(hours=int(rul))
    return next_maintenance


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PREDICTIVE MAINTENANCE SYSTEM - TFT MODEL")
    print("="*60 + "\n")
    
    # Note: This is a demonstration structure
    # In actual use, load your data here
    print("📝 To use this system:")
    print("1. Load your data: df = pd.read_csv('your_data.csv')")
    print("2. Split into train/val/test sets")
    print("3. Create preprocessor and preprocess data")
    print("4. Create datasets and dataloaders")
    print("5. Initialize and train the model")
    print("6. Make predictions with uncertainty")
    print("\nSee example usage in the training script.")
