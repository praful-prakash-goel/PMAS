"""
Temporal Fusion Transformer (TFT) for RUL Prediction
Incorporates static and time-varying features with attention mechanism
"""

import torch
import torch.nn as nn


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network - learns to select relevant features"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.grn = GatedResidualNetwork(input_size, hidden_size, output_size, dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        weights = self.grn(x)
        weights = self.softmax(weights)
        output = x * weights
        return output, weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) - core building block"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate_fc = nn.Linear(hidden_size, output_size)
        self.gate_activation = nn.Sigmoid()
        
        if input_size != output_size:
            self.skip_fc = nn.Linear(input_size, output_size)
        else:
            self.skip_fc = None
        
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context=None):
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x
        
        hidden = self.elu(self.fc1(x))
        
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_fc(context)
        
        out = self.fc2(hidden)
        gate = self.gate_activation(self.gate_fc(hidden))
        out = out * gate
        
        out = self.layer_norm(skip + self.dropout(out))
        return out


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for RUL Prediction
    
    Architecture:
    1. Static covariate encoder (machine_id)
    2. Time-varying feature processing
    3. LSTM for temporal dependencies
    4. Multi-head attention
    5. GRN for final prediction
    """
    
    def __init__(self, num_static_vars, static_embedding_dim, time_varying_dim,
                 hidden_size, lstm_layers, dropout, attention_heads, output_size=1):
        super().__init__()
        
        self.num_static_vars = num_static_vars
        self.static_embedding_dim = static_embedding_dim
        self.time_varying_dim = time_varying_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        
        # Static covariate embedding (for machine_id)
        self.static_embedding = nn.Embedding(num_static_vars, static_embedding_dim)
        
        # Static encoder
        self.static_encoder = GatedResidualNetwork(
            static_embedding_dim, hidden_size, hidden_size, dropout
        )
        
        # Variable selection for time-varying features
        self.time_varying_vsn = VariableSelectionNetwork(
            time_varying_dim, hidden_size, time_varying_dim, dropout
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=time_varying_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Post-attention processing
        self.post_attention_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        # Gate to combine static and temporal information
        self.static_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout, context_size=hidden_size
        )
        
        # Output layers
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        self.output_fc = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, static_features, time_varying_features):
        """
        Forward pass
        
        Args:
            static_features: [batch_size] - machine IDs
            time_varying_features: [batch_size, seq_len, time_varying_dim]
        
        Returns:
            predictions: [batch_size]
        """
        batch_size, seq_len, _ = time_varying_features.shape
        
        # 1. Static covariate encoding
        static_embed = self.static_embedding(static_features)
        static_encoded = self.static_encoder(static_embed)
        
        # Initialize LSTM hidden state with static context
        h0 = static_encoded.unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # 2. Variable selection for time-varying features
        time_varying_selected, _ = self.time_varying_vsn(time_varying_features)
        
        # 3. LSTM processing
        lstm_out, _ = self.lstm(time_varying_selected, (h0, c0))
        
        # 4. Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        attn_out = self.post_attention_grn(attn_out + lstm_out)
        
        # 5. Take last timestep
        final_temporal = attn_out[:, -1, :]
        
        # 6. Static enrichment
        enriched = self.static_enrichment(final_temporal, context=static_encoded)
        
        # 7. Output prediction
        output = self.output_grn(enriched)
        output = self.dropout(output)
        prediction = self.output_fc(output)
        
        return prediction.squeeze(-1)
