import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, List, Dict

class Chomp1d(nn.Module):
    """
    Module that performs causal padding removal.
    Used to ensure causal convolutions in TCN.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    """
    A single temporal block in the TCN, consisting of:
    - Dilated causal convolution
    - Normalization
    - Activation
    - Dropout
    - Residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        
        # Add residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with multiple stacked temporal blocks,
    each with increasing dilation factor.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal convolution
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, 
                             stride=1, dilation=dilation_size, 
                             padding=padding, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Input shape: (batch, features, sequence_length)
        Output shape: (batch, channels[-1], sequence_length)
        """
        return self.network(x)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the self-attention mechanism.
    """
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer to not be considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-heads
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out_proj(context)
        
        return output, attn_weights

class TCNWithAttention(nn.Module):
    """
    Complete model architecture that combines:
    1. Temporal Convolutional Network (TCN)
    2. Self-Attention mechanism
    3. Cross-Attention mechanism for feature relationships
    4. Output layer for binary classification
    """
    def __init__(self, input_size, num_channels, seq_len, 
                 kernel_size=3, dropout=0.2, attention_size=128, 
                 num_heads=4, positional_encoding=True):
        """
        Args:
            input_size: Number of input features
            num_channels: List of hidden channel sizes for TCN
            seq_len: Length of input sequence (time steps)
            kernel_size: Kernel size for TCN
            dropout: Dropout rate
            attention_size: Dimension of attention mechanism
            num_heads: Number of attention heads
            positional_encoding: Whether to use positional encoding
        """
        super(TCNWithAttention, self).__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.attention_size = attention_size
        self.num_heads = num_heads
        
        # TCN to process temporal features
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        # Projection layer after TCN
        self.tcn_projection = nn.Linear(num_channels[-1], attention_size)
        
        # Positional encoding for self-attention
        self.use_positional_encoding = positional_encoding
        if positional_encoding:
            self.pos_encoder = PositionalEncoding(attention_size, max_len=seq_len)
        
        # Self-attention for temporal relationships
        self.self_attention = MultiHeadAttention(attention_size, num_heads, dropout)
        
        # Cross-attention for feature relationships
        self.cross_attention = MultiHeadAttention(attention_size, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(attention_size)
        self.norm2 = nn.LayerNorm(attention_size)
        
        # Feed-forward layers
        self.ff_layer = nn.Sequential(
            nn.Linear(attention_size, attention_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_size * 4, attention_size),
            nn.Dropout(dropout)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(attention_size, attention_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, sequence_length, features]
        
        Returns:
            Tensor of shape [batch_size, 1] - probability of accident
        """
        batch_size, seq_len, features = x.size()
        
        # TCN expects [batch, features, sequence]
        x_tcn = x.transpose(1, 2)
        
        # Pass through TCN
        tcn_output = self.tcn(x_tcn)
        
        # Back to [batch, sequence, channels]
        tcn_output = tcn_output.transpose(1, 2)
        
        # Project TCN output to attention dimension
        projected = self.tcn_projection(tcn_output)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            projected = self.pos_encoder(projected)
        
        # Apply self-attention (temporal relationships)
        self_attn_output, _ = self.self_attention(projected, projected, projected)
        
        # Add residual connection and normalize
        attn_output = self.norm1(projected + self_attn_output)
        
        # Apply cross-attention (feature relationships)
        # Here we transpose the sequence and feature dimensions for cross-attention
        cross_query = attn_output.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        cross_attn_output, _ = self.cross_attention(cross_query, attn_output, attn_output)
        
        # Add residual connection and normalize
        cross_output = self.norm2(attn_output + cross_attn_output)
        
        # Apply feed-forward layers
        ff_output = self.ff_layer(cross_output)
        final_output = self.norm2(cross_output + ff_output)
        
        # Global average pooling across sequence
        pooled = final_output.mean(dim=1)
        
        # Final classification
        output = self.classifier(pooled)
        
        return output
    
    def predict(self, x, threshold=0.5):
        """
        Make binary predictions based on probability threshold.
        
        Args:
            x: Input tensor
            threshold: Probability threshold for positive class
            
        Returns:
            Binary predictions (0 or 1)
        """
        with torch.no_grad():
            probs = self.forward(x)
            predictions = (probs >= threshold).float()
            return predictions


class NPPADModel:
    """
    Wrapper class for the TCN with Attention model specifically for NPPAD data.
    Handles preprocessing and provides convenient training/inference methods.
    """
    def __init__(self, config=None):
        """
        Initialize the model with the given configuration.
        
        Args:
            config: Dictionary with model configuration
        """
        self.config = {
            'input_size': 97,              # Default number of features
            'num_channels': [64, 128, 256, 128],  # TCN channel sizes
            'seq_len': 30,                 # Default sequence length (5 min at 10 sec intervals)
            'kernel_size': 3,              # TCN kernel size
            'dropout': 0.2,                # Dropout rate
            'attention_size': 128,         # Attention dimension
            'num_heads': 4,                # Number of attention heads
            'positional_encoding': True,   # Use positional encoding
            'learning_rate': 0.001,        # Learning rate
            'batch_size': 32,              # Batch size
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Device
            'sequence_stride': 1           # Stride for inference
        }
        
        # Update config with provided values
        if config:
            self.config.update(config)
        
        # Create the model
        self.model = TCNWithAttention(
            input_size=self.config['input_size'],
            num_channels=self.config['num_channels'],
            seq_len=self.config['seq_len'],
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout'],
            attention_size=self.config['attention_size'],
            num_heads=self.config['num_heads'],
            positional_encoding=self.config['positional_encoding']
        )
        
        # Move model to device
        self.model.to(self.config['device'])
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Binary cross entropy loss
        self.criterion = nn.BCELoss()
        
    def update_config(self, new_config):
        """
        Update the model configuration.
        Requires model to be re-initialized.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        self.config.update(new_config)
        
        # Re-create the model with new config
        self.model = TCNWithAttention(
            input_size=self.config['input_size'],
            num_channels=self.config['num_channels'],
            seq_len=self.config['seq_len'],
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout'],
            attention_size=self.config['attention_size'],
            num_heads=self.config['num_heads'],
            positional_encoding=self.config['positional_encoding']
        )
        
        # Move model to device
        self.model.to(self.config['device'])
        
        # Re-initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
    
    def load_weights(self, file_path):
        """
        Load model weights from file.
        
        Args:
            file_path: Path to the saved model weights
        """
        checkpoint = torch.load(file_path, map_location=self.config['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def save_weights(self, file_path):
        """
        Save model weights to file.
        
        Args:
            file_path: Path to save the model weights
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, file_path)
        
    def train_step(self, x_batch, y_batch):
        """
        Perform a single training step.
        
        Args:
            x_batch: Input batch of shape [batch_size, seq_len, features]
            y_batch: Target batch of shape [batch_size, 1]
            
        Returns:
            Loss value
        """
        # Convert to tensor and move to device
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
        if not isinstance(y_batch, torch.Tensor):
            y_batch = torch.tensor(y_batch, dtype=torch.float32)
            
        x_batch = x_batch.to(self.config['device'])
        y_batch = y_batch.to(self.config['device'])
        
        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        
        y_pred = self.model(x_batch)
        
        # Ensure shapes match
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)
            
        # Calculate loss
        loss = self.criterion(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, x_val, y_val):
        """
        Validate the model on validation data.
        
        Args:
            x_val: Validation data of shape [n_samples, seq_len, features]
            y_val: Validation targets of shape [n_samples, 1]
            
        Returns:
            Dictionary with validation metrics
        """
        # Convert to tensor and move to device
        if not isinstance(x_val, torch.Tensor):
            x_val = torch.tensor(x_val, dtype=torch.float32)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.tensor(y_val, dtype=torch.float32)
            
        x_val = x_val.to(self.config['device'])
        y_val = y_val.to(self.config['device'])
        
        # Ensure shape
        if y_val.dim() == 1:
            y_val = y_val.unsqueeze(1)
        
        # Evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            y_pred = self.model(x_val)
            
            # Calculate loss
            loss = self.criterion(y_pred, y_val)
            
            # Calculate metrics
            y_pred_binary = (y_pred >= 0.5).float()
            accuracy = (y_pred_binary == y_val).float().mean().item()
            
            # Extract values for metrics
            y_true = y_val.cpu().numpy()
            y_scores = y_pred.cpu().numpy()
            y_pred_binary = y_pred_binary.cpu().numpy()
            
            # Calculate additional metrics
            tp = np.sum((y_true == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true == 0) & (y_pred_binary == 1))
            tn = np.sum((y_true == 0) & (y_pred_binary == 0))
            fn = np.sum((y_true == 1) & (y_pred_binary == 0))
            
            # Handle division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            return metrics
    
    def predict(self, x, threshold=0.5):
        """
        Make predictions on new data.
        
        Args:
            x: Input data of shape [n_samples, seq_len, features]
            threshold: Probability threshold for positive class
            
        Returns:
            Predicted probabilities and binary predictions
        """
        # Convert to tensor and move to device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        x = x.to(self.config['device'])
        
        # Evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            probs = self.model(x)
            
            # Binary predictions
            predictions = (probs >= threshold).float()
            
            # Move back to CPU for numpy conversion
            probs = probs.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            return probs, predictions
    
    def real_time_inference(self, data_stream, window_size=None, stride=None):
        """
        Perform inference on a continuous data stream.
        
        Args:
            data_stream: Array-like of shape [time_steps, features]
            window_size: Size of the sliding window (default: from config)
            stride: Stride for the sliding window (default: from config)
            
        Returns:
            Array of prediction probabilities for each time step
        """
        if window_size is None:
            window_size = self.config['seq_len']
        if stride is None:
            stride = self.config['sequence_stride']
            
        # Ensure data is numpy array
        if not isinstance(data_stream, np.ndarray):
            data_stream = np.array(data_stream)
            
        # Get dimensions
        n_timesteps, n_features = data_stream.shape
        
        # Can't make predictions with fewer timesteps than window size
        if n_timesteps < window_size:
            return np.zeros(n_timesteps)
        
        # Prepare output array (all zeros initially)
        predictions = np.zeros(n_timesteps)
        
        # Sliding window inference
        for t in range(0, n_timesteps - window_size + 1, stride):
            # Extract window
            window = data_stream[t:t+window_size]
            
            # Add batch dimension
            batch = np.expand_dims(window, axis=0)
            
            # Make prediction
            prob, _ = self.predict(batch)
            
            # Store prediction for the future time step
            if t + window_size < n_timesteps:
                predictions[t + window_size] = prob[0, 0]
                
        return predictions

# Example usage
if __name__ == "__main__":
    # Create example data
    batch_size = 8
    seq_len = 30
    n_features = 97
    
    x = torch.randn(batch_size, seq_len, n_features)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Initialize model
    config = {
        'input_size': n_features,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'num_channels': [64, 128, 256, 128]
    }
    
    model = NPPADModel(config)
    
    # Forward pass
    y_pred = model.model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")
    
    # Training step
    loss = model.train_step(x, y)
    print(f"Training loss: {loss:.4f}")
    
    # Validation
    metrics = model.validate(x, y)
    print(f"Validation metrics: {metrics}")
    
    # Prediction
    probs, preds = model.predict(x)
    print(f"Prediction shape: {probs.shape}") 