import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.dropout1,
                               self.conv2, self.bn2, nn.ReLU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * x, dim=1)  # (batch_size, input_dim)
        return attended, attention_weights

class TCNWithAttention(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNWithAttention, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        
        # Calculate padding to maintain sequence length
        self.padding = (kernel_size - 1) * 2
        
        # Temporal blocks
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation,
                                     padding=self.padding, dropout=dropout))
        
        self.tcn = nn.Sequential(*layers)
        
        # Attention layer
        self.attention = AttentionLayer(num_channels[-1])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        
        # Apply TCN
        x = self.tcn(x)  # (batch_size, num_channels[-1], seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_channels[-1])
        
        # Apply attention
        x, attention_weights = self.attention(x)  # (batch_size, num_channels[-1])
        
        # Output layer
        x = self.output_layer(x)
        return x, attention_weights

def create_model(input_size, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2):
    """
    Create a TCN with Attention model.
    
    Args:
        input_size (int): Number of input features
        num_channels (list): List of number of channels in each temporal block
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout rate
        
    Returns:
        TCNWithAttention: The model
    """
    model = TCNWithAttention(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
    return model 