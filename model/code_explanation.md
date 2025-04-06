# Phase 2: Model Implementation Explanation

## Overview
This document explains the implementation of the Temporal Convolutional Network (TCN) with Attention Mechanism for predicting Reactor Scram events in nuclear power plant accidents.

## Model Architecture

### 1. Temporal Block (`TemporalBlock` class)
The basic building block of the TCN:
```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
```
- **Purpose**: Processes temporal data with dilated convolutions
- **Components**:
  - Two 1D convolutional layers with batch normalization and ReLU
  - Dropout for regularization
  - Residual connection (downsample) to maintain information flow
- **Key Features**:
  - Dilation increases receptive field exponentially
  - Residual connections help with gradient flow
  - Batch normalization stabilizes training

### 2. Attention Layer (`AttentionLayer` class)
Implements the attention mechanism:
```python
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
```
- **Purpose**: Focuses on important temporal features
- **Components**:
  - Two linear layers with tanh activation
  - Softmax for attention weights
- **Process**:
  1. Computes attention scores for each timestep
  2. Applies softmax to get attention weights
  3. Weighted sum of features based on attention

### 3. Main Model (`TCNWithAttention` class)
Combines TCN with attention:
```python
class TCNWithAttention(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
```
- **Architecture**:
  1. Multiple temporal blocks with increasing dilation
  2. Attention layer on processed features
  3. Output layer for final prediction
- **Data Flow**:
  1. Input: (batch_size, seq_len, input_size)
  2. TCN processing: (batch_size, num_channels[-1], seq_len)
  3. Attention: (batch_size, num_channels[-1])
  4. Output: (batch_size, 1)

## Training Process

### 1. Data Loading
```python
def load_data(data_dir="processed_data"):
```
- Loads preprocessed data from .npy files
- Converts numpy arrays to PyTorch tensors
- Splits into train/validation/test sets

### 2. Model Training (`ModelTrainer` class)
Handles the training process:
```python
class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.001):
```
- **Components**:
  - Binary Cross Entropy loss
  - Adam optimizer
  - Training and validation loops
- **Features**:
  - Early stopping based on validation AUC
  - Model checkpointing
  - Comprehensive metrics tracking

### 3. Training Loop
```python
def main():
    # Training setup
    num_epochs = 50
    patience = 10
```
- **Process**:
  1. Train for one epoch
  2. Validate model
  3. Save best model based on validation AUC
  4. Early stopping if no improvement
- **Metrics Tracked**:
  - Loss (BCE)
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC

## Model Parameters

### Default Configuration
```python
model = create_model(
    input_size=input_size,
    num_channels=[32, 64, 128],
    kernel_size=3,
    dropout=0.2
)
```
- **Architecture**:
  - 3 temporal blocks with 32, 64, and 128 channels
  - Kernel size of 3 for convolutions
  - Dropout rate of 0.2
- **Training**:
  - Batch size: 32
  - Learning rate: 0.001
  - Early stopping patience: 10 epochs

## Key Features

1. **Temporal Processing**:
   - Handles sequential data effectively
   - Captures long-range dependencies through dilation
   - Maintains temporal order information

2. **Attention Mechanism**:
   - Focuses on relevant time steps
   - Improves prediction accuracy
   - Provides interpretability through attention weights

3. **Training Stability**:
   - Batch normalization
   - Dropout regularization
   - Residual connections
   - Early stopping

4. **Performance Monitoring**:
   - Comprehensive metrics
   - Validation-based model selection
   - Test set evaluation

## Usage

1. **Model Creation**:
```python
model = create_model(input_size=97)  # 97 features
```

2. **Training**:
```python
python model/train.py
```

3. **Model Loading**:
```python
model.load_state_dict(torch.load('model/best_model.pth'))
``` 