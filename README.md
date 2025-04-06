# Nuclear Power Plant Accident Detection (NPPAD)

A deep learning project for predicting Reactor Scram events in nuclear power plant accidents using Temporal Convolutional Networks (TCN) with Attention Mechanism.

## Project Overview

This project aims to develop an early warning system for nuclear power plant accidents by analyzing operational parameters and predicting Reactor Scram events. The system uses a combination of Temporal Convolutional Networks and Attention Mechanism to process time series data and identify patterns that precede accidents.

## Project Structure

```
Nuclear-Project/
├── data/
│   └── NPPAD/                    # Original dataset
│       ├── LOCA/                 # Loss of Coolant Accident
│       ├── SGBTR/               # Steam Generator Tube Rupture
│       └── ...                   # Other accident types
├── processed_data/              # Preprocessed data
│   ├── X_train.npy             # Training features
│   ├── y_train.npy             # Training labels
│   ├── X_val.npy               # Validation features
│   ├── y_val.npy               # Validation labels
│   ├── X_test.npy              # Test features
│   └── y_test.npy              # Test labels
├── model/
│   ├── tcn_attention.py        # Model architecture
│   ├── train.py                # Training script
│   ├── best_model.pth          # Saved model
│   └── code_explanation.md     # Implementation details
├── data_loader.py              # Data loading and preprocessing
├── test_data_loader.py         # Data loader tests
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Model Architecture

The project implements a Temporal Convolutional Network (TCN) with Attention Mechanism:

1. **Temporal Convolutional Network**
   - Multiple temporal blocks with increasing dilation
   - Residual connections for better gradient flow
   - Batch normalization and dropout for regularization

2. **Attention Mechanism**
   - Focuses on important temporal features
   - Improves prediction accuracy
   - Provides interpretability through attention weights

3. **Output Layer**
   - Binary classification for Reactor Scram prediction
   - Sigmoid activation for probability output

## Features

- **Data Processing**
  - Handles 97 operational parameters
  - Processes time series data
  - Normalizes features
  - Creates sliding windows

- **Model Training**
  - Early stopping based on validation AUC
  - Model checkpointing
  - Comprehensive metrics tracking
  - GPU support

- **Performance Metrics**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the dataset:
```bash
python test_data_loader.py
```

4. Train the model:
```bash
python model/train.py
```

## Dependencies

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- Pandas 1.3.0+
- Matplotlib 3.4.0+
- Seaborn 0.11.0+
- scikit-learn 0.24.0+

## Model Parameters

- **Architecture**:
  - Input size: 97 features
  - Temporal blocks: [32, 64, 128] channels
  - Kernel size: 3
  - Dropout rate: 0.2

- **Training**:
  - Batch size: 32
  - Learning rate: 0.001
  - Early stopping patience: 10 epochs
  - Maximum epochs: 50

## Usage

1. **Data Preparation**:
```python
from data_loader import DataLoader
loader = DataLoader()
loader.prepare_dataset()
```

2. **Model Training**:
```python
python model/train.py
```

3. **Model Loading**:
```python
from model.tcn_attention import create_model
model = create_model(input_size=97)
model.load_state_dict(torch.load('model/best_model.pth'))
```

## Project Phases

1. **Phase 1: Data Preparation** ✅
   - Data loading and preprocessing
   - Feature engineering
   - Dataset splitting

2. **Phase 2: Model Implementation** ✅
   - TCN architecture
   - Attention mechanism
   - Training pipeline

3. **Phase 3: Model Evaluation** (Next)
   - Performance metrics
   - Visualization
   - Error analysis

4. **Phase 4: Deployment** (Future)
   - Model optimization
   - API development
   - System integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 