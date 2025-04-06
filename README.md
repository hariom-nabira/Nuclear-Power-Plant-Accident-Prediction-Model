# Nuclear Power Plant Accident Prediction System

This project implements a machine learning system to predict Reactor Scram events in nuclear power plants, using time-series data from plant operational parameters.

## Project Overview

The system analyzes continuous time-series data from nuclear power plant simulations and predicts potential Reactor Scram events within the next 5 minutes. It uses a Temporal Convolutional Network (TCN) with Attention Mechanism to process the multivariate time-series data.

## Dataset

The project uses the Nuclear Power Plant Accident Data (NPPAD) dataset, which contains:
- 12 different accident types (LOCA, SGBTR, LR, MD, SGATR, SLBIC, LOCAC, RI, FLB, LLB, SLBOC, RW)
- 100 simulations per accident type with varying severity (1% to 100%)
- Each simulation has:
  - Time series data (CSV): ~97 operational parameters recorded at 10-second intervals
  - Transient Report (TXT): Detailed event log with timestamps of key events, including Reactor Scram

## Project Structure

```
├── data_loader.py         # Handles loading and organizing dataset files
├── time_series_processor.py  # Creates sliding windows and engineered features
├── data_validator.py      # Validates data quality and consistency
├── preprocess_data.py     # Main preprocessing pipeline
├── model.py               # TCN with Attention model implementation
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── inference.py           # Real-time inference script
├── code_documentation.md  # Detailed documentation of code structure
├── project_analysis.md    # Project analysis and implementation plan
├── processed_data/        # Processed dataset files (after running preprocessing)
├── models/                # Saved model weights and configurations
├── plots/                 # Training curves and evaluation plots
└── evaluation/            # Detailed evaluation results
```

## Getting Started

### Prerequisites

```
Python 3.8+
numpy
pandas
scikit-learn
scipy
torch
matplotlib
seaborn
tqdm
joblib
h5py
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/npp-accident-prediction.git
cd npp-accident-prediction
```

2. Install required packages
```bash
pip install -r requirements.txt
```

### Data Preprocessing

To preprocess the NPPAD dataset:

```bash
python preprocess_data.py --data_dir NPPAD --output_dir processed_data
```

This will:
- Load all simulation data
- Create sliding windows with labels
- Add engineered features
- Split data into train/validation/test sets
- Save the processed data to the specified output directory

## Project Phases

### Phase 1: Data Preparation

The data preparation phase includes:

1. **Data Loading**: Read CSV files and extract Reactor Scram timestamps from Transient Reports
2. **Feature Engineering**: Normalize data and add derived features like rates of change and rolling statistics
3. **Sliding Window Creation**: Create fixed-size windows (30 time steps = 5 minutes) with corresponding labels
4. **Data Validation**: Check for data quality issues, consistency, and proper label distribution

### Phase 2: Model Development

The model development phase includes:

1. **Model Architecture**: Implementation of a Temporal Convolutional Network (TCN) with attention mechanisms
2. **Training Pipeline**: Setting up the training loop with early stopping and validation
3. **Evaluation Metrics**: Implementation of comprehensive evaluation metrics for binary classification
4. **Real-Time Inference**: Development of a simulation environment for real-time inference

## Model Architecture

The model architecture consists of:

1. **Temporal Convolutional Network (TCN)**:
   - Dilated causal convolutions to capture long-range dependencies
   - Residual connections to facilitate gradient flow
   - Increasing receptive field to cover longer time periods

2. **Attention Mechanisms**:
   - Self-attention for temporal relationships within each parameter
   - Cross-attention for relationships between different parameters

3. **Classification Head**:
   - Feed-forward layers for binary classification (Scram/No Scram)

## Usage

### Training

```bash
python train.py --data_dir processed_data --model_dir models --batch_size 32 --num_epochs 50
```

Training options:
- `--data_dir`: Directory with preprocessed data
- `--model_dir`: Directory to save models
- `--plot_dir`: Directory to save plots
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Initial learning rate
- `--patience`: Patience for early stopping
- `--tcn_channels`: Comma-separated list of TCN channel sizes
- `--kernel_size`: Kernel size for TCN
- `--dropout`: Dropout rate
- `--attention_size`: Dimension of attention mechanism
- `--num_heads`: Number of attention heads
- `--no_cuda`: Disable CUDA
- `--seed`: Random seed

### Evaluation

```bash
python evaluate.py --model_path models/best_model.pt --data_dir processed_data --output_dir evaluation
```

Evaluation options:
- `--model_path`: Path to saved model weights
- `--data_dir`: Directory with preprocessed data
- `--output_dir`: Directory to save evaluation results

### Real-Time Inference

```bash
python inference.py --model_path models/best_model.pt --data_path test_dataset.pkl --output_dir inference_results
```

Inference options:
- `--model_path`: Path to saved model weights
- `--data_path`: Path to simulation data (pickle or CSV)
- `--output_dir`: Directory to save inference results
- `--window_size`: Size of the sliding window for prediction
- `--stride`: Stride for sliding window
- `--threshold`: Threshold for binary prediction
- `--time_delay`: Delay between processing steps (seconds)
- `--no_plot`: Disable real-time plotting

## License

[MIT License](LICENSE)

## Acknowledgements

- The NPPAD dataset used in this project
- Contributors to the open-source libraries used in this project 