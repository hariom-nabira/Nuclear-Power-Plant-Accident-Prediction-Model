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
├── model.py               # TCN with Attention model implementation (coming soon)
├── train.py               # Model training script (coming soon)
├── evaluate.py            # Model evaluation script (coming soon)
├── inference.py           # Real-time inference script (coming soon)
├── code_documentation.md  # Detailed documentation of code structure
├── project_analysis.md    # Project analysis and implementation plan
└── processed_data/        # Processed dataset files (after running preprocessing)
```

## Getting Started

### Prerequisites

```
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
torch (coming soon)
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

## Data Preparation Phase

The data preparation phase includes:

1. **Data Loading**: Read CSV files and extract Reactor Scram timestamps from Transient Reports
2. **Feature Engineering**: Normalize data and add derived features like rates of change and rolling statistics
3. **Sliding Window Creation**: Create fixed-size windows (30 time steps = 5 minutes) with corresponding labels
4. **Data Validation**: Check for data quality issues, consistency, and proper label distribution

## Model Architecture (Coming Soon)

The model architecture will consist of:
- Temporal Convolutional Network (TCN) for capturing temporal patterns
- Self-attention mechanism for learning dependencies between time steps
- Cross-attention for understanding relationships between different parameters

## Usage

### Training (Coming Soon)

```bash
python train.py --data_dir processed_data --model_dir models
```

### Evaluation (Coming Soon)

```bash
python evaluate.py --model_path models/best_model.pth --data_dir processed_data
```

### Inference (Coming Soon)

```bash
python inference.py --model_path models/best_model.pth --input_data sample_data.csv
```

## License

[MIT License](LICENSE)

## Acknowledgements

- The NPPAD dataset used in this project
- Contributors to the open-source libraries used in this project 