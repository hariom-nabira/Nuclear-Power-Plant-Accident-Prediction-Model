# Nuclear Power Plant Accident Detection System

This project implements a real-time accident detection system for nuclear power plants using a Temporal Convolutional Network (TCN) with Attention Mechanism. The system analyzes continuous time-series data of operational parameters to predict potential Reactor Scram events within the next 5 minutes.

## Dataset

The NPPAD (Nuclear Power Plant Accident Dataset) contains simulations of 12 different accident types:
- LOCA (Loss of Coolant Accident)
- SGBTR (Steam Generator Break Tube Rupture)
- LR (Load Rejection)
- MD (Main Steam Line Break)
- SGATR (Steam Generator Auxiliary Feedwater Trip)
- SLBIC (Steam Line Break Inside Containment)
- LOCAC (Loss of Coolant Accident with Containment)
- RI (Reactor Trip)
- FLB (Feedwater Line Break)
- LLB (Loop Line Break)
- SLBOC (Steam Line Break Outside Containment)
- RW (Reactor Water Level)

Each accident type has 100 simulations with varying severity (1% to 100%). Each simulation contains:
1. Time series data (97 operational parameters, 10-second intervals)
2. Transient Report (detailed event log with Reactor Scram timestamps)

## Project Structure

```
.
├── data_loader.py          # Data loading and preprocessing
├── test_data_loader.py     # Data validation and analysis
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
└── processed_data/        # Directory for processed data
    ├── X_train.npy
    ├── X_val.npy
    ├── X_test.npy
    ├── y_train.npy
    ├── y_val.npy
    └── y_test.npy
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the NPPAD dataset in the project root directory:
```
NPPAD/
├── LOCA/
│   ├── 1.csv
│   ├── 1TransientReport.txt
│   └── ...
├── SGBTR/
│   ├── 1.csv
│   ├── 1TransientReport.txt
│   └── ...
└── ...
```

## Data Preparation

Run the data preparation script to process the dataset:
```bash
python test_data_loader.py
```

This will:
1. Load and preprocess the data
2. Generate sliding windows (5-minute lookback)
3. Create derived features
4. Split data into train/validation/test sets
5. Save processed data in the `processed_data` directory
6. Generate analysis plots:
   - `label_distribution.png`: Distribution of accident/non-accident labels
   - `feature_statistics.png`: Statistics of input features

## Data Processing Details

The data processing pipeline includes:
1. Loading time series data from CSV files
2. Extracting Reactor Scram timestamps from Transient Reports
3. Creating sliding windows (30 time steps = 5 minutes)
4. Normalizing parameters to [0,1] range
5. Generating derived features:
   - Rate of change for critical parameters
   - Rolling statistics (mean, std) for key parameters
   - Parameter correlations within each window

## Next Steps

1. Implement the TCN model architecture
2. Add attention mechanism
3. Develop training pipeline
4. Implement real-time prediction system 