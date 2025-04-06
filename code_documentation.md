# Nuclear Power Plant Accident Prediction - Code Documentation

This document explains the code structure and implementation details for the NPP accident prediction system.

## Table of Contents
1. [Data Preparation](#data-preparation)
   - [Data Loader](#data-loader)
   - [Time Series Processing](#time-series-processing)
   - [Feature Engineering](#feature-engineering)
   - [Data Validation](#data-validation)
   - [Data Preprocessing Pipeline](#data-preprocessing-pipeline)

## Data Preparation

### Data Loader (`data_loader.py`)
The data loader module is responsible for reading and organizing the NPPAD dataset.

#### Key Components:
- **NPPADDataLoader**: Main class for loading data from the NPPAD dataset
  - `__init__(data_dir)`: Initialize with dataset directory path
  - `_get_accident_types()`: Identify all accident types from directory structure
  - `_create_data_mapping()`: Create a mapping of all data files for each accident type and simulation
  - `extract_scram_timestamp(report_path)`: Extract the timestamp of Reactor Scram from a Transient Report
  - `load_simulation_data(accident_type, sim_num)`: Load a specific simulation's data and extract Scram timestamp
  - `get_all_simulation_info()`: Create a DataFrame with information about all simulations
  - `load_all_simulations()`: Load all simulations data and their Scram timestamps
  - `get_dataset_statistics()`: Get basic statistics about the dataset

#### Data Flow:
1. Scan the directory structure to identify accident types (folders like LOCA, SGBTR, etc.)
2. For each accident type, identify CSV files (simulation data) and corresponding Transient Reports
3. Create a mapping structure that links each simulation to its data files
4. Provide methods to load specific simulations or all simulations

#### Usage:
```python
data_loader = NPPADDataLoader("NPPAD")
df, scram_time = data_loader.load_simulation_data("LOCA", 1)
all_data = data_loader.load_all_simulations()
```

### Time Series Processing (`time_series_processor.py`)
This module handles the creation of sliding windows from the time series data and performs feature engineering.

#### Key Components:
- **TimeSeriesProcessor**: Main class for processing time series data
  - `__init__(window_size, prediction_horizon, stride)`: Initialize with window parameters
  - `_normalize_dataframe(df)`: Normalize all numerical columns to [0,1] range
  - `_add_engineered_features(df)`: Add engineered features like rates of change and rolling statistics
  - `create_sliding_windows(df, scram_time)`: Create sliding windows from time series data
  - `process_simulation(df, scram_time)`: Process a single simulation
  - `create_sequences_dataset(data_dict)`: Process all simulations to create a dataset for training

#### Data Flow:
1. Normalize the raw time series data to [0,1] range
2. Add engineered features (rates of change, rolling statistics)
3. Create sliding windows of specified size
4. Label each window based on whether Reactor Scram occurs within the prediction horizon
5. Compile all sequences into a dataset suitable for training

#### Window Creation Logic:
- Each window contains `window_size` time steps (default: 30 steps = 5 minutes at 10-sec intervals)
- Windows are created with a stride of `stride` time steps (default: 1 step = 10 seconds)
- For each window, a binary label is created:
  - 1 if Reactor Scram occurs within the next `prediction_horizon` time steps
  - 0 otherwise

#### Usage:
```python
processor = TimeSeriesProcessor(window_size=30, prediction_horizon=30, stride=1)
processed_data = processor.process_simulation(df, scram_time)
full_dataset = processor.create_sequences_dataset(all_simulations_data)
```

### Data Validation (`data_validator.py`)
This module provides tools to validate both raw and processed data, ensuring quality and consistency.

#### Key Components:
- **DataValidator**: Main class for data validation
  - `check_missing_values(df)`: Check for missing values in the dataframe
  - `check_data_consistency(df)`: Check data consistency issues
  - `check_label_distribution(labels)`: Check the distribution of labels
  - `check_dataset_structure(dataset)`: Check the structure of the processed dataset
  - `plot_label_distribution(dataset, by_accident_type)`: Visualize label distribution
  - `validate_simulation(df, scram_time)`: Validate a single simulation
  - `validate_dataset(dataset)`: Validate the processed dataset

#### Validation Checks:
1. Missing Value Detection: Identifies and quantifies missing values
2. Data Consistency: Checks for monotonically increasing time, constant columns, and outliers
3. Label Distribution: Analyzes class balance and imbalance ratios
4. Dataset Structure: Validates shape consistency, metadata completeness, and feature alignment
5. Simulation Validation: Ensures Scram time falls within simulation time range

#### Usage:
```python
validator = DataValidator()
validation_results = validator.validate_simulation(df, scram_time)
dataset_validation = validator.validate_dataset(processed_dataset)
```

### Data Preprocessing Pipeline (`preprocess_data.py`)
This script integrates all the data processing components into a complete pipeline.

#### Key Components:
- `preprocess_dataset()`: Main function that orchestrates the entire preprocessing workflow
  - Loads all simulation data
  - Processes data into sequences
  - Validates the dataset
  - Creates train/val/test splits
  - Saves processed data to disk

#### Data Flow:
1. Load all simulations using `NPPADDataLoader`
2. Process the data into sequences using `TimeSeriesProcessor`
3. Validate the dataset using `DataValidator`
4. Split the data into training, validation, and test sets
   - Splitting is done at the simulation level to prevent data leakage
   - Stratification by accident type ensures balanced representation
5. Save the processed data, feature names, scaler, and statistics

#### Train/Val/Test Split Strategy:
- Split by simulation rather than by individual sequences
- Stratify by accident type to ensure balanced representation
- Default split: 70% training, 15% validation, 15% testing

#### Usage:
```python
stats = preprocess_dataset(
    data_dir="NPPAD",
    output_dir="processed_data",
    window_size=30,
    prediction_horizon=30,
    stride=1
)
```

## Command-line Interface
The preprocessing pipeline can be run from the command line with customizable parameters:

```bash
python preprocess_data.py --data_dir NPPAD --output_dir processed_data --window_size 30 --prediction_horizon 30 --stride 1
```

Arguments:
- `--data_dir`: Path to NPPAD dataset
- `--output_dir`: Output directory for processed data
- `--window_size`: Number of time steps in window (30 = 5 min)
- `--prediction_horizon`: Number of steps to look ahead for Scram
- `--stride`: Step size between consecutive windows
- `--test_size`: Proportion of data for testing
- `--val_size`: Proportion of data for validation
- `--random_state`: Random seed for reproducibility

## Model Architecture

## Training Pipeline

## Evaluation Metrics

## System Integration 