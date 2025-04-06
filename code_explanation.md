# Code Explanation

This document explains the implementation details of the Nuclear Power Plant Accident Detection System.

## Phase 1: Data Preparation

### DataLoader Class
The `DataLoader` class is responsible for:
1. Loading and organizing the NPPAD dataset
2. Parsing CSV files containing time series data
3. Extracting Reactor Scram timestamps from Transient Report files
4. Creating a unified data structure for training

Key components:
- `__init__`: Initializes the data loader with dataset paths and configuration
- `load_accident_data`: Loads data for a specific accident type
- `parse_transient_report`: Extracts Reactor Scram timestamps from report files
- `create_windows`: Implements sliding window mechanism for time series data
- `normalize_data`: Normalizes parameters to [0,1] range
- `generate_features`: Creates derived features from raw parameters

### Data Processing Pipeline
The pipeline follows these steps:
1. Load raw data from CSV and TXT files
2. Extract Reactor Scram events
3. Create sliding windows with 5-minute lookback
4. Generate labels based on future Reactor Scram events
5. Normalize data and create derived features
6. Validate data quality and consistency

### Feature Engineering
The following derived features are created:
1. Rate of change for critical parameters
2. Rolling statistics (mean, standard deviation)
3. Parameter correlations within each window
4. Temporal derivatives for key measurements

### Data Validation
The validation process checks for:
1. Missing values and data consistency
2. Label distribution balance
3. Temporal continuity
4. Parameter value ranges
5. Correlation patterns 