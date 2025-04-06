# Nuclear Power Plant Accident Data Preprocessing

This repository contains scripts for preprocessing nuclear power plant accident simulation data. The preprocessing involves:

1. Scanning TransientReport.txt files to find accident timestamps (Reactor Scram or Core Meltdown)
2. Adding a "label" column to the corresponding CSV files:
   - 0 for normal operation (before accident_time - 180 seconds)
   - 1 for potential accident (after that point)
3. Aligning all CSV files to have the same column structure

## Requirements

- Python 3.6+
- pandas
- numpy
- tqdm

Install required packages:

```bash
pip install pandas numpy tqdm
```

## Scripts

### Basic Script (`preprocess_simulations.py`)

This script provides basic functionality for preprocessing the simulation data:

```bash
python preprocess_simulations.py
```

### Advanced Script (`preprocess_simulations_advanced.py`)

This script offers more advanced features:
- Parallel processing for faster execution
- Memory optimization for large datasets
- Detailed logging and error reporting
- Command-line options for customization

Usage:

```bash
python preprocess_simulations_advanced.py --root-dir NPPAD --workers 8
```

Options:
- `--root-dir`: Root directory containing simulations (default: "NPPAD")
- `--workers`: Maximum number of worker processes (default: 8)
- `--no-align`: Skip CSV alignment step
- `--align-only`: Only perform CSV alignment

## Output Files

The advanced script generates several output files:
- `preprocess_log_*.log`: Detailed log of the preprocessing operation
- `processing_results.json`: Detailed results of the processing operation
- `column_info.json`: Information about columns across all CSV files
- `alignment_results.json`: Results of the column alignment process

## Directory Structure

The scripts expect the following directory structure:

```
NPPAD/
├── ACCIDENT_TYPE_1/
│   ├── 1.csv
│   ├── 1Transient Report.txt
│   ├── 2.csv
│   ├── 2Transient Report.txt
│   └── ...
├── ACCIDENT_TYPE_2/
│   ├── 1.csv
│   ├── 1Transient Report.txt
│   ├── 2.csv
│   ├── 2Transient Report.txt
│   └── ...
└── ...
```

## Example Workflow

1. Basic preprocessing:
   ```bash
   python preprocess_simulations.py
   ```

2. Advanced preprocessing with custom options:
   ```bash
   python preprocess_simulations_advanced.py --root-dir /path/to/NPPAD --workers 16
   ```

3. Alignment only (if you've already labeled the data):
   ```bash
   python preprocess_simulations_advanced.py --align-only
   ```

## Label Format

The preprocessing adds the following columns to each CSV file:
- `label`: 0 for normal operation, 1 for potential accident
- `accident_timestamp`: The timestamp when the accident occurred
- `accident_type`: The type of accident (Reactor Scram or Core Meltdown)

This labeling can be used for training machine learning models to predict nuclear power plant accidents before they occur. 