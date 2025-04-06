import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
import json
import time
from sklearn.model_selection import train_test_split

from data_loader import NPPADDataLoader
from time_series_processor import TimeSeriesProcessor
from data_validator import DataValidator

def preprocess_dataset(data_dir: str, 
                      output_dir: str = 'processed_data',
                      window_size: int = 30, 
                      prediction_horizon: int = 30, 
                      stride: int = 1, 
                      test_size: float = 0.15,
                      validation_size: float = 0.15,
                      random_state: int = 42) -> Dict:
    """
    Preprocess the NPPAD dataset and create train/val/test splits.
    
    Args:
        data_dir (str): Path to the NPPAD dataset
        output_dir (str): Directory to save processed data
        window_size (int): Number of time steps in each window
        prediction_horizon (int): Number of time steps to look ahead for Scram events
        stride (int): Step size between consecutive windows
        test_size (float): Proportion of data to use for testing
        validation_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        Dict: Dictionary with processing stats
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    data_loader = NPPADDataLoader(data_dir)
    processor = TimeSeriesProcessor(window_size, prediction_horizon, stride)
    validator = DataValidator()
    
    print(f"Found {len(data_loader.accident_types)} accident types")
    
    # Get dataset statistics
    stats = data_loader.get_dataset_statistics()
    print(f"Total simulations: {stats['total_simulations']}")
    print(f"Simulations with Reactor Scram: {stats['simulations_with_scram']}")
    
    # Load all simulations
    print("Loading simulation data...")
    all_data = data_loader.load_all_simulations()
    
    # Process data into sequences
    print("Processing data into sequences...")
    dataset = processor.create_sequences_dataset(all_data)
    
    print(f"Created dataset with {len(dataset['X'])} sequences, {len(dataset['feature_names'])} features")
    print(f"Positive label rate: {np.mean(dataset['y']) * 100:.2f}%")
    
    # Validate dataset
    print("Validating dataset...")
    validation_results = validator.validate_dataset(dataset)
    
    if not validation_results.get('structure', {}).get('is_valid', False):
        raise ValueError(f"Dataset validation failed: {validation_results.get('structure', {}).get('validation_message', 'Unknown error')}")
    
    # Split into train/val/test
    # We want to split by simulation rather than individual sequences to prevent data leakage
    print("Creating train/val/test splits...")
    
    # Extract metadata
    metadata = dataset['metadata']
    unique_simulations = list(set((m['accident_type'], m['simulation']) for m in metadata))
    
    # Split unique simulations
    train_sims, test_sims = train_test_split(
        unique_simulations, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[acc_type for acc_type, _ in unique_simulations]  # Stratify by accident type
    )
    
    train_sims, val_sims = train_test_split(
        train_sims,
        test_size=validation_size / (1 - test_size),  # Adjust to get correct proportion
        random_state=random_state,
        stratify=[acc_type for acc_type, _ in train_sims]  # Stratify by accident type
    )
    
    # Create masks for sequences
    train_mask = np.array([
        (m['accident_type'], m['simulation']) in train_sims 
        for m in metadata
    ])
    
    val_mask = np.array([
        (m['accident_type'], m['simulation']) in val_sims 
        for m in metadata
    ])
    
    test_mask = np.array([
        (m['accident_type'], m['simulation']) in test_sims 
        for m in metadata
    ])
    
    # Create splits
    X_train = dataset['X'][train_mask]
    y_train = dataset['y'][train_mask]
    
    X_val = dataset['X'][val_mask]
    y_val = dataset['y'][val_mask]
    
    X_test = dataset['X'][test_mask]
    y_test = dataset['y'][test_mask]
    
    metadata_train = [m for i, m in enumerate(metadata) if train_mask[i]]
    metadata_val = [m for i, m in enumerate(metadata) if val_mask[i]]
    metadata_test = [m for i, m in enumerate(metadata) if test_mask[i]]
    
    # Create split datasets
    train_dataset = {
        'X': X_train,
        'y': y_train,
        'metadata': metadata_train,
        'feature_names': dataset['feature_names']
    }
    
    val_dataset = {
        'X': X_val,
        'y': y_val,
        'metadata': metadata_val,
        'feature_names': dataset['feature_names']
    }
    
    test_dataset = {
        'X': X_test,
        'y': y_test,
        'metadata': metadata_test,
        'feature_names': dataset['feature_names']
    }
    
    # Save processed data
    print("Saving processed data...")
    
    with open(os.path.join(output_dir, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open(os.path.join(output_dir, 'val_dataset.pkl'), 'wb') as f:
        pickle.dump(val_dataset, f)
    
    with open(os.path.join(output_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)
        
    # Save feature names separately for easy access
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(dataset['feature_names'], f)
    
    # Save scaler for future use
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(processor.scaler, f)
    
    # Save processing parameters
    processing_params = {
        'window_size': window_size,
        'prediction_horizon': prediction_horizon,
        'stride': stride,
        'test_size': test_size,
        'validation_size': validation_size,
        'random_state': random_state
    }
    
    with open(os.path.join(output_dir, 'processing_params.json'), 'w') as f:
        json.dump(processing_params, f)
    
    # Compile statistics
    dataset_stats = {
        'total_sequences': len(dataset['X']),
        'total_features': len(dataset['feature_names']),
        'train_sequences': len(X_train),
        'val_sequences': len(X_val),
        'test_sequences': len(X_test),
        'positive_rate_train': float(np.mean(y_train)),
        'positive_rate_val': float(np.mean(y_val)),
        'positive_rate_test': float(np.mean(y_test)),
        'train_simulations': len(train_sims),
        'val_simulations': len(val_sims),
        'test_simulations': len(test_sims),
        'processing_time_seconds': time.time() - start_time
    }
    
    # Save statistics
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    print(f"Data preprocessing completed in {dataset_stats['processing_time_seconds']:.2f} seconds")
    print(f"Processed data saved to {output_dir}")
    
    return dataset_stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess NPPAD dataset for Reactor Scram prediction")
    parser.add_argument("--data_dir", type=str, default="NPPAD", help="Path to NPPAD dataset")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory for processed data")
    parser.add_argument("--window_size", type=int, default=30, help="Number of time steps in window (30 = 5 min)")
    parser.add_argument("--prediction_horizon", type=int, default=30, help="Number of steps to look ahead for Scram")
    parser.add_argument("--stride", type=int, default=1, help="Step size between consecutive windows")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion of data for testing")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of data for validation")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    stats = preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        stride=args.stride,
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_state
    )
    
    # Print key statistics
    print("\nDataset Statistics:")
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Train/Val/Test split: {stats['train_sequences']}/{stats['val_sequences']}/{stats['test_sequences']}")
    print(f"Positive label rate (Train/Val/Test): {stats['positive_rate_train']*100:.2f}%/{stats['positive_rate_val']*100:.2f}%/{stats['positive_rate_test']*100:.2f}%") 