import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
import json
import time
from sklearn.model_selection import train_test_split
import gc  # For garbage collection

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
                      random_state: int = 42,
                      batch_mode: bool = True) -> Dict:
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
        batch_mode (bool): Whether to process data in batches to save memory
    
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
    
    if batch_mode:
        # Process one accident type at a time
        return batch_process_dataset(
            data_loader, processor, validator, output_dir, 
            test_size, validation_size, random_state
        )
    else:
        # Process all data at once (may cause memory issues)
        return standard_process_dataset(
            data_loader, processor, validator, output_dir,
            test_size, validation_size, random_state
        )

def batch_process_dataset(data_loader: NPPADDataLoader, 
                         processor: TimeSeriesProcessor,
                         validator: DataValidator,
                         output_dir: str,
                         test_size: float = 0.15,
                         validation_size: float = 0.15,
                         random_state: int = 42) -> Dict:
    """
    Process the dataset in batches (one accident type at a time) to reduce memory usage.
    
    Args:
        data_loader (NPPADDataLoader): Data loader instance
        processor (TimeSeriesProcessor): Time series processor instance
        validator (DataValidator): Data validator instance
        output_dir (str): Output directory
        test_size (float): Proportion of data for testing
        validation_size (float): Proportion of data for validation
        random_state (int): Random seed
        
    Returns:
        Dict: Processing statistics
    """
    print("Using batch processing mode to conserve memory")
    
    # Create directory for intermediate files
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each accident type separately
    accident_stats = {}
    all_features = set()
    all_sequences = 0
    accident_sequence_counts = {}
    
    for accident_type in data_loader.accident_types:
        acc_start_time = time.time()
        print(f"\nProcessing {accident_type}...")
        
        # Load data for this accident type only
        accident_data = {}
        for sim_num in data_loader.data_mapping[accident_type]:
            try:
                df, scram_time = data_loader.load_simulation_data(accident_type, sim_num)
                accident_data[sim_num] = (df, scram_time)
            except Exception as e:
                print(f"Error loading simulation {sim_num} for {accident_type}: {e}")
                continue
                
        if not accident_data:
            print(f"No valid data for {accident_type}, skipping")
            continue
        
        # Process this accident type
        try:
            # Fit the scaler on the first accident type
            if not processor.scaler_fitted:
                # We need to scan all accident types first to get feature names
                sample_data = {}
                for acc_type in data_loader.accident_types:
                    sample_data[acc_type] = {}
                    for sim_num in list(data_loader.data_mapping[acc_type])[:2]:  # Just get a couple of samples
                        try:
                            df, scram_time = data_loader.load_simulation_data(acc_type, sim_num)
                            sample_data[acc_type][sim_num] = (df, scram_time)
                        except:
                            continue
                
                # Fit scaler on sample data
                processor.fit_scaler(sample_data)
                del sample_data
                gc.collect()
            
            # Process the data for this accident type
            processed = processor.process_in_batches({accident_type: accident_data})
            
            # Update stats
            all_features.update(processed['feature_names'])
            seq_count = len(processed['X'])
            all_sequences += seq_count
            accident_sequence_counts[accident_type] = seq_count
            
            # Save this accident type's data
            acc_file = os.path.join(temp_dir, f"{accident_type}.pkl")
            with open(acc_file, 'wb') as f:
                pickle.dump(processed, f)
                
            print(f"Saved {seq_count} sequences for {accident_type}")
            print(f"Processing time: {time.time() - acc_start_time:.2f} seconds")
            
            # Clean up to free memory
            del processed
            del accident_data
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {accident_type}: {e}")
            continue
    
    # Now split the data
    print("\nSplitting data into train/val/test sets...")
    
    # Get all accident types that were successfully processed
    processed_accident_types = [f.split('.')[0] for f in os.listdir(temp_dir) if f.endswith('.pkl')]
    
    if not processed_accident_types:
        raise ValueError("No accident types were successfully processed")
    
    # Gather information about all simulations
    all_simulations = []
    for acc_type in processed_accident_types:
        # Load the processed data
        with open(os.path.join(temp_dir, f"{acc_type}.pkl"), 'rb') as f:
            processed = pickle.load(f)
        
        # Extract unique simulations
        sims = set((m['accident_type'], m['simulation']) for m in processed['metadata'])
        all_simulations.extend(list(sims))
    
    # Split simulations
    train_sims, test_sims = train_test_split(
        all_simulations, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[acc_type for acc_type, _ in all_simulations]  # Stratify by accident type
    )
    
    train_sims, val_sims = train_test_split(
        train_sims,
        test_size=validation_size / (1 - test_size),  # Adjust to get correct proportion
        random_state=random_state,
        stratify=[acc_type for acc_type, _ in train_sims]  # Stratify by accident type
    )
    
    # Now create the train/val/test sets
    X_train, y_train, metadata_train = [], [], []
    X_val, y_val, metadata_val = [], [], []
    X_test, y_test, metadata_test = [], [], []
    
    for acc_type in processed_accident_types:
        # Load the processed data
        with open(os.path.join(temp_dir, f"{acc_type}.pkl"), 'rb') as f:
            processed = pickle.load(f)
        
        # Create masks for this accident type
        meta = processed['metadata']
        X = processed['X']
        y = processed['y']
        
        train_mask = np.array([(m['accident_type'], m['simulation']) in train_sims for m in meta])
        val_mask = np.array([(m['accident_type'], m['simulation']) in val_sims for m in meta])
        test_mask = np.array([(m['accident_type'], m['simulation']) in test_sims for m in meta])
        
        # Split data
        if np.any(train_mask):
            X_train.append(X[train_mask])
            y_train.append(y[train_mask])
            metadata_train.extend([m for i, m in enumerate(meta) if train_mask[i]])
            
        if np.any(val_mask):
            X_val.append(X[val_mask])
            y_val.append(y[val_mask])
            metadata_val.extend([m for i, m in enumerate(meta) if val_mask[i]])
            
        if np.any(test_mask):
            X_test.append(X[test_mask])
            y_test.append(y[test_mask])
            metadata_test.extend([m for i, m in enumerate(meta) if test_mask[i]])
            
        # Clean up
        del processed
        gc.collect()
    
    # Concatenate data
    X_train = np.concatenate(X_train, axis=0) if X_train else np.array([])
    y_train = np.concatenate(y_train, axis=0) if y_train else np.array([])
    
    X_val = np.concatenate(X_val, axis=0) if X_val else np.array([])
    y_val = np.concatenate(y_val, axis=0) if y_val else np.array([])
    
    X_test = np.concatenate(X_test, axis=0) if X_test else np.array([])
    y_test = np.concatenate(y_test, axis=0) if y_test else np.array([])
    
    # Save the feature names
    feature_names = processor.feature_names_list
    
    # Create the datasets
    train_dataset = {
        'X': X_train,
        'y': y_train,
        'metadata': metadata_train,
        'feature_names': feature_names
    }
    
    val_dataset = {
        'X': X_val,
        'y': y_val,
        'metadata': metadata_val,
        'feature_names': feature_names
    }
    
    test_dataset = {
        'X': X_test,
        'y': y_test,
        'metadata': metadata_test,
        'feature_names': feature_names
    }
    
    # Save the processed data
    print("Saving final datasets...")
    
    with open(os.path.join(output_dir, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open(os.path.join(output_dir, 'val_dataset.pkl'), 'wb') as f:
        pickle.dump(val_dataset, f)
    
    with open(os.path.join(output_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)
    
    # Save feature names separately
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(processor.scaler, f)
    
    # Save processing parameters
    processing_params = {
        'window_size': processor.window_size,
        'prediction_horizon': processor.prediction_horizon,
        'stride': processor.stride,
        'test_size': test_size,
        'validation_size': validation_size,
        'random_state': random_state
    }
    
    with open(os.path.join(output_dir, 'processing_params.json'), 'w') as f:
        json.dump(processing_params, f)
    
    # Compile statistics
    dataset_stats = {
        'total_sequences': int(len(X_train) + len(X_val) + len(X_test)),
        'total_features': len(feature_names),
        'train_sequences': int(len(X_train)),
        'val_sequences': int(len(X_val)),
        'test_sequences': int(len(X_test)),
        'positive_rate_train': float(np.mean(y_train)) if len(y_train) > 0 else 0,
        'positive_rate_val': float(np.mean(y_val)) if len(y_val) > 0 else 0,
        'positive_rate_test': float(np.mean(y_test)) if len(y_test) > 0 else 0,
        'train_simulations': len(train_sims),
        'val_simulations': len(val_sims),
        'test_simulations': len(test_sims),
        'accident_types': processed_accident_types,
        'sequences_per_accident': accident_sequence_counts,
        'processing_time_seconds': time.time() - start_time
    }
    
    # Save statistics
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    print(f"Data preprocessing completed in {dataset_stats['processing_time_seconds']:.2f} seconds")
    print(f"Processed data saved to {output_dir}")
    
    return dataset_stats

def standard_process_dataset(data_loader: NPPADDataLoader, 
                            processor: TimeSeriesProcessor,
                            validator: DataValidator,
                            output_dir: str,
                            test_size: float = 0.15,
                            validation_size: float = 0.15,
                            random_state: int = 42) -> Dict:
    """
    Process the entire dataset at once - may cause memory issues with large datasets.
    
    Args:
        data_loader (NPPADDataLoader): Data loader instance
        processor (TimeSeriesProcessor): Time series processor instance
        validator (DataValidator): Data validator instance
        output_dir (str): Output directory
        test_size (float): Proportion of data for testing
        validation_size (float): Proportion of data for validation
        random_state (int): Random seed
        
    Returns:
        Dict: Processing statistics
    """
    # Load all simulations
    print("Loading all simulation data...")
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
        'window_size': processor.window_size,
        'prediction_horizon': processor.prediction_horizon,
        'stride': processor.stride,
        'test_size': test_size,
        'validation_size': validation_size,
        'random_state': random_state
    }
    
    with open(os.path.join(output_dir, 'processing_params.json'), 'w') as f:
        json.dump(processing_params, f)
    
    # Compile statistics
    accident_type_counts = {}
    for m in metadata:
        acc_type = m['accident_type']
        if acc_type not in accident_type_counts:
            accident_type_counts[acc_type] = 0
        accident_type_counts[acc_type] += 1
    
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
        'sequences_per_accident': accident_type_counts,
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
    parser.add_argument("--batch_mode", action="store_true", help="Process in batches to conserve memory")
    
    args = parser.parse_args()
    
    stats = preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        stride=args.stride,
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_state,
        batch_mode=args.batch_mode
    )
    
    # Print key statistics
    print("\nDataset Statistics:")
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Train/Val/Test split: {stats['train_sequences']}/{stats['val_sequences']}/{stats['test_sequences']}")
    print(f"Positive label rate (Train/Val/Test): {stats['positive_rate_train']*100:.2f}%/{stats['positive_rate_val']*100:.2f}%/{stats['positive_rate_test']*100:.2f}%") 