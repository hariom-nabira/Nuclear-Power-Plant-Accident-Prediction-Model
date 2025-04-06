import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_root: str, window_size: int = 30, stride: int = 1):
        """
        Initialize the DataLoader for NPPAD dataset.
        
        Args:
            data_root (str): Root directory containing the NPPAD dataset
            window_size (int): Number of time steps in each window (default: 30 for 5 minutes)
            stride (int): Number of time steps to move the window (default: 1)
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root directory not found: {data_root}")
            
        logger.info(f"Initializing DataLoader with root directory: {data_root}")
        self.window_size = window_size
        self.stride = stride
        self.accident_types = [
            'LOCA'
        ]
        
        # Find available accident types
        self.available_types = [acc_type for acc_type in self.accident_types 
                              if (self.data_root / acc_type).exists()]
        
        if not self.available_types:
            raise ValueError(f"No accident type directories found in {data_root}")
            
        logger.info(f"Found {len(self.available_types)} accident types: {self.available_types}")
        missing_types = set(self.accident_types) - set(self.available_types)
        if missing_types:
            logger.warning(f"Missing accident type directories: {missing_types}")
        
    def load_accident_data(self, accident_type: str) -> Tuple[List[pd.DataFrame], List[float]]:
        """
        Load data for a specific accident type.
        
        Args:
            accident_type (str): Type of accident (e.g., 'LOCA', 'SGBTR')
            
        Returns:
            Tuple containing:
            - List of DataFrames with time series data
            - List of Reactor Scram timestamps
        """
        accident_dir = self.data_root / accident_type
        if not accident_dir.exists():
            logger.error(f"Accident type directory not found: {accident_type}")
            return [], []
            
        logger.info(f"\nLoading data for {accident_type}:")
        logger.info(f"Directory: {accident_dir}")
        
        # List all files in directory
        all_files = list(accident_dir.glob("*"))
        logger.info(f"Total files found: {len(all_files)}")
        logger.info(f"First few files: {[f.name for f in all_files[:5]]}")
        
        data_files = []
        scram_timestamps = []
        
        # Try different patterns for TransientReport files
        txt_patterns = [
            "*TransientReport.txt",
            "*Transient Report.txt",
            "*TransientReport*.txt",
            "*Transient Report*.txt"
        ]
        
        # Load data for each simulation (1-100)
        for sim_num in range(1, 101):
            csv_path = accident_dir / f"{sim_num}.csv"
            
            # Try to find matching TransientReport file
            txt_path = None
            for pattern in txt_patterns:
                potential_txt = list(accident_dir.glob(pattern.replace("*", str(sim_num))))
                if potential_txt:
                    txt_path = potential_txt[0]
                    break
            
            if not (csv_path.exists() and txt_path and txt_path.exists()):
                logger.warning(f"Missing files for simulation {sim_num} in {accident_type}")
                logger.warning(f"  CSV path: {csv_path} (exists: {csv_path.exists()})")
                logger.warning(f"  TXT path: {txt_path} (exists: {txt_path.exists() if txt_path else False})")
                continue
                
            try:
                # Load time series data
                df = pd.read_csv(csv_path)
                if df.empty:
                    logger.warning(f"Empty CSV file: {csv_path}")
                    continue
                    
                data_files.append(df)
                
                # Extract Reactor Scram timestamp
                scram_time = self.parse_transient_report(txt_path)
                scram_timestamps.append(scram_time)
                
                logger.info(f"Successfully loaded simulation {sim_num}")
                
            except Exception as e:
                logger.error(f"Error loading simulation {sim_num} in {accident_type}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(data_files)} simulations for {accident_type}")
        return data_files, scram_timestamps
    
    def parse_transient_report(self, report_path: Path) -> float:
        """
        Parse the Transient Report file to extract Reactor Scram timestamp.
        
        Args:
            report_path (Path): Path to the Transient Report file
            
        Returns:
            float: Timestamp of Reactor Scram event in seconds
        """
        try:
            with open(report_path, 'r') as f:
                content = f.read()
                
            # Look for Reactor Scram event in the report
            scram_lines = [line for line in content.split('\n') 
                          if 'Reactor Scram' in line]
            
            if not scram_lines:
                logger.warning(f"No Reactor Scram event found in {report_path}")
                return float('inf')
                
            # Extract timestamp from the first Reactor Scram event
            # Assuming timestamp is in seconds at the start of the line
            try:
                timestamp = float(scram_lines[0].split()[0])
                logger.info(f"Found Reactor Scram at timestamp {timestamp} in {report_path}")
                return timestamp
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing timestamp in {report_path}: {str(e)}")
                return float('inf')
                
        except Exception as e:
            logger.error(f"Error reading report file {report_path}: {str(e)}")
            return float('inf')
    
    def create_windows(self, df: pd.DataFrame, scram_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data and generate labels.
        
        Args:
            df (pd.DataFrame): Time series data
            scram_time (float): Reactor Scram timestamp
            
        Returns:
            Tuple containing:
            - Array of windows (shape: n_windows × window_size × n_features)
            - Array of labels (1 if Reactor Scram occurs in next 5 minutes)
        """
        try:
            # Convert DataFrame to numpy array for faster processing
            data = df.values
            
            # Calculate number of windows
            n_windows = (len(data) - self.window_size) // self.stride + 1
            
            if n_windows <= 0:
                logger.warning(f"Not enough data points for window size {self.window_size}")
                return np.array([]), np.array([])
            
            # Create strided view of the data for efficient window creation
            strided_data = np.lib.stride_tricks.sliding_window_view(
                data,
                window_shape=(self.window_size, data.shape[1])
            )[::self.stride]
            
            # Calculate end times for all windows at once
            window_end_times = np.arange(self.window_size, len(data) + 1, self.stride) * 10
            
            # Create labels (1 if Reactor Scram occurs in next 5 minutes)
            labels = np.logical_and(
                window_end_times <= scram_time,
                scram_time <= window_end_times + 300
            ).astype(np.int32)
            
            return strided_data, labels
            
        except Exception as e:
            logger.error(f"Error creating windows: {str(e)}")
            return np.array([]), np.array([])
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0,1] range using min-max scaling.
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            np.ndarray: Normalized data
        """
        try:
            if data.size == 0:
                return data
                
            # Reshape to 2D for normalization
            n_samples = data.shape[0] * data.shape[1]
            n_features = data.shape[2]
            data_2d = data.reshape(n_samples, n_features)
            
            # Calculate min and max for each feature
            min_vals = np.min(data_2d, axis=0)
            max_vals = np.max(data_2d, axis=0)
            
            # Avoid division by zero
            max_vals[max_vals == min_vals] = 1
            
            # Normalize
            normalized = (data_2d - min_vals) / (max_vals - min_vals)
            
            # Reshape back to 3D
            return normalized.reshape(data.shape)
            
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return data
    
    def generate_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Generate derived features from raw parameters.
        
        Args:
            windows (np.ndarray): Input windows array
            
        Returns:
            np.ndarray: Array with additional features
        """
        try:
            if windows.size == 0:
                return windows
                
            n_windows, n_timesteps, n_features = windows.shape
            
            # Calculate rate of change
            roc = np.zeros_like(windows)
            roc[:, 1:] = windows[:, 1:] - windows[:, :-1]
            
            # Calculate rolling statistics
            window_size = 5  # 50 seconds
            rolling_mean = np.zeros_like(windows)
            rolling_std = np.zeros_like(windows)
            
            for i in range(window_size, n_timesteps):
                rolling_mean[:, i] = np.mean(windows[:, i-window_size:i], axis=1)
                rolling_std[:, i] = np.std(windows[:, i-window_size:i], axis=1)
                
            # Combine all features
            features = np.concatenate([
                windows,
                roc,
                rolling_mean,
                rolling_std
            ], axis=2)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            return windows
    
    def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the complete dataset for training.
        
        Returns:
            Tuple containing:
            - Array of all windows with features
            - Array of all labels
        """
        all_windows = []
        all_labels = []
        loaded_types = []
        total_simulations = 0
        
        for accident_type in self.available_types:
            logger.info(f"\nProcessing {accident_type}...")
            
            # Load data for this accident type
            data_files, scram_times = self.load_accident_data(accident_type)
            
            if not data_files:
                logger.warning(f"No data loaded for {accident_type}")
                continue
                
            # Process each simulation
            for df, scram_time in zip(data_files, scram_times):
                # Create windows and labels
                windows, labels = self.create_windows(df, scram_time)
                
                if windows.size == 0:
                    continue
                    
                # Normalize data
                windows = self.normalize_data(windows)
                
                # Generate additional features
                windows = self.generate_features(windows)
                
                all_windows.append(windows)
                all_labels.append(labels)
                total_simulations += 1
                
                # Log progress every 10 simulations
                if total_simulations % 10 == 0:
                    logger.info(f"Processed {total_simulations} simulations...")
            
            if data_files:
                loaded_types.append(accident_type)
        
        if not all_windows:
            raise ValueError("No data was successfully loaded. Please check the data directory structure and file contents.")
        
        # Combine all data
        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        logger.info("\nDataset Summary:")
        logger.info(f"Successfully loaded data from {len(loaded_types)} accident types: {loaded_types}")
        logger.info(f"Total simulations processed: {total_simulations}")
        logger.info(f"Total number of windows: {len(X)}")
        logger.info(f"Window shape: {X.shape}")
        logger.info(f"Number of features: {X.shape[2]}")
        logger.info(f"Label distribution: {np.unique(y, return_counts=True)}")
        
        return X, y 