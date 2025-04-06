import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.preprocessing import MinMaxScaler
import warnings

class TimeSeriesProcessor:
    """
    Process time series data from NPP simulations.
    This class handles:
    1. Sliding window creation
    2. Feature engineering
    3. Label generation based on Reactor Scram events
    """
    
    def __init__(self, window_size: int = 30, prediction_horizon: int = 30, stride: int = 1):
        """
        Initialize the time series processor.
        
        Args:
            window_size (int): Number of time steps in each input window (default: 30 = 5 min at 10-sec intervals)
            prediction_horizon (int): Number of time steps to look ahead for Scram events (default: 30 = 5 min)
            stride (int): Step size between consecutive windows (default: 1 = 10 seconds)
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all numerical columns in the dataframe to [0,1] range.
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            
        Returns:
            pd.DataFrame: Normalized dataframe
        """
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Fit scaler if not already fit
        if self.feature_columns is None:
            self.feature_columns = numerical_cols
            self.scaler.fit(df[numerical_cols])
        
        # Transform the data
        df_normalized = df.copy()
        df_normalized[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df_normalized
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional engineered features
        """
        df_with_features = df.copy()
        
        # Calculate rate of change for all parameters
        for col in self.feature_columns:
            # Skip non-numeric columns
            if col not in df.select_dtypes(include=['float64', 'int64']).columns:
                continue
                
            # Rate of change (first derivative)
            df_with_features[f"{col}_rate"] = df[col].diff().fillna(0)
            
            # Rolling statistics (mean and std over 1 minute = 6 time steps)
            df_with_features[f"{col}_mean_1min"] = df[col].rolling(window=6, min_periods=1).mean()
            df_with_features[f"{col}_std_1min"] = df[col].rolling(window=6, min_periods=1).std().fillna(0)
        
        # Fill any missing values from the engineered features
        df_with_features.fillna(0, inplace=True)
        
        return df_with_features
    
    def create_sliding_windows(self, 
                             df: pd.DataFrame, 
                             scram_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from the time series data.
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            scram_time (Optional[float]): Timestamp of Reactor Scram in seconds
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - X: Array of sliding windows with shape (n_windows, window_size, n_features)
                - y: Binary labels indicating if Scram occurs within the prediction horizon
        """
        # Normalize the data
        df_normalized = self._normalize_dataframe(df)
        
        # Add engineered features
        df_processed = self._add_engineered_features(df_normalized)
        
        # Get the time column name (assume it's the first column)
        time_col = df.columns[0]
        
        # Extract features
        features = df_processed.drop(columns=[time_col])
        feature_names = features.columns.tolist()
        
        # Convert to numpy arrays
        time_values = df[time_col].values
        feature_values = features.values
        
        # Create sliding windows
        n_samples = len(df)
        n_features = feature_values.shape[1]
        
        # Calculate number of windows
        n_windows = max(0, n_samples - self.window_size - self.prediction_horizon + 1)
        
        if n_windows <= 0:
            warnings.warn("Data sequence too short to create any windows")
            return np.array([]), np.array([])
        
        # Initialize arrays
        X = np.zeros((n_windows, self.window_size, n_features))
        y = np.zeros(n_windows)
        
        # Fill the arrays
        for i in range(0, n_windows, self.stride):
            # Input window
            X[i] = feature_values[i:i+self.window_size]
            
            # Label: Check if Scram occurs in prediction horizon
            if scram_time is not None:
                window_end_time = time_values[i + self.window_size - 1]
                prediction_horizon_end = window_end_time + (self.prediction_horizon * 10)  # Assuming 10-sec intervals
                
                # If Scram time is within prediction horizon
                if window_end_time < scram_time <= prediction_horizon_end:
                    y[i] = 1
        
        return X, y
    
    def process_simulation(self, 
                         df: pd.DataFrame, 
                         scram_time: Optional[float] = None) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Process a single simulation, creating sliding windows and labels.
        
        Args:
            df (pd.DataFrame): Simulation data
            scram_time (Optional[float]): Timestamp of Reactor Scram
            
        Returns:
            Dict: Dictionary containing:
                - X: Input windows
                - y: Labels
                - feature_names: List of feature names
        """
        X, y = self.create_sliding_windows(df, scram_time)
        
        # Get all feature names (original + engineered)
        df_normalized = self._normalize_dataframe(df)
        df_processed = self._add_engineered_features(df_normalized)
        time_col = df.columns[0]
        feature_names = df_processed.drop(columns=[time_col]).columns.tolist()
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names
        }
    
    def create_sequences_dataset(self, 
                              data_dict: Dict[str, Dict[int, Tuple[pd.DataFrame, Optional[float]]]]) -> Dict:
        """
        Process all simulations to create a dataset for training.
        
        Args:
            data_dict: Dictionary with structure {accident_type: {sim_num: (df, scram_time)}}
            
        Returns:
            Dict: Dictionary containing concatenated X, y arrays and metadata
        """
        all_X = []
        all_y = []
        metadata = []
        
        for accident_type, simulations in data_dict.items():
            for sim_num, (df, scram_time) in simulations.items():
                processed = self.process_simulation(df, scram_time)
                
                if len(processed['X']) > 0:
                    all_X.append(processed['X'])
                    all_y.append(processed['y'])
                    
                    # Add metadata for each window
                    n_windows = len(processed['X'])
                    metadata.extend([{
                        'accident_type': accident_type,
                        'simulation': sim_num,
                        'window_idx': i
                    } for i in range(n_windows)])
        
        if not all_X:
            raise ValueError("No valid sequences were created from the input data")
        
        # Concatenate all arrays
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        return {
            'X': X_combined,
            'y': y_combined,
            'metadata': metadata,
            'feature_names': processed['feature_names'] if all_X else []
        }

# Example usage
if __name__ == "__main__":
    from data_loader import NPPADDataLoader
    
    # Load a single simulation
    data_loader = NPPADDataLoader("NPPAD")
    accident_type = data_loader.accident_types[0]
    sim_num = list(data_loader.data_mapping[accident_type].keys())[0]
    df, scram_time = data_loader.load_simulation_data(accident_type, sim_num)
    
    # Process the simulation
    processor = TimeSeriesProcessor(window_size=30, prediction_horizon=30, stride=1)
    processed_data = processor.process_simulation(df, scram_time)
    
    print(f"Created {len(processed_data['X'])} sequences")
    print(f"Positive labels: {np.sum(processed_data['y'])}")
    print(f"Number of features: {processed_data['X'].shape[2]}")
    print(f"First 5 feature names: {processed_data['feature_names'][:5]}") 