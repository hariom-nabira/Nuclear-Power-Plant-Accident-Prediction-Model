import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Set
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
        self.all_feature_columns = set()  # Track all feature columns across all files
        self.scaler_fitted = False
        self.feature_names_list = []  # Master list of all feature names in consistent order
    
    def collect_all_features(self, data_dict: Dict[str, Dict[int, Tuple[pd.DataFrame, Optional[float]]]]) -> Set[str]:
        """
        Collect all possible feature names across all dataframes.
        
        Args:
            data_dict: Dictionary with structure {accident_type: {sim_num: (df, scram_time)}}
            
        Returns:
            Set[str]: Set of all feature column names
        """
        all_features = set()
        time_cols = set()
        
        for accident_type, simulations in data_dict.items():
            for sim_num, (df, _) in simulations.items():
                # Assume first column is time
                time_col = df.columns[0]
                time_cols.add(time_col)
                
                # Add all numeric columns
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                all_features.update(numeric_cols)
                
                # Only process a few files to avoid excessive memory usage
                if len(all_features) > 0 and len(time_cols) == 1:
                    break
        
        # Remove time column from features
        all_features = all_features - time_cols
        
        return all_features
    
    def fit_scaler(self, data_dict: Dict[str, Dict[int, Tuple[pd.DataFrame, Optional[float]]]]) -> None:
        """
        Fit the scaler on a combined dataset of all features.
        
        Args:
            data_dict: Dictionary with structure {accident_type: {sim_num: (df, scram_time)}}
        """
        # Collect feature names from all dataframes
        self.all_feature_columns = self.collect_all_features(data_dict)
        self.feature_columns = list(self.all_feature_columns)
        
        # Create a sample dataframe with all features to fit the scaler
        combined_data = []
        
        # Collect sample data from each simulation (first 10 rows)
        for accident_type, simulations in data_dict.items():
            for sim_num, (df, _) in simulations.items():
                sample_rows = min(10, len(df))
                time_col = df.columns[0]
                
                # Only keep numeric columns that are in our feature set
                df_numeric = df.select_dtypes(include=['float64', 'int64'])
                df_features = df_numeric.loc[:, df_numeric.columns.isin(self.feature_columns)]
                
                combined_data.append(df_features.iloc[:sample_rows])
                
                # Only use a few simulations to avoid excessive memory usage
                if len(combined_data) >= 10:
                    break
            
            if len(combined_data) >= 10:
                break
        
        if combined_data:
            # Combine all samples
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Make sure all feature columns exist (some may be missing)
            for col in self.feature_columns:
                if col not in combined_df.columns:
                    combined_df[col] = 0.0
            
            # Fit the scaler on the combined dataframe
            self.scaler.fit(combined_df[self.feature_columns])
            self.scaler_fitted = True
            
            print(f"Fitted scaler on {len(self.feature_columns)} features")
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all numerical columns in the dataframe to [0,1] range.
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            
        Returns:
            pd.DataFrame: Normalized dataframe
        """
        # If scaler isn't fitted, this is likely a single file processing case
        if not self.scaler_fitted:
            # Get numeric columns from this dataframe
            time_col = df.columns[0]  # Assume first column is time
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # Remove time column if it's numeric
            if time_col in numerical_cols:
                numerical_cols.remove(time_col)
                
            self.feature_columns = numerical_cols
            self.all_feature_columns = set(numerical_cols)
            
            # Fit the scaler on this dataframe
            self.scaler.fit(df[numerical_cols])
            self.scaler_fitted = True
        
        # Create a normalized copy
        df_normalized = df.copy()
        
        # Get time column
        time_col = df.columns[0]  # Assume first column is time
        
        # Prepare dataframe for transformation
        transform_cols = []
        for col in self.feature_columns:
            if col in df.columns:
                transform_cols.append(col)
            else:
                # Add missing columns with zeros
                df_normalized[col] = 0.0
                transform_cols.append(col)
        
        # Apply transformation
        df_normalized[transform_cols] = self.scaler.transform(df_normalized[transform_cols])
        
        return df_normalized
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional engineered features
        """
        # Create a copy of the original dataframe
        df_base = df.copy()
        
        # Initialize dictionaries to collect new features
        rate_features = {}
        mean_features = {}
        std_features = {}
        
        # Calculate features for all parameters at once
        for col in self.feature_columns:
            # Skip columns not in this dataframe
            if col not in df.columns:
                continue
            
            # Skip non-numeric columns
            if col not in df.select_dtypes(include=['float64', 'int64']).columns:
                continue
                
            # Rate of change (first derivative)
            rate_features[f"{col}_rate"] = df[col].diff().fillna(0)
            
            # Rolling statistics (mean and std over 1 minute = 6 time steps)
            mean_features[f"{col}_mean_1min"] = df[col].rolling(window=6, min_periods=1).mean()
            std_features[f"{col}_std_1min"] = df[col].rolling(window=6, min_periods=1).std().fillna(0)
        
        # Create dataframes for each feature type
        df_rates = pd.DataFrame(rate_features, index=df.index)
        df_means = pd.DataFrame(mean_features, index=df.index)
        df_stds = pd.DataFrame(std_features, index=df.index)
        
        # Concatenate all features at once
        df_with_features = pd.concat([df_base, df_rates, df_means, df_stds], axis=1)
        
        # Fill any missing values from the engineered features
        df_with_features.fillna(0, inplace=True)
        
        return df_with_features
    
    def get_all_possible_features(self, data_dict: Dict[str, Dict[int, Tuple[pd.DataFrame, Optional[float]]]]) -> List[str]:
        """
        Get all possible feature names that would be created after feature engineering.
        
        Args:
            data_dict: Dictionary with structure {accident_type: {sim_num: (df, scram_time)}}
            
        Returns:
            List[str]: List of all possible feature names
        """
        # Get base features first
        base_features = list(self.all_feature_columns)
        
        # Generate all possible derived feature names
        all_features = []
        all_features.extend(base_features)
        all_features.extend([f"{col}_rate" for col in base_features])
        all_features.extend([f"{col}_mean_1min" for col in base_features])
        all_features.extend([f"{col}_std_1min" for col in base_features])
        
        return all_features
    
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
        
        # Make sure we have all expected features
        if self.feature_names_list:
            for col in self.feature_names_list:
                if col not in df_processed.columns:
                    df_processed[col] = 0.0
                    
            # Keep only the master list of features and in the same order
            features = df_processed[self.feature_names_list]
        else:
            # Extract features - first time through
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
        
        return X, y, feature_names
    
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
        X, y, feature_names = self.create_sliding_windows(df, scram_time)
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names
        }
    
    def process_in_batches(self, data_dict):
        all_X = []
        all_y = []
        metadata = []
        
        # First, extract the inner dictionary for the accident type
        # Get the only accident type key (there should be just one in this method)
        if len(data_dict) != 1:
            print(f"Warning: Expected 1 accident type, got {len(data_dict)}")
        
        accident_type = list(data_dict.keys())[0]
        simulations_dict = data_dict[accident_type]
        
        # Now process each simulation
        for sim_num, (df, scram_time) in simulations_dict.items():
            processed = self.process_simulation(df, scram_time)
            
            if len(processed['X']) > 0:
                # Ensure all sequences have the correct feature dimension
                if self.feature_names_list and processed['X'].shape[2] != len(self.feature_names_list):
                    # This should never happen if feature_names_list is being used properly
                    print(f"Warning: Simulation {sim_num} has {processed['X'].shape[2]} features instead of {len(self.feature_names_list)}")
                    # Skip this one to avoid dimension errors
                    continue
                
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
            'feature_names': self.feature_names_list
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
        # First, fit the scaler on all possible features
        if not self.scaler_fitted:
            self.fit_scaler(data_dict)
            
            # Create a master list of all possible features after feature engineering
            self.feature_names_list = self.get_all_possible_features(data_dict)
            print(f"Created master feature list with {len(self.feature_names_list)} features")
        
        all_X = []
        all_y = []
        all_metadata = []
        
        # Process one accident type at a time to reduce memory usage
        for accident_type, simulations in data_dict.items():
            print(f"Processing {accident_type}: {len(simulations)} simulations")
            
            # Create a subset dictionary with just this accident type
            accident_data = {accident_type: simulations}
            
            try:
                # Process this accident type
                processed = self.process_in_batches(accident_data)
                
                # Collect results
                all_X.append(processed['X'])
                all_y.append(processed['y'])
                all_metadata.extend(processed['metadata'])
                
                print(f"  Processed {len(processed['X'])} sequences with {processed['X'].shape[2]} features")
            except Exception as e:
                print(f"  Error processing {accident_type}: {e}")
                continue
        
        if not all_X:
            raise ValueError("No valid sequences were created from any accident type")
        
        # Concatenate all arrays
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        return {
            'X': X_combined,
            'y': y_combined,
            'metadata': all_metadata,
            'feature_names': self.feature_names_list
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