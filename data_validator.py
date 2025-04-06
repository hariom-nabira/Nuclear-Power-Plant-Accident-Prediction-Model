import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class DataValidator:
    """
    Data validator for the NPP dataset.
    This class checks data quality, consistency, and distribution.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        pass
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, Union[int, float, Dict]]:
        """
        Check for missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Dictionary with missing value statistics
        """
        total_cells = np.product(df.shape)
        missing_cells = df.isna().sum().sum()
        
        missing_by_column = df.isna().sum()
        columns_with_missing = missing_by_column[missing_by_column > 0]
        
        return {
            'total_missing': missing_cells,
            'missing_percentage': (missing_cells / total_cells) * 100,
            'columns_with_missing': columns_with_missing.to_dict() if not columns_with_missing.empty else {}
        }
    
    def check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Union[bool, str, List]]:
        """
        Check data consistency issues such as:
        - Monotonically increasing time
        - Constant/zero columns
        - Outliers
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Dictionary with consistency check results
        """
        results = {
            'time_monotonic': True,
            'time_monotonic_message': '',
            'constant_columns': [],
            'zero_dominant_columns': [],
            'outlier_columns': []
        }
        
        # Check time monotonicity (assuming first column is time)
        time_col = df.columns[0]
        if not df[time_col].is_monotonic_increasing:
            results['time_monotonic'] = False
            results['time_monotonic_message'] = f"Time column '{time_col}' is not monotonically increasing"
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                results['constant_columns'].append(col)
        
        # Check for zero-dominant columns (>95% zeros)
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            zero_percentage = (df[col] == 0).mean() * 100
            if zero_percentage > 95:
                results['zero_dominant_columns'].append(col)
        
        # Simple outlier detection (values outside 3 standard deviations)
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:  # Avoid division by zero
                outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]
                if len(outliers) > 0.01 * len(df):  # More than 1% outliers
                    results['outlier_columns'].append(col)
        
        return results
    
    def check_label_distribution(self, labels: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Check the distribution of labels.
        
        Args:
            labels (np.ndarray): Array of binary labels
            
        Returns:
            Dict: Dictionary with label distribution statistics
        """
        total_samples = len(labels)
        positive_samples = np.sum(labels)
        negative_samples = total_samples - positive_samples
        
        return {
            'total_samples': total_samples,
            'positive_samples': int(positive_samples),
            'negative_samples': int(negative_samples),
            'positive_percentage': (positive_samples / total_samples) * 100 if total_samples > 0 else 0,
            'negative_percentage': (negative_samples / total_samples) * 100 if total_samples > 0 else 0,
            'imbalance_ratio': (negative_samples / positive_samples) if positive_samples > 0 else float('inf')
        }
    
    def check_dataset_structure(self, dataset: Dict) -> Dict[str, Union[bool, str, Dict]]:
        """
        Check the structure of the processed dataset.
        
        Args:
            dataset (Dict): The processed dataset dictionary
            
        Returns:
            Dict: Dictionary with structure validation results
        """
        results = {
            'is_valid': True,
            'validation_message': '',
            'details': {}
        }
        
        # Check required keys
        required_keys = ['X', 'y', 'feature_names', 'metadata']
        missing_keys = [key for key in required_keys if key not in dataset]
        
        if missing_keys:
            results['is_valid'] = False
            results['validation_message'] = f"Missing required keys: {missing_keys}"
            return results
        
        # Check shapes
        X_shape = dataset['X'].shape
        y_shape = dataset['y'].shape
        
        results['details']['X_shape'] = X_shape
        results['details']['y_shape'] = y_shape
        results['details']['n_features'] = X_shape[2] if len(X_shape) == 3 else 0
        results['details']['n_feature_names'] = len(dataset['feature_names'])
        
        # Validate shape consistency
        if len(X_shape) != 3:
            results['is_valid'] = False
            results['validation_message'] = f"X should be 3D but has shape {X_shape}"
        elif X_shape[0] != y_shape[0]:
            results['is_valid'] = False
            results['validation_message'] = f"X and y have inconsistent first dimension: {X_shape[0]} vs {y_shape[0]}"
        elif len(dataset['metadata']) != X_shape[0]:
            results['is_valid'] = False
            results['validation_message'] = "Metadata length doesn't match X first dimension"
        elif X_shape[2] != results['details']['n_feature_names']:
            results['is_valid'] = False
            results['validation_message'] = "Number of features doesn't match feature_names length"
        
        # Check metadata structure
        if results['is_valid'] and dataset['metadata']:
            metadata_keys = set(dataset['metadata'][0].keys())
            required_metadata_keys = {'accident_type', 'simulation', 'window_idx'}
            if not required_metadata_keys.issubset(metadata_keys):
                results['is_valid'] = False
                results['validation_message'] = f"Metadata missing required keys: {required_metadata_keys - metadata_keys}"
        
        # Check label distribution
        results['details']['label_distribution'] = self.check_label_distribution(dataset['y'])
        
        return results
    
    def plot_label_distribution(self, dataset: Dict, by_accident_type: bool = False) -> None:
        """
        Plot the distribution of labels in the dataset.
        
        Args:
            dataset (Dict): The processed dataset dictionary
            by_accident_type (bool): Whether to break down distribution by accident type
        """
        y = dataset['y']
        metadata = dataset['metadata']
        
        if not by_accident_type:
            # Overall distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(x=y)
            plt.title('Label Distribution')
            plt.xlabel('Label (1 = Scram in next 5 minutes)')
            plt.ylabel('Count')
            plt.xticks([0, 1], ['No Scram', 'Scram'])
            plt.show()
        else:
            # Distribution by accident type
            accident_types = [m['accident_type'] for m in metadata]
            accident_type_labels = [(t, int(y[i])) for i, t in enumerate(accident_types)]
            
            df = pd.DataFrame(accident_type_labels, columns=['accident_type', 'label'])
            
            plt.figure(figsize=(12, 8))
            sns.countplot(x='accident_type', hue='label', data=df)
            plt.title('Label Distribution by Accident Type')
            plt.xlabel('Accident Type')
            plt.ylabel('Count')
            plt.legend(['No Scram', 'Scram'])
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def validate_simulation(self, 
                           df: pd.DataFrame, 
                           scram_time: Optional[float] = None) -> Dict[str, Union[Dict, str]]:
        """
        Validate a single simulation.
        
        Args:
            df (pd.DataFrame): Simulation data
            scram_time (Optional[float]): Timestamp of Reactor Scram
            
        Returns:
            Dict: Dictionary with validation results
        """
        results = {
            'missing_values': self.check_missing_values(df),
            'consistency': self.check_data_consistency(df),
            'scram_time_valid': True,
        }
        
        # Check if scram time is within simulation time range
        if scram_time is not None:
            time_col = df.columns[0]
            min_time = df[time_col].min()
            max_time = df[time_col].max()
            
            if not (min_time <= scram_time <= max_time):
                results['scram_time_valid'] = False
                results['scram_time_message'] = f"Scram time {scram_time} is outside simulation time range [{min_time}, {max_time}]"
        
        return results
    
    def validate_dataset(self, dataset: Dict) -> Dict:
        """
        Validate the processed dataset.
        
        Args:
            dataset (Dict): The processed dataset dictionary
            
        Returns:
            Dict: Dictionary with validation results
        """
        structure_validation = self.check_dataset_structure(dataset)
        
        if not structure_validation['is_valid']:
            return structure_validation
        
        # Check for potential data leakage
        metadata = dataset['metadata']
        accident_sim_pairs = [(m['accident_type'], m['simulation']) for m in metadata]
        windows_per_simulation = Counter(accident_sim_pairs)
        
        # Check class balance by accident type
        accident_types = [m['accident_type'] for m in metadata]
        accident_type_counts = Counter(accident_types)
        
        y = dataset['y']
        positive_by_type = {}
        for i, m in enumerate(metadata):
            acc_type = m['accident_type']
            if acc_type not in positive_by_type:
                positive_by_type[acc_type] = 0
            positive_by_type[acc_type] += y[i]
        
        return {
            'structure': structure_validation,
            'windows_per_simulation': dict(windows_per_simulation),
            'samples_per_accident_type': dict(accident_type_counts),
            'positive_samples_by_type': positive_by_type
        }

# Example usage
if __name__ == "__main__":
    from data_loader import NPPADDataLoader
    from time_series_processor import TimeSeriesProcessor
    
    # Load a single simulation
    data_loader = NPPADDataLoader("NPPAD")
    accident_type = data_loader.accident_types[0]
    sim_num = list(data_loader.data_mapping[accident_type].keys())[0]
    df, scram_time = data_loader.load_simulation_data(accident_type, sim_num)
    
    # Validate raw data
    validator = DataValidator()
    validation_results = validator.validate_simulation(df, scram_time)
    print("Simulation Validation Results:")
    print(f"Missing values: {validation_results['missing_values']}")
    print(f"Consistency checks: {validation_results['consistency']}")
    
    # Process the simulation
    processor = TimeSeriesProcessor(window_size=30, prediction_horizon=30, stride=1)
    processed_data = processor.process_simulation(df, scram_time)
    
    # Check label distribution
    label_distribution = validator.check_label_distribution(processed_data['y'])
    print(f"\nLabel Distribution: {label_distribution}") 