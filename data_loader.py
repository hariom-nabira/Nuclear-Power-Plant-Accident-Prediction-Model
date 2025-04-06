import os
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union

class NPPADDataLoader:
    """
    Data loader for the Nuclear Power Plant Accident Data (NPPAD) dataset.
    This class handles loading CSV files and Transient Reports, and extracts
    Reactor Scram timestamps.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the NPPAD dataset directory
        """
        self.data_dir = data_dir
        self.accident_types = self._get_accident_types()
        self.data_mapping = self._create_data_mapping()
        
    def _get_accident_types(self) -> List[str]:
        """
        Get all accident types from the dataset directory.
        
        Returns:
            List[str]: List of accident type names
        """
        return [d for d in os.listdir(self.data_dir) 
                if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
    
    def _create_data_mapping(self) -> Dict[str, Dict[int, Dict[str, str]]]:
        """
        Create a mapping of all data files for each accident type and simulation.
        
        Returns:
            Dict: Nested dictionary with structure:
                  {accident_type: {simulation_number: {'csv': csv_path, 'report': report_path}}}
        """
        data_mapping = {}
        
        for accident_type in self.accident_types:
            accident_dir = os.path.join(self.data_dir, accident_type)
            data_mapping[accident_type] = {}
            
            # Get all CSV files
            csv_files = [f for f in os.listdir(accident_dir) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                # Extract simulation number
                sim_num = int(re.search(r'(\d+)\.csv', csv_file).group(1))
                
                # Find corresponding transient report
                report_file = f"{sim_num}Transient Report.txt"
                report_path = os.path.join(accident_dir, report_file)
                
                if os.path.exists(report_path):
                    data_mapping[accident_type][sim_num] = {
                        'csv': os.path.join(accident_dir, csv_file),
                        'report': report_path
                    }
        
        return data_mapping
    
    def extract_scram_timestamp(self, report_path: str) -> Optional[float]:
        """
        Extract the Reactor Scram timestamp from a Transient Report.
        
        Args:
            report_path (str): Path to the Transient Report file
            
        Returns:
            Optional[float]: Timestamp of Reactor Scram in seconds, or None if not found
        """
        try:
            with open(report_path, 'r') as f:
                for line in f:
                    # Look for Reactor Scram in the report
                    if 'Reactor Scram' in line:
                        # Extract time (handle different possible formats)
                        time_match = re.search(r'(\d+(?:\.\d+)?)\s*sec', line)
                        if time_match:
                            return float(time_match.group(1))
            return None
        except Exception as e:
            print(f"Error extracting Scram timestamp from {report_path}: {e}")
            return None
    
    def load_simulation_data(self, accident_type: str, sim_num: int) -> Tuple[pd.DataFrame, Optional[float]]:
        """
        Load a specific simulation's data and extract Scram timestamp.
        
        Args:
            accident_type (str): Type of accident (e.g., 'LOCA')
            sim_num (int): Simulation number
            
        Returns:
            Tuple[pd.DataFrame, Optional[float]]: 
                - DataFrame with time series data
                - Reactor Scram timestamp or None if not found
        """
        if accident_type not in self.data_mapping or sim_num not in self.data_mapping[accident_type]:
            raise ValueError(f"Data for accident type {accident_type}, simulation {sim_num} not found")
        
        file_paths = self.data_mapping[accident_type][sim_num]
        
        # Load CSV data
        df = pd.read_csv(file_paths['csv'])
        
        # Extract Scram timestamp
        scram_time = self.extract_scram_timestamp(file_paths['report'])
        
        return df, scram_time
    
    def get_all_simulation_info(self) -> pd.DataFrame:
        """
        Create a DataFrame with information about all simulations including their Scram times.
        
        Returns:
            pd.DataFrame: DataFrame with columns for accident type, simulation number,
                         severity (based on sim_num), and Scram timestamp
        """
        data = []
        
        for accident_type in self.accident_types:
            for sim_num in self.data_mapping[accident_type]:
                report_path = self.data_mapping[accident_type][sim_num]['report']
                scram_time = self.extract_scram_timestamp(report_path)
                
                data.append({
                    'accident_type': accident_type,
                    'simulation_number': sim_num,
                    'severity': sim_num,  # Assuming sim_num reflects severity
                    'scram_time': scram_time
                })
        
        return pd.DataFrame(data)
    
    def load_all_simulations(self) -> Dict[str, Dict[int, Tuple[pd.DataFrame, Optional[float]]]]:
        """
        Load all simulations data and their Scram timestamps.
        Warning: This can be memory-intensive.
        
        Returns:
            Dict: Nested dictionary with structure:
                  {accident_type: {simulation_number: (dataframe, scram_time)}}
        """
        all_data = {}
        
        for accident_type in self.accident_types:
            all_data[accident_type] = {}
            for sim_num in self.data_mapping[accident_type]:
                df, scram_time = self.load_simulation_data(accident_type, sim_num)
                all_data[accident_type][sim_num] = (df, scram_time)
        
        return all_data
    
    def get_dataset_statistics(self) -> Dict[str, Union[int, Dict]]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dict: Dictionary with statistics about the dataset
        """
        stats = {
            'num_accident_types': len(self.accident_types),
            'accident_types': self.accident_types,
            'simulations_per_type': {},
            'simulations_with_scram': 0,
            'total_simulations': 0
        }
        
        sim_info = self.get_all_simulation_info()
        stats['total_simulations'] = len(sim_info)
        stats['simulations_with_scram'] = sum(~sim_info['scram_time'].isna())
        
        for accident_type in self.accident_types:
            type_sims = sim_info[sim_info['accident_type'] == accident_type]
            stats['simulations_per_type'][accident_type] = {
                'total': len(type_sims),
                'with_scram': sum(~type_sims['scram_time'].isna()),
                'severity_range': (type_sims['severity'].min(), type_sims['severity'].max())
            }
            
        return stats

# Example usage
if __name__ == "__main__":
    data_loader = NPPADDataLoader("NPPAD")
    sim_info = data_loader.get_all_simulation_info()
    print(f"Found {len(sim_info)} simulations across {len(data_loader.accident_types)} accident types")
    
    # Print statistics
    stats = data_loader.get_dataset_statistics()
    print(f"Dataset Statistics: {stats}")
    
    # Load data for a specific simulation
    accident_type = data_loader.accident_types[0]
    sim_num = list(data_loader.data_mapping[accident_type].keys())[0]
    df, scram_time = data_loader.load_simulation_data(accident_type, sim_num)
    
    print(f"\nSample data for {accident_type}, simulation {sim_num}:")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}...")
    print(f"Scram time: {scram_time} seconds") 