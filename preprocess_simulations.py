import os
import re
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

def find_accident_timestamp(report_file):
    """
    Parse the TransientReport file to find the timestamp of an accident (Reactor Scram or Core Meltdown)
    Returns the timestamp in seconds or None if no accident found
    """
    try:
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Look for Reactor Scram or Core Meltdown events
        scram_match = re.search(r'(\d+\.?\d*)\s*sec,\s*Reactor\s*Scram', content)
        meltdown_match = re.search(r'(\d+\.?\d*)\s*sec,\s*Core\s*Meltdown', content)
        
        # Use the first match found (if any)
        if scram_match:
            return float(scram_match.group(1))
        elif meltdown_match:
            return float(meltdown_match.group(1))
        else:
            return None
    except Exception as e:
        print(f"Error processing {report_file}: {e}")
        return None

def add_labels_to_csv(csv_file, accident_time):
    """
    Add a 'label' column to the CSV file based on the accident timestamp
    0 = normal operation (before accident_time - 180 seconds)
    1 = potential accident (after that point)
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Ensure TIME column exists
        if 'TIME' not in df.columns:
            print(f"Warning: TIME column not found in {csv_file}")
            return False
        
        # Define threshold time (accident time minus 180 seconds)
        threshold_time = max(0, accident_time - 180)
        
        # Add label column (0 for normal, 1 for potential accident)
        df['label'] = np.where(df['TIME'] < threshold_time, 0, 1)
        
        # Save the modified CSV
        df.to_csv(csv_file, index=False)
        return True
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return False

def align_csv_dimensions(csv_files):
    """
    Align dimensions of all CSV files by finding the common columns
    and ensuring all files have the same columns
    """
    try:
        # Find common columns across all files
        all_columns = set()
        common_columns = None
        
        # First pass: identify common columns
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if common_columns is None:
                    common_columns = set(df.columns)
                else:
                    common_columns &= set(df.columns)
                all_columns |= set(df.columns)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        if not common_columns:
            print("No common columns found across files")
            return
        
        # Make sure 'TIME' and 'label' are included
        required_columns = {'TIME', 'label'}
        final_columns = list(common_columns | required_columns)
        
        # Second pass: standardize columns
        for csv_file in tqdm(csv_files, desc="Aligning CSV dimensions"):
            try:
                df = pd.read_csv(csv_file)
                
                # Ensure all required columns exist
                for col in required_columns:
                    if col not in df.columns and col != 'label':  # 'label' might be added later
                        print(f"Warning: Required column {col} missing in {csv_file}")
                
                # Add missing columns with NaN values
                for col in final_columns:
                    if col not in df.columns and col != 'label':
                        df[col] = np.nan
                
                # Select only the final columns and save
                df = df[final_columns]
                df.to_csv(csv_file, index=False)
            except Exception as e:
                print(f"Error standardizing {csv_file}: {e}")
    except Exception as e:
        print(f"Error in align_csv_dimensions: {e}")

def process_all_simulations(root_dir):
    """
    Process all simulations in the given directory structure
    """
    # Find all simulation folders
    simulation_dirs = []
    for dirpath, _, _ in os.walk(root_dir):
        if os.path.basename(dirpath) != "NPPAD":  # Skip the root NPPAD directory
            simulation_dirs.append(dirpath)
    
    if not simulation_dirs:
        simulation_dirs = [root_dir]  # Use root directory if no subdirectories found
    
    total_processed = 0
    total_labeled = 0
    
    # Process each simulation directory
    for sim_dir in simulation_dirs:
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(sim_dir, "*.csv"))
        
        print(f"Processing {len(csv_files)} simulations in {sim_dir}")
        
        for csv_file in tqdm(csv_files, desc=f"Processing {os.path.basename(sim_dir)}"):
            # Construct the corresponding TransientReport filename
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            report_file = os.path.join(sim_dir, f"{base_name}Transient Report.txt")
            
            # Check if the report file exists
            if not os.path.exists(report_file):
                print(f"Warning: No TransientReport file found for {csv_file}")
                continue
            
            # Find accident timestamp
            accident_time = find_accident_timestamp(report_file)
            
            if accident_time is not None:
                # Add labels to the CSV
                success = add_labels_to_csv(csv_file, accident_time)
                if success:
                    total_labeled += 1
            else:
                print(f"No accident found in {report_file}")
            
            total_processed += 1
    
    print(f"Total processed: {total_processed}, Successfully labeled: {total_labeled}")
    
    # After labeling, align dimensions of all CSV files
    all_csv_files = []
    for sim_dir in simulation_dirs:
        all_csv_files.extend(glob.glob(os.path.join(sim_dir, "*.csv")))
    
    print(f"Aligning dimensions for {len(all_csv_files)} CSV files")
    align_csv_dimensions(all_csv_files)

if __name__ == "__main__":
    # Specify the root directory containing all simulations
    root_dir = os.path.join(os.getcwd(), "NPPAD")
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} not found")
    else:
        print(f"Starting preprocessing of simulations in {root_dir}")
        process_all_simulations(root_dir)
        print("Preprocessing complete!") 