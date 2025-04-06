import numpy as np
import pandas as pd
from data_loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_directory_structure():
    """Check and report the directory structure."""
    nppad_dir = Path("NPPAD")
    
    if not nppad_dir.exists():
        logger.error("NPPAD directory not found!")
        logger.info("Please create the following directory structure:")
        logger.info("""
NPPAD/
├── LOCA/
│   ├── 1.csv
│   ├── 1TransientReport.txt
│   └── ...
├── SGBTR/
│   ├── 1.csv
│   ├── 1TransientReport.txt
│   └── ...
└── ...
        """)
        return False
    
    # Check each accident type directory
    accident_types = ['LOCA']
    
    logger.info("\nChecking directory structure:")
    found_data = False
    for acc_type in accident_types:
        acc_dir = nppad_dir / acc_type
        if acc_dir.exists():
            # Count CSV and TXT files
            csv_files = list(acc_dir.glob("*.csv"))
            # Try different patterns for TransientReport files
            txt_patterns = [
                "*TransientReport.txt",
                "*Transient Report.txt",
                "*TransientReport*.txt",
                "*Transient Report*.txt"
            ]
            txt_files = []
            for pattern in txt_patterns:
                txt_files.extend(list(acc_dir.glob(pattern)))
            
            logger.info(f"\n{acc_type}:")
            logger.info(f"  - Directory exists: Yes")
            logger.info(f"  - Number of CSV files: {len(csv_files)}")
            logger.info(f"  - Number of TransientReport files: {len(txt_files)}")
            
            if csv_files and txt_files:
                found_data = True
                # Check first file pair
                first_csv = csv_files[0]
                first_txt = txt_files[0]
                
                try:
                    df = pd.read_csv(first_csv)
                    logger.info(f"  - First CSV file ({first_csv.name}) shape: {df.shape}")
                    logger.info(f"  - CSV columns: {list(df.columns)[:5]}...")  # Show first 5 columns
                except Exception as e:
                    logger.error(f"  - Error reading first CSV file: {str(e)}")
                
                try:
                    with open(first_txt, 'r') as f:
                        content = f.read()
                    logger.info(f"  - First TransientReport file ({first_txt.name}) size: {len(content)} bytes")
                    # Show first few lines of the report
                    first_lines = content.split('\n')[:3]
                    logger.info(f"  - First few lines of report:\n    " + "\n    ".join(first_lines))
                except Exception as e:
                    logger.error(f"  - Error reading first TransientReport file: {str(e)}")
            else:
                logger.warning(f"  - Missing required files (CSV or TransientReport)")
                if csv_files:
                    logger.info(f"  - Found CSV files: {[f.name for f in csv_files[:3]]}...")
                if txt_files:
                    logger.info(f"  - Found TransientReport files: {[f.name for f in txt_files[:3]]}...")
                # List all files in directory to help debug
                all_files = list(acc_dir.glob("*"))
                logger.info(f"  - All files in directory: {[f.name for f in all_files[:5]]}...")
        else:
            logger.warning(f"\n{acc_type}: Directory not found")
    
    if not found_data:
        logger.error("\nNo valid data files found in any accident type directory!")
        return False
    
    return True

def analyze_data_distribution(y):
    """Analyze and plot the distribution of labels."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    logger.info("\nLabel Distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / total) * 100
        logger.info(f"Label {label}: {count} samples ({percentage:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(8, 6))
    plt.bar(unique, counts)
    plt.title("Distribution of Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("label_distribution.png")
    plt.close()

def analyze_feature_statistics(X):
    """Analyze and plot statistics of features."""
    n_features = X.shape[2]
    means = np.mean(X, axis=(0, 1))  # Mean across all windows and timesteps
    stds = np.std(X, axis=(0, 1))    # Std across all windows and timesteps
    
    logger.info("\nFeature Statistics:")
    logger.info(f"Number of features: {n_features}")
    logger.info(f"Mean of means: {np.mean(means):.4f}")
    logger.info(f"Mean of stds: {np.mean(stds):.4f}")
    
    # Plot feature statistics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(means)
    plt.title("Feature Means")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Value")
    
    plt.subplot(1, 2, 2)
    plt.plot(stds)
    plt.title("Feature Standard Deviations")
    plt.xlabel("Feature Index")
    plt.ylabel("Standard Deviation")
    
    plt.tight_layout()
    plt.savefig("feature_statistics.png")
    plt.close()

def main():
    # Create processed_data directory if it doesn't exist
    os.makedirs("processed_data", exist_ok=True)
    
    # Check directory structure first
    if not check_directory_structure():
        return
    
    # Initialize data loader
    data_loader = DataLoader(data_root="NPPAD")
    
    try:
        # Prepare dataset
        logger.info("\nLoading and preparing dataset...")
        X, y = data_loader.prepare_dataset()
        
        # Print dataset information
        logger.info("\nDataset Information:")
        logger.info(f"Total number of windows: {len(X)}")
        logger.info(f"Window shape: {X.shape}")
        logger.info(f"Number of features: {X.shape[2]}")
        
        # Analyze data distribution
        analyze_data_distribution(y)
        
        # Analyze feature statistics
        analyze_feature_statistics(X)
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info("\nData Split:")
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Save processed data
        np.save("processed_data/X_train.npy", X_train)
        np.save("processed_data/X_val.npy", X_val)
        np.save("processed_data/X_test.npy", X_test)
        np.save("processed_data/y_train.npy", y_train)
        np.save("processed_data/y_val.npy", y_val)
        np.save("processed_data/y_test.npy", y_test)
        
        logger.info("\nData preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()