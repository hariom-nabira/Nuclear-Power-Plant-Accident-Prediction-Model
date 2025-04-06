import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_display_data():
    try:
        # Load the data
        print("\nLoading processed data...")
        data_dir = "processed_data"
        
        # Load train, val, test sets
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # Display basic information
        print("\nData Shapes:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        print("\nData Types:")
        print(f"X_train dtype: {X_train.dtype}")
        print(f"y_train dtype: {y_train.dtype}")
        
        # Display label distribution for each set
        print("\nLabel Distribution:")
        print("Training set:")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for val, count in zip(unique_train, counts_train):
            print(f"Label {val}: {count} samples ({count/len(y_train)*100:.2f}%)")
            
        print("\nValidation set:")
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        for val, count in zip(unique_val, counts_val):
            print(f"Label {val}: {count} samples ({count/len(y_val)*100:.2f}%)")
            
        print("\nTest set:")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        for val, count in zip(unique_test, counts_test):
            print(f"Label {val}: {count} samples ({count/len(y_test)*100:.2f}%)")
        
        # Display sample window from training set
        print("\nSample Window (First window from training set):")
        print(f"Shape: {X_train[0].shape}")
        print("\nFirst few values:")
        print(X_train[0][:5, :5])  # Show first 5 timesteps and 5 features
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Feature value distributions for each set
        plt.subplot(2, 2, 1)
        plt.hist(X_train.reshape(-1, X_train.shape[-1]).mean(axis=0), bins=50, alpha=0.5, label='Train')
        plt.hist(X_val.reshape(-1, X_val.shape[-1]).mean(axis=0), bins=50, alpha=0.5, label='Val')
        plt.hist(X_test.reshape(-1, X_test.shape[-1]).mean(axis=0), bins=50, alpha=0.5, label='Test')
        plt.title('Distribution of Feature Means')
        plt.xlabel('Mean Value')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot 2: Label distribution for training set
        plt.subplot(2, 2, 2)
        sns.countplot(x=y_train)
        plt.title('Label Distribution (Training)')
        plt.xlabel('Label')
        plt.ylabel('Count')
        
        # Plot 3: Label distribution for validation set
        plt.subplot(2, 2, 3)
        sns.countplot(x=y_val)
        plt.title('Label Distribution (Validation)')
        plt.xlabel('Label')
        plt.ylabel('Count')
        
        # Plot 4: Label distribution for test set
        plt.subplot(2, 2, 4)
        sns.countplot(x=y_test)
        plt.title('Label Distribution (Test)')
        plt.xlabel('Label')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('data_visualization.png')
        print("\nVisualization saved as 'data_visualization.png'")
        
        # Display feature statistics for training set
        print("\nFeature Statistics (Training Set):")
        feature_stats = pd.DataFrame(X_train.reshape(-1, X_train.shape[-1])).describe()
        print(feature_stats)
        
        # Calculate and display dataset split ratios
        total_samples = len(y_train) + len(y_val) + len(y_test)
        print("\nDataset Split Ratios:")
        print(f"Training: {len(y_train)/total_samples*100:.1f}%")
        print(f"Validation: {len(y_val)/total_samples*100:.1f}%")
        print(f"Test: {len(y_test)/total_samples*100:.1f}%")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find .npy files. Make sure the processed_data directory exists and contains the required files.")
        print(f"Missing file: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    load_and_display_data() 