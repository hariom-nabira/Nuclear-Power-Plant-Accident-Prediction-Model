import logging
from data_loader import DataLoader
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize DataLoader
        logger.info("Initializing DataLoader...")
        data_loader = DataLoader(
            data_root="NPPAD",  # Path to your NPPAD directory
            window_size=30,     # 5 minutes of data (30 * 10s intervals)
            stride=1            # 1 timestep overlap between windows
        )
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        X, y = data_loader.prepare_dataset()
        
        # Save processed data
        logger.info("Saving processed data...")
        np.save('processed_data.npy', X)
        np.save('labels.npy', y)
        
        logger.info(f"Successfully processed data:")
        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")
        logger.info(f"Number of positive samples: {np.sum(y)}")
        logger.info(f"Number of negative samples: {len(y) - np.sum(y)}")
        
    except Exception as e:
        logger.error(f"Error running data loader: {str(e)}")
        raise

if __name__ == "__main__":
    main() 