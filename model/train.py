import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging
from tcn_attention import create_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, _ = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return total_loss / len(train_loader), accuracy, precision, recall, f1
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs, _ = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                all_preds.extend((outputs > 0.5).float().cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        
        return total_loss / len(val_loader), accuracy, precision, recall, f1, auc

def load_data(data_dir="processed_data"):
    """Load the processed data from .npy files."""
    logger.info("Loading data...")
    
    # Load train, val, test sets
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    input_size = X_train.shape[2]  # Number of features
    model = create_model(
        input_size=input_size,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2
    ).to(device)
    
    # Create trainer
    trainer = ModelTrainer(model, device)
    
    # Training loop
    num_epochs = 50
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = trainer.validate(val_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'model/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('model/best_model.pth'))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = trainer.validate(test_loader)
    
    logger.info("\nTest Results:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_prec:.4f}")
    logger.info(f"Test Recall: {test_rec:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main() 