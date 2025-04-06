import os
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd

from model import NPPADModel

def load_dataset(data_dir, batch_size=32, num_workers=4):
    """
    Load the preprocessed datasets.
    
    Args:
        data_dir: Directory with preprocessed data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with dataloaders and dataset information
    """
    print(f"Loading datasets from {data_dir}...")
    
    # Load feature names
    try:
        with open(os.path.join(data_dir, 'feature_names.json'), 'r') as f:
            feature_names = json.load(f)
    except FileNotFoundError:
        print("Warning: feature_names.json not found")
        feature_names = None
    
    # Load training data
    try:
        with open(os.path.join(data_dir, 'train_dataset.pkl'), 'rb') as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Training dataset not found in {data_dir}")
    
    # Load validation data
    try:
        with open(os.path.join(data_dir, 'val_dataset.pkl'), 'rb') as f:
            val_data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Validation dataset not found in {data_dir}")
    
    # Load test data
    try:
        with open(os.path.join(data_dir, 'test_dataset.pkl'), 'rb') as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test dataset not found in {data_dir}")
    
    # Load processing parameters
    try:
        with open(os.path.join(data_dir, 'processing_params.json'), 'r') as f:
            processing_params = json.load(f)
    except FileNotFoundError:
        print("Warning: processing_params.json not found")
        processing_params = {}
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(train_data['X'], dtype=torch.float32),
        torch.tensor(train_data['y'], dtype=torch.float32)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(val_data['X'], dtype=torch.float32),
        torch.tensor(val_data['y'], dtype=torch.float32)
    )
    
    test_dataset = TensorDataset(
        torch.tensor(test_data['X'], dtype=torch.float32),
        torch.tensor(test_data['y'], dtype=torch.float32)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset stats
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of features: {train_data['X'].shape[2]}")
    print(f"Sequence length: {train_data['X'].shape[1]}")
    
    # Positive rate
    train_pos_rate = np.mean(train_data['y'])
    val_pos_rate = np.mean(val_data['y'])
    test_pos_rate = np.mean(test_data['y'])
    
    print(f"Positive rate - Train: {train_pos_rate:.2%}, Val: {val_pos_rate:.2%}, Test: {test_pos_rate:.2%}")
    
    # Return data
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_names': feature_names,
        'processing_params': processing_params,
        'train_metadata': train_data.get('metadata', None),
        'val_metadata': val_data.get('metadata', None),
        'test_metadata': test_data.get('metadata', None),
        'num_features': train_data['X'].shape[2],
        'seq_len': train_data['X'].shape[1],
        'pos_rates': {
            'train': train_pos_rate,
            'val': val_pos_rate,
            'test': test_pos_rate
        }
    }

def train_model(model, train_loader, val_loader, config):
    """
    Train the model.
    
    Args:
        model: NPPADModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        
    Returns:
        Dictionary with training history
    """
    # Get parameters from config
    num_epochs = config.get('num_epochs', 50)
    learning_rate = config.get('learning_rate', 0.001)
    patience = config.get('patience', 10)
    min_delta = config.get('min_delta', 0.001)
    model_dir = config.get('model_dir', 'models')
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.model.train()
        train_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_progress):
            # Perform training step
            loss = model.train_step(data, target)
            train_loss += loss
            
            # Update progress bar
            train_progress.set_postfix({'loss': f'{loss:.4f}'})
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader)
        val_loss = val_metrics['loss']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['learning_rate'].append(model.optimizer.param_groups[0]['lr'])
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_metrics['accuracy']:.4f} - "
              f"Val Precision: {val_metrics['precision']:.4f} - "
              f"Val Recall: {val_metrics['recall']:.4f} - "
              f"Val F1: {val_metrics['f1_score']:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss - min_delta:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            model.save_weights(best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save history
        with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    # Load best model for evaluation
    print(f"Loading best model from epoch {best_epoch+1}")
    model.load_weights(best_model_path)
    
    return history

def evaluate_model(model, data_loader, threshold=0.5):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: NPPADModel instance
        data_loader: DataLoader for evaluation
        threshold: Decision threshold for binary classification
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set model to evaluation mode
    model.model.eval()
    
    # Collect all predictions and targets
    all_probs = []
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Evaluating'):
            # Move data to device
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)
                
            data = data.to(model.config['device'])
            target = target.to(model.config['device'])
            
            # Ensure correct shape
            if target.dim() == 1:
                target = target.unsqueeze(1)
            
            # Forward pass
            output = model.model(data)
            
            # Calculate loss
            loss = model.criterion(output, target)
            total_loss += loss.item()
            
            # Make predictions
            probs = output.cpu().numpy()
            preds = (probs >= threshold).astype(np.float32)
            targets = target.cpu().numpy()
            
            # Collect results
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_targets)
    
    # Confusion matrix values
    tp = np.sum((all_targets == 1) & (all_preds == 1))
    fp = np.sum((all_targets == 0) & (all_preds == 1))
    tn = np.sum((all_targets == 0) & (all_preds == 0))
    fn = np.sum((all_targets == 1) & (all_preds == 0))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
    
    # Calculate AUC and AUPRC
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    # Return metrics
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        },
        'pr_curve': {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist()
        },
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'targets': all_targets.tolist()
    }

def plot_learning_curves(history, output_dir='plots'):
    """
    Plot training and validation learning curves.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation metrics
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()
    
    print(f"Learning curves saved to {os.path.join(output_dir, 'learning_curves.png')}")

def plot_roc_pr_curves(evaluation, output_dir='plots'):
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        evaluation: Dictionary with evaluation results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ROC curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(evaluation['roc_curve']['fpr'], evaluation['roc_curve']['tpr'], 
             label=f'ROC curve (AUC = {evaluation["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(evaluation['pr_curve']['recall'], evaluation['pr_curve']['precision'],
             label=f'PR curve (AUC = {evaluation["pr_auc"]:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves.png'))
    plt.close()
    
    print(f"ROC and PR curves saved to {os.path.join(output_dir, 'roc_pr_curves.png')}")

def train_and_evaluate(data_dir, config):
    """
    Train and evaluate the model.
    
    Args:
        data_dir: Directory with preprocessed data
        config: Configuration dictionary
        
    Returns:
        Dictionary with training results
    """
    # Load datasets
    data = load_dataset(data_dir, batch_size=config.get('batch_size', 32))
    
    # Update configuration with dataset information
    config['input_size'] = data['num_features']
    config['seq_len'] = data['seq_len']
    
    # Initialize model
    model = NPPADModel(config)
    
    # Create output directories
    model_dir = config.get('model_dir', 'models')
    plot_dir = config.get('plot_dir', 'plots')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train model
    print("\nStarting model training...")
    history = train_model(model, data['train_loader'], data['val_loader'], config)
    
    # Plot learning curves
    plot_learning_curves(history, plot_dir)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_evaluation = evaluate_model(model, data['test_loader'])
    
    # Print test results
    print(f"\nTest Results:")
    print(f"Loss: {test_evaluation['loss']:.4f}")
    print(f"Accuracy: {test_evaluation['accuracy']:.4f}")
    print(f"Precision: {test_evaluation['precision']:.4f}")
    print(f"Recall: {test_evaluation['recall']:.4f}")
    print(f"F1 Score: {test_evaluation['f1_score']:.4f}")
    print(f"ROC AUC: {test_evaluation['roc_auc']:.4f}")
    print(f"PR AUC: {test_evaluation['pr_auc']:.4f}")
    
    # Plot ROC and PR curves
    plot_roc_pr_curves(test_evaluation, plot_dir)
    
    # Save test evaluation results
    with open(os.path.join(model_dir, 'test_evaluation.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key in ['roc_curve', 'pr_curve']:
            for subkey in test_evaluation[key]:
                if isinstance(test_evaluation[key][subkey], np.ndarray):
                    test_evaluation[key][subkey] = test_evaluation[key][subkey].tolist()
                    
        json.dump(test_evaluation, f, indent=2)
    
    # Save a copy of the best model as final model
    model.save_weights(os.path.join(model_dir, 'final_model.pt'))
    
    return {
        'history': history,
        'test_evaluation': test_evaluation,
        'config': config,
        'model': model
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train NPPAD Reactor Scram prediction model')
    parser.add_argument('--data_dir', type=str, default='processed_data', 
                        help='Directory with preprocessed data')
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='Directory to save models')
    parser.add_argument('--plot_dir', type=str, default='plots', 
                        help='Directory to save plots')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience for early stopping')
    parser.add_argument('--tcn_channels', type=str, default='64,128,256,128', 
                        help='Comma-separated list of TCN channel sizes')
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help='Kernel size for TCN')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout rate')
    parser.add_argument('--attention_size', type=int, default=128, 
                        help='Dimension of attention mechanism')
    parser.add_argument('--num_heads', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create configuration
    config = {
        'model_dir': args.model_dir,
        'plot_dir': args.plot_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'num_channels': [int(c) for c in args.tcn_channels.split(',')],
        'kernel_size': args.kernel_size,
        'dropout': args.dropout,
        'attention_size': args.attention_size,
        'num_heads': args.num_heads,
        'device': 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda',
        'seed': args.seed
    }
    
    print(f"Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train and evaluate model
    results = train_and_evaluate(args.data_dir, config)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 