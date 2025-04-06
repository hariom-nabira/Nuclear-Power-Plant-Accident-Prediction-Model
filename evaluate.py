import os
import argparse
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from model import NPPADModel
from train import load_dataset, evaluate_model, plot_roc_pr_curves

def plot_confusion_matrix(y_true, y_pred, output_dir='plots'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Scram', 'Scram'],
                yticklabels=['No Scram', 'Scram'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def plot_score_distribution(y_true, y_scores, output_dir='plots'):
    """
    Plot the distribution of prediction scores for each class.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Separate scores by true class
    scores_positive = y_scores[y_true == 1]
    scores_negative = y_scores[y_true == 0]
    
    # Plot histograms
    plt.hist(scores_negative, bins=20, alpha=0.5, label='No Scram', color='blue', density=True)
    plt.hist(scores_positive, bins=20, alpha=0.5, label='Scram', color='red', density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Prediction Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
    plt.close()
    
    print(f"Score distribution saved to {os.path.join(output_dir, 'score_distribution.png')}")

def plot_threshold_impact(y_true, y_scores, output_dir='plots'):
    """
    Plot the impact of different threshold values on model metrics.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate thresholds
    thresholds = np.arange(0.05, 0.95, 0.05)
    
    # Initialize metrics
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Confusion matrix elements
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Metrics
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracy, label='Accuracy', marker='o')
    plt.plot(thresholds, precision, label='Precision', marker='s')
    plt.plot(thresholds, recall, label='Recall', marker='^')
    plt.plot(thresholds, f1, label='F1 Score', marker='d')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Threshold on Model Metrics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_impact.png'))
    plt.close()
    
    print(f"Threshold impact analysis saved to {os.path.join(output_dir, 'threshold_impact.png')}")
    
    # Find optimal threshold for F1
    optimal_idx = np.argmax(f1)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold for F1 score: {optimal_threshold:.2f}")
    print(f"Metrics at optimal threshold:")
    print(f"  Accuracy: {accuracy[optimal_idx]:.4f}")
    print(f"  Precision: {precision[optimal_idx]:.4f}")
    print(f"  Recall: {recall[optimal_idx]:.4f}")
    print(f"  F1 Score: {f1[optimal_idx]:.4f}")
    
    return optimal_threshold

def plot_time_to_event(metadata, y_scores, output_dir='plots'):
    """
    Analyze prediction scores as a function of time to event.
    
    Args:
        metadata: Metadata with time information
        y_scores: Prediction scores
        output_dir: Directory to save plots
    """
    if metadata is None or 'time_to_event' not in metadata:
        print("Warning: No time-to-event data available for plotting")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract time to event
    time_to_event = metadata['time_to_event']
    
    # Group by time bins
    time_bins = np.arange(-300, 10, 10)  # 10-second bins from -300 to 0
    bin_indices = np.digitize(time_to_event, time_bins)
    
    # Calculate average score per bin
    bin_scores = []
    bin_stds = []
    valid_bins = []
    
    for i in range(1, len(time_bins)):
        bin_mask = (bin_indices == i)
        if np.sum(bin_mask) > 0:
            bin_scores.append(np.mean(y_scores[bin_mask]))
            bin_stds.append(np.std(y_scores[bin_mask]))
            valid_bins.append(time_bins[i-1])
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.errorbar(valid_bins, bin_scores, yerr=bin_stds, fmt='o-', capsize=5)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.xlabel('Time to Event (seconds)')
    plt.ylabel('Average Prediction Score')
    plt.title('Prediction Score vs. Time to Scram Event')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_to_event.png'))
    plt.close()
    
    print(f"Time-to-event analysis saved to {os.path.join(output_dir, 'time_to_event.png')}")

def evaluate_saved_model(model_path, data_dir, output_dir='evaluation'):
    """
    Evaluate a saved model on test data.
    
    Args:
        model_path: Path to saved model weights
        data_dir: Directory with preprocessed data
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configuration
    try:
        model_dir = os.path.dirname(model_path)
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Model config not found, using default configuration")
        config = {}
    
    # Load dataset
    data = load_dataset(data_dir, batch_size=config.get('batch_size', 32))
    
    # Update config with dataset information
    config['input_size'] = data['num_features']
    config['seq_len'] = data['seq_len']
    
    # Initialize model
    model = NPPADModel(config)
    
    # Load weights
    try:
        model.load_weights(model_path)
        print(f"Loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_evaluation = evaluate_model(model, data['test_loader'])
    
    # Save evaluation results
    evaluation_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(evaluation_path, 'w') as f:
        for key in ['roc_curve', 'pr_curve']:
            for subkey in test_evaluation[key]:
                if isinstance(test_evaluation[key][subkey], np.ndarray):
                    test_evaluation[key][subkey] = test_evaluation[key][subkey].tolist()
        json.dump(test_evaluation, f, indent=2)
    
    print(f"Saved evaluation results to {evaluation_path}")
    
    # Print test results
    print("\nTest Results:")
    print(f"Loss: {test_evaluation['loss']:.4f}")
    print(f"Accuracy: {test_evaluation['accuracy']:.4f}")
    print(f"Precision: {test_evaluation['precision']:.4f}")
    print(f"Recall: {test_evaluation['recall']:.4f}")
    print(f"F1 Score: {test_evaluation['f1_score']:.4f}")
    print(f"ROC AUC: {test_evaluation['roc_auc']:.4f}")
    print(f"PR AUC: {test_evaluation['pr_auc']:.4f}")
    
    # Generate classification report
    y_true = test_evaluation['targets']
    y_pred = test_evaluation['predictions']
    report = classification_report(y_true, y_pred, target_names=['No Scram', 'Scram'])
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate plots
    plot_roc_pr_curves(test_evaluation, output_dir)
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_score_distribution(y_true, test_evaluation['probabilities'], output_dir)
    optimal_threshold = plot_threshold_impact(y_true, test_evaluation['probabilities'], output_dir)
    
    # Time-to-event analysis if metadata is available
    if data['test_metadata'] is not None:
        plot_time_to_event(data['test_metadata'], test_evaluation['probabilities'], output_dir)
    
    # Save optimal threshold
    with open(os.path.join(output_dir, 'optimal_threshold.json'), 'w') as f:
        json.dump({'optimal_threshold': float(optimal_threshold)}, f, indent=2)
    
    return test_evaluation

def main():
    parser = argparse.ArgumentParser(description='Evaluate NPPAD model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model weights')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluate_saved_model(args.model_path, args.data_dir, args.output_dir)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main() 