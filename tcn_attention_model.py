import os
import numpy as np
import pandas as pd
import gc  # For garbage collection

# Try to use GPU but with memory growth to avoid allocating all memory at once
import tensorflow as tf

print("Configuring TensorFlow...")
# Memory growth needs to be set before GPUs are initialized
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Found {len(physical_devices)} GPUs")
        for device in physical_devices:
            # Don't pre-allocate memory; allocate as-needed
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        
        # Only use first GPU to avoid issues
        if len(physical_devices) > 1:
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            print(f"Using only GPU: {physical_devices[0]}")
        
        # Limit memory usage to a percentage of GPU memory (adjust as needed)
        # This prevents TensorFlow from using all GPU memory
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.compat.v1.keras.backend.set_session(sess)
        print("GPU memory limited to 90%")
    else:
        print("No GPUs found, using CPU")
except Exception as e:
    print(f"Error configuring GPU: {e}")
    print("Falling back to CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU if GPU config fails

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Activation, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Reshape, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from collections import Counter
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
CONFIG = {
    'data_dir': 'NPPAD',                 # Directory containing processed CSV files
    'sequence_length': 18,               # 3 minutes of history (10sec intervals = 18 points)
    'prediction_horizon': 1,             # Binary prediction (will accident happen in next 180s)
    'k_folds': 3,                        # Number of folds for cross-validation
    'batch_size': 64,                    # Batch size for training
    'epochs': 10,                        # Maximum number of epochs (reduced to 20)
    'patience': 5,                       # Early stopping patience 
    'tcn_filters': [64, 128, 128],       # Filters for TCN layers
    'tcn_kernel_size': 3,                # Kernel size for TCN
    'tcn_dilations': [1, 2, 4, 8],       # Dilation rates for TCN
    'attention_heads': 4,                # Number of attention heads
    'dropout_rate': 0.3,                 # Dropout rate
    'learning_rate': 0.001,              # Learning rate
    'test_size': 0.2,                    # Proportion of data for testing
    'val_size': 0.2,                     # Proportion of training data for validation
    'model_dir': 'models',               # Directory to save models
    'results_dir': 'results',            # Directory to save results
    'class_weight': {0: 1, 1: 2},        # Weight for handling class imbalance
    'use_gpu': True,                     # Using GPU with memory management
    'sample_size': 100,                  # Process only 200 files for testing
    'verbose': 1,                        # Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    'data_chunk_size': 10000             # Process data in chunks to avoid memory issues
}

def create_directories():
    """Create necessary directories for saving models and results"""
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(CONFIG['results_dir'], 'figures'), exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess all CSV data from the NPPAD directory"""
    print("Loading and preprocessing data...")
    
    # Find all CSV files
    all_files = []
    for root, _, _ in os.walk(CONFIG['data_dir']):
        files = glob.glob(os.path.join(root, '*.csv'))
        all_files.extend(files)
    
    print(f"Found {len(all_files)} CSV files")
    
    # Load a small sample to determine feature dimensionality
    sample_df = pd.read_csv(all_files[0])
    
    # Skip non-feature columns
    non_feature_cols = ['TIME', 'label', 'accident_timestamp', 'accident_type']
    feature_cols = [col for col in sample_df.columns if col not in non_feature_cols]
    
    print(f"Found {len(feature_cols)} feature columns")
    
    # Load data in chunks to manage memory
    all_sequences = []
    all_labels = []
    accident_types = []
    
    # Limit the number of files for debugging if sample_size is set
    if CONFIG['sample_size'] is not None:
        print(f"Using a sample of {CONFIG['sample_size']} files for testing")
        all_files = all_files[:CONFIG['sample_size']]
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            
            # Skip files with too few rows
            if len(df) < CONFIG['sequence_length'] + CONFIG['prediction_horizon']:
                continue
                
            # Extract features and labels
            features = df[feature_cols].values
            times = df['TIME'].values
            labels = df['label'].values
            
            # Record accident types for analysis
            if 1 in labels:
                accident_type = df['accident_type'].iloc[0]
                if isinstance(accident_type, str):
                    accident_types.append(accident_type)
            
            # Create sequences with sliding window
            for i in range(len(df) - CONFIG['sequence_length'] - CONFIG['prediction_horizon'] + 1):
                # Ensure we're using 10-second intervals (check TIME column)
                if i > 0 and abs(times[i] - times[i-1] - 10.0) > 1e-5:
                    continue
                
                seq = features[i:i+CONFIG['sequence_length']]
                
                # Label is 1 if any point in the prediction horizon has label 1
                target_labels = labels[i+CONFIG['sequence_length']:
                                       i+CONFIG['sequence_length']+CONFIG['prediction_horizon']]
                target = 1 if 1 in target_labels else 0
                
                all_sequences.append(seq)
                all_labels.append(target)
                
                # Process in chunks to avoid memory issues
                if len(all_sequences) >= CONFIG['data_chunk_size']:
                    print(f"Processed {len(all_sequences)} sequences so far...")
                    chunk_X = np.array(all_sequences)
                    chunk_y = np.array(all_labels)
                    yield chunk_X, chunk_y, feature_cols, False
                    all_sequences = []
                    all_labels = []
                    gc.collect()  # Force garbage collection
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Convert remaining sequences to numpy arrays
    if all_sequences:
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        print(f"Created {len(X)} sequences in final chunk")
        print(f"Class distribution: {Counter(y)}")
        
        # Print accident type distribution
        if accident_types:
            print("Accident type distribution:")
            for acc_type, count in Counter(accident_types).items():
                print(f"  {acc_type}: {count}")
        
        yield X, y, feature_cols, True
    else:
        print("No sequences created in final chunk")
        yield None, None, feature_cols, True

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate):
    """TCN residual block with dilated causal convolutions"""
    prev_x = x
    
    # Layer normalization
    x = LayerNormalization()(x)
    
    # Dilated causal convolution
    x = Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               padding='causal',
               dilation_rate=dilation_rate,
               activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Second dilated causal convolution
    x = Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               padding='causal',
               dilation_rate=dilation_rate,
               activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # If dimensions don't match, transform the input
    if prev_x.shape[-1] != nb_filters:
        prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    
    # Residual connection
    res = prev_x + x
    return res

def attention_block(x, num_heads, key_dim):
    """Multi-head self-attention block"""
    # Self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x, x)
    
    # Skip connection
    return x + attention_output

def build_tcn_attention_model(input_shape):
    """Build TCN model with attention mechanism"""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # TCN blocks with increasing dilation rates
    for i, (nb_filters, dilation_rate) in enumerate(
            zip(CONFIG['tcn_filters'], CONFIG['tcn_dilations'])):
        x = residual_block(
            x, 
            dilation_rate=dilation_rate,
            nb_filters=nb_filters,
            kernel_size=CONFIG['tcn_kernel_size'],
            dropout_rate=CONFIG['dropout_rate']
        )
    
    # Attention mechanism
    x = attention_block(x, CONFIG['attention_heads'], key_dim=CONFIG['tcn_filters'][-1]//CONFIG['attention_heads'])
    
    # Global pooling to reduce sequence dimension
    x = GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with mixed precision for faster GPU training
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    
    # Use mixed precision if on GPU
    if CONFIG['use_gpu'] and tf.config.list_physical_devices('GPU'):
        # Mixed precision uses float16 for most ops but keeps float32 for critical ops
        print("Using mixed precision for faster GPU training")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model

def plot_training_history(history, fold=None):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    
    # Save figure
    plt_path = os.path.join(CONFIG['results_dir'], 'figures', 
                           f'training_history{"_fold"+str(fold) if fold is not None else ""}.png')
    plt.savefig(plt_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, fold=None):
    """Plot confusion matrix"""
    # Convert lists to numpy arrays if they aren't already
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Apply threshold to predictions
    y_pred_binary = (y_pred_arr > 0.5).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Accident'],
                yticklabels=['Normal', 'Accident'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure
    plt_path = os.path.join(CONFIG['results_dir'], 'figures', 
                           f'confusion_matrix{"_fold"+str(fold) if fold is not None else ""}.png')
    plt.savefig(plt_path)
    plt.close()

def train_with_kfold(all_data):
    """Train the model with k-fold cross-validation"""
    print(f"Starting {CONFIG['k_folds']}-fold cross-validation...")
    
    # Combine all data chunks into X and y
    X, y = combine_data_chunks(all_data)
    
    if X is None or len(X) == 0:
        print("No data to train on!")
        return None
    
    # Initialize k-fold
    kfold = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    
    # Initialize results tracking
    fold_results = []
    all_val_predictions = []
    all_val_true = []
    
    # Train and evaluate for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nTraining fold {fold+1}/{CONFIG['k_folds']}")
        
        tf.keras.backend.clear_session()  # Clear memory between folds
        
        # Train fold with error handling
        try:
            fold_result = _train_fold(fold, X, y, train_idx, val_idx, all_val_predictions, all_val_true)
            fold_results.append(fold_result)
            print(f"Fold {fold+1} results: {fold_result}")
        except Exception as e:
            print(f"Error in fold {fold+1}: {e}")
            continue
            
        # Force garbage collection
        gc.collect()
    
    # Calculate overall performance
    if all_val_true and all_val_predictions:
        # Convert lists to numpy arrays
        all_val_true_arr = np.array(all_val_true)
        all_val_predictions_arr = np.array(all_val_predictions)
        
        # Calculate AUC
        overall_auc = roc_auc_score(all_val_true_arr, all_val_predictions_arr)
        
        # Calculate binary predictions
        binary_predictions = (all_val_predictions_arr > 0.5).astype(int)
        
        # Generate classification report
        report = classification_report(all_val_true_arr, binary_predictions, output_dict=True)
        
        # Save results
        results = {
            'config': {k: str(v) if isinstance(v, (dict, list)) else v for k, v in CONFIG.items()},
            'fold_results': fold_results,
            'overall_auc': float(overall_auc),
            'classification_report': report
        }
        
        with open(os.path.join(CONFIG['results_dir'], 'kfold_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Plot overall confusion matrix
        plot_confusion_matrix(all_val_true, all_val_predictions)
        
        return results
    else:
        print("No validation results collected, cannot compute overall metrics")
        return None

def combine_data_chunks(data_chunks):
    """Combine data chunks into a single numpy array"""
    if not data_chunks:
        return None, None
        
    X_chunks = [chunk[0] for chunk in data_chunks if chunk[0] is not None]
    y_chunks = [chunk[1] for chunk in data_chunks if chunk[1] is not None]
    
    if not X_chunks or not y_chunks:
        return None, None
        
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    
    print(f"Combined data shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Final class distribution: {Counter(y)}")
    
    return X, y

def _train_fold(fold, X, y, train_idx, val_idx, all_val_predictions, all_val_true):
    """Helper function to train a single fold"""
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Scale features using Min-Max scaling
    scaler = MinMaxScaler()
    X_train_fold_reshaped = X_train_fold.reshape(-1, X_train_fold.shape[-1])
    X_val_fold_reshaped = X_val_fold.reshape(-1, X_val_fold.shape[-1])
    
    X_train_fold_scaled = scaler.fit_transform(X_train_fold_reshaped)
    X_val_fold_scaled = scaler.transform(X_val_fold_reshaped)
    
    # Reshape back to 3D
    X_train_fold = X_train_fold_scaled.reshape(X_train_fold.shape)
    X_val_fold = X_val_fold_scaled.reshape(X_val_fold.shape)
    
    # Build model
    model = build_tcn_attention_model((X_train_fold.shape[1], X_train_fold.shape[2]))
    
    if fold == 0:
        # Print model summary for the first fold
        model.summary()
        try:
            plot_model(model, to_file=os.path.join(CONFIG['results_dir'], 'model_architecture.png'), 
                       show_shapes=True)
        except Exception as e:
            print(f"Could not generate model plot: {e}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_dir'], f'model_fold_{fold+1}.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_data=(X_val_fold, y_val_fold),
        callbacks=callbacks,
        class_weight=CONFIG['class_weight'],
        verbose=CONFIG['verbose']
    )
    train_time = time.time() - start_time
    
    # Plot training history
    plot_training_history(history, fold+1)
    
    # Evaluate on validation set
    val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(
        X_val_fold, y_val_fold, verbose=0)
    
    # Get predictions
    val_pred = model.predict(X_val_fold, batch_size=CONFIG['batch_size'], verbose=0)
    
    # Flatten prediction array before extending the list
    val_pred_flat = val_pred.flatten()
    all_val_predictions.extend(val_pred_flat.tolist())
    all_val_true.extend(y_val_fold.tolist())
    
    # Plot confusion matrix for this fold
    try:
        plot_confusion_matrix(y_val_fold, val_pred, fold+1)
    except Exception as e:
        print(f"Error plotting confusion matrix for fold {fold+1}: {e}")
    
    # Save fold results
    fold_result = {
        'fold': fold + 1,
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_auc': float(val_auc),
        'training_time': train_time,
        'best_epoch': len(history.history['loss']) - CONFIG['patience']
    }
    
    return fold_result

def train_final_model(all_data):
    """Train final model on all data with a proper test split"""
    print("\nTraining final model...")
    
    # Combine all data chunks into X and y
    X, y = combine_data_chunks(all_data)
    
    if X is None or len(X) == 0:
        print("No data to train final model on!")
        return None, None
        
    tf.keras.backend.clear_session()  # Clear memory before final model
    
    try:
        return _train_final_model_internal(X, y)
    except Exception as e:
        print(f"Error in final model training: {e}")
        return None, None

def _train_final_model_internal(X, y):
    """Internal function for training the final model (to be used with device context)"""
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=42, stratify=y
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=CONFIG['val_size'], random_state=42, stratify=y_train
    )
    
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D
    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)
    X_test = X_test_scaled.reshape(X_test.shape)
    
    # Save scaler for future use
    import joblib
    joblib.dump(scaler, os.path.join(CONFIG['model_dir'], 'scaler.pkl'))
    
    # Build model
    model = build_tcn_attention_model((X_train.shape[1], X_train.shape[2]))
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_dir'], 'final_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=CONFIG['class_weight'],
        verbose=CONFIG['verbose']
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(
        X_test, y_test, batch_size=CONFIG['batch_size'], verbose=0
    )
    
    # Get predictions
    test_pred = model.predict(X_test, batch_size=CONFIG['batch_size'], verbose=0)
    
    # Plot confusion matrix - ensure arrays are properly formatted
    try:
        plot_confusion_matrix(y_test, test_pred)
    except Exception as e:
        print(f"Error plotting confusion matrix for final model: {e}")
    
    # Generate classification report
    test_pred_np = np.array(test_pred)
    binary_predictions = (test_pred_np > 0.5).astype(int)
    
    report = classification_report(y_test, binary_predictions, output_dict=True)
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'classification_report': report
    }
    
    with open(os.path.join(CONFIG['results_dir'], 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    print(f"Test results: {test_results}")
    
    # Save model in SavedModel format for deployment
    model.save(os.path.join(CONFIG['model_dir'], 'final_model_saved'))
    
    print("Final model training complete!")
    
    return model, test_results

def main():
    """Main function to execute the training pipeline"""
    # Create directories
    create_directories()
    
    try:
        # Load and preprocess data in chunks
        all_data = []
        feature_cols = None
        
        # Process data in chunks to avoid memory issues
        for X_chunk, y_chunk, cols, is_last_chunk in load_and_preprocess_data():
            if X_chunk is not None and y_chunk is not None:
                all_data.append((X_chunk, y_chunk))
                
                if feature_cols is None:
                    feature_cols = cols
            
            # Break if we're just testing with a small sample
            if is_last_chunk:
                break
                
        if not all_data:
            print("No data was loaded!")
            return
            
        # Save feature columns for future reference
        if feature_cols:
            with open(os.path.join(CONFIG['model_dir'], 'feature_columns.json'), 'w') as f:
                json.dump(feature_cols, f)
        
        # Train with k-fold cross-validation
        try:
            print("Starting cross-validation...")
            kfold_results = train_with_kfold(all_data)
            print("Cross-validation completed successfully")
        except Exception as e:
            print(f"Error during cross-validation: {e}")
            import traceback
            traceback.print_exc()
            kfold_results = None
        
        # Train final model
        try:
            print("Starting final model training...")
            final_model, test_results = train_final_model(all_data)
            print("Final model training completed successfully")
        except Exception as e:
            print(f"Error during final model training: {e}")
            import traceback
            traceback.print_exc()
            final_model, test_results = None, None
        
        print("\nTraining pipeline complete!")
        if kfold_results:
            print(f"Overall AUC across folds: {kfold_results.get('overall_auc', 'N/A')}")
        
        if test_results:
            print(f"Final model test accuracy: {test_results.get('test_accuracy', 'N/A')}")
            print(f"Final model test AUC: {test_results.get('test_auc', 'N/A')}")
        
        return final_model
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise 