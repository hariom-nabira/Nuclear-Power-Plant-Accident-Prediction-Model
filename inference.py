import os
import argparse
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque
import pickle

from model import NPPADModel

class RealTimeInferenceEngine:
    """
    Engine for real-time inference with a trained NPPAD model.
    This simulates a real-time monitoring system for nuclear power plant parameters.
    """
    def __init__(self, model_path, stream_data_path, window_size=30, stride=1, threshold=0.5):
        """
        Initialize the real-time inference engine.
        
        Args:
            model_path: Path to the saved model
            stream_data_path: Path to data to use for simulation
            window_size: Size of the sliding window for prediction
            stride: Stride for sliding window
            threshold: Threshold for binary prediction
        """
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold
        
        # Load model
        self.load_model(model_path)
        
        # Load data for simulation
        self.load_simulation_data(stream_data_path)
        
        # Initialize buffers
        self.data_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=100)  # Store last 100 timestamps
        self.score_buffer = deque(maxlen=100)  # Store last 100 prediction scores
        self.alert_buffer = deque(maxlen=100)  # Store last 100 alert states
        
        # Initialize state
        self.current_index = 0
        self.alert_active = False
        self.alert_start_time = None
        
    def load_model(self, model_path):
        """
        Load the trained model.
        
        Args:
            model_path: Path to the saved model
        """
        print(f"Loading model from {model_path}...")
        
        # Load model configuration
        try:
            model_dir = os.path.dirname(model_path)
            with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print("Model config not found, using default configuration")
            config = {}
        
        # Check for optimal threshold
        try:
            with open(os.path.join(model_dir, '../evaluation/optimal_threshold.json'), 'r') as f:
                threshold_data = json.load(f)
                self.threshold = threshold_data.get('optimal_threshold', self.threshold)
                print(f"Using optimal threshold: {self.threshold}")
        except FileNotFoundError:
            print(f"Using default threshold: {self.threshold}")
        
        # Initialize model
        self.model = NPPADModel(config)
        
        # Load weights
        try:
            self.model.load_weights(model_path)
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")
        
        # Set model to evaluation mode
        self.model.model.eval()
        
    def load_simulation_data(self, data_path):
        """
        Load data for simulation.
        
        Args:
            data_path: Path to the data file
        """
        print(f"Loading simulation data from {data_path}...")
        
        try:
            # Try loading as pickle first (for test dataset)
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, dict) and 'X' in data and 'y' in data:
                self.simulation_data = data['X']
                self.simulation_labels = data['y']
                self.feature_names = data.get('feature_names', None)
                
                print(f"Loaded {len(self.simulation_data)} samples from pickle file")
                
            else:
                raise ValueError("Unexpected data format in pickle file")
                
        except (pickle.UnpicklingError, EOFError):
            # Try loading as CSV
            try:
                df = pd.read_csv(data_path)
                
                # Assuming first column is timestamp or ID, last column is label
                if 'label' in df.columns:
                    label_col = 'label'
                else:
                    label_col = df.columns[-1]
                    
                self.simulation_labels = df[label_col].values
                
                # Remove label column for features
                feature_cols = [col for col in df.columns if col != label_col]
                
                # Save feature names
                self.feature_names = feature_cols
                
                # Convert to numpy array
                self.simulation_data = df[feature_cols].values
                
                print(f"Loaded {len(self.simulation_data)} samples from CSV file")
                
            except Exception as e:
                raise RuntimeError(f"Error loading simulation data: {e}")
                
        # Normalize if needed
        if np.max(np.abs(self.simulation_data)) > 100:
            print("Normalizing simulation data...")
            self.simulation_data = (self.simulation_data - np.mean(self.simulation_data, axis=0)) / np.std(self.simulation_data, axis=0)
    
    def get_next_data_point(self):
        """
        Get the next data point from the simulation data.
        
        Returns:
            Tuple of (data_point, label, done)
        """
        if self.current_index >= len(self.simulation_data):
            return None, None, True
        
        data_point = self.simulation_data[self.current_index]
        label = self.simulation_labels[self.current_index] if self.simulation_labels is not None else None
        
        self.current_index += 1
        
        return data_point, label, False
    
    def make_prediction(self, window_data):
        """
        Make a prediction using the current window of data.
        
        Args:
            window_data: Window of data for prediction
            
        Returns:
            Tuple of (prediction_score, binary_prediction)
        """
        # Convert to tensor
        data = np.expand_dims(window_data, axis=0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            probs, preds = self.model.predict(data, threshold=self.threshold)
            
        return probs[0][0], preds[0][0]
    
    def update_buffers(self, timestamp, data_point, score, prediction):
        """
        Update the data and prediction buffers.
        
        Args:
            timestamp: Current timestamp
            data_point: Current data point
            score: Prediction score
            prediction: Binary prediction
        """
        # Update data buffer
        self.data_buffer.append(data_point)
        
        # Update time and prediction buffers
        self.time_buffer.append(timestamp)
        self.score_buffer.append(score)
        self.alert_buffer.append(prediction)
        
        # Update alert state
        if prediction == 1:
            if not self.alert_active:
                self.alert_active = True
                self.alert_start_time = timestamp
        else:
            if self.alert_active:
                self.alert_active = False
                self.alert_start_time = None
    
    def run_simulation(self, output_dir=None, plot=True, time_delay=0.1):
        """
        Run the simulation with the provided data.
        
        Args:
            output_dir: Directory to save results
            plot: Whether to show real-time plot
            time_delay: Delay between processing steps (seconds)
            
        Returns:
            Dictionary with simulation results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize buffers
        self.data_buffer.clear()
        self.time_buffer.clear()
        self.score_buffer.clear()
        self.alert_buffer.clear()
        
        # Reset state
        self.current_index = 0
        self.alert_active = False
        self.alert_start_time = None
        
        # Simulation results
        results = {
            'timestamps': [],
            'scores': [],
            'predictions': [],
            'true_labels': [],
            'alerts': []
        }
        
        # Setup plot if requested
        if plot:
            plt.ion()  # Enable interactive mode
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle('Nuclear Power Plant Anomaly Detection', fontsize=16)
            
            # Prediction plot
            line1, = ax1.plot([], [], 'b-', label='Prediction Score')
            ax1.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold:.2f})')
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Probability of Scram')
            ax1.grid(True)
            ax1.legend()
            
            # Alert plot
            line2, = ax2.plot([], [], 'ro', label='Alert')
            ax2.set_xlim(0, 100)
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Alert State')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Normal', 'ALERT'])
            ax2.grid(True)
            
            # Status text
            status_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                                  bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.pause(0.1)
        
        # Fill buffer with initial data
        print("Filling initial data buffer...")
        for _ in range(self.window_size):
            data_point, label, done = self.get_next_data_point()
            if done:
                print("Not enough data for initial buffer")
                return results
            self.data_buffer.append(data_point)
        
        print(f"Starting simulation with {len(self.simulation_data) - self.window_size} steps...")
        start_time = time.time()
        
        # Main simulation loop
        while True:
            # Get current data window
            window_data = np.array(list(self.data_buffer))
            
            # Make prediction
            score, prediction = self.make_prediction(window_data)
            
            # Get next data point
            data_point, label, done = self.get_next_data_point()
            if done:
                break
            
            # Get current timestamp
            timestamp = self.current_index - 1
            
            # Update buffers
            self.update_buffers(timestamp, data_point, score, prediction)
            
            # Store results
            results['timestamps'].append(timestamp)
            results['scores'].append(float(score))
            results['predictions'].append(int(prediction))
            results['true_labels'].append(int(label) if label is not None else None)
            results['alerts'].append(self.alert_active)
            
            # Print status
            if self.current_index % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Step {self.current_index}/{len(self.simulation_data)} | "
                      f"Score: {score:.4f} | Alert: {self.alert_active} | "
                      f"Elapsed: {elapsed:.2f}s")
            
            # Update plot if enabled
            if plot and self.current_index % 5 == 0:
                # Update data
                x_data = list(self.time_buffer)
                y_data = list(self.score_buffer)
                alert_data = list(self.alert_buffer)
                
                # Update lines
                line1.set_data(x_data, y_data)
                line2.set_data(x_data, [1 if a else 0 for a in alert_data])
                
                # Update x limits to show last 100 points
                if len(x_data) > 0:
                    ax1.set_xlim(max(0, x_data[-1] - 100), max(100, x_data[-1]))
                    ax2.set_xlim(max(0, x_data[-1] - 100), max(100, x_data[-1]))
                
                # Update status text
                if self.alert_active:
                    status_text.set_text(f'ALERT: Potential scram event detected! (Score: {score:.4f})')
                    status_text.set_color('red')
                else:
                    status_text.set_text(f'Normal operation (Score: {score:.4f})')
                    status_text.set_color('green')
                
                plt.draw()
                plt.pause(0.01)
            
            # Add delay to simulate real-time processing
            if time_delay > 0:
                time.sleep(time_delay)
        
        # Calculate final metrics
        if None not in results['true_labels']:
            # Calculate accuracy
            accuracy = np.mean(np.array(results['predictions']) == np.array(results['true_labels']))
            print(f"\nSimulation completed - Accuracy: {accuracy:.4f}")
            
            # Count early detections
            early_detections = 0
            total_events = 0
            
            for i in range(len(results['true_labels'])):
                if i > 0 and results['true_labels'][i] == 1 and results['true_labels'][i-1] == 0:
                    total_events += 1
                    
                    # Check if we detected it early
                    for j in range(max(0, i-30), i):
                        if results['predictions'][j] == 1:
                            early_detections += 1
                            break
            
            if total_events > 0:
                print(f"Early detection rate: {early_detections}/{total_events} events ({early_detections/total_events:.2%})")
        
        # Save results if specified
        if output_dir:
            # Save prediction results
            with open(os.path.join(output_dir, 'simulation_results.json'), 'w') as f:
                json.dump({
                    'timestamps': results['timestamps'],
                    'scores': results['scores'],
                    'predictions': results['predictions'],
                    'true_labels': results['true_labels'],
                    'alerts': results['alerts'],
                }, f, indent=2)
            
            # Generate final plot
            plt.figure(figsize=(12, 8))
            
            # Scores
            plt.subplot(2, 1, 1)
            plt.plot(results['timestamps'], results['scores'], 'b-', label='Prediction Score')
            plt.axhline(y=self.threshold, color='r', linestyle='--', 
                      label=f'Threshold ({self.threshold:.2f})')
            
            # True labels if available
            if None not in results['true_labels']:
                plt.plot(results['timestamps'], results['true_labels'], 'g-', label='True Label')
            
            plt.xlabel('Time Step')
            plt.ylabel('Probability of Scram')
            plt.title('Reactor Scram Prediction')
            plt.legend()
            plt.grid(True)
            
            # Alerts
            plt.subplot(2, 1, 2)
            plt.plot(results['timestamps'], results['predictions'], 'ro', markersize=3, label='Predictions')
            
            # Fill alert regions
            alert_regions = []
            start_idx = None
            
            for i, alert in enumerate(results['alerts']):
                if alert and start_idx is None:
                    start_idx = i
                elif not alert and start_idx is not None:
                    alert_regions.append((results['timestamps'][start_idx], results['timestamps'][i-1]))
                    start_idx = None
            
            if start_idx is not None:
                alert_regions.append((results['timestamps'][start_idx], results['timestamps'][-1]))
            
            for start, end in alert_regions:
                plt.axvspan(start, end, color='r', alpha=0.3)
            
            # True labels if available
            if None not in results['true_labels']:
                plt.plot(results['timestamps'], results['true_labels'], 'g-', label='True Label')
                
            plt.xlabel('Time Step')
            plt.ylabel('Alert State')
            plt.title('Scram Alerts')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'simulation_plot.png'))
            plt.close()
            
            print(f"Saved simulation results to {output_dir}")
        
        # Close plot
        if plot:
            plt.ioff()
            plt.close()
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Run real-time inference simulation for NPPAD model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model weights')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to simulation data (pickle or CSV)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--window_size', type=int, default=30,
                        help='Size of the sliding window for prediction')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    parser.add_argument('--time_delay', type=float, default=0.1,
                        help='Delay between processing steps (seconds)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable real-time plotting')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = RealTimeInferenceEngine(
        model_path=args.model_path,
        stream_data_path=args.data_path,
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold
    )
    
    # Run simulation
    engine.run_simulation(
        output_dir=args.output_dir,
        plot=not args.no_plot,
        time_delay=args.time_delay
    )
    
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main() 