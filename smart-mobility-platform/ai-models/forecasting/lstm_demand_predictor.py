"""
LSTM-based Demand and Congestion Forecasting Model
Predicts passenger demand and congestion levels using deep learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MobilityTimeSeriesDataset(Dataset):
    """Custom dataset for mobility time series data"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMDemandPredictor(nn.Module):
    """LSTM model for demand and congestion prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 1, 
                 dropout: float = 0.2):
        super(LSTMDemandPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        output = self.relu(self.fc1(last_output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

class TransformerDemandPredictor(nn.Module):
    """Transformer-based model for demand prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 6, 
                 output_size: int = 1, dropout: float = 0.1):
        super(TransformerDemandPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Output projection
        output = self.output_layer(self.dropout(pooled))
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)].transpose(0, 1)
        return self.dropout(x)

class MobilityForecastingPipeline:
    """Complete forecasting pipeline for mobility data"""
    
    def __init__(self, sequence_length: int = 24, 
                 prediction_horizon: int = 6,
                 model_type: str = 'lstm'):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.model = None
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_cols: List[str],
                    feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training"""
        
        self.target_columns = target_cols
        
        if feature_cols is None:
            # Auto-generate features
            feature_cols = self._generate_time_features(df)
        
        self.feature_columns = feature_cols + target_cols
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').copy()
        
        # Create feature matrix
        feature_data = df_sorted[self.feature_columns].values
        target_data = df_sorted[target_cols].values
        
        # Scale features
        self.scalers['features'] = MinMaxScaler()
        self.scalers['targets'] = MinMaxScaler()
        
        scaled_features = self.scalers['features'].fit_transform(feature_data)
        scaled_targets = self.scalers['targets'].fit_transform(target_data)
        
        # Create sequences
        sequences, targets = self._create_sequences(
            scaled_features, scaled_targets
        )
        
        return sequences, targets
    
    def _generate_time_features(self, df: pd.DataFrame) -> List[str]:
        """Generate time-based features"""
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) |
            ((df['hour'] >= 17) & (df['hour'] <= 19))
        ).astype(int)
        
        time_features = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
            'month_sin', 'month_cos', 'is_weekend', 'is_rush_hour'
        ]
        
        return time_features
    
    def _create_sequences(self, features: np.ndarray, 
                         targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = features[i:i + self.sequence_length]
            
            # Target (future values)
            if self.prediction_horizon == 1:
                target = targets[i + self.sequence_length]
            else:
                target = targets[i + self.sequence_length:
                               i + self.sequence_length + self.prediction_horizon]
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets)
    
    def build_model(self, input_size: int):
        """Build the forecasting model"""
        output_size = len(self.target_columns) * self.prediction_horizon
        
        if self.model_type == 'lstm':
            self.model = LSTMDemandPredictor(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                output_size=output_size,
                dropout=0.2
            )
        elif self.model_type == 'transformer':
            self.model = TransformerDemandPredictor(
                input_size=input_size,
                d_model=128,
                nhead=8,
                num_layers=6,
                output_size=output_size,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, sequences: np.ndarray, targets: np.ndarray,
              validation_split: float = 0.2, epochs: int = 100,
              batch_size: int = 32, learning_rate: float = 0.001):
        """Train the forecasting model"""
        
        # Split data
        split_idx = int(len(sequences) * (1 - validation_split))
        
        train_sequences = sequences[:split_idx]
        train_targets = targets[:split_idx]
        val_sequences = sequences[split_idx:]
        val_targets = targets[split_idx:]
        
        # Create datasets
        train_dataset = MobilityTimeSeriesDataset(train_sequences, train_targets)
        val_dataset = MobilityTimeSeriesDataset(val_sequences, val_targets)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Build model if not already built
        if self.model is None:
            input_size = sequences.shape[2]
            self.build_model(input_size)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_sequences, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets.view(outputs.shape))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    outputs = self.model(batch_sequences)
                    loss = criterion(outputs, batch_targets.view(outputs.shape))
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        
        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(sequences)
            predictions = self.model(sequences_tensor)
            predictions_np = predictions.numpy()
        
        # Inverse transform predictions
        if self.prediction_horizon == 1:
            predictions_reshaped = predictions_np
        else:
            predictions_reshaped = predictions_np.reshape(-1, len(self.target_columns))
        
        predictions_scaled = self.scalers['targets'].inverse_transform(predictions_reshaped)
        
        return predictions_scaled
    
    def evaluate(self, sequences: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(sequences)
        
        # Inverse transform targets
        if targets.ndim == 3:
            targets_reshaped = targets.reshape(-1, len(self.target_columns))
        else:
            targets_reshaped = targets
        
        targets_scaled = self.scalers['targets'].inverse_transform(targets_reshaped)
        
        # Calculate metrics
        metrics = {}
        
        for i, col in enumerate(self.target_columns):
            y_true = targets_scaled[:, i]
            y_pred = predictions[:, i] if predictions.ndim > 1 else predictions.flatten()
            
            metrics[f'{col}_mae'] = mean_absolute_error(y_true, y_pred)
            metrics[f'{col}_mse'] = mean_squared_error(y_true, y_pred)
            metrics[f'{col}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics[f'{col}_r2'] = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            metrics[f'{col}_mape'] = mape
        
        return metrics
    
    def forecast_future(self, last_sequence: np.ndarray, 
                       steps_ahead: int = 24) -> np.ndarray:
        """Forecast multiple steps into the future"""
        self.model.eval()
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps_ahead):
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                prediction = self.model(sequence_tensor)
                prediction_np = prediction.numpy().flatten()
            
            forecasts.append(prediction_np)
            
            # Update sequence for next prediction
            # Add prediction to sequence and remove oldest entry
            new_row = current_sequence[-1].copy()
            for i, col_idx in enumerate(range(-len(self.target_columns), 0)):
                new_row[col_idx] = prediction_np[i]
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        forecasts = np.array(forecasts)
        
        # Inverse transform
        forecasts_scaled = self.scalers['targets'].inverse_transform(forecasts)
        
        return forecasts_scaled
    
    def _plot_training_history(self, train_losses: List[float], 
                              val_losses: List[float]):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, sequences: np.ndarray, targets: np.ndarray,
                        n_samples: int = 5):
        """Plot predictions vs actual values"""
        predictions = self.predict(sequences[:n_samples])
        
        # Inverse transform targets
        if targets.ndim == 3:
            targets_reshaped = targets[:n_samples].reshape(-1, len(self.target_columns))
        else:
            targets_reshaped = targets[:n_samples]
        
        targets_scaled = self.scalers['targets'].inverse_transform(targets_reshaped)
        
        fig, axes = plt.subplots(len(self.target_columns), 1, 
                                figsize=(15, 5 * len(self.target_columns)))
        
        if len(self.target_columns) == 1:
            axes = [axes]
        
        for i, col in enumerate(self.target_columns):
            axes[i].plot(targets_scaled[:, i], label='Actual', marker='o')
            axes[i].plot(predictions[:, i], label='Predicted', marker='s')
            axes[i].set_title(f'Predictions vs Actual - {col}')
            axes[i].set_xlabel('Sample')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model and scalers"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'scalers': self.scalers
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scalers"""
        model_data = torch.load(filepath)
        
        self.model_type = model_data['model_type']
        self.sequence_length = model_data['sequence_length']
        self.prediction_horizon = model_data['prediction_horizon']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        self.scalers = model_data['scalers']
        
        # Rebuild model
        input_size = len(self.feature_columns)
        self.build_model(input_size)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath}")

# Example usage and training script
if __name__ == "__main__":
    # Load sample data (replace with actual data loading)
    # df = pd.read_sql("SELECT * FROM processed_mobility_data", db_connection)
    
    # For demonstration, create synthetic data
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    # Simulate demand patterns
    hour_pattern = np.sin(2 * np.pi * np.arange(1000) / 24) * 50 + 100
    weekly_pattern = np.sin(2 * np.pi * np.arange(1000) / (24 * 7)) * 20
    noise = np.random.normal(0, 10, 1000)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'passenger_demand': hour_pattern + weekly_pattern + noise,
        'congestion_level': np.random.uniform(20, 80, 1000),
        'route_id': np.random.choice(['R1', 'R2', 'R3'], 1000)
    })
    
    # Initialize forecasting pipeline
    pipeline = MobilityForecastingPipeline(
        sequence_length=24,  # 24 hours of history
        prediction_horizon=6,  # Predict 6 hours ahead
        model_type='lstm'
    )
    
    # Prepare data
    target_cols = ['passenger_demand', 'congestion_level']
    sequences, targets = pipeline.prepare_data(df, target_cols)
    
    print(f"Data prepared: {sequences.shape} sequences, {targets.shape} targets")
    
    # Train model
    train_losses, val_losses = pipeline.train(
        sequences, targets,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Evaluate model
    metrics = pipeline.evaluate(sequences[-100:], targets[-100:])
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make future forecasts
    last_sequence = sequences[-1]
    future_forecasts = pipeline.forecast_future(last_sequence, steps_ahead=24)
    print(f"\nFuture forecasts shape: {future_forecasts.shape}")
    
    # Plot predictions
    pipeline.plot_predictions(sequences[-20:], targets[-20:], n_samples=5)
    
    # Save model
    pipeline.save_model('mobility_demand_predictor.pth')