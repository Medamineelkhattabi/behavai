"""
Anomaly Detection System for Smart Mobility Platform
Detects unusual patterns in passenger flows, vehicle behavior, and system performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AutoEncoder(nn.Module):
    """Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1][1:] + [input_dim]
        
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if hidden_dim != input_dim else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class LSTMAutoEncoder(nn.Module):
    """LSTM Autoencoder for time series anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, sequence_length: int = 24):
        super(LSTMAutoEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Use last encoded state as input to decoder
        decoder_input = encoded[:, -1, :].unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode
        decoded, _ = self.decoder_lstm(decoder_input)
        
        # Output
        output = self.output_layer(decoded)
        
        return output

class MobilityAnomalyDetector:
    """Comprehensive anomaly detection system for mobility data"""
    
    def __init__(self, detection_methods: List[str] = None):
        if detection_methods is None:
            detection_methods = ['isolation_forest', 'autoencoder', 'lstm_autoencoder']
        
        self.detection_methods = detection_methods
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_columns = []
        
        # Anomaly types
        self.anomaly_types = {
            'demand_surge': 'Unexpected increase in passenger demand',
            'demand_drop': 'Unexpected decrease in passenger demand',
            'congestion_anomaly': 'Unusual traffic congestion patterns',
            'vehicle_breakdown': 'Vehicle performance anomalies',
            'route_disruption': 'Route service disruptions',
            'temporal_anomaly': 'Time-based pattern anomalies',
            'spatial_anomaly': 'Location-based anomalies'
        }
    
    def prepare_data(self, df: pd.DataFrame, 
                    feature_cols: List[str] = None,
                    time_col: str = 'timestamp') -> pd.DataFrame:
        """Prepare data for anomaly detection"""
        
        if feature_cols is None:
            # Auto-select numerical columns
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_col in feature_cols:
                feature_cols.remove(time_col)
        
        self.feature_columns = feature_cols
        
        # Sort by time
        df_sorted = df.sort_values(time_col).copy()
        
        # Create time-based features
        if time_col in df_sorted.columns:
            df_sorted = self._add_temporal_features(df_sorted, time_col)
        
        # Create rolling statistics
        df_sorted = self._add_rolling_features(df_sorted, feature_cols)
        
        # Create lag features
        df_sorted = self._add_lag_features(df_sorted, feature_cols)
        
        return df_sorted
    
    def _add_temporal_features(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Add temporal features for anomaly detection"""
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Basic time features
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['month'] = df[time_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) |
            ((df['hour'] >= 17) & (df['hour'] <= 19))
        ).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Add temporal features to feature columns
        temporal_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                           'is_weekend', 'is_rush_hour']
        self.feature_columns.extend(temporal_features)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, 
                            feature_cols: List[str],
                            windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """Add rolling statistics features"""
        df = df.copy()
        
        for col in feature_cols:
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
                
                # Z-score (deviation from rolling mean)
                df[f'{col}_zscore_{window}'] = (
                    (df[col] - df[f'{col}_rolling_mean_{window}']) / 
                    (df[f'{col}_rolling_std_{window}'] + 1e-8)
                )
                
                # Add to feature columns
                self.feature_columns.extend([
                    f'{col}_rolling_mean_{window}',
                    f'{col}_rolling_std_{window}',
                    f'{col}_zscore_{window}'
                ])
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, 
                         feature_cols: List[str],
                         lags: List[int] = [1, 2, 6, 12]) -> pd.DataFrame:
        """Add lag features"""
        df = df.copy()
        
        for col in feature_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                self.feature_columns.append(f'{col}_lag_{lag}')
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def fit_isolation_forest(self, X: np.ndarray, contamination: float = 0.1):
        """Fit Isolation Forest model"""
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model.fit(X_scaled)
        
        self.models['isolation_forest'] = model
        self.scalers['isolation_forest'] = scaler
        
        # Calculate threshold
        scores = model.decision_function(X_scaled)
        self.thresholds['isolation_forest'] = np.percentile(scores, contamination * 100)
    
    def fit_one_class_svm(self, X: np.ndarray, nu: float = 0.1):
        """Fit One-Class SVM model"""
        model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=nu
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model.fit(X_scaled)
        
        self.models['one_class_svm'] = model
        self.scalers['one_class_svm'] = scaler
    
    def fit_autoencoder(self, X: np.ndarray, epochs: int = 100, 
                       batch_size: int = 32, threshold_percentile: float = 95):
        """Fit Autoencoder model"""
        input_dim = X.shape[1]
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create model
        model = AutoEncoder(input_dim)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            # Batch training
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Autoencoder Epoch {epoch}, Loss: {loss.item():.6f}')
        
        # Calculate reconstruction errors for threshold
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            threshold = torch.percentile(reconstruction_errors, threshold_percentile)
        
        self.models['autoencoder'] = model
        self.scalers['autoencoder'] = scaler
        self.thresholds['autoencoder'] = threshold.item()
    
    def fit_lstm_autoencoder(self, X: np.ndarray, sequence_length: int = 24,
                           epochs: int = 100, batch_size: int = 16,
                           threshold_percentile: float = 95):
        """Fit LSTM Autoencoder for time series anomaly detection"""
        
        # Create sequences
        sequences = self._create_sequences(X, sequence_length)
        
        if len(sequences) == 0:
            print("Not enough data for sequence creation")
            return
        
        input_dim = sequences.shape[2]
        
        # Scale features
        scaler = MinMaxScaler()
        # Reshape for scaling
        original_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = scaler.fit_transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(original_shape)
        
        # Create model
        model = LSTMAutoEncoder(input_dim, sequence_length=sequence_length)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(sequences_scaled)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / (len(X_tensor) // batch_size + 1)
                print(f'LSTM Autoencoder Epoch {epoch}, Avg Loss: {avg_loss:.6f}')
        
        # Calculate reconstruction errors for threshold
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            reconstruction_errors = torch.mean(
                torch.mean((X_tensor - reconstructed) ** 2, dim=2), dim=1
            )
            threshold = torch.percentile(reconstruction_errors, threshold_percentile)
        
        self.models['lstm_autoencoder'] = model
        self.scalers['lstm_autoencoder'] = scaler
        self.thresholds['lstm_autoencoder'] = threshold.item()
    
    def _create_sequences(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for LSTM autoencoder"""
        sequences = []
        
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        
        return np.array(sequences)
    
    def detect_anomalies(self, X: np.ndarray, 
                        methods: List[str] = None) -> Dict[str, np.ndarray]:
        """Detect anomalies using specified methods"""
        if methods is None:
            methods = self.detection_methods
        
        anomalies = {}
        
        for method in methods:
            if method not in self.models:
                print(f"Model {method} not fitted. Skipping...")
                continue
            
            if method == 'isolation_forest':
                anomalies[method] = self._detect_isolation_forest(X)
            elif method == 'one_class_svm':
                anomalies[method] = self._detect_one_class_svm(X)
            elif method == 'autoencoder':
                anomalies[method] = self._detect_autoencoder(X)
            elif method == 'lstm_autoencoder':
                anomalies[method] = self._detect_lstm_autoencoder(X)
        
        return anomalies
    
    def _detect_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        model = self.models['isolation_forest']
        scaler = self.scalers['isolation_forest']
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Convert to binary (1 = anomaly, 0 = normal)
        return (predictions == -1).astype(int)
    
    def _detect_one_class_svm(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using One-Class SVM"""
        model = self.models['one_class_svm']
        scaler = self.scalers['one_class_svm']
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Convert to binary (1 = anomaly, 0 = normal)
        return (predictions == -1).astype(int)
    
    def _detect_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using Autoencoder"""
        model = self.models['autoencoder']
        scaler = self.scalers['autoencoder']
        threshold = self.thresholds['autoencoder']
        
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        return (reconstruction_errors.numpy() > threshold).astype(int)
    
    def _detect_lstm_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using LSTM Autoencoder"""
        model = self.models['lstm_autoencoder']
        scaler = self.scalers['lstm_autoencoder']
        threshold = self.thresholds['lstm_autoencoder']
        
        # Create sequences
        sequence_length = model.sequence_length
        sequences = self._create_sequences(X, sequence_length)
        
        if len(sequences) == 0:
            return np.zeros(len(X))
        
        # Scale sequences
        original_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = scaler.transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(original_shape)
        
        X_tensor = torch.FloatTensor(sequences_scaled)
        
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            reconstruction_errors = torch.mean(
                torch.mean((X_tensor - reconstructed) ** 2, dim=2), dim=1
            )
        
        # Extend anomalies to original length
        anomalies_sequences = (reconstruction_errors.numpy() > threshold).astype(int)
        
        # Map back to original indices
        anomalies = np.zeros(len(X))
        for i, is_anomaly in enumerate(anomalies_sequences):
            if is_anomaly:
                # Mark the entire sequence as potentially anomalous
                start_idx = i
                end_idx = min(i + sequence_length, len(X))
                anomalies[start_idx:end_idx] = 1
        
        return anomalies
    
    def classify_anomaly_type(self, X: np.ndarray, anomaly_indices: np.ndarray,
                            original_features: List[str]) -> Dict[int, str]:
        """Classify the type of detected anomalies"""
        anomaly_types = {}
        
        for idx in np.where(anomaly_indices)[0]:
            features = X[idx]
            anomaly_type = self._determine_anomaly_type(features, original_features)
            anomaly_types[idx] = anomaly_type
        
        return anomaly_types
    
    def _determine_anomaly_type(self, features: np.ndarray, 
                              feature_names: List[str]) -> str:
        """Determine the type of anomaly based on feature values"""
        # This is a simplified classification - in practice, you'd use more sophisticated methods
        
        # Find features with highest deviation
        feature_dict = dict(zip(feature_names, features))
        
        # Check for demand-related anomalies
        demand_features = [f for f in feature_names if 'passenger' in f.lower() or 'demand' in f.lower()]
        if demand_features:
            demand_values = [feature_dict.get(f, 0) for f in demand_features]
            avg_demand = np.mean(demand_values)
            
            if avg_demand > np.percentile(demand_values, 90):
                return 'demand_surge'
            elif avg_demand < np.percentile(demand_values, 10):
                return 'demand_drop'
        
        # Check for congestion anomalies
        congestion_features = [f for f in feature_names if 'speed' in f.lower() or 'congestion' in f.lower()]
        if congestion_features:
            congestion_values = [feature_dict.get(f, 0) for f in congestion_features]
            if any(v < 10 for v in congestion_values):  # Very low speed
                return 'congestion_anomaly'
        
        # Check for temporal anomalies
        temporal_features = [f for f in feature_names if 'hour' in f.lower() or 'time' in f.lower()]
        if temporal_features:
            return 'temporal_anomaly'
        
        # Default
        return 'general_anomaly'
    
    def ensemble_detection(self, X: np.ndarray, 
                          voting_threshold: float = 0.5) -> np.ndarray:
        """Ensemble anomaly detection using multiple methods"""
        all_anomalies = self.detect_anomalies(X)
        
        if not all_anomalies:
            return np.zeros(len(X))
        
        # Voting mechanism
        anomaly_votes = np.zeros(len(X))
        
        for method, anomalies in all_anomalies.items():
            anomaly_votes += anomalies
        
        # Normalize votes
        anomaly_votes = anomaly_votes / len(all_anomalies)
        
        # Apply threshold
        ensemble_anomalies = (anomaly_votes >= voting_threshold).astype(int)
        
        return ensemble_anomalies
    
    def calculate_anomaly_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate anomaly scores for each method"""
        scores = {}
        
        for method in self.detection_methods:
            if method not in self.models:
                continue
            
            if method == 'isolation_forest':
                model = self.models[method]
                scaler = self.scalers[method]
                X_scaled = scaler.transform(X)
                scores[method] = -model.decision_function(X_scaled)  # Higher = more anomalous
            
            elif method == 'autoencoder':
                model = self.models[method]
                scaler = self.scalers[method]
                X_scaled = scaler.transform(X)
                X_tensor = torch.FloatTensor(X_scaled)
                
                model.eval()
                with torch.no_grad():
                    reconstructed = model(X_tensor)
                    reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
                
                scores[method] = reconstruction_errors.numpy()
        
        return scores
    
    def visualize_anomalies(self, df: pd.DataFrame, anomalies: np.ndarray,
                          time_col: str = 'timestamp', 
                          value_cols: List[str] = None):
        """Visualize detected anomalies"""
        if value_cols is None:
            value_cols = [col for col in df.columns if col != time_col][:3]  # First 3 numeric columns
        
        fig, axes = plt.subplots(len(value_cols), 1, figsize=(15, 4 * len(value_cols)))
        if len(value_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(value_cols):
            # Plot time series
            axes[i].plot(df[time_col], df[col], alpha=0.7, label='Normal')
            
            # Highlight anomalies
            anomaly_indices = np.where(anomalies)[0]
            if len(anomaly_indices) > 0:
                axes[i].scatter(df[time_col].iloc[anomaly_indices], 
                               df[col].iloc[anomaly_indices],
                               color='red', s=50, label='Anomaly', zorder=5)
            
            axes[i].set_title(f'Anomalies in {col}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(col)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_anomaly_report(self, df: pd.DataFrame, 
                              anomalies: Dict[str, np.ndarray],
                              time_col: str = 'timestamp') -> Dict[str, Any]:
        """Generate comprehensive anomaly report"""
        report = {
            'detection_summary': {},
            'temporal_analysis': {},
            'severity_analysis': {},
            'method_comparison': {}
        }
        
        # Detection summary
        for method, anomaly_array in anomalies.items():
            total_anomalies = np.sum(anomaly_array)
            anomaly_rate = total_anomalies / len(anomaly_array) * 100
            
            report['detection_summary'][method] = {
                'total_anomalies': int(total_anomalies),
                'anomaly_rate_percent': round(anomaly_rate, 2),
                'first_anomaly': df[time_col].iloc[np.where(anomaly_array)[0][0]].isoformat() if total_anomalies > 0 else None,
                'last_anomaly': df[time_col].iloc[np.where(anomaly_array)[0][-1]].isoformat() if total_anomalies > 0 else None
            }
        
        # Temporal analysis
        if anomalies:
            ensemble_anomalies = self.ensemble_detection(df[self.feature_columns].values)
            anomaly_times = df[time_col].iloc[np.where(ensemble_anomalies)[0]]
            
            if len(anomaly_times) > 0:
                anomaly_hours = anomaly_times.dt.hour.value_counts().sort_index()
                anomaly_days = anomaly_times.dt.dayofweek.value_counts().sort_index()
                
                report['temporal_analysis'] = {
                    'peak_anomaly_hours': anomaly_hours.head(3).to_dict(),
                    'peak_anomaly_days': anomaly_days.head(3).to_dict(),
                    'total_anomaly_periods': len(anomaly_times)
                }
        
        return report
    
    def save_models(self, filepath_prefix: str):
        """Save all trained models"""
        import joblib
        
        for method, model in self.models.items():
            if method in ['autoencoder', 'lstm_autoencoder']:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_class': model.__class__.__name__,
                    'threshold': self.thresholds.get(method)
                }, f"{filepath_prefix}_{method}.pth")
            else:
                joblib.dump(model, f"{filepath_prefix}_{method}.pkl")
        
        # Save scalers and metadata
        joblib.dump(self.scalers, f"{filepath_prefix}_scalers.pkl")
        joblib.dump({
            'thresholds': self.thresholds,
            'feature_columns': self.feature_columns,
            'detection_methods': self.detection_methods
        }, f"{filepath_prefix}_metadata.pkl")
        
        print(f"Models saved with prefix: {filepath_prefix}")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    # Normal patterns with some anomalies
    normal_demand = 100 + 50 * np.sin(2 * np.pi * np.arange(1000) / 24)  # Daily pattern
    normal_speed = 30 + 10 * np.sin(2 * np.pi * np.arange(1000) / (24 * 7))  # Weekly pattern
    
    # Inject anomalies
    anomaly_indices = [100, 250, 400, 600, 800]
    demand_data = normal_demand.copy()
    speed_data = normal_speed.copy()
    
    for idx in anomaly_indices:
        demand_data[idx] = demand_data[idx] * 3  # Demand surge
        speed_data[idx] = speed_data[idx] * 0.3  # Speed drop
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'passenger_demand': demand_data + np.random.normal(0, 10, 1000),
        'average_speed': speed_data + np.random.normal(0, 3, 1000),
        'congestion_level': np.random.uniform(20, 80, 1000),
        'occupancy_rate': np.random.uniform(30, 90, 1000)
    })
    
    # Initialize anomaly detector
    detector = MobilityAnomalyDetector()
    
    # Prepare data
    processed_df = detector.prepare_data(df, 
        feature_cols=['passenger_demand', 'average_speed', 'congestion_level', 'occupancy_rate'])
    
    # Extract features (remove NaN rows)
    X = processed_df[detector.feature_columns].dropna().values
    
    print(f"Data prepared: {X.shape}")
    
    # Fit models
    print("Fitting Isolation Forest...")
    detector.fit_isolation_forest(X, contamination=0.05)
    
    print("Fitting Autoencoder...")
    detector.fit_autoencoder(X, epochs=50)
    
    print("Fitting LSTM Autoencoder...")
    detector.fit_lstm_autoencoder(X, epochs=50, sequence_length=24)
    
    # Detect anomalies
    print("Detecting anomalies...")
    anomalies = detector.detect_anomalies(X)
    
    # Ensemble detection
    ensemble_anomalies = detector.ensemble_detection(X, voting_threshold=0.3)
    
    print(f"Anomalies detected:")
    for method, anomaly_array in anomalies.items():
        print(f"  {method}: {np.sum(anomaly_array)} anomalies")
    print(f"  Ensemble: {np.sum(ensemble_anomalies)} anomalies")
    
    # Visualize results
    detector.visualize_anomalies(
        processed_df.dropna(), 
        ensemble_anomalies,
        value_cols=['passenger_demand', 'average_speed']
    )
    
    # Generate report
    report = detector.generate_anomaly_report(
        processed_df.dropna(), 
        anomalies
    )
    
    print("\nAnomaly Detection Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Save models
    detector.save_models("mobility_anomaly_detector")
    
    print("Anomaly detection system setup completed!")