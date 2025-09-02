"""
Graph Neural Network for Passenger Flow Prediction
Predicts passenger flows across the mobility network using graph-based deep learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for flow prediction"""
    
    def __init__(self, node_features: int, edge_features: int = 0,
                 hidden_dim: int = 64, output_dim: int = 1,
                 num_heads: int = 4, num_layers: int = 3,
                 dropout: float = 0.2):
        super(GraphAttentionNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node feature preprocessing
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Edge feature preprocessing (if available)
        if edge_features > 0:
            self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        else:
            self.edge_embedding = None
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim, heads=num_heads, 
                   dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, 
                       heads=num_heads, dropout=dropout, concat=True)
            )
        
        # Final layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, 
                   heads=1, dropout=dropout, concat=False)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Node embedding
        x = F.relu(self.node_embedding(x))
        
        # GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling if batch is provided (for graph-level prediction)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output
        out = self.output_layers(x)
        
        return out

class SpatioTemporalGNN(nn.Module):
    """Spatio-temporal GNN combining spatial and temporal patterns"""
    
    def __init__(self, node_features: int, temporal_features: int,
                 spatial_hidden: int = 64, temporal_hidden: int = 64,
                 output_dim: int = 1, dropout: float = 0.2):
        super(SpatioTemporalGNN, self).__init__()
        
        # Spatial component (GAT)
        self.spatial_gnn = GraphAttentionNetwork(
            node_features=node_features,
            hidden_dim=spatial_hidden,
            output_dim=spatial_hidden,
            dropout=dropout
        )
        
        # Temporal component (LSTM)
        self.temporal_lstm = nn.LSTM(
            input_size=temporal_features,
            hidden_size=temporal_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(spatial_hidden + temporal_hidden, 
                     (spatial_hidden + temporal_hidden) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((spatial_hidden + temporal_hidden) // 2, output_dim)
        )
        
    def forward(self, x_spatial, edge_index, x_temporal, edge_attr=None):
        # Spatial features
        spatial_out = self.spatial_gnn(x_spatial, edge_index, edge_attr)
        
        # Temporal features
        temporal_out, _ = self.temporal_lstm(x_temporal)
        temporal_out = temporal_out[:, -1, :]  # Use last output
        
        # Ensure compatible dimensions
        if spatial_out.size(0) != temporal_out.size(0):
            # Repeat temporal features for each node
            temporal_out = temporal_out.repeat(spatial_out.size(0), 1)
        
        # Fusion
        combined = torch.cat([spatial_out, temporal_out], dim=1)
        output = self.fusion(combined)
        
        return output

class FlowPredictionGNN(nn.Module):
    """Specialized GNN for flow prediction between node pairs"""
    
    def __init__(self, node_features: int, edge_features: int = 0,
                 hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.2):
        super(FlowPredictionGNN, self).__init__()
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Flow prediction head
        self.flow_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated source and target embeddings
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, flow_edges):
        # Node embeddings
        x = self.node_encoder(x)
        
        # GNN propagation
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))
        
        # Flow prediction for specified edges
        source_embeddings = x[flow_edges[0]]
        target_embeddings = x[flow_edges[1]]
        
        # Concatenate source and target embeddings
        flow_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        # Predict flows
        flows = self.flow_predictor(flow_embeddings)
        
        return flows

class MobilityGNNPipeline:
    """Complete pipeline for GNN-based mobility prediction"""
    
    def __init__(self, model_type: str = 'gat'):
        self.model_type = model_type
        self.model = None
        self.node_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.node_mapping = {}
        self.reverse_node_mapping = {}
        
    def prepare_graph_data(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                          flows_df: pd.DataFrame = None) -> List[Data]:
        """Prepare graph data for training"""
        
        # Create node mapping
        unique_nodes = sorted(nodes_df['node_id'].unique())
        self.node_mapping = {node: i for i, node in enumerate(unique_nodes)}
        self.reverse_node_mapping = {i: node for node, i in self.node_mapping.items()}
        
        # Prepare node features
        node_feature_cols = [col for col in nodes_df.columns 
                           if col not in ['node_id', 'timestamp']]
        
        # Group by timestamp to create temporal snapshots
        graph_data_list = []
        
        if 'timestamp' in nodes_df.columns:
            # Temporal graphs
            for timestamp, group in nodes_df.groupby('timestamp'):
                graph_data = self._create_single_graph(
                    group, edges_df, flows_df, node_feature_cols, timestamp
                )
                if graph_data is not None:
                    graph_data_list.append(graph_data)
        else:
            # Single static graph
            graph_data = self._create_single_graph(
                nodes_df, edges_df, flows_df, node_feature_cols
            )
            if graph_data is not None:
                graph_data_list.append(graph_data)
        
        return graph_data_list
    
    def _create_single_graph(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                           flows_df: pd.DataFrame, node_feature_cols: List[str],
                           timestamp=None) -> Optional[Data]:
        """Create a single graph data object"""
        
        # Node features
        node_features = []
        node_targets = []
        
        for node_id in sorted(self.node_mapping.keys()):
            node_data = nodes_df[nodes_df['node_id'] == node_id]
            
            if node_data.empty:
                # Fill with zeros if node not present in this timestamp
                node_features.append(np.zeros(len(node_feature_cols)))
                node_targets.append(0)
            else:
                features = node_data[node_feature_cols].values[0]
                node_features.append(features)
                
                # Target (e.g., passenger count, congestion level)
                if 'target' in node_data.columns:
                    node_targets.append(node_data['target'].values[0])
                else:
                    node_targets.append(0)
        
        node_features = np.array(node_features)
        node_targets = np.array(node_targets)
        
        # Edge index
        edge_index = []
        edge_attr = []
        
        for _, edge in edges_df.iterrows():
            source = edge['source']
            target = edge['target']
            
            if source in self.node_mapping and target in self.node_mapping:
                source_idx = self.node_mapping[source]
                target_idx = self.node_mapping[target]
                
                edge_index.append([source_idx, target_idx])
                
                # Edge attributes (if available)
                edge_features = []
                for col in edges_df.columns:
                    if col not in ['source', 'target', 'timestamp']:
                        edge_features.append(edge[col])
                
                edge_attr.append(edge_features if edge_features else [1.0])
        
        if not edge_index:
            return None
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None
        
        # Flow edges (for flow prediction)
        flow_edges = None
        flow_targets = None
        
        if flows_df is not None and not flows_df.empty:
            flow_edge_list = []
            flow_target_list = []
            
            for _, flow in flows_df.iterrows():
                source = flow['origin_stop_id']
                target = flow['destination_stop_id']
                
                if source in self.node_mapping and target in self.node_mapping:
                    source_idx = self.node_mapping[source]
                    target_idx = self.node_mapping[target]
                    
                    flow_edge_list.append([source_idx, target_idx])
                    flow_target_list.append(flow['passenger_count'])
            
            if flow_edge_list:
                flow_edges = torch.tensor(flow_edge_list, dtype=torch.long).t().contiguous()
                flow_targets = torch.tensor(flow_target_list, dtype=torch.float)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(node_targets, dtype=torch.float),
            flow_edges=flow_edges,
            flow_targets=flow_targets
        )
        
        return data
    
    def build_model(self, node_features: int, edge_features: int = 0):
        """Build the GNN model"""
        
        if self.model_type == 'gat':
            self.model = GraphAttentionNetwork(
                node_features=node_features,
                edge_features=edge_features,
                hidden_dim=64,
                output_dim=1,
                num_heads=4,
                num_layers=3
            )
        elif self.model_type == 'flow_gnn':
            self.model = FlowPredictionGNN(
                node_features=node_features,
                edge_features=edge_features,
                hidden_dim=64,
                num_layers=3
            )
        elif self.model_type == 'spatiotemporal':
            self.model = SpatioTemporalGNN(
                node_features=node_features,
                temporal_features=10,  # Configurable
                spatial_hidden=64,
                temporal_hidden=64
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, graph_data_list: List[Data], validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the GNN model"""
        
        # Split data
        split_idx = int(len(graph_data_list) * (1 - validation_split))
        train_data = graph_data_list[:split_idx]
        val_data = graph_data_list[split_idx:]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Build model if not already built
        if self.model is None:
            sample_data = graph_data_list[0]
            node_features = sample_data.x.shape[1]
            edge_features = sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 0
            self.build_model(node_features, edge_features)
        
        # Loss and optimizer
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
            train_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                if self.model_type == 'flow_gnn' and batch.flow_edges is not None:
                    # Flow prediction
                    outputs = self.model(batch.x, batch.edge_index, batch.flow_edges)
                    targets = batch.flow_targets.unsqueeze(1)
                else:
                    # Node prediction
                    outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    targets = batch.y.unsqueeze(1)
                
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if self.model_type == 'flow_gnn' and batch.flow_edges is not None:
                        outputs = self.model(batch.x, batch.edge_index, batch.flow_edges)
                        targets = batch.flow_targets.unsqueeze(1)
                    else:
                        outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                        targets = batch.y.unsqueeze(1)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batches += 1
            
            train_loss /= max(train_batches, 1)
            val_loss /= max(val_batches, 1)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses
    
    def predict(self, graph_data_list: List[Data]) -> np.ndarray:
        """Make predictions on graph data"""
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for data in graph_data_list:
                if self.model_type == 'flow_gnn' and data.flow_edges is not None:
                    output = self.model(data.x, data.edge_index, data.flow_edges)
                else:
                    output = self.model(data.x, data.edge_index, data.edge_attr)
                
                predictions.append(output.numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(self, graph_data_list: List[Data]) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(graph_data_list)
        
        # Collect targets
        targets = []
        for data in graph_data_list:
            if self.model_type == 'flow_gnn' and data.flow_targets is not None:
                targets.append(data.flow_targets.numpy())
            else:
                targets.append(data.y.numpy())
        
        targets = np.concatenate(targets, axis=0)
        
        # Calculate metrics
        mae = mean_absolute_error(targets, predictions.flatten())
        mse = mean_squared_error(targets, predictions.flatten())
        rmse = np.sqrt(mse)
        
        # MAPE
        mape = np.mean(np.abs((targets - predictions.flatten()) / (targets + 1e-8))) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
        return metrics
    
    def visualize_predictions(self, graph_data_list: List[Data], n_samples: int = 5):
        """Visualize predictions vs actual values"""
        predictions = self.predict(graph_data_list[:n_samples])
        
        targets = []
        for data in graph_data_list[:n_samples]:
            if self.model_type == 'flow_gnn' and data.flow_targets is not None:
                targets.append(data.flow_targets.numpy())
            else:
                targets.append(data.y.numpy())
        
        targets = np.concatenate(targets, axis=0)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(targets, predictions.flatten(), alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        
        plt.subplot(2, 2, 2)
        residuals = targets - predictions.flatten()
        plt.scatter(predictions.flatten(), residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        plt.subplot(2, 2, 4)
        sample_indices = range(min(50, len(targets)))
        plt.plot(sample_indices, targets[:len(sample_indices)], 'o-', label='Actual', alpha=0.7)
        plt.plot(sample_indices, predictions.flatten()[:len(sample_indices)], 's-', label='Predicted', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Sample Predictions')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'node_mapping': self.node_mapping,
            'reverse_node_mapping': self.reverse_node_mapping,
            'node_scaler': self.node_scaler,
            'target_scaler': self.target_scaler
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = torch.load(filepath)
        
        self.model_type = model_data['model_type']
        self.node_mapping = model_data['node_mapping']
        self.reverse_node_mapping = model_data['reverse_node_mapping']
        self.node_scaler = model_data['node_scaler']
        self.target_scaler = model_data['target_scaler']
        
        # Model will need to be rebuilt with correct architecture
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Sample nodes (stops/stations)
    nodes_data = []
    for i in range(50):
        nodes_data.append({
            'node_id': f'stop_{i}',
            'latitude': np.random.uniform(25.0, 25.5),
            'longitude': np.random.uniform(55.0, 55.5),
            'degree_centrality': np.random.uniform(0, 1),
            'betweenness_centrality': np.random.uniform(0, 1),
            'passenger_count': np.random.poisson(100),
            'target': np.random.uniform(0, 100)  # Target variable
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Sample edges (connections)
    edges_data = []
    for i in range(80):
        source = f'stop_{np.random.randint(0, 50)}'
        target = f'stop_{np.random.randint(0, 50)}'
        if source != target:
            edges_data.append({
                'source': source,
                'target': target,
                'travel_time': np.random.uniform(1, 20),
                'distance': np.random.uniform(0.1, 5.0)
            })
    
    edges_df = pd.DataFrame(edges_data)
    
    # Sample flows
    flows_data = []
    for i in range(100):
        flows_data.append({
            'origin_stop_id': f'stop_{np.random.randint(0, 50)}',
            'destination_stop_id': f'stop_{np.random.randint(0, 50)}',
            'passenger_count': np.random.poisson(20)
        })
    
    flows_df = pd.DataFrame(flows_data)
    
    # Initialize GNN pipeline
    pipeline = MobilityGNNPipeline(model_type='gat')
    
    # Prepare graph data
    graph_data_list = pipeline.prepare_graph_data(nodes_df, edges_df, flows_df)
    
    print(f"Created {len(graph_data_list)} graph snapshots")
    
    # Train model
    train_losses, val_losses = pipeline.train(
        graph_data_list,
        epochs=50,
        batch_size=1,  # Small batch size for graph data
        learning_rate=0.001
    )
    
    # Evaluate model
    metrics = pipeline.evaluate(graph_data_list[-20:])
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize predictions
    pipeline.visualize_predictions(graph_data_list[-10:])
    
    # Save model
    pipeline.save_model('mobility_gnn_model.pth')