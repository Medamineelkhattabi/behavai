"""
Graph-Based Mobility Network Builder
Creates and analyzes graph representations of transportation networks
Nodes = stops/stations, Edges = routes, Weights = travel time/occupancy
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class MobilityNetworkGraph:
    def __init__(self, db_connection_string: str):
        self.db_connection = db_connection_string
        self.graphs = {}  # Store different graph representations
        self.node_features = {}
        self.edge_features = {}
        self.temporal_graphs = {}
        
    def load_network_data(self):
        """Load all network-related data from database"""
        queries = {
            'stops': "SELECT * FROM stops",
            'routes': "SELECT * FROM routes", 
            'gps_data': """
                SELECT vehicle_id, route_id, stop_sequence, latitude, longitude, 
                       speed, occupancy_level, timestamp
                FROM gps_data 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                ORDER BY vehicle_id, timestamp
            """,
            'passenger_flows': """
                SELECT origin_stop_id, destination_stop_id, route_id, 
                       passenger_count, avg_travel_time, time_window_start
                FROM passenger_flows
                WHERE time_window_start >= NOW() - INTERVAL '7 days'
            """,
            'turnstile_data': """
                SELECT stop_id, direction, count, timestamp
                FROM turnstile_data
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            """
        }
        
        self.data = {}
        for key, query in queries.items():
            print(f"Loading {key}...")
            self.data[key] = pd.read_sql(query, self.db_connection)
        
        print("Network data loaded successfully")
        
    def build_base_network(self) -> nx.Graph:
        """Build base network graph with stops as nodes"""
        G = nx.Graph()
        
        # Add nodes (stops)
        for _, stop in self.data['stops'].iterrows():
            G.add_node(
                stop['stop_id'],
                name=stop['name'],
                latitude=stop['latitude'],
                longitude=stop['longitude'],
                transport_mode=stop['transport_mode_id'],
                zone=stop.get('zone', 'unknown')
            )
        
        # Add edges based on route connections
        route_connections = self._extract_route_connections()
        
        for connection in route_connections:
            origin, destination, route_data = connection
            
            if G.has_edge(origin, destination):
                # Update existing edge with additional route
                G[origin][destination]['routes'].append(route_data['route_id'])
                G[origin][destination]['avg_travel_time'] = np.mean([
                    G[origin][destination]['avg_travel_time'],
                    route_data['avg_travel_time']
                ])
            else:
                G.add_edge(
                    origin, destination,
                    routes=[route_data['route_id']],
                    avg_travel_time=route_data['avg_travel_time'],
                    distance=route_data.get('distance', 0),
                    transport_mode=route_data.get('transport_mode', 'unknown')
                )
        
        self.graphs['base'] = G
        print(f"Base network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _extract_route_connections(self) -> List[Tuple]:
        """Extract route connections from GPS data"""
        connections = []
        
        # Group by vehicle and route to get sequential stops
        for (vehicle_id, route_id), group in self.data['gps_data'].groupby(['vehicle_id', 'route_id']):
            if group['stop_sequence'].isna().all():
                continue
                
            # Sort by stop sequence
            group_sorted = group.sort_values('stop_sequence')
            
            # Create connections between consecutive stops
            for i in range(len(group_sorted) - 1):
                current_stop = group_sorted.iloc[i]
                next_stop = group_sorted.iloc[i + 1]
                
                if pd.notna(current_stop['stop_sequence']) and pd.notna(next_stop['stop_sequence']):
                    # Calculate travel time
                    time_diff = (pd.to_datetime(next_stop['timestamp']) - 
                               pd.to_datetime(current_stop['timestamp'])).total_seconds()
                    
                    if time_diff > 0:  # Valid travel time
                        connections.append((
                            f"stop_{int(current_stop['stop_sequence'])}",
                            f"stop_{int(next_stop['stop_sequence'])}",
                            {
                                'route_id': route_id,
                                'avg_travel_time': time_diff,
                                'avg_speed': current_stop['speed'],
                                'transport_mode': 'bus'  # Default, can be enhanced
                            }
                        ))
        
        return connections
    
    def build_weighted_networks(self):
        """Build multiple weighted versions of the network"""
        base_graph = self.graphs.get('base', self.build_base_network())
        
        # Travel time weighted network
        G_time = base_graph.copy()
        for u, v, data in G_time.edges(data=True):
            G_time[u][v]['weight'] = data.get('avg_travel_time', 1)
        self.graphs['travel_time'] = G_time
        
        # Passenger flow weighted network
        G_flow = base_graph.copy()
        
        # Calculate passenger flows for edges
        flow_weights = self._calculate_flow_weights()
        
        for u, v, data in G_flow.edges(data=True):
            edge_key = f"{u}_{v}"
            G_flow[u][v]['weight'] = flow_weights.get(edge_key, 1)
            G_flow[u][v]['passenger_flow'] = flow_weights.get(edge_key, 0)
        
        self.graphs['passenger_flow'] = G_flow
        
        # Combined weighted network (travel time + passenger flow)
        G_combined = base_graph.copy()
        for u, v, data in G_combined.edges(data=True):
            time_weight = G_time[u][v]['weight']
            flow_weight = G_flow[u][v]['weight']
            
            # Normalize and combine weights
            combined_weight = 0.6 * time_weight + 0.4 * flow_weight
            G_combined[u][v]['weight'] = combined_weight
        
        self.graphs['combined'] = G_combined
        
        print("Weighted networks created successfully")
    
    def _calculate_flow_weights(self) -> Dict[str, float]:
        """Calculate passenger flow weights for edges"""
        flow_weights = {}
        
        if 'passenger_flows' in self.data and not self.data['passenger_flows'].empty:
            flow_data = self.data['passenger_flows'].groupby(
                ['origin_stop_id', 'destination_stop_id']
            )['passenger_count'].sum().reset_index()
            
            for _, row in flow_data.iterrows():
                edge_key = f"{row['origin_stop_id']}_{row['destination_stop_id']}"
                flow_weights[edge_key] = row['passenger_count']
        
        return flow_weights
    
    def build_temporal_networks(self, time_windows: List[str] = None):
        """Build time-specific network snapshots"""
        if time_windows is None:
            time_windows = ['morning_rush', 'midday', 'evening_rush', 'night']
        
        time_filters = {
            'morning_rush': (7, 10),
            'midday': (10, 16), 
            'evening_rush': (16, 20),
            'night': (20, 7)
        }
        
        for window in time_windows:
            if window not in time_filters:
                continue
                
            start_hour, end_hour = time_filters[window]
            
            # Filter data for time window
            if start_hour < end_hour:
                time_mask = (
                    (pd.to_datetime(self.data['gps_data']['timestamp']).dt.hour >= start_hour) &
                    (pd.to_datetime(self.data['gps_data']['timestamp']).dt.hour < end_hour)
                )
            else:  # Night window crosses midnight
                time_mask = (
                    (pd.to_datetime(self.data['gps_data']['timestamp']).dt.hour >= start_hour) |
                    (pd.to_datetime(self.data['gps_data']['timestamp']).dt.hour < end_hour)
                )
            
            filtered_gps = self.data['gps_data'][time_mask]
            
            # Build temporal network
            G_temporal = self._build_network_from_gps(filtered_gps)
            self.temporal_graphs[window] = G_temporal
            
            print(f"Temporal network '{window}' created: "
                  f"{G_temporal.number_of_nodes()} nodes, {G_temporal.number_of_edges()} edges")
    
    def _build_network_from_gps(self, gps_data: pd.DataFrame) -> nx.Graph:
        """Build network from filtered GPS data"""
        G = nx.Graph()
        
        # Add nodes from stops data
        for _, stop in self.data['stops'].iterrows():
            G.add_node(
                stop['stop_id'],
                name=stop['name'],
                latitude=stop['latitude'],
                longitude=stop['longitude']
            )
        
        # Add edges from GPS trajectories
        connections = self._extract_route_connections_from_data(gps_data)
        
        for origin, destination, edge_data in connections:
            if G.has_node(origin) and G.has_node(destination):
                G.add_edge(origin, destination, **edge_data)
        
        return G
    
    def _extract_route_connections_from_data(self, gps_data: pd.DataFrame) -> List[Tuple]:
        """Extract connections from specific GPS data"""
        connections = []
        
        for (vehicle_id, route_id), group in gps_data.groupby(['vehicle_id', 'route_id']):
            if group['stop_sequence'].isna().all():
                continue
                
            group_sorted = group.sort_values(['timestamp', 'stop_sequence'])
            
            for i in range(len(group_sorted) - 1):
                current = group_sorted.iloc[i]
                next_stop = group_sorted.iloc[i + 1]
                
                if (pd.notna(current['stop_sequence']) and 
                    pd.notna(next_stop['stop_sequence'])):
                    
                    time_diff = (pd.to_datetime(next_stop['timestamp']) - 
                               pd.to_datetime(current['timestamp'])).total_seconds()
                    
                    if time_diff > 0:
                        connections.append((
                            f"stop_{int(current['stop_sequence'])}",
                            f"stop_{int(next_stop['stop_sequence'])}",
                            {
                                'travel_time': time_diff,
                                'avg_speed': current['speed'],
                                'avg_occupancy': current['occupancy_level'],
                                'route_id': route_id
                            }
                        ))
        
        return connections
    
    def calculate_network_metrics(self) -> Dict[str, Dict]:
        """Calculate comprehensive network metrics"""
        metrics = {}
        
        for graph_name, G in self.graphs.items():
            if G.number_of_nodes() == 0:
                continue
                
            graph_metrics = {
                'basic_metrics': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'density': nx.density(G),
                    'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
                },
                'connectivity_metrics': {
                    'is_connected': nx.is_connected(G),
                    'number_of_components': nx.number_connected_components(G),
                    'average_clustering': nx.average_clustering(G),
                    'transitivity': nx.transitivity(G)
                },
                'centrality_metrics': {
                    'degree_centrality': nx.degree_centrality(G),
                    'betweenness_centrality': nx.betweenness_centrality(G),
                    'closeness_centrality': nx.closeness_centrality(G),
                    'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000)
                }
            }
            
            # Calculate shortest path metrics if connected
            if nx.is_connected(G):
                graph_metrics['path_metrics'] = {
                    'average_shortest_path_length': nx.average_shortest_path_length(G),
                    'diameter': nx.diameter(G),
                    'radius': nx.radius(G)
                }
            
            metrics[graph_name] = graph_metrics
        
        self.network_metrics = metrics
        return metrics
    
    def identify_critical_nodes(self, graph_name: str = 'base', top_k: int = 10) -> Dict[str, List]:
        """Identify critical nodes based on centrality measures"""
        if graph_name not in self.graphs:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        G = self.graphs[graph_name]
        metrics = self.network_metrics.get(graph_name, {}).get('centrality_metrics', {})
        
        if not metrics:
            self.calculate_network_metrics()
            metrics = self.network_metrics[graph_name]['centrality_metrics']
        
        critical_nodes = {}
        
        for centrality_type, centrality_values in metrics.items():
            # Sort nodes by centrality value
            sorted_nodes = sorted(centrality_values.items(), 
                                key=lambda x: x[1], reverse=True)
            critical_nodes[centrality_type] = sorted_nodes[:top_k]
        
        return critical_nodes
    
    def detect_communities(self, graph_name: str = 'base') -> Dict[str, int]:
        """Detect communities in the network"""
        if graph_name not in self.graphs:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        G = self.graphs[graph_name]
        
        # Use Louvain algorithm for community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
        except ImportError:
            # Fallback to greedy modularity communities
            communities = nx.community.greedy_modularity_communities(G)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
        
        self.communities = partition
        return partition
    
    def analyze_robustness(self, graph_name: str = 'base') -> Dict[str, float]:
        """Analyze network robustness to node/edge removal"""
        if graph_name not in self.graphs:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        G = self.graphs[graph_name].copy()
        original_components = nx.number_connected_components(G)
        
        robustness_metrics = {}
        
        # Node removal robustness
        critical_nodes = self.identify_critical_nodes(graph_name, top_k=5)
        degree_critical = [node for node, _ in critical_nodes['degree_centrality']]
        
        # Test removal of top degree nodes
        G_test = G.copy()
        for node in degree_critical:
            G_test.remove_node(node)
            new_components = nx.number_connected_components(G_test)
            robustness_metrics[f'remove_{node}'] = new_components / original_components
        
        # Random node removal
        import random
        nodes = list(G.nodes())
        random_nodes = random.sample(nodes, min(5, len(nodes)))
        
        G_random = G.copy()
        for node in random_nodes:
            if G_random.has_node(node):
                G_random.remove_node(node)
        
        robustness_metrics['random_removal'] = (
            nx.number_connected_components(G_random) / original_components
        )
        
        return robustness_metrics
    
    def visualize_network(self, graph_name: str = 'base', 
                         layout: str = 'spring', 
                         node_color_by: str = 'degree',
                         save_path: Optional[str] = None):
        """Visualize the network graph"""
        if graph_name not in self.graphs:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        G = self.graphs[graph_name]
        
        plt.figure(figsize=(20, 16))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'geographic' and all('latitude' in G.nodes[node] for node in G.nodes()):
            pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) 
                   for node in G.nodes()}
        else:
            pos = nx.circular_layout(G)
        
        # Node coloring
        if node_color_by == 'degree':
            node_colors = [G.degree(node) for node in G.nodes()]
        elif node_color_by == 'betweenness':
            centrality = nx.betweenness_centrality(G)
            node_colors = [centrality[node] for node in G.nodes()]
        else:
            node_colors = 'lightblue'
        
        # Node sizes based on degree
        node_sizes = [300 + G.degree(node) * 50 for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8, cmap='viridis')
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
        
        # Add labels for important nodes
        important_nodes = dict(sorted(dict(G.degree()).items(), 
                                    key=lambda x: x[1], reverse=True)[:10])
        nx.draw_networkx_labels(G, pos, labels=important_nodes, font_size=8)
        
        plt.title(f'Mobility Network Graph - {graph_name}', size=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_network_map(self, graph_name: str = 'base'):
        """Create interactive network visualization"""
        if graph_name not in self.graphs:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        G = self.graphs[graph_name]
        
        # Prepare node data
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            hoverinfo='text',
            marker=dict(size=[], color=[], colorscale='Viridis', showscale=True)
        )
        
        # Prepare edge data
        edge_trace = go.Scatter(x=[], y=[], mode='lines', 
                               line=dict(width=1, color='gray'),
                               hoverinfo='none')
        
        # Add geographic coordinates if available
        for node in G.nodes():
            if 'latitude' in G.nodes[node] and 'longitude' in G.nodes[node]:
                x, y = G.nodes[node]['longitude'], G.nodes[node]['latitude']
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                node_trace['text'] += tuple([G.nodes[node].get('name', node)])
                node_trace['marker']['size'] += tuple([10 + G.degree(node) * 2])
                node_trace['marker']['color'] += tuple([G.degree(node)])
        
        # Add edges
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['longitude'], G.nodes[edge[0]]['latitude']
            x1, y1 = G.nodes[edge[1]]['longitude'], G.nodes[edge[1]]['latitude']
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Interactive Mobility Network - {graph_name}',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Mobility network graph visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def export_graph_features(self, graph_name: str = 'base') -> pd.DataFrame:
        """Export graph features for ML models"""
        if graph_name not in self.graphs:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        G = self.graphs[graph_name]
        
        # Calculate all centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector_centrality = {node: 0 for node in G.nodes()}
        
        # Create feature dataframe
        features_data = []
        for node in G.nodes():
            node_data = {
                'node_id': node,
                'degree': G.degree(node),
                'degree_centrality': degree_centrality[node],
                'betweenness_centrality': betweenness_centrality[node],
                'closeness_centrality': closeness_centrality[node],
                'eigenvector_centrality': eigenvector_centrality[node],
                'clustering_coefficient': nx.clustering(G, node)
            }
            
            # Add geographic features if available
            if 'latitude' in G.nodes[node]:
                node_data['latitude'] = G.nodes[node]['latitude']
                node_data['longitude'] = G.nodes[node]['longitude']
            
            features_data.append(node_data)
        
        features_df = pd.DataFrame(features_data)
        return features_df
    
    def generate_network_report(self) -> str:
        """Generate comprehensive network analysis report"""
        if not self.network_metrics:
            self.calculate_network_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("MOBILITY NETWORK ANALYSIS REPORT")
        report.append("=" * 60)
        
        for graph_name, metrics in self.network_metrics.items():
            report.append(f"\n{graph_name.upper()} NETWORK:")
            report.append("-" * 30)
            
            # Basic metrics
            basic = metrics['basic_metrics']
            report.append(f"Nodes: {basic['nodes']}")
            report.append(f"Edges: {basic['edges']}")
            report.append(f"Density: {basic['density']:.4f}")
            report.append(f"Average Degree: {basic['average_degree']:.2f}")
            
            # Connectivity
            connectivity = metrics['connectivity_metrics']
            report.append(f"Connected: {connectivity['is_connected']}")
            report.append(f"Components: {connectivity['number_of_components']}")
            report.append(f"Average Clustering: {connectivity['average_clustering']:.4f}")
            
            # Path metrics (if available)
            if 'path_metrics' in metrics:
                path = metrics['path_metrics']
                report.append(f"Average Path Length: {path['average_shortest_path_length']:.2f}")
                report.append(f"Diameter: {path['diameter']}")
            
            # Top central nodes
            critical_nodes = self.identify_critical_nodes(graph_name, top_k=3)
            report.append(f"\nTop 3 nodes by degree centrality:")
            for node, centrality in critical_nodes['degree_centrality']:
                node_name = self.graphs[graph_name].nodes[node].get('name', node)
                report.append(f"  {node_name}: {centrality:.4f}")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    # Initialize network builder
    network_builder = MobilityNetworkGraph(
        "postgresql://mobility_user:mobility_pass@localhost:5432/mobility_platform"
    )
    
    # Load data and build networks
    network_builder.load_network_data()
    network_builder.build_base_network()
    network_builder.build_weighted_networks()
    network_builder.build_temporal_networks()
    
    # Calculate metrics
    metrics = network_builder.calculate_network_metrics()
    
    # Generate report
    report = network_builder.generate_network_report()
    print(report)
    
    # Visualize network
    network_builder.visualize_network('base', layout='geographic')
    
    # Export features for ML
    features = network_builder.export_graph_features('base')
    print(f"\nExported {len(features)} node features for ML models")