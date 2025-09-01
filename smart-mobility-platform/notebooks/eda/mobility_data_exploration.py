"""
Comprehensive Exploratory Data Analysis for Smart Mobility Platform
Deep analysis of mobility patterns, peak demand windows, and anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import folium
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MobilityDataExplorer:
    def __init__(self, db_connection_string):
        """Initialize with database connection"""
        self.db_connection = db_connection_string
        self.data_cache = {}
        
    def load_data(self, force_reload=False):
        """Load all necessary data for analysis"""
        if not force_reload and self.data_cache:
            return self.data_cache
            
        # Load data from database
        queries = {
            'gps_data': """
                SELECT * FROM gps_data 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
            """,
            'ticketing_data': """
                SELECT * FROM ticketing_data 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
            """,
            'turnstile_data': """
                SELECT * FROM turnstile_data 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
            """,
            'weather_data': """
                SELECT * FROM weather_data 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
            """,
            'special_events': """
                SELECT * FROM special_events 
                WHERE start_time >= NOW() - INTERVAL '30 days'
            """,
            'stops': "SELECT * FROM stops",
            'routes': "SELECT * FROM routes",
            'vehicles': "SELECT * FROM vehicles"
        }
        
        for key, query in queries.items():
            print(f"Loading {key}...")
            self.data_cache[key] = pd.read_sql(query, self.db_connection)
            
        # Add derived features
        self._add_temporal_features()
        
        return self.data_cache
    
    def _add_temporal_features(self):
        """Add temporal features to datasets"""
        for key in ['gps_data', 'ticketing_data', 'turnstile_data', 'weather_data']:
            if key in self.data_cache:
                df = self.data_cache[key]
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                df['is_weekend'] = df['day_of_week'].isin([5, 6])
                df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    
    def analyze_temporal_patterns(self):
        """Comprehensive temporal pattern analysis"""
        print("=== TEMPORAL PATTERN ANALYSIS ===")
        
        # Hourly patterns
        self._analyze_hourly_patterns()
        
        # Daily patterns
        self._analyze_daily_patterns()
        
        # Seasonal patterns
        self._analyze_seasonal_patterns()
        
        # Rush hour analysis
        self._analyze_rush_hours()
    
    def _analyze_hourly_patterns(self):
        """Analyze patterns by hour of day"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # GPS data by hour
        gps_hourly = self.data_cache['gps_data'].groupby('hour').agg({
            'vehicle_id': 'nunique',
            'occupancy_level': 'mean',
            'speed': 'mean'
        })
        
        axes[0, 0].plot(gps_hourly.index, gps_hourly['vehicle_id'], marker='o')
        axes[0, 0].set_title('Active Vehicles by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Active Vehicles')
        
        # Ticketing patterns
        ticketing_hourly = self.data_cache['ticketing_data'].groupby(['hour', 'transaction_type']).size().unstack()
        ticketing_hourly.plot(kind='bar', ax=axes[0, 1], stacked=True)
        axes[0, 1].set_title('Ticketing Transactions by Hour')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Transactions')
        
        # Occupancy patterns
        axes[1, 0].plot(gps_hourly.index, gps_hourly['occupancy_level'], marker='o', color='red')
        axes[1, 0].set_title('Average Occupancy Level by Hour')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Occupancy (%)')
        
        # Speed patterns
        axes[1, 1].plot(gps_hourly.index, gps_hourly['speed'], marker='o', color='green')
        axes[1, 1].set_title('Average Speed by Hour')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Average Speed (km/h)')
        
        plt.tight_layout()
        plt.show()
        
        # Print peak hours
        peak_occupancy_hour = gps_hourly['occupancy_level'].idxmax()
        peak_activity_hour = gps_hourly['vehicle_id'].idxmax()
        
        print(f"Peak occupancy hour: {peak_occupancy_hour}:00")
        print(f"Peak activity hour: {peak_activity_hour}:00")
    
    def _analyze_daily_patterns(self):
        """Analyze patterns by day of week"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Daily ridership
        daily_ridership = self.data_cache['ticketing_data'].groupby('day_of_week').size()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        axes[0, 0].bar(day_names, daily_ridership.values)
        axes[0, 0].set_title('Daily Ridership Patterns')
        axes[0, 0].set_ylabel('Number of Transactions')
        
        # Weekend vs Weekday patterns
        weekend_comparison = self.data_cache['gps_data'].groupby(['is_weekend', 'hour']).agg({
            'occupancy_level': 'mean',
            'speed': 'mean'
        })
        
        for is_weekend in [False, True]:
            label = 'Weekend' if is_weekend else 'Weekday'
            data = weekend_comparison.loc[is_weekend]
            axes[0, 1].plot(data.index, data['occupancy_level'], 
                           label=label, marker='o')
        
        axes[0, 1].set_title('Occupancy: Weekday vs Weekend')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Average Occupancy (%)')
        axes[0, 1].legend()
        
        # Daily congestion patterns
        daily_congestion = self.data_cache['gps_data'].groupby('day_of_week').agg({
            'speed': 'mean',
            'occupancy_level': 'mean'
        })
        
        axes[1, 0].bar(day_names, daily_congestion['speed'].values, alpha=0.7)
        axes[1, 0].set_title('Average Speed by Day')
        axes[1, 0].set_ylabel('Average Speed (km/h)')
        
        axes[1, 1].bar(day_names, daily_congestion['occupancy_level'].values, 
                      alpha=0.7, color='orange')
        axes[1, 1].set_title('Average Occupancy by Day')
        axes[1, 1].set_ylabel('Average Occupancy (%)')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_seasonal_patterns(self):
        """Analyze seasonal and monthly patterns"""
        # Add month column
        for key in ['gps_data', 'ticketing_data']:
            if key in self.data_cache:
                self.data_cache[key]['month'] = pd.to_datetime(self.data_cache[key]['timestamp']).dt.month
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Monthly ridership trends
        monthly_ridership = self.data_cache['ticketing_data'].groupby('month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        axes[0, 0].plot(monthly_ridership.index, monthly_ridership.values, marker='o')
        axes[0, 0].set_title('Monthly Ridership Trends')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Number of Transactions')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].set_xticklabels(month_names[:len(monthly_ridership)])
        
        # Monthly occupancy patterns
        monthly_occupancy = self.data_cache['gps_data'].groupby('month')['occupancy_level'].mean()
        axes[0, 1].plot(monthly_occupancy.index, monthly_occupancy.values, 
                       marker='o', color='red')
        axes[0, 1].set_title('Monthly Average Occupancy')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Occupancy (%)')
        
        # Weather correlation
        if 'weather_data' in self.data_cache and not self.data_cache['weather_data'].empty:
            weather_mobility = self._correlate_weather_mobility()
            
            axes[1, 0].scatter(weather_mobility['temperature'], 
                             weather_mobility['ridership'], alpha=0.6)
            axes[1, 0].set_title('Temperature vs Ridership')
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].set_ylabel('Daily Ridership')
            
            axes[1, 1].scatter(weather_mobility['precipitation'], 
                             weather_mobility['ridership'], alpha=0.6, color='blue')
            axes[1, 1].set_title('Precipitation vs Ridership')
            axes[1, 1].set_xlabel('Precipitation (mm)')
            axes[1, 1].set_ylabel('Daily Ridership')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_rush_hours(self):
        """Deep dive into rush hour patterns"""
        print("\n=== RUSH HOUR ANALYSIS ===")
        
        rush_hour_data = self.data_cache['gps_data'][self.data_cache['gps_data']['is_rush_hour']]
        non_rush_data = self.data_cache['gps_data'][~self.data_cache['gps_data']['is_rush_hour']]
        
        print(f"Rush hour average occupancy: {rush_hour_data['occupancy_level'].mean():.2f}%")
        print(f"Non-rush hour average occupancy: {non_rush_data['occupancy_level'].mean():.2f}%")
        print(f"Rush hour average speed: {rush_hour_data['speed'].mean():.2f} km/h")
        print(f"Non-rush hour average speed: {non_rush_data['speed'].mean():.2f} km/h")
        
        # Rush hour heatmap
        rush_hour_heatmap = self.data_cache['gps_data'].groupby(['hour', 'day_of_week'])['occupancy_level'].mean().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(rush_hour_heatmap, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Occupancy Level Heatmap (Hour vs Day of Week)')
        plt.xlabel('Day of Week (0=Monday)')
        plt.ylabel('Hour of Day')
        plt.show()
    
    def analyze_spatial_patterns(self):
        """Analyze spatial mobility patterns"""
        print("\n=== SPATIAL PATTERN ANALYSIS ===")
        
        # Route analysis
        self._analyze_route_performance()
        
        # Stop analysis
        self._analyze_stop_patterns()
        
        # Spatial clustering
        self._analyze_spatial_clusters()
    
    def _analyze_route_performance(self):
        """Analyze performance by route"""
        route_stats = self.data_cache['gps_data'].groupby('route_id').agg({
            'speed': ['mean', 'std'],
            'occupancy_level': ['mean', 'max'],
            'vehicle_id': 'nunique'
        }).round(2)
        
        route_stats.columns = ['avg_speed', 'speed_std', 'avg_occupancy', 
                              'max_occupancy', 'vehicle_count']
        
        # Top and bottom performing routes
        print("Top 5 routes by average speed:")
        print(route_stats.nlargest(5, 'avg_speed')[['avg_speed', 'avg_occupancy']])
        
        print("\nTop 5 most congested routes (highest occupancy):")
        print(route_stats.nlargest(5, 'avg_occupancy')[['avg_speed', 'avg_occupancy']])
        
        # Route performance visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Speed vs Occupancy scatter
        axes[0, 0].scatter(route_stats['avg_speed'], route_stats['avg_occupancy'], 
                          alpha=0.6, s=route_stats['vehicle_count']*10)
        axes[0, 0].set_xlabel('Average Speed (km/h)')
        axes[0, 0].set_ylabel('Average Occupancy (%)')
        axes[0, 0].set_title('Route Performance: Speed vs Occupancy')
        
        # Top routes by occupancy
        top_routes = route_stats.nlargest(10, 'avg_occupancy')
        axes[0, 1].barh(range(len(top_routes)), top_routes['avg_occupancy'].values)
        axes[0, 1].set_yticks(range(len(top_routes)))
        axes[0, 1].set_yticklabels(top_routes.index)
        axes[0, 1].set_xlabel('Average Occupancy (%)')
        axes[0, 1].set_title('Top 10 Routes by Occupancy')
        
        # Speed distribution
        axes[1, 0].hist(route_stats['avg_speed'], bins=20, alpha=0.7)
        axes[1, 0].set_xlabel('Average Speed (km/h)')
        axes[1, 0].set_ylabel('Number of Routes')
        axes[1, 0].set_title('Distribution of Route Average Speeds')
        
        # Occupancy distribution
        axes[1, 1].hist(route_stats['avg_occupancy'], bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Average Occupancy (%)')
        axes[1, 1].set_ylabel('Number of Routes')
        axes[1, 1].set_title('Distribution of Route Average Occupancy')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_stop_patterns(self):
        """Analyze patterns at individual stops"""
        # Merge stop data with turnstile data
        stop_activity = pd.merge(self.data_cache['turnstile_data'], 
                               self.data_cache['stops'], 
                               left_on='stop_id', right_on='stop_id')
        
        # Calculate stop metrics
        stop_metrics = stop_activity.groupby('stop_id').agg({
            'count': 'sum',
            'name': 'first',
            'latitude': 'first',
            'longitude': 'first'
        })
        
        stop_metrics['total_passengers'] = stop_metrics['count']
        
        # Busiest stops
        print("Top 10 busiest stops:")
        busiest_stops = stop_metrics.nlargest(10, 'total_passengers')
        print(busiest_stops[['name', 'total_passengers']])
        
        # Create interactive map
        self._create_stop_activity_map(stop_metrics)
    
    def _create_stop_activity_map(self, stop_metrics):
        """Create interactive map showing stop activity"""
        # Calculate center coordinates
        center_lat = stop_metrics['latitude'].mean()
        center_lon = stop_metrics['longitude'].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add stop markers
        max_passengers = stop_metrics['total_passengers'].max()
        
        for idx, row in stop_metrics.iterrows():
            # Normalize marker size
            marker_size = (row['total_passengers'] / max_passengers) * 20 + 5
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=marker_size,
                popup=f"{row['name']}<br>Passengers: {row['total_passengers']}",
                color='red',
                fillColor='red',
                fillOpacity=0.6
            ).add_to(m)
        
        # Save map
        m.save('stop_activity_map.html')
        print("Stop activity map saved as 'stop_activity_map.html'")
    
    def _analyze_spatial_clusters(self):
        """Identify spatial clusters of high activity"""
        from sklearn.cluster import DBSCAN
        
        # Prepare data for clustering
        stop_coords = self.data_cache['stops'][['latitude', 'longitude']].values
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=0.01, min_samples=3).fit(stop_coords)
        
        # Add cluster labels to stops
        self.data_cache['stops']['cluster'] = clustering.labels_
        
        # Analyze clusters
        cluster_stats = self.data_cache['stops'].groupby('cluster').agg({
            'latitude': ['mean', 'count'],
            'longitude': 'mean',
            'name': 'first'
        })
        
        print(f"Identified {len(cluster_stats)} spatial clusters")
        print("Cluster centers and sizes:")
        print(cluster_stats)
    
    def analyze_anomalies(self):
        """Detect and analyze anomalies in mobility data"""
        print("\n=== ANOMALY ANALYSIS ===")
        
        # Statistical anomalies in occupancy
        self._detect_occupancy_anomalies()
        
        # Speed anomalies
        self._detect_speed_anomalies()
        
        # Temporal anomalies
        self._detect_temporal_anomalies()
    
    def _detect_occupancy_anomalies(self):
        """Detect anomalies in occupancy levels"""
        # Calculate rolling statistics
        gps_sorted = self.data_cache['gps_data'].sort_values('timestamp')
        gps_sorted['rolling_mean'] = gps_sorted.groupby('vehicle_id')['occupancy_level'].rolling(window=10).mean().values
        gps_sorted['rolling_std'] = gps_sorted.groupby('vehicle_id')['occupancy_level'].rolling(window=10).std().values
        
        # Calculate z-scores
        gps_sorted['z_score'] = (gps_sorted['occupancy_level'] - gps_sorted['rolling_mean']) / gps_sorted['rolling_std']
        
        # Identify anomalies (z-score > 3)
        anomalies = gps_sorted[abs(gps_sorted['z_score']) > 3]
        
        print(f"Detected {len(anomalies)} occupancy anomalies")
        
        # Visualize anomalies
        plt.figure(figsize=(15, 8))
        
        # Sample a vehicle for visualization
        sample_vehicle = gps_sorted['vehicle_id'].iloc[0]
        vehicle_data = gps_sorted[gps_sorted['vehicle_id'] == sample_vehicle].head(100)
        
        plt.plot(vehicle_data['timestamp'], vehicle_data['occupancy_level'], 
                label='Occupancy Level', alpha=0.7)
        plt.plot(vehicle_data['timestamp'], vehicle_data['rolling_mean'], 
                label='Rolling Mean', color='red')
        
        # Mark anomalies
        vehicle_anomalies = vehicle_data[abs(vehicle_data['z_score']) > 3]
        if not vehicle_anomalies.empty:
            plt.scatter(vehicle_anomalies['timestamp'], vehicle_anomalies['occupancy_level'], 
                       color='red', s=100, label='Anomalies')
        
        plt.title(f'Occupancy Anomalies - Vehicle {sample_vehicle}')
        plt.xlabel('Time')
        plt.ylabel('Occupancy Level (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _detect_speed_anomalies(self):
        """Detect speed anomalies"""
        # Identify unusually low or high speeds
        speed_stats = self.data_cache['gps_data']['speed'].describe()
        
        # Define anomaly thresholds
        low_speed_threshold = speed_stats['25%'] - 1.5 * (speed_stats['75%'] - speed_stats['25%'])
        high_speed_threshold = speed_stats['75%'] + 1.5 * (speed_stats['75%'] - speed_stats['25%'])
        
        speed_anomalies = self.data_cache['gps_data'][
            (self.data_cache['gps_data']['speed'] < low_speed_threshold) |
            (self.data_cache['gps_data']['speed'] > high_speed_threshold)
        ]
        
        print(f"Detected {len(speed_anomalies)} speed anomalies")
        print(f"Low speed threshold: {low_speed_threshold:.2f} km/h")
        print(f"High speed threshold: {high_speed_threshold:.2f} km/h")
    
    def _detect_temporal_anomalies(self):
        """Detect temporal anomalies in ridership patterns"""
        # Daily ridership patterns
        daily_ridership = self.data_cache['ticketing_data'].groupby('date').size()
        
        # Calculate rolling statistics
        daily_ridership = daily_ridership.sort_index()
        rolling_mean = daily_ridership.rolling(window=7).mean()
        rolling_std = daily_ridership.rolling(window=7).std()
        
        # Calculate z-scores
        z_scores = (daily_ridership - rolling_mean) / rolling_std
        
        # Identify anomalous days
        anomalous_days = daily_ridership[abs(z_scores) > 2]
        
        print(f"Detected {len(anomalous_days)} anomalous days in ridership")
        
        # Visualize
        plt.figure(figsize=(15, 8))
        plt.plot(daily_ridership.index, daily_ridership.values, 
                label='Daily Ridership', alpha=0.7)
        plt.plot(rolling_mean.index, rolling_mean.values, 
                label='7-day Rolling Mean', color='red')
        
        # Mark anomalous days
        if not anomalous_days.empty:
            plt.scatter(anomalous_days.index, anomalous_days.values, 
                       color='red', s=100, label='Anomalous Days')
        
        plt.title('Daily Ridership Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Number of Passengers')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _correlate_weather_mobility(self):
        """Correlate weather conditions with mobility patterns"""
        # Aggregate daily data
        daily_weather = self.data_cache['weather_data'].groupby('date').agg({
            'temperature': 'mean',
            'precipitation': 'sum',
            'wind_speed': 'mean',
            'weather_condition': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
        })
        
        daily_ridership = self.data_cache['ticketing_data'].groupby('date').size()
        
        # Merge weather and ridership data
        weather_mobility = pd.merge(daily_weather, daily_ridership.to_frame('ridership'), 
                                  left_index=True, right_index=True, how='inner')
        
        return weather_mobility
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*50)
        print("SMART MOBILITY PLATFORM - INSIGHTS REPORT")
        print("="*50)
        
        # Load data if not already loaded
        if not self.data_cache:
            self.load_data()
        
        # Key metrics
        total_vehicles = self.data_cache['gps_data']['vehicle_id'].nunique()
        total_stops = len(self.data_cache['stops'])
        total_routes = self.data_cache['gps_data']['route_id'].nunique()
        total_transactions = len(self.data_cache['ticketing_data'])
        
        print(f"Total Vehicles Tracked: {total_vehicles}")
        print(f"Total Stops: {total_stops}")
        print(f"Total Routes: {total_routes}")
        print(f"Total Transactions: {total_transactions}")
        
        # Peak demand insights
        peak_hour = self.data_cache['gps_data'].groupby('hour')['occupancy_level'].mean().idxmax()
        peak_day = self.data_cache['gps_data'].groupby('day_of_week')['occupancy_level'].mean().idxmax()
        
        print(f"\nPeak demand hour: {peak_hour}:00")
        print(f"Peak demand day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][peak_day]}")
        
        # Performance insights
        avg_speed = self.data_cache['gps_data']['speed'].mean()
        avg_occupancy = self.data_cache['gps_data']['occupancy_level'].mean()
        
        print(f"\nAverage vehicle speed: {avg_speed:.2f} km/h")
        print(f"Average occupancy level: {avg_occupancy:.2f}%")
        
        # Recommendations
        print("\n" + "="*30)
        print("RECOMMENDATIONS")
        print("="*30)
        
        if avg_occupancy > 70:
            print("• High occupancy detected - consider increasing frequency during peak hours")
        
        if avg_speed < 20:
            print("• Low average speed detected - investigate traffic congestion causes")
        
        rush_hour_occupancy = self.data_cache['gps_data'][self.data_cache['gps_data']['is_rush_hour']]['occupancy_level'].mean()
        if rush_hour_occupancy > avg_occupancy * 1.5:
            print("• Significant rush hour congestion - implement dynamic scheduling")
        
        print("• Deploy predictive models to anticipate demand surges")
        print("• Implement real-time passenger information systems")
        print("• Consider route optimization based on demand patterns")

# Example usage
if __name__ == "__main__":
    # Initialize explorer
    explorer = MobilityDataExplorer("postgresql://mobility_user:mobility_pass@localhost:5432/mobility_platform")
    
    # Load data
    data = explorer.load_data()
    
    # Run comprehensive analysis
    explorer.analyze_temporal_patterns()
    explorer.analyze_spatial_patterns()
    explorer.analyze_anomalies()
    
    # Generate insights report
    explorer.generate_insights_report()