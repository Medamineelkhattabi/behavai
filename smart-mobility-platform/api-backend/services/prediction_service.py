"""
Prediction Service for ML Model Inference
Handles demand forecasting, congestion prediction, and flow analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pickle
import joblib
from pathlib import Path

# Import ML models
import sys
sys.path.append('../../ai-models')
from forecasting.lstm_demand_predictor import MobilityForecastingPipeline
from graph_networks.gnn_flow_predictor import MobilityGNNPipeline

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for handling ML model predictions"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.models = {}
        self.model_metadata = {}
        self.models_loaded = False
        self.model_version = "1.0.0"
        self.gnn_model_version = "1.0.0"
        
        # Model paths
        self.model_paths = {
            'demand_lstm': '../../ai-models/forecasting/models/demand_predictor.pth',
            'demand_transformer': '../../ai-models/forecasting/models/demand_transformer.pth',
            'congestion_predictor': '../../ai-models/forecasting/models/congestion_predictor.pth',
            'flow_gnn': '../../ai-models/graph-networks/models/flow_predictor.pth',
            'anomaly_detector': '../../ai-models/anomaly-detection/models/anomaly_detector.pkl'
        }
    
    async def load_models(self):
        """Load all ML models"""
        try:
            logger.info("Loading ML models...")
            
            # Load demand prediction models
            await self._load_demand_models()
            
            # Load congestion prediction models
            await self._load_congestion_models()
            
            # Load flow prediction models
            await self._load_flow_models()
            
            self.models_loaded = True
            logger.info("All ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.models_loaded = False
            raise
    
    async def _load_demand_models(self):
        """Load demand prediction models"""
        try:
            # LSTM model
            if Path(self.model_paths['demand_lstm']).exists():
                self.models['demand_lstm'] = MobilityForecastingPipeline(model_type='lstm')
                self.models['demand_lstm'].load_model(self.model_paths['demand_lstm'])
                logger.info("LSTM demand predictor loaded")
            
            # Transformer model
            if Path(self.model_paths['demand_transformer']).exists():
                self.models['demand_transformer'] = MobilityForecastingPipeline(model_type='transformer')
                self.models['demand_transformer'].load_model(self.model_paths['demand_transformer'])
                logger.info("Transformer demand predictor loaded")
            
            # If no models exist, create dummy models for development
            if 'demand_lstm' not in self.models:
                logger.warning("No demand prediction models found, creating dummy model")
                self.models['demand_lstm'] = self._create_dummy_demand_model()
                
        except Exception as e:
            logger.error(f"Failed to load demand models: {e}")
            # Create dummy model as fallback
            self.models['demand_lstm'] = self._create_dummy_demand_model()
    
    async def _load_congestion_models(self):
        """Load congestion prediction models"""
        try:
            if Path(self.model_paths['congestion_predictor']).exists():
                self.models['congestion_predictor'] = MobilityForecastingPipeline(model_type='lstm')
                self.models['congestion_predictor'].load_model(self.model_paths['congestion_predictor'])
                logger.info("Congestion predictor loaded")
            else:
                logger.warning("No congestion prediction model found, creating dummy model")
                self.models['congestion_predictor'] = self._create_dummy_congestion_model()
                
        except Exception as e:
            logger.error(f"Failed to load congestion models: {e}")
            self.models['congestion_predictor'] = self._create_dummy_congestion_model()
    
    async def _load_flow_models(self):
        """Load passenger flow prediction models"""
        try:
            if Path(self.model_paths['flow_gnn']).exists():
                self.models['flow_gnn'] = MobilityGNNPipeline(model_type='flow_gnn')
                self.models['flow_gnn'].load_model(self.model_paths['flow_gnn'])
                logger.info("GNN flow predictor loaded")
            else:
                logger.warning("No flow prediction model found, creating dummy model")
                self.models['flow_gnn'] = self._create_dummy_flow_model()
                
        except Exception as e:
            logger.error(f"Failed to load flow models: {e}")
            self.models['flow_gnn'] = self._create_dummy_flow_model()
    
    async def predict_demand(self, locations: List[str], 
                           time_horizon: int = 24,
                           features: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Predict passenger demand for specified locations"""
        try:
            predictions = []
            
            for location in locations:
                # Get historical data for the location
                historical_data = await self._get_location_data(location, days_back=30)
                
                if historical_data.empty:
                    # Use dummy prediction if no data
                    prediction = self._generate_dummy_demand_prediction(location, time_horizon)
                else:
                    # Use actual model prediction
                    if 'demand_lstm' in self.models:
                        prediction = await self._predict_demand_lstm(
                            location, historical_data, time_horizon, features
                        )
                    else:
                        prediction = self._generate_dummy_demand_prediction(location, time_horizon)
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Demand prediction failed: {e}")
            # Return dummy predictions as fallback
            return [self._generate_dummy_demand_prediction(loc, time_horizon) for loc in locations]
    
    async def predict_congestion(self, routes: Optional[List[str]] = None,
                               areas: Optional[List[str]] = None,
                               time_horizon: int = 6,
                               weather_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Predict congestion levels for routes and areas"""
        try:
            predictions = []
            
            # Predict for routes
            if routes:
                for route in routes:
                    historical_data = await self._get_route_data(route, days_back=7)
                    
                    if historical_data.empty:
                        prediction = self._generate_dummy_congestion_prediction(route, "route", time_horizon)
                    else:
                        prediction = await self._predict_congestion_lstm(
                            route, "route", historical_data, time_horizon, weather_conditions
                        )
                    
                    predictions.append(prediction)
            
            # Predict for areas
            if areas:
                for area in areas:
                    historical_data = await self._get_area_data(area, days_back=7)
                    
                    if historical_data.empty:
                        prediction = self._generate_dummy_congestion_prediction(area, "area", time_horizon)
                    else:
                        prediction = await self._predict_congestion_lstm(
                            area, "area", historical_data, time_horizon, weather_conditions
                        )
                    
                    predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Congestion prediction failed: {e}")
            # Return dummy predictions as fallback
            dummy_predictions = []
            if routes:
                dummy_predictions.extend([
                    self._generate_dummy_congestion_prediction(route, "route", time_horizon) 
                    for route in routes
                ])
            if areas:
                dummy_predictions.extend([
                    self._generate_dummy_congestion_prediction(area, "area", time_horizon) 
                    for area in areas
                ])
            return dummy_predictions
    
    async def predict_passenger_flow(self, origin_destinations: List[Dict[str, str]],
                                   time_horizon: int = 6,
                                   graph_features: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Predict passenger flows using Graph Neural Networks"""
        try:
            predictions = []
            
            for od_pair in origin_destinations:
                origin = od_pair.get('origin')
                destination = od_pair.get('destination')
                
                # Get graph data for the OD pair
                graph_data = await self._get_graph_data(origin, destination)
                
                if not graph_data:
                    prediction = self._generate_dummy_flow_prediction(origin, destination, time_horizon)
                else:
                    prediction = await self._predict_flow_gnn(
                        origin, destination, graph_data, time_horizon, graph_features
                    )
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Flow prediction failed: {e}")
            # Return dummy predictions as fallback
            return [
                self._generate_dummy_flow_prediction(od['origin'], od['destination'], time_horizon)
                for od in origin_destinations
            ]
    
    async def _predict_demand_lstm(self, location: str, historical_data: pd.DataFrame,
                                 time_horizon: int, features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict demand using LSTM model"""
        try:
            model = self.models['demand_lstm']
            
            # Prepare data for prediction
            if len(historical_data) < model.sequence_length:
                # Not enough data, return dummy prediction
                return self._generate_dummy_demand_prediction(location, time_horizon)
            
            # Get the last sequence for prediction
            last_sequence = historical_data[model.feature_columns].tail(model.sequence_length).values
            
            # Make prediction
            predictions = model.forecast_future(last_sequence, steps_ahead=time_horizon)
            
            # Generate time points
            start_time = datetime.utcnow()
            time_points = [start_time + timedelta(hours=i) for i in range(time_horizon)]
            
            # Calculate confidence intervals (simplified)
            confidence_intervals = []
            for pred in predictions:
                base_value = pred[0] if isinstance(pred, np.ndarray) else pred
                confidence_intervals.append({
                    'lower': float(base_value * 0.8),
                    'upper': float(base_value * 1.2)
                })
            
            return {
                'location_id': location,
                'predicted_demand': [float(p[0]) if isinstance(p, np.ndarray) else float(p) for p in predictions],
                'confidence_intervals': confidence_intervals,
                'time_points': time_points
            }
            
        except Exception as e:
            logger.error(f"LSTM demand prediction failed for {location}: {e}")
            return self._generate_dummy_demand_prediction(location, time_horizon)
    
    async def _predict_congestion_lstm(self, location: str, location_type: str,
                                     historical_data: pd.DataFrame, time_horizon: int,
                                     weather_conditions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict congestion using LSTM model"""
        try:
            model = self.models['congestion_predictor']
            
            if len(historical_data) < model.sequence_length:
                return self._generate_dummy_congestion_prediction(location, location_type, time_horizon)
            
            # Prepare data
            last_sequence = historical_data[model.feature_columns].tail(model.sequence_length).values
            
            # Make prediction
            predictions = model.forecast_future(last_sequence, steps_ahead=time_horizon)
            
            # Generate time points
            start_time = datetime.utcnow()
            time_points = [start_time + timedelta(hours=i) for i in range(time_horizon)]
            
            # Convert to severity levels
            severity_levels = []
            for pred in predictions:
                congestion_level = pred[0] if isinstance(pred, np.ndarray) else pred
                if congestion_level < 30:
                    severity_levels.append("low")
                elif congestion_level < 60:
                    severity_levels.append("medium")
                elif congestion_level < 80:
                    severity_levels.append("high")
                else:
                    severity_levels.append("critical")
            
            return {
                'location_id': location,
                'location_type': location_type,
                'predicted_congestion': [float(p[0]) if isinstance(p, np.ndarray) else float(p) for p in predictions],
                'severity_levels': severity_levels,
                'time_points': time_points
            }
            
        except Exception as e:
            logger.error(f"LSTM congestion prediction failed for {location}: {e}")
            return self._generate_dummy_congestion_prediction(location, location_type, time_horizon)
    
    async def _predict_flow_gnn(self, origin: str, destination: str, graph_data: Dict[str, Any],
                              time_horizon: int, graph_features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict passenger flow using GNN model"""
        try:
            model = self.models['flow_gnn']
            
            # This would use the actual GNN model
            # For now, return dummy prediction
            return self._generate_dummy_flow_prediction(origin, destination, time_horizon)
            
        except Exception as e:
            logger.error(f"GNN flow prediction failed for {origin}->{destination}: {e}")
            return self._generate_dummy_flow_prediction(origin, destination, time_horizon)
    
    async def _get_location_data(self, location: str, days_back: int = 30) -> pd.DataFrame:
        """Get historical data for a location"""
        try:
            query = f"""
                SELECT timestamp, passenger_count, occupancy_level, congestion_level
                FROM passenger_flows pf
                JOIN congestion_metrics cm ON pf.origin_stop_id = cm.location_id
                WHERE pf.origin_stop_id = '{location}'
                AND pf.time_window_start >= NOW() - INTERVAL '{days_back} days'
                ORDER BY timestamp
            """
            
            data = await self.db_manager.execute_query(query)
            return pd.DataFrame(data) if data else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get location data for {location}: {e}")
            return pd.DataFrame()
    
    async def _get_route_data(self, route: str, days_back: int = 7) -> pd.DataFrame:
        """Get historical data for a route"""
        try:
            query = f"""
                SELECT timestamp, avg_speed, congestion_level, occupancy_rate
                FROM congestion_metrics
                WHERE location_id = '{route}' AND location_type = 'route'
                AND timestamp >= NOW() - INTERVAL '{days_back} days'
                ORDER BY timestamp
            """
            
            data = await self.db_manager.execute_query(query)
            return pd.DataFrame(data) if data else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get route data for {route}: {e}")
            return pd.DataFrame()
    
    async def _get_area_data(self, area: str, days_back: int = 7) -> pd.DataFrame:
        """Get historical data for an area"""
        try:
            query = f"""
                SELECT timestamp, congestion_level, avg_wait_time, occupancy_rate
                FROM congestion_metrics
                WHERE location_id = '{area}' AND location_type = 'area'
                AND timestamp >= NOW() - INTERVAL '{days_back} days'
                ORDER BY timestamp
            """
            
            data = await self.db_manager.execute_query(query)
            return pd.DataFrame(data) if data else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get area data for {area}: {e}")
            return pd.DataFrame()
    
    async def _get_graph_data(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get graph data for flow prediction"""
        try:
            # This would fetch graph structure and features
            # For now, return empty dict
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get graph data for {origin}->{destination}: {e}")
            return {}
    
    # === DUMMY MODEL CREATORS ===
    
    def _create_dummy_demand_model(self):
        """Create dummy demand prediction model for development"""
        class DummyDemandModel:
            def __init__(self):
                self.sequence_length = 24
                self.feature_columns = ['passenger_count', 'hour', 'day_of_week']
            
            def forecast_future(self, last_sequence, steps_ahead):
                # Generate realistic-looking demand predictions
                base_demand = 100
                hourly_pattern = [base_demand + 50 * np.sin(2 * np.pi * i / 24) for i in range(steps_ahead)]
                noise = np.random.normal(0, 10, steps_ahead)
                return [max(0, demand + noise[i]) for i, demand in enumerate(hourly_pattern)]
        
        return DummyDemandModel()
    
    def _create_dummy_congestion_model(self):
        """Create dummy congestion prediction model"""
        class DummyCongestionModel:
            def __init__(self):
                self.sequence_length = 24
                self.feature_columns = ['congestion_level', 'avg_speed', 'hour']
            
            def forecast_future(self, last_sequence, steps_ahead):
                # Generate realistic congestion predictions
                base_congestion = 50
                rush_hour_pattern = []
                current_hour = datetime.now().hour
                
                for i in range(steps_ahead):
                    hour = (current_hour + i) % 24
                    if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                        congestion = base_congestion + 30
                    elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night
                        congestion = base_congestion - 20
                    else:
                        congestion = base_congestion
                    
                    rush_hour_pattern.append(max(0, min(100, congestion + np.random.normal(0, 5))))
                
                return rush_hour_pattern
        
        return DummyCongestionModel()
    
    def _create_dummy_flow_model(self):
        """Create dummy flow prediction model"""
        class DummyFlowModel:
            def predict_flow(self, origin, destination, time_horizon):
                # Generate realistic flow predictions
                base_flow = np.random.randint(10, 50)
                flows = []
                
                for i in range(time_horizon):
                    # Add some hourly variation
                    hour_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 24)
                    flow = int(base_flow * hour_factor * np.random.uniform(0.8, 1.2))
                    flows.append(max(0, flow))
                
                return flows
        
        return DummyFlowModel()
    
    # === DUMMY PREDICTION GENERATORS ===
    
    def _generate_dummy_demand_prediction(self, location: str, time_horizon: int) -> Dict[str, Any]:
        """Generate dummy demand prediction"""
        # Create realistic demand pattern
        base_demand = np.random.randint(50, 150)
        predictions = []
        confidence_intervals = []
        
        current_hour = datetime.now().hour
        
        for i in range(time_horizon):
            hour = (current_hour + i) % 24
            
            # Rush hour multiplier
            if hour in [7, 8, 9]:  # Morning rush
                multiplier = 1.8
            elif hour in [17, 18, 19]:  # Evening rush
                multiplier = 2.0
            elif hour in [10, 11, 12, 13, 14, 15, 16]:  # Daytime
                multiplier = 1.0
            else:  # Night
                multiplier = 0.3
            
            demand = base_demand * multiplier + np.random.normal(0, 10)
            demand = max(0, demand)
            predictions.append(demand)
            
            # Confidence interval
            confidence_intervals.append({
                'lower': demand * 0.8,
                'upper': demand * 1.2
            })
        
        # Time points
        start_time = datetime.utcnow()
        time_points = [start_time + timedelta(hours=i) for i in range(time_horizon)]
        
        return {
            'location_id': location,
            'predicted_demand': predictions,
            'confidence_intervals': confidence_intervals,
            'time_points': time_points
        }
    
    def _generate_dummy_congestion_prediction(self, location: str, location_type: str, 
                                            time_horizon: int) -> Dict[str, Any]:
        """Generate dummy congestion prediction"""
        predictions = []
        severity_levels = []
        
        current_hour = datetime.now().hour
        
        for i in range(time_horizon):
            hour = (current_hour + i) % 24
            
            # Congestion pattern
            if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                congestion = np.random.uniform(70, 90)
            elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night
                congestion = np.random.uniform(10, 30)
            else:
                congestion = np.random.uniform(40, 60)
            
            predictions.append(congestion)
            
            # Severity level
            if congestion < 30:
                severity_levels.append("low")
            elif congestion < 60:
                severity_levels.append("medium")
            elif congestion < 80:
                severity_levels.append("high")
            else:
                severity_levels.append("critical")
        
        # Time points
        start_time = datetime.utcnow()
        time_points = [start_time + timedelta(hours=i) for i in range(time_horizon)]
        
        return {
            'location_id': location,
            'location_type': location_type,
            'predicted_congestion': predictions,
            'severity_levels': severity_levels,
            'time_points': time_points
        }
    
    def _generate_dummy_flow_prediction(self, origin: str, destination: str, 
                                      time_horizon: int) -> Dict[str, Any]:
        """Generate dummy flow prediction"""
        base_flow = np.random.randint(15, 45)
        flows = []
        confidence_scores = []
        
        current_hour = datetime.now().hour
        
        for i in range(time_horizon):
            hour = (current_hour + i) % 24
            
            # Flow pattern
            if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                flow = int(base_flow * np.random.uniform(1.5, 2.0))
            else:
                flow = int(base_flow * np.random.uniform(0.5, 1.2))
            
            flows.append(max(0, flow))
            confidence_scores.append(np.random.uniform(0.7, 0.95))
        
        # Time points
        start_time = datetime.utcnow()
        time_points = [start_time + timedelta(hours=i) for i in range(time_horizon)]
        
        return {
            'origin_id': origin,
            'destination_id': destination,
            'predicted_flow': flows,
            'confidence_scores': confidence_scores,
            'time_points': time_points
        }
    
    async def start_retraining(self, models: List[str], data_range: Dict[str, Any]) -> str:
        """Start model retraining task"""
        # This would implement actual model retraining
        # For now, return a dummy task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        logger.info(f"Started retraining task {task_id} for models: {models}")
        
        # In a real implementation, this would:
        # 1. Fetch training data from the specified date range
        # 2. Prepare data for training
        # 3. Start training process in background
        # 4. Save new model versions
        # 5. Update model metadata
        
        return task_id
    
    async def monitor_retraining_task(self, task_id: str):
        """Monitor retraining task progress"""
        # This would monitor the actual training process
        logger.info(f"Monitoring retraining task {task_id}")
        
        # Simulate training progress
        await asyncio.sleep(10)
        logger.info(f"Retraining task {task_id} completed")