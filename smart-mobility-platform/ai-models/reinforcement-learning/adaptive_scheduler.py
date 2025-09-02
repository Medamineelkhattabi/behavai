"""
Reinforcement Learning Agent for Adaptive Scheduling and Dynamic Dispatch
Optimizes vehicle schedules and dispatch decisions based on real-time conditions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class Vehicle:
    """Vehicle representation"""
    vehicle_id: str
    capacity: int
    current_location: Tuple[float, float]
    route_id: str
    occupancy: int = 0
    speed: float = 0.0
    fuel_level: float = 100.0
    maintenance_due: bool = False
    
@dataclass
class Stop:
    """Stop/Station representation"""
    stop_id: str
    location: Tuple[float, float]
    waiting_passengers: int = 0
    expected_arrivals: int = 0
    priority: float = 1.0

@dataclass
class Route:
    """Route representation"""
    route_id: str
    stops: List[str]
    base_schedule: List[float]  # Base travel times between stops
    current_delays: List[float] = None

class MobilitySchedulingEnv(gym.Env):
    """Custom environment for mobility scheduling optimization"""
    
    def __init__(self, vehicles: List[Vehicle], stops: List[Stop], 
                 routes: List[Route], time_horizon: int = 24):
        super(MobilitySchedulingEnv, self).__init__()
        
        self.vehicles = {v.vehicle_id: v for v in vehicles}
        self.stops = {s.stop_id: s for s in stops}
        self.routes = {r.route_id: r for r in routes}
        
        self.time_horizon = time_horizon
        self.current_time = 0
        self.time_step = 0.25  # 15-minute intervals
        
        # State space: [time, vehicle_states, stop_states, weather, events]
        vehicle_state_size = len(vehicles) * 6  # location(2), occupancy, speed, fuel, maintenance
        stop_state_size = len(stops) * 3  # waiting_passengers, expected_arrivals, priority
        time_features = 4  # hour, day_of_week, is_rush_hour, is_weekend
        external_features = 5  # weather, special_events, congestion_level, etc.
        
        self.state_size = time_features + vehicle_state_size + stop_state_size + external_features
        
        # Action space: [vehicle_schedule_adjustments, dispatch_decisions]
        # For each vehicle: [schedule_adjustment (-2 to +2 intervals), dispatch_now (0/1)]
        self.action_size = len(vehicles) * 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_size,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-2, 0] * len(vehicles)), 
            high=np.array([2, 1] * len(vehicles)), 
            dtype=np.float32
        )
        
        # Simulation state
        self.passenger_demand_history = []
        self.congestion_history = []
        self.weather_conditions = {'temperature': 25, 'precipitation': 0, 'wind_speed': 5}
        self.special_events = []
        
        # Performance metrics
        self.total_passengers_served = 0
        self.total_waiting_time = 0
        self.total_fuel_consumed = 0
        self.total_delays = 0
        self.service_level = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_time = 0
        self.time_step_count = 0
        
        # Reset vehicles to initial state
        for vehicle in self.vehicles.values():
            vehicle.occupancy = 0
            vehicle.fuel_level = 100.0
            vehicle.maintenance_due = False
        
        # Reset stops
        for stop in self.stops.values():
            stop.waiting_passengers = np.random.poisson(5)
            stop.expected_arrivals = np.random.poisson(10)
        
        # Reset metrics
        self.total_passengers_served = 0
        self.total_waiting_time = 0
        self.total_fuel_consumed = 0
        self.total_delays = 0
        
        return self._get_state(), {}
    
    def step(self, action):
        """Execute one time step in the environment"""
        action = np.array(action).reshape(-1, 2)  # [vehicle_id, [schedule_adj, dispatch]]
        
        # Process actions for each vehicle
        rewards = []
        for i, (vehicle_id, vehicle) in enumerate(self.vehicles.items()):
            schedule_adjustment = action[i, 0]
            dispatch_decision = action[i, 1] > 0.5
            
            # Apply scheduling adjustment
            vehicle_reward = self._process_vehicle_action(
                vehicle, schedule_adjustment, dispatch_decision
            )
            rewards.append(vehicle_reward)
        
        # Simulate environment dynamics
        self._simulate_passenger_dynamics()
        self._simulate_traffic_conditions()
        self._update_vehicle_states()
        
        # Calculate global reward
        total_reward = sum(rewards) + self._calculate_system_reward()
        
        # Update time
        self.current_time += self.time_step
        self.time_step_count += 1
        
        # Check termination
        done = self.current_time >= self.time_horizon
        
        # Additional info
        info = {
            'passengers_served': self.total_passengers_served,
            'average_waiting_time': self.total_waiting_time / max(1, self.total_passengers_served),
            'fuel_efficiency': self.total_fuel_consumed / max(1, len(self.vehicles)),
            'service_level': self.service_level
        }
        
        return self._get_state(), total_reward, done, False, info
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        state = []
        
        # Time features
        hour = (self.current_time % 24)
        day_of_week = int(self.current_time // 24) % 7
        is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        
        state.extend([hour / 24, day_of_week / 7, is_rush_hour, is_weekend])
        
        # Vehicle states
        for vehicle in self.vehicles.values():
            state.extend([
                vehicle.current_location[0] / 100,  # Normalized coordinates
                vehicle.current_location[1] / 100,
                vehicle.occupancy / vehicle.capacity,
                vehicle.speed / 60,  # Normalized speed
                vehicle.fuel_level / 100,
                1 if vehicle.maintenance_due else 0
            ])
        
        # Stop states
        for stop in self.stops.values():
            state.extend([
                min(stop.waiting_passengers / 50, 1),  # Normalized waiting passengers
                min(stop.expected_arrivals / 100, 1),  # Normalized expected arrivals
                stop.priority
            ])
        
        # External conditions
        state.extend([
            self.weather_conditions['temperature'] / 40,
            self.weather_conditions['precipitation'] / 10,
            self.weather_conditions['wind_speed'] / 30,
            len(self.special_events) / 5,  # Number of active events
            self._get_congestion_level() / 100
        ])
        
        return np.array(state, dtype=np.float32)
    
    def _process_vehicle_action(self, vehicle: Vehicle, 
                               schedule_adjustment: float, 
                               dispatch_decision: bool) -> float:
        """Process actions for a single vehicle"""
        reward = 0
        
        # Schedule adjustment reward/penalty
        if abs(schedule_adjustment) > 0.1:
            # Penalty for frequent schedule changes
            reward -= abs(schedule_adjustment) * 0.1
            
            # Reward for appropriate adjustments based on conditions
            congestion = self._get_congestion_level()
            demand = self._get_current_demand()
            
            if congestion > 70 and schedule_adjustment > 0:
                reward += 0.5  # Reward for increasing frequency during congestion
            elif demand < 30 and schedule_adjustment < 0:
                reward += 0.3  # Reward for reducing frequency during low demand
        
        # Dispatch decision reward
        if dispatch_decision:
            # Calculate benefit of immediate dispatch
            nearby_demand = self._calculate_nearby_demand(vehicle)
            if nearby_demand > vehicle.capacity * 0.7:
                reward += 1.0  # High reward for dispatching when demand is high
            else:
                reward -= 0.5  # Penalty for unnecessary dispatch
        
        return reward
    
    def _simulate_passenger_dynamics(self):
        """Simulate passenger arrival and departure"""
        hour = int(self.current_time % 24)
        
        # Demand patterns based on time of day
        base_demand_multiplier = self._get_demand_multiplier(hour)
        
        for stop in self.stops.values():
            # New passenger arrivals
            base_arrivals = np.random.poisson(2 * base_demand_multiplier)
            
            # Weather and event effects
            weather_effect = self._get_weather_demand_effect()
            event_effect = self._get_event_demand_effect(stop)
            
            new_arrivals = int(base_arrivals * weather_effect * event_effect)
            stop.waiting_passengers += new_arrivals
            stop.expected_arrivals = max(0, stop.expected_arrivals - new_arrivals)
            
            # Passenger departure (boarding vehicles)
            for vehicle in self.vehicles.values():
                if self._is_vehicle_at_stop(vehicle, stop):
                    boardings = min(
                        stop.waiting_passengers,
                        vehicle.capacity - vehicle.occupancy
                    )
                    
                    stop.waiting_passengers -= boardings
                    vehicle.occupancy += boardings
                    self.total_passengers_served += boardings
                    
                    # Calculate waiting time (simplified)
                    avg_waiting_time = stop.waiting_passengers * 0.5
                    self.total_waiting_time += avg_waiting_time
    
    def _simulate_traffic_conditions(self):
        """Simulate traffic and congestion effects"""
        hour = int(self.current_time % 24)
        
        # Rush hour effects
        if hour in [7, 8, 9, 17, 18, 19]:
            congestion_multiplier = 1.5
        elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:
            congestion_multiplier = 0.7
        else:
            congestion_multiplier = 1.0
        
        # Weather effects on traffic
        if self.weather_conditions['precipitation'] > 5:
            congestion_multiplier *= 1.3
        
        # Update vehicle speeds based on congestion
        for vehicle in self.vehicles.values():
            base_speed = 30  # km/h
            vehicle.speed = base_speed / congestion_multiplier
            
            # Fuel consumption based on speed and congestion
            fuel_consumption = 0.1 * (1 + (1 / congestion_multiplier - 1) * 0.5)
            vehicle.fuel_level = max(0, vehicle.fuel_level - fuel_consumption)
            self.total_fuel_consumed += fuel_consumption
    
    def _update_vehicle_states(self):
        """Update vehicle positions and states"""
        for vehicle in self.vehicles.values():
            # Simple movement simulation
            if vehicle.speed > 0:
                # Move vehicle along route (simplified)
                route = self.routes.get(vehicle.route_id)
                if route:
                    # Update position based on speed and time
                    distance_moved = vehicle.speed * self.time_step
                    # Simplified position update (would be more complex in reality)
                    
            # Maintenance checks
            if vehicle.fuel_level < 10:
                vehicle.maintenance_due = True
            
            # Drop off passengers (simplified)
            if np.random.random() < 0.3:  # 30% chance of passenger drop-off
                dropoffs = min(vehicle.occupancy, np.random.poisson(2))
                vehicle.occupancy -= dropoffs
    
    def _calculate_system_reward(self) -> float:
        """Calculate system-wide performance reward"""
        reward = 0
        
        # Service level reward
        total_waiting = sum(stop.waiting_passengers for stop in self.stops.values())
        if total_waiting < 50:
            reward += 2.0
        elif total_waiting > 200:
            reward -= 1.0
        
        # Fuel efficiency reward
        avg_fuel_level = sum(v.fuel_level for v in self.vehicles.values()) / len(self.vehicles)
        if avg_fuel_level > 50:
            reward += 0.5
        
        # Fleet utilization reward
        active_vehicles = sum(1 for v in self.vehicles.values() if v.occupancy > 0)
        utilization = active_vehicles / len(self.vehicles)
        reward += utilization * 1.0
        
        return reward
    
    def _get_demand_multiplier(self, hour: int) -> float:
        """Get demand multiplier based on hour"""
        # Peak hours: 7-9 AM and 5-7 PM
        if hour in [7, 8, 9]:
            return 2.0
        elif hour in [17, 18, 19]:
            return 2.5
        elif hour in [10, 11, 12, 13, 14, 15, 16]:
            return 1.0
        else:
            return 0.3
    
    def _get_weather_demand_effect(self) -> float:
        """Calculate weather effect on demand"""
        effect = 1.0
        
        # Rain increases demand
        if self.weather_conditions['precipitation'] > 2:
            effect *= 1.3
        
        # Extreme temperatures affect demand
        temp = self.weather_conditions['temperature']
        if temp < 5 or temp > 40:
            effect *= 1.2
        
        return effect
    
    def _get_event_demand_effect(self, stop: Stop) -> float:
        """Calculate special event effect on stop demand"""
        effect = 1.0
        
        for event in self.special_events:
            # Distance-based effect (simplified)
            if event.get('location') and stop.location:
                # Would calculate actual distance in real implementation
                distance = abs(event['location'][0] - stop.location[0]) + abs(event['location'][1] - stop.location[1])
                if distance < 0.1:  # Very close to event
                    effect *= event.get('impact_multiplier', 2.0)
        
        return effect
    
    def _get_congestion_level(self) -> float:
        """Get current congestion level"""
        hour = int(self.current_time % 24)
        base_congestion = 30
        
        if hour in [7, 8, 9, 17, 18, 19]:
            base_congestion = 80
        elif hour in [10, 11, 12, 13, 14, 15, 16]:
            base_congestion = 50
        
        # Add weather effects
        if self.weather_conditions['precipitation'] > 5:
            base_congestion += 20
        
        return min(100, base_congestion)
    
    def _get_current_demand(self) -> float:
        """Get current system-wide demand level"""
        total_waiting = sum(stop.waiting_passengers for stop in self.stops.values())
        return min(100, total_waiting)
    
    def _calculate_nearby_demand(self, vehicle: Vehicle) -> int:
        """Calculate passenger demand near vehicle"""
        nearby_demand = 0
        
        for stop in self.stops.values():
            # Distance calculation (simplified)
            distance = abs(vehicle.current_location[0] - stop.location[0]) + \
                      abs(vehicle.current_location[1] - stop.location[1])
            
            if distance < 0.1:  # Within 0.1 units
                nearby_demand += stop.waiting_passengers
        
        return nearby_demand
    
    def _is_vehicle_at_stop(self, vehicle: Vehicle, stop: Stop) -> bool:
        """Check if vehicle is at stop"""
        distance = abs(vehicle.current_location[0] - stop.location[0]) + \
                  abs(vehicle.current_location[1] - stop.location[1])
        return distance < 0.05  # Threshold for being "at" stop

class AdaptiveScheduler:
    """Main class for adaptive scheduling using RL"""
    
    def __init__(self, vehicles: List[Vehicle], stops: List[Stop], 
                 routes: List[Route], algorithm: str = 'PPO'):
        self.env = MobilitySchedulingEnv(vehicles, stops, routes)
        self.algorithm = algorithm
        self.model = None
        self.training_history = []
        
    def create_model(self, **kwargs):
        """Create RL model"""
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                verbose=1,
                learning_rate=kwargs.get('learning_rate', 3e-4),
                n_steps=kwargs.get('n_steps', 2048),
                batch_size=kwargs.get('batch_size', 64),
                n_epochs=kwargs.get('n_epochs', 10),
                gamma=kwargs.get('gamma', 0.99),
                clip_range=kwargs.get('clip_range', 0.2)
            )
        elif self.algorithm == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                self.env,
                verbose=1,
                learning_rate=kwargs.get('learning_rate', 1e-4),
                buffer_size=kwargs.get('buffer_size', 100000),
                learning_starts=kwargs.get('learning_starts', 1000),
                batch_size=kwargs.get('batch_size', 32),
                gamma=kwargs.get('gamma', 0.99)
            )
        elif self.algorithm == 'A2C':
            self.model = A2C(
                'MlpPolicy',
                self.env,
                verbose=1,
                learning_rate=kwargs.get('learning_rate', 7e-4),
                n_steps=kwargs.get('n_steps', 5),
                gamma=kwargs.get('gamma', 0.99)
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return self.model
    
    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000):
        """Train the RL model"""
        if self.model is None:
            self.create_model()
        
        # Create evaluation environment
        eval_env = MobilitySchedulingEnv(
            list(self.env.vehicles.values()),
            list(self.env.stops.values()),
            list(self.env.routes.values())
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        return self.model
    
    def predict_schedule_adjustments(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal schedule adjustments"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert state to observation format
        obs = self._state_to_observation(current_state)
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to schedule adjustments
        adjustments = self._action_to_adjustments(action)
        
        return adjustments
    
    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to observation array"""
        # This would need to match the observation space format
        # Implementation depends on the specific state format
        obs = []
        
        # Time features
        current_time = state.get('current_time', 0)
        hour = (current_time % 24) / 24
        day_of_week = (int(current_time // 24) % 7) / 7
        is_rush_hour = 1 if int(current_time % 24) in [7, 8, 9, 17, 18, 19] else 0
        is_weekend = 1 if int(current_time // 24) % 7 >= 5 else 0
        
        obs.extend([hour, day_of_week, is_rush_hour, is_weekend])
        
        # Vehicle states
        vehicles = state.get('vehicles', {})
        for vehicle_id in sorted(vehicles.keys()):
            vehicle = vehicles[vehicle_id]
            obs.extend([
                vehicle.get('latitude', 0) / 100,
                vehicle.get('longitude', 0) / 100,
                vehicle.get('occupancy', 0) / vehicle.get('capacity', 1),
                vehicle.get('speed', 0) / 60,
                vehicle.get('fuel_level', 100) / 100,
                1 if vehicle.get('maintenance_due', False) else 0
            ])
        
        # Stop states
        stops = state.get('stops', {})
        for stop_id in sorted(stops.keys()):
            stop = stops[stop_id]
            obs.extend([
                min(stop.get('waiting_passengers', 0) / 50, 1),
                min(stop.get('expected_arrivals', 0) / 100, 1),
                stop.get('priority', 1.0)
            ])
        
        # External conditions
        weather = state.get('weather', {})
        obs.extend([
            weather.get('temperature', 25) / 40,
            weather.get('precipitation', 0) / 10,
            weather.get('wind_speed', 5) / 30,
            len(state.get('special_events', [])) / 5,
            state.get('congestion_level', 30) / 100
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _action_to_adjustments(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert model action to schedule adjustments"""
        adjustments = {}
        
        action = action.reshape(-1, 2)
        
        for i, vehicle_id in enumerate(sorted(self.env.vehicles.keys())):
            schedule_adj = action[i, 0]
            dispatch_decision = action[i, 1] > 0.5
            
            adjustments[vehicle_id] = {
                'schedule_adjustment': float(schedule_adj),
                'immediate_dispatch': bool(dispatch_decision)
            }
        
        return adjustments
    
    def evaluate_performance(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        total_rewards = []
        performance_metrics = {
            'passengers_served': [],
            'average_waiting_time': [],
            'fuel_efficiency': [],
            'service_level': []
        }
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            for key in performance_metrics:
                if key in info:
                    performance_metrics[key].append(info[key])
        
        # Calculate average metrics
        results = {
            'average_reward': np.mean(total_rewards),
            'reward_std': np.std(total_rewards)
        }
        
        for key, values in performance_metrics.items():
            if values:
                results[f'avg_{key}'] = np.mean(values)
                results[f'std_{key}'] = np.std(values)
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(filepath)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(filepath)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(filepath)
        
        print(f"Model loaded from {filepath}")
    
    def visualize_performance(self, results: Dict[str, float]):
        """Visualize performance metrics"""
        metrics = {k: v for k, v in results.items() if not k.endswith('_std')}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward plot
        axes[0, 0].bar(['Average Reward'], [results['average_reward']])
        axes[0, 0].errorbar(['Average Reward'], [results['average_reward']], 
                           yerr=[results['reward_std']], fmt='o', color='red')
        axes[0, 0].set_title('Average Episode Reward')
        axes[0, 0].set_ylabel('Reward')
        
        # Performance metrics
        perf_metrics = ['avg_passengers_served', 'avg_average_waiting_time', 
                       'avg_fuel_efficiency', 'avg_service_level']
        perf_values = [results.get(m, 0) for m in perf_metrics]
        perf_labels = ['Passengers\nServed', 'Avg Waiting\nTime', 
                      'Fuel\nEfficiency', 'Service\nLevel']
        
        axes[0, 1].bar(perf_labels, perf_values)
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Value')
        
        # Additional visualizations can be added here
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample vehicles
    vehicles = [
        Vehicle(f"vehicle_{i}", capacity=50, 
               current_location=(np.random.uniform(0, 10), np.random.uniform(0, 10)),
               route_id=f"route_{i % 3}")
        for i in range(10)
    ]
    
    # Create sample stops
    stops = [
        Stop(f"stop_{i}", 
             location=(np.random.uniform(0, 10), np.random.uniform(0, 10)),
             waiting_passengers=np.random.randint(0, 20))
        for i in range(20)
    ]
    
    # Create sample routes
    routes = [
        Route(f"route_{i}", 
              stops=[f"stop_{j}" for j in range(i*5, (i+1)*5)],
              base_schedule=[2.0, 3.0, 2.5, 4.0, 3.5])
        for i in range(3)
    ]
    
    # Initialize scheduler
    scheduler = AdaptiveScheduler(vehicles, stops, routes, algorithm='PPO')
    
    # Train model
    print("Training adaptive scheduler...")
    scheduler.train(total_timesteps=50000)
    
    # Evaluate performance
    print("Evaluating performance...")
    results = scheduler.evaluate_performance(n_episodes=10)
    
    print("\nPerformance Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Visualize results
    scheduler.visualize_performance(results)
    
    # Save model
    scheduler.save_model("adaptive_scheduler_model")
    
    # Test prediction
    sample_state = {
        'current_time': 8.5,  # 8:30 AM
        'vehicles': {v.vehicle_id: {
            'latitude': v.current_location[0],
            'longitude': v.current_location[1],
            'occupancy': v.occupancy,
            'capacity': v.capacity,
            'speed': v.speed,
            'fuel_level': v.fuel_level,
            'maintenance_due': v.maintenance_due
        } for v in vehicles},
        'stops': {s.stop_id: {
            'waiting_passengers': s.waiting_passengers,
            'expected_arrivals': s.expected_arrivals,
            'priority': s.priority
        } for s in stops},
        'weather': {'temperature': 25, 'precipitation': 0, 'wind_speed': 5},
        'special_events': [],
        'congestion_level': 60
    }
    
    adjustments = scheduler.predict_schedule_adjustments(sample_state)
    print(f"\nPredicted schedule adjustments: {adjustments}")
    
    print("Adaptive scheduler training and evaluation completed!")