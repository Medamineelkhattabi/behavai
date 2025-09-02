"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

# === ENUMS ===

class TransportMode(str, Enum):
    BUS = "bus"
    FERRY = "ferry"
    TRAIN = "train"
    METRO = "metro"
    TRAM = "tram"

class AnomalyType(str, Enum):
    DEMAND_SURGE = "demand_surge"
    DEMAND_DROP = "demand_drop"
    CONGESTION_ANOMALY = "congestion_anomaly"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    ROUTE_DISRUPTION = "route_disruption"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    SPATIAL_ANOMALY = "spatial_anomaly"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# === BASE MODELS ===

class BaseResponse(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    message: Optional[str] = None

class Location(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    
class TimeRange(BaseModel):
    start_time: datetime
    end_time: datetime
    
    @validator('end_time')
    def end_after_start(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v

# === HEALTH CHECK ===

class HealthResponse(BaseResponse):
    status: str
    database_status: bool
    redis_status: bool
    models_status: Dict[str, bool]

# === PREDICTION MODELS ===

class DemandPredictionRequest(BaseModel):
    locations: List[str] = Field(..., description="List of location IDs")
    time_horizon: int = Field(24, ge=1, le=168, description="Prediction horizon in hours")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features")

class DemandPrediction(BaseModel):
    location_id: str
    predicted_demand: List[float]
    confidence_intervals: List[Dict[str, float]]
    time_points: List[datetime]

class DemandPredictionResponse(BaseResponse):
    predictions: List[DemandPrediction]
    model_version: str
    confidence_scores: List[float]

class CongestionPredictionRequest(BaseModel):
    routes: Optional[List[str]] = Field(None, description="Route IDs")
    areas: Optional[List[str]] = Field(None, description="Area IDs")
    time_horizon: int = Field(6, ge=1, le=48, description="Prediction horizon in hours")
    weather_conditions: Optional[Dict[str, Any]] = None

class CongestionPrediction(BaseModel):
    location_id: str
    location_type: str  # 'route' or 'area'
    predicted_congestion: List[float]
    severity_levels: List[str]
    time_points: List[datetime]

class CongestionPredictionResponse(BaseResponse):
    predictions: List[CongestionPrediction]
    model_version: str

class FlowPredictionRequest(BaseModel):
    origin_destinations: List[Dict[str, str]] = Field(..., description="Origin-destination pairs")
    time_horizon: int = Field(6, ge=1, le=24, description="Prediction horizon in hours")
    graph_features: Optional[Dict[str, Any]] = None

class FlowPrediction(BaseModel):
    origin_id: str
    destination_id: str
    predicted_flow: List[int]
    confidence_scores: List[float]
    time_points: List[datetime]

class FlowPredictionResponse(BaseResponse):
    flow_predictions: List[FlowPrediction]
    model_version: str

# === ANOMALY DETECTION MODELS ===

class AnomalyDetectionRequest(BaseModel):
    data: Dict[str, List[float]] = Field(..., description="Time series data")
    detection_methods: List[str] = Field(["isolation_forest", "autoencoder"], 
                                       description="Anomaly detection methods")
    sensitivity: float = Field(0.1, ge=0.01, le=0.5, description="Detection sensitivity")

class Anomaly(BaseModel):
    timestamp: datetime
    location_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    score: float = Field(..., ge=0, le=1)
    description: str
    affected_metrics: List[str]

class AnomalyDetectionResponse(BaseResponse):
    anomalies: List[Anomaly]
    detection_methods_used: List[str]

class AnomalyAlert(BaseModel):
    alert_id: str
    timestamp: datetime
    location_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    description: str
    status: str = "active"
    resolved_at: Optional[datetime] = None

# === OPTIMIZATION MODELS ===

class ScheduleOptimizationRequest(BaseModel):
    routes: List[str] = Field(..., description="Routes to optimize")
    constraints: Dict[str, Any] = Field({}, description="Optimization constraints")
    objectives: List[str] = Field(["minimize_waiting_time", "maximize_efficiency"],
                                description="Optimization objectives")

class ScheduleOptimizationResponse(BaseResponse):
    task_id: str
    status: TaskStatus
    estimated_completion_time: datetime

class OptimizedSchedule(BaseModel):
    route_id: str
    original_schedule: List[datetime]
    optimized_schedule: List[datetime]
    expected_improvement: Dict[str, float]
    confidence_score: float

class OptimizationTaskStatus(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = Field(..., ge=0, le=100)
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[List[OptimizedSchedule]] = None
    error_message: Optional[str] = None

class DispatchRecommendationRequest(BaseModel):
    current_state: Dict[str, Any] = Field(..., description="Current system state")
    demand_forecast: Dict[str, List[float]] = Field(..., description="Demand forecast")
    constraints: Dict[str, Any] = Field({}, description="Dispatch constraints")

class DispatchRecommendation(BaseModel):
    vehicle_id: str
    action: str  # 'dispatch', 'wait', 'redirect'
    target_location: Optional[str] = None
    priority: int = Field(..., ge=1, le=10)
    reasoning: str
    expected_benefit: float

class DispatchRecommendationResponse(BaseResponse):
    recommendations: List[DispatchRecommendation]

# === REAL-TIME DATA MODELS ===

class VehicleStatus(BaseModel):
    vehicle_id: str
    route_id: str
    transport_mode: TransportMode
    current_location: Location
    occupancy_level: int = Field(..., ge=0, le=100)
    speed: float = Field(..., ge=0)
    heading: Optional[float] = Field(None, ge=0, le=360)
    next_stop: Optional[str] = None
    estimated_arrival: Optional[datetime] = None
    status: str = "active"  # active, maintenance, out_of_service
    last_updated: datetime

class StopStatus(BaseModel):
    stop_id: str
    name: str
    location: Location
    waiting_passengers: int = Field(..., ge=0)
    expected_arrivals: List[Dict[str, Any]]  # vehicle arrivals
    congestion_level: int = Field(..., ge=0, le=100)
    accessibility_status: str = "normal"
    last_updated: datetime

class CongestionOverview(BaseModel):
    overall_congestion_level: int = Field(..., ge=0, le=100)
    congested_routes: List[str]
    congested_areas: List[str]
    average_speed: float
    incidents: List[Dict[str, Any]]
    timestamp: datetime

# === HISTORICAL DATA MODELS ===

class HistoricalDataResponse(BaseResponse):
    data: List[Dict[str, Any]]
    start_date: datetime
    end_date: datetime
    aggregation: str
    total_records: int

class KPIResponse(BaseResponse):
    kpis: Dict[str, float]
    period: str

# === SYSTEM MANAGEMENT MODELS ===

class RetrainingRequest(BaseModel):
    models: List[str] = Field(..., description="Models to retrain")
    data_range: TimeRange = Field(..., description="Data range for training")

class RetrainingResponse(BaseResponse):
    task_id: str
    status: TaskStatus
    models: List[str]

class SystemMetricsResponse(BaseResponse):
    metrics: Dict[str, Any]

# === PASSENGER-FACING MODELS ===

class TravelRecommendation(BaseModel):
    route_options: List[Dict[str, Any]]
    estimated_duration: int  # minutes
    estimated_cost: Optional[float] = None
    congestion_info: str
    accessibility_info: str
    confidence_score: float = Field(..., ge=0, le=1)

class TravelRecommendationResponse(BaseResponse):
    recommendations: List[TravelRecommendation]

class ServiceDisruption(BaseModel):
    disruption_id: str
    title: str
    description: str
    affected_routes: List[str]
    affected_stops: List[str]
    severity: SeverityLevel
    start_time: datetime
    expected_end_time: Optional[datetime] = None
    alternative_routes: List[str] = []
    status: str = "active"

# === GRAPH NETWORK MODELS ===

class NetworkNode(BaseModel):
    node_id: str
    name: str
    location: Location
    node_type: str
    features: Dict[str, float]

class NetworkEdge(BaseModel):
    source_id: str
    target_id: str
    weight: float
    edge_type: str
    features: Dict[str, float]

class GraphNetworkRequest(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    prediction_type: str = "flow"

class GraphNetworkResponse(BaseResponse):
    predictions: Dict[str, Any]
    model_version: str

# === WEATHER AND EVENTS ===

class WeatherConditions(BaseModel):
    location: str
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    weather_condition: str
    timestamp: datetime

class SpecialEvent(BaseModel):
    event_id: str
    name: str
    description: Optional[str] = None
    event_type: str
    start_time: datetime
    end_time: datetime
    location: str
    expected_attendance: Optional[int] = None
    transport_impact_level: int = Field(..., ge=1, le=5)

# === BATCH PROCESSING ===

class BatchProcessingRequest(BaseModel):
    job_type: str
    parameters: Dict[str, Any]
    data_sources: List[str]
    output_destination: str

class BatchProcessingResponse(BaseResponse):
    job_id: str
    status: TaskStatus
    estimated_completion: datetime

# === VALIDATION MODELS ===

class ValidationRequest(BaseModel):
    model_type: str
    validation_data: Dict[str, Any]
    metrics: List[str] = ["mae", "rmse", "mape"]

class ValidationResponse(BaseResponse):
    model_type: str
    validation_metrics: Dict[str, float]
    performance_summary: str

# === EXPORT MODELS ===

class ExportRequest(BaseModel):
    data_type: str
    date_range: TimeRange
    format: str = "csv"  # csv, json, parquet
    filters: Optional[Dict[str, Any]] = None

class ExportResponse(BaseResponse):
    download_url: str
    file_size: int
    expires_at: datetime