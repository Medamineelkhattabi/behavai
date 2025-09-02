"""
FastAPI Backend for Smart Mobility Platform
Provides REST API endpoints for ML model inference, data access, and system management
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import redis
import asyncio
import logging
from contextlib import asynccontextmanager

# Internal imports
from .models.database import DatabaseManager
from .models.schemas import *
from .services.prediction_service import PredictionService
from .services.anomaly_service import AnomalyService
from .services.optimization_service import OptimizationService
from .services.real_time_service import RealTimeService
from .utils.auth import verify_token
from .utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
db_manager = None
prediction_service = None
anomaly_service = None
optimization_service = None
real_time_service = None
metrics_collector = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global db_manager, prediction_service, anomaly_service, optimization_service
    global real_time_service, metrics_collector, redis_client
    
    logger.info("Starting Smart Mobility Platform API...")
    
    # Initialize database connection
    db_manager = DatabaseManager()
    await db_manager.connect()
    
    # Initialize Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Initialize services
    prediction_service = PredictionService(db_manager)
    anomaly_service = AnomalyService(db_manager)
    optimization_service = OptimizationService(db_manager)
    real_time_service = RealTimeService(redis_client, db_manager)
    metrics_collector = MetricsCollector()
    
    # Load ML models
    await prediction_service.load_models()
    await anomaly_service.load_models()
    await optimization_service.load_models()
    
    # Start background tasks
    asyncio.create_task(real_time_service.start_monitoring())
    
    logger.info("Smart Mobility Platform API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Smart Mobility Platform API...")
    await db_manager.disconnect()
    if redis_client:
        redis_client.close()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Smart Mobility Platform API",
    description="AI-powered behavior-aware smart mobility platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        user_info = verify_token(credentials.credentials)
        return user_info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = await db_manager.health_check()
        
        # Check Redis connection
        redis_status = redis_client.ping()
        
        # Check ML models
        models_status = {
            "prediction_models": prediction_service.models_loaded,
            "anomaly_models": anomaly_service.models_loaded,
            "optimization_models": optimization_service.models_loaded
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            database_status=db_status,
            redis_status=redis_status,
            models_status=models_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# === PREDICTION ENDPOINTS ===

@app.post("/api/v1/predictions/demand", response_model=DemandPredictionResponse)
async def predict_demand(
    request: DemandPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict passenger demand for specified locations and time periods"""
    try:
        predictions = await prediction_service.predict_demand(
            locations=request.locations,
            time_horizon=request.time_horizon,
            features=request.features
        )
        
        return DemandPredictionResponse(
            predictions=predictions,
            model_version=prediction_service.model_version,
            confidence_scores=predictions.get('confidence_scores', []),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Demand prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction service error")

@app.post("/api/v1/predictions/congestion", response_model=CongestionPredictionResponse)
async def predict_congestion(
    request: CongestionPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict congestion levels for routes and areas"""
    try:
        predictions = await prediction_service.predict_congestion(
            routes=request.routes,
            areas=request.areas,
            time_horizon=request.time_horizon,
            weather_conditions=request.weather_conditions
        )
        
        return CongestionPredictionResponse(
            predictions=predictions,
            model_version=prediction_service.model_version,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Congestion prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction service error")

@app.post("/api/v1/predictions/flow", response_model=FlowPredictionResponse)
async def predict_passenger_flow(
    request: FlowPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict passenger flows using Graph Neural Networks"""
    try:
        predictions = await prediction_service.predict_passenger_flow(
            origin_destinations=request.origin_destinations,
            time_horizon=request.time_horizon,
            graph_features=request.graph_features
        )
        
        return FlowPredictionResponse(
            flow_predictions=predictions,
            model_version=prediction_service.gnn_model_version,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Flow prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction service error")

# === ANOMALY DETECTION ENDPOINTS ===

@app.post("/api/v1/anomalies/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Detect anomalies in mobility data"""
    try:
        anomalies = await anomaly_service.detect_anomalies(
            data=request.data,
            detection_methods=request.detection_methods,
            sensitivity=request.sensitivity
        )
        
        return AnomalyDetectionResponse(
            anomalies=anomalies,
            detection_methods_used=request.detection_methods,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail="Anomaly detection service error")

@app.get("/api/v1/anomalies/recent", response_model=List[AnomalyAlert])
async def get_recent_anomalies(
    hours: int = Query(24, description="Hours to look back"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    current_user: dict = Depends(get_current_user)
):
    """Get recent anomaly alerts"""
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        anomalies = await anomaly_service.get_recent_anomalies(
            since=since,
            severity_filter=severity
        )
        
        return anomalies
    except Exception as e:
        logger.error(f"Failed to get recent anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve anomalies")

# === OPTIMIZATION ENDPOINTS ===

@app.post("/api/v1/optimization/schedule", response_model=ScheduleOptimizationResponse)
async def optimize_schedule(
    request: ScheduleOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Optimize vehicle schedules using reinforcement learning"""
    try:
        # Start optimization task in background
        task_id = await optimization_service.start_schedule_optimization(
            routes=request.routes,
            constraints=request.constraints,
            objectives=request.objectives
        )
        
        # Add background task to monitor progress
        background_tasks.add_task(
            optimization_service.monitor_optimization_task,
            task_id
        )
        
        return ScheduleOptimizationResponse(
            task_id=task_id,
            status="started",
            estimated_completion_time=datetime.utcnow() + timedelta(minutes=30),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Schedule optimization failed: {e}")
        raise HTTPException(status_code=500, detail="Optimization service error")

@app.get("/api/v1/optimization/schedule/{task_id}", response_model=OptimizationTaskStatus)
async def get_optimization_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get status of optimization task"""
    try:
        status = await optimization_service.get_task_status(task_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")

@app.post("/api/v1/optimization/dispatch", response_model=DispatchRecommendationResponse)
async def get_dispatch_recommendations(
    request: DispatchRecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get real-time dispatch recommendations"""
    try:
        recommendations = await optimization_service.get_dispatch_recommendations(
            current_state=request.current_state,
            demand_forecast=request.demand_forecast,
            constraints=request.constraints
        )
        
        return DispatchRecommendationResponse(
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Dispatch recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Optimization service error")

# === REAL-TIME DATA ENDPOINTS ===

@app.get("/api/v1/realtime/vehicles", response_model=List[VehicleStatus])
async def get_vehicle_status(
    route_id: Optional[str] = Query(None, description="Filter by route"),
    current_user: dict = Depends(get_current_user)
):
    """Get real-time vehicle status"""
    try:
        vehicles = await real_time_service.get_vehicle_status(route_filter=route_id)
        return vehicles
    except Exception as e:
        logger.error(f"Failed to get vehicle status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve vehicle status")

@app.get("/api/v1/realtime/stops", response_model=List[StopStatus])
async def get_stop_status(
    area: Optional[str] = Query(None, description="Filter by area"),
    current_user: dict = Depends(get_current_user)
):
    """Get real-time stop/station status"""
    try:
        stops = await real_time_service.get_stop_status(area_filter=area)
        return stops
    except Exception as e:
        logger.error(f"Failed to get stop status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stop status")

@app.get("/api/v1/realtime/congestion", response_model=CongestionOverview)
async def get_congestion_overview(
    current_user: dict = Depends(get_current_user)
):
    """Get system-wide congestion overview"""
    try:
        overview = await real_time_service.get_congestion_overview()
        return overview
    except Exception as e:
        logger.error(f"Failed to get congestion overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve congestion data")

# === HISTORICAL DATA ENDPOINTS ===

@app.get("/api/v1/data/demand/historical", response_model=HistoricalDataResponse)
async def get_historical_demand(
    start_date: datetime = Query(..., description="Start date"),
    end_date: datetime = Query(..., description="End date"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    aggregation: str = Query("hourly", description="Aggregation level"),
    current_user: dict = Depends(get_current_user)
):
    """Get historical demand data"""
    try:
        data = await db_manager.get_historical_demand(
            start_date=start_date,
            end_date=end_date,
            location_id=location_id,
            aggregation=aggregation
        )
        
        return HistoricalDataResponse(
            data=data,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
            total_records=len(data)
        )
    except Exception as e:
        logger.error(f"Failed to get historical demand: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve historical data")

@app.get("/api/v1/data/performance/kpis", response_model=KPIResponse)
async def get_kpis(
    period: str = Query("24h", description="Time period"),
    current_user: dict = Depends(get_current_user)
):
    """Get key performance indicators"""
    try:
        kpis = await metrics_collector.get_kpis(period=period)
        return KPIResponse(
            kpis=kpis,
            period=period,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get KPIs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve KPIs")

# === SYSTEM MANAGEMENT ENDPOINTS ===

@app.post("/api/v1/system/retrain", response_model=RetrainingResponse)
async def trigger_model_retraining(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger model retraining"""
    try:
        # Check user permissions
        if current_user.get("role") not in ["admin", "data_scientist"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        task_id = await prediction_service.start_retraining(
            models=request.models,
            data_range=request.data_range
        )
        
        background_tasks.add_task(
            prediction_service.monitor_retraining_task,
            task_id
        )
        
        return RetrainingResponse(
            task_id=task_id,
            status="started",
            models=request.models,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail="Retraining service error")

@app.get("/api/v1/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    current_user: dict = Depends(get_current_user)
):
    """Get system performance metrics"""
    try:
        metrics = await metrics_collector.get_system_metrics()
        return SystemMetricsResponse(
            metrics=metrics,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")

# === PASSENGER-FACING ENDPOINTS ===

@app.get("/api/v1/passenger/recommendations", response_model=TravelRecommendationResponse)
async def get_travel_recommendations(
    origin: str = Query(..., description="Origin stop/station"),
    destination: str = Query(..., description="Destination stop/station"),
    departure_time: Optional[datetime] = Query(None, description="Preferred departure time"),
    preferences: Optional[str] = Query("fastest", description="Route preferences")
):
    """Get travel recommendations for passengers"""
    try:
        recommendations = await real_time_service.get_travel_recommendations(
            origin=origin,
            destination=destination,
            departure_time=departure_time or datetime.utcnow(),
            preferences=preferences
        )
        
        return TravelRecommendationResponse(
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get travel recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

@app.get("/api/v1/passenger/disruptions", response_model=List[ServiceDisruption])
async def get_service_disruptions(
    area: Optional[str] = Query(None, description="Filter by area")
):
    """Get current service disruptions"""
    try:
        disruptions = await real_time_service.get_service_disruptions(area_filter=area)
        return disruptions
    except Exception as e:
        logger.error(f"Failed to get service disruptions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve disruptions")

# === WebSocket ENDPOINTS ===

@app.websocket("/ws/realtime")
async def websocket_realtime_updates(websocket):
    """WebSocket endpoint for real-time updates"""
    await real_time_service.handle_websocket_connection(websocket)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )