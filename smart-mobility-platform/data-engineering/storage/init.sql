-- Smart Mobility Platform Database Schema

-- Core infrastructure tables
CREATE TABLE IF NOT EXISTS transport_modes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stops (
    id SERIAL PRIMARY KEY,
    stop_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    transport_mode_id INTEGER REFERENCES transport_modes(id),
    zone VARCHAR(50),
    accessibility_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS routes (
    id SERIAL PRIMARY KEY,
    route_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    transport_mode_id INTEGER REFERENCES transport_modes(id),
    route_type VARCHAR(50),
    color VARCHAR(7),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(100) NOT NULL UNIQUE,
    transport_mode_id INTEGER REFERENCES transport_modes(id),
    capacity INTEGER,
    vehicle_type VARCHAR(50),
    manufacturer VARCHAR(100),
    year_manufactured INTEGER,
    accessibility_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Real-time data tables
CREATE TABLE IF NOT EXISTS gps_data (
    id BIGSERIAL PRIMARY KEY,
    vehicle_id VARCHAR(100) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    speed DECIMAL(5, 2),
    heading DECIMAL(5, 2),
    timestamp TIMESTAMP NOT NULL,
    route_id VARCHAR(100),
    stop_sequence INTEGER,
    occupancy_level INTEGER CHECK (occupancy_level >= 0 AND occupancy_level <= 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ais_data (
    id BIGSERIAL PRIMARY KEY,
    vessel_id VARCHAR(100) NOT NULL,
    mmsi VARCHAR(20),
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    speed_over_ground DECIMAL(5, 2),
    course_over_ground DECIMAL(5, 2),
    heading DECIMAL(5, 2),
    timestamp TIMESTAMP NOT NULL,
    vessel_type VARCHAR(50),
    destination VARCHAR(200),
    eta TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sensor_data (
    id BIGSERIAL PRIMARY KEY,
    sensor_id VARCHAR(100) NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    location_id VARCHAR(100),
    value DECIMAL(10, 4) NOT NULL,
    unit VARCHAR(20),
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticketing_data (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL UNIQUE,
    card_id VARCHAR(100),
    stop_id VARCHAR(100),
    route_id VARCHAR(100),
    vehicle_id VARCHAR(100),
    transaction_type VARCHAR(20) NOT NULL, -- 'tap_in', 'tap_out', 'purchase'
    amount DECIMAL(10, 2),
    timestamp TIMESTAMP NOT NULL,
    passenger_category VARCHAR(50), -- 'adult', 'student', 'senior', 'child'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turnstile_data (
    id BIGSERIAL PRIMARY KEY,
    turnstile_id VARCHAR(100) NOT NULL,
    stop_id VARCHAR(100) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'entry', 'exit'
    count INTEGER NOT NULL DEFAULT 1,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weather and external data
CREATE TABLE IF NOT EXISTS weather_data (
    id BIGSERIAL PRIMARY KEY,
    location VARCHAR(100) NOT NULL,
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    precipitation DECIMAL(5, 2),
    wind_speed DECIMAL(5, 2),
    wind_direction DECIMAL(5, 2),
    visibility DECIMAL(5, 2),
    weather_condition VARCHAR(100),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS special_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    event_type VARCHAR(50),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    location VARCHAR(200),
    expected_attendance INTEGER,
    impact_radius DECIMAL(10, 2), -- in kilometers
    transport_impact_level INTEGER CHECK (transport_impact_level >= 1 AND transport_impact_level <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processed and aggregated data
CREATE TABLE IF NOT EXISTS passenger_flows (
    id BIGSERIAL PRIMARY KEY,
    origin_stop_id VARCHAR(100) NOT NULL,
    destination_stop_id VARCHAR(100) NOT NULL,
    route_id VARCHAR(100),
    passenger_count INTEGER NOT NULL,
    avg_travel_time DECIMAL(8, 2),
    time_window_start TIMESTAMP NOT NULL,
    time_window_end TIMESTAMP NOT NULL,
    day_of_week INTEGER,
    is_holiday BOOLEAN DEFAULT FALSE,
    weather_condition VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS congestion_metrics (
    id BIGSERIAL PRIMARY KEY,
    location_id VARCHAR(100) NOT NULL,
    location_type VARCHAR(20) NOT NULL, -- 'stop', 'route', 'area'
    congestion_level DECIMAL(5, 2) NOT NULL CHECK (congestion_level >= 0 AND congestion_level <= 100),
    avg_wait_time DECIMAL(8, 2),
    avg_travel_time DECIMAL(8, 2),
    occupancy_rate DECIMAL(5, 2),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI Model predictions and results
CREATE TABLE IF NOT EXISTS demand_predictions (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    location_id VARCHAR(100) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL, -- 'passenger_count', 'congestion_level'
    predicted_value DECIMAL(10, 4) NOT NULL,
    confidence_score DECIMAL(5, 4),
    prediction_horizon INTEGER, -- minutes into the future
    prediction_timestamp TIMESTAMP NOT NULL,
    actual_value DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomaly_detections (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    location_id VARCHAR(100) NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL,
    severity_score DECIMAL(5, 4) NOT NULL CHECK (severity_score >= 0 AND severity_score <= 1),
    description TEXT,
    detected_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization results
CREATE TABLE IF NOT EXISTS schedule_optimizations (
    id BIGSERIAL PRIMARY KEY,
    route_id VARCHAR(100) NOT NULL,
    optimization_type VARCHAR(50) NOT NULL,
    original_schedule JSONB,
    optimized_schedule JSONB,
    expected_improvement DECIMAL(5, 2),
    implementation_date DATE,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_gps_data_vehicle_timestamp ON gps_data(vehicle_id, timestamp);
CREATE INDEX idx_gps_data_timestamp ON gps_data(timestamp);
CREATE INDEX idx_ais_data_vessel_timestamp ON ais_data(vessel_id, timestamp);
CREATE INDEX idx_sensor_data_sensor_timestamp ON sensor_data(sensor_id, timestamp);
CREATE INDEX idx_ticketing_data_timestamp ON ticketing_data(timestamp);
CREATE INDEX idx_turnstile_data_stop_timestamp ON turnstile_data(stop_id, timestamp);
CREATE INDEX idx_weather_data_location_timestamp ON weather_data(location, timestamp);
CREATE INDEX idx_passenger_flows_time_window ON passenger_flows(time_window_start, time_window_end);
CREATE INDEX idx_congestion_metrics_location_timestamp ON congestion_metrics(location_id, timestamp);
CREATE INDEX idx_demand_predictions_location_timestamp ON demand_predictions(location_id, prediction_timestamp);

-- Insert initial transport modes
INSERT INTO transport_modes (name, description) VALUES
('bus', 'Public bus transportation'),
('ferry', 'Water transportation via ferry'),
('train', 'Rail transportation'),
('metro', 'Urban metro/subway system'),
('tram', 'Light rail/tram system')
ON CONFLICT (name) DO NOTHING;