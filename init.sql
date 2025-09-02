-- Smart Mobility Platform Database Schema (Simplified for Windows setup)

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS routes (
    id SERIAL PRIMARY KEY,
    route_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    transport_mode_id INTEGER REFERENCES transport_modes(id),
    route_type VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(100) NOT NULL UNIQUE,
    transport_mode_id INTEGER REFERENCES transport_modes(id),
    capacity INTEGER,
    vehicle_type VARCHAR(50),
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
    occupancy_level INTEGER CHECK (occupancy_level >= 0 AND occupancy_level <= 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS passenger_flows (
    id BIGSERIAL PRIMARY KEY,
    origin_stop_id VARCHAR(100) NOT NULL,
    destination_stop_id VARCHAR(100) NOT NULL,
    route_id VARCHAR(100),
    passenger_count INTEGER NOT NULL,
    avg_travel_time DECIMAL(8, 2),
    time_window_start TIMESTAMP NOT NULL,
    time_window_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial transport modes
INSERT INTO transport_modes (name, description) VALUES
('bus', 'Public bus transportation'),
('ferry', 'Water transportation via ferry'),
('train', 'Rail transportation'),
('metro', 'Urban metro/subway system'),
('tram', 'Light rail/tram system')
ON CONFLICT (name) DO NOTHING;

-- Insert sample data for testing
INSERT INTO stops (stop_id, name, latitude, longitude, transport_mode_id) VALUES
('stop_001', 'Central Station', 25.2048, 55.2708, 1),
('stop_002', 'Airport Terminal', 25.2532, 55.3657, 1),
('stop_003', 'Mall of Emirates', 25.1190, 55.2000, 1),
('stop_004', 'Dubai Marina', 25.0800, 55.1400, 2),
('stop_005', 'Business Bay', 25.1900, 55.2600, 3)
ON CONFLICT (stop_id) DO NOTHING;

INSERT INTO routes (route_id, name, transport_mode_id) VALUES
('route_001', 'City Center Line', 1),
('route_002', 'Airport Express', 1),
('route_003', 'Marina Ferry', 2),
('route_004', 'Metro Red Line', 3),
('route_005', 'Tram Green Line', 5)
ON CONFLICT (route_id) DO NOTHING;