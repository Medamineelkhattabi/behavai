"""
Kafka Consumer for Real-time Data Ingestion
Handles GPS, AIS, sensor, and ticketing data streams
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'mobility_platform')
    username: str = os.getenv('DB_USER', 'mobility_user')
    password: str = os.getenv('DB_PASSWORD', 'mobility_pass')

@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    group_id: str = os.getenv('KAFKA_GROUP_ID', 'mobility-platform-consumer')
    auto_offset_reset: str = 'latest'

class MobilityDataConsumer:
    def __init__(self, db_config: DatabaseConfig, kafka_config: KafkaConfig):
        self.db_config = db_config
        self.kafka_config = kafka_config
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Topic handlers mapping
        self.topic_handlers = {
            'gps-data': self._handle_gps_data,
            'ais-data': self._handle_ais_data,
            'sensor-data': self._handle_sensor_data,
            'ticketing-data': self._handle_ticketing_data,
            'turnstile-data': self._handle_turnstile_data,
            'weather-data': self._handle_weather_data,
            'special-events': self._handle_special_events
        }
    
    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.database,
            user=self.db_config.username,
            password=self.db_config.password,
            cursor_factory=RealDictCursor
        )
    
    def _handle_gps_data(self, data: Dict[str, Any]) -> bool:
        """Handle GPS data from buses and other vehicles"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO gps_data 
                        (vehicle_id, latitude, longitude, speed, heading, timestamp, 
                         route_id, stop_sequence, occupancy_level)
                        VALUES (%(vehicle_id)s, %(latitude)s, %(longitude)s, %(speed)s, 
                               %(heading)s, %(timestamp)s, %(route_id)s, %(stop_sequence)s, 
                               %(occupancy_level)s)
                    """, data)
                    conn.commit()
            
            # Cache latest position in Redis for real-time queries
            cache_key = f"vehicle:{data['vehicle_id']}:latest"
            self.redis_client.hset(cache_key, mapping={
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'timestamp': data['timestamp'],
                'occupancy_level': data.get('occupancy_level', 0)
            })
            self.redis_client.expire(cache_key, 300)  # 5 minutes TTL
            
            return True
        except Exception as e:
            logger.error(f"Error handling GPS data: {e}")
            return False
    
    def _handle_ais_data(self, data: Dict[str, Any]) -> bool:
        """Handle AIS data from ferries"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO ais_data 
                        (vessel_id, mmsi, latitude, longitude, speed_over_ground, 
                         course_over_ground, heading, timestamp, vessel_type, 
                         destination, eta)
                        VALUES (%(vessel_id)s, %(mmsi)s, %(latitude)s, %(longitude)s, 
                               %(speed_over_ground)s, %(course_over_ground)s, %(heading)s, 
                               %(timestamp)s, %(vessel_type)s, %(destination)s, %(eta)s)
                    """, data)
                    conn.commit()
            
            # Cache ferry position
            cache_key = f"ferry:{data['vessel_id']}:latest"
            self.redis_client.hset(cache_key, mapping={
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'timestamp': data['timestamp'],
                'destination': data.get('destination', ''),
                'eta': data.get('eta', '')
            })
            self.redis_client.expire(cache_key, 300)
            
            return True
        except Exception as e:
            logger.error(f"Error handling AIS data: {e}")
            return False
    
    def _handle_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Handle sensor data from train/metro systems"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO sensor_data 
                        (sensor_id, sensor_type, location_id, value, unit, timestamp, metadata)
                        VALUES (%(sensor_id)s, %(sensor_type)s, %(location_id)s, 
                               %(value)s, %(unit)s, %(timestamp)s, %(metadata)s)
                    """, data)
                    conn.commit()
            
            # Cache critical sensor readings
            if data['sensor_type'] in ['occupancy', 'temperature', 'door_status']:
                cache_key = f"sensor:{data['sensor_id']}:latest"
                self.redis_client.hset(cache_key, mapping={
                    'value': data['value'],
                    'timestamp': data['timestamp'],
                    'sensor_type': data['sensor_type']
                })
                self.redis_client.expire(cache_key, 600)  # 10 minutes TTL
            
            return True
        except Exception as e:
            logger.error(f"Error handling sensor data: {e}")
            return False
    
    def _handle_ticketing_data(self, data: Dict[str, Any]) -> bool:
        """Handle ticketing and card tap data"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO ticketing_data 
                        (transaction_id, card_id, stop_id, route_id, vehicle_id, 
                         transaction_type, amount, timestamp, passenger_category)
                        VALUES (%(transaction_id)s, %(card_id)s, %(stop_id)s, %(route_id)s, 
                               %(vehicle_id)s, %(transaction_type)s, %(amount)s, 
                               %(timestamp)s, %(passenger_category)s)
                    """, data)
                    conn.commit()
            
            # Update real-time passenger count
            if data['transaction_type'] in ['tap_in', 'tap_out']:
                self._update_passenger_count(data)
            
            return True
        except Exception as e:
            logger.error(f"Error handling ticketing data: {e}")
            return False
    
    def _handle_turnstile_data(self, data: Dict[str, Any]) -> bool:
        """Handle turnstile entry/exit counts"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO turnstile_data 
                        (turnstile_id, stop_id, direction, count, timestamp)
                        VALUES (%(turnstile_id)s, %(stop_id)s, %(direction)s, 
                               %(count)s, %(timestamp)s)
                    """, data)
                    conn.commit()
            
            # Update station occupancy metrics
            self._update_station_occupancy(data)
            
            return True
        except Exception as e:
            logger.error(f"Error handling turnstile data: {e}")
            return False
    
    def _handle_weather_data(self, data: Dict[str, Any]) -> bool:
        """Handle weather feed data"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO weather_data 
                        (location, temperature, humidity, precipitation, wind_speed, 
                         wind_direction, visibility, weather_condition, timestamp)
                        VALUES (%(location)s, %(temperature)s, %(humidity)s, %(precipitation)s, 
                               %(wind_speed)s, %(wind_direction)s, %(visibility)s, 
                               %(weather_condition)s, %(timestamp)s)
                    """, data)
                    conn.commit()
            
            # Cache current weather
            cache_key = f"weather:{data['location']}:current"
            self.redis_client.hset(cache_key, mapping=data)
            self.redis_client.expire(cache_key, 1800)  # 30 minutes TTL
            
            return True
        except Exception as e:
            logger.error(f"Error handling weather data: {e}")
            return False
    
    def _handle_special_events(self, data: Dict[str, Any]) -> bool:
        """Handle special events data"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO special_events 
                        (event_id, name, description, event_type, start_time, end_time, 
                         location, expected_attendance, impact_radius, transport_impact_level)
                        VALUES (%(event_id)s, %(name)s, %(description)s, %(event_type)s, 
                               %(start_time)s, %(end_time)s, %(location)s, %(expected_attendance)s, 
                               %(impact_radius)s, %(transport_impact_level)s)
                        ON CONFLICT (event_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        start_time = EXCLUDED.start_time,
                        end_time = EXCLUDED.end_time,
                        expected_attendance = EXCLUDED.expected_attendance
                    """, data)
                    conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error handling special events data: {e}")
            return False
    
    def _update_passenger_count(self, data: Dict[str, Any]):
        """Update real-time passenger counts"""
        cache_key = f"passenger_count:{data['stop_id']}:current"
        if data['transaction_type'] == 'tap_in':
            self.redis_client.incr(cache_key)
        elif data['transaction_type'] == 'tap_out':
            self.redis_client.decr(cache_key)
        self.redis_client.expire(cache_key, 3600)  # 1 hour TTL
    
    def _update_station_occupancy(self, data: Dict[str, Any]):
        """Update station occupancy metrics"""
        cache_key = f"station_occupancy:{data['stop_id']}:current"
        current_count = self.redis_client.get(cache_key) or 0
        
        if data['direction'] == 'entry':
            new_count = int(current_count) + data['count']
        else:  # exit
            new_count = max(0, int(current_count) - data['count'])
        
        self.redis_client.set(cache_key, new_count, ex=3600)
    
    async def start_consuming(self):
        """Start consuming messages from Kafka topics"""
        consumer = KafkaConsumer(
            *self.topic_handlers.keys(),
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            group_id=self.kafka_config.group_id,
            auto_offset_reset=self.kafka_config.auto_offset_reset,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info(f"Started consuming from topics: {list(self.topic_handlers.keys())}")
        
        try:
            for message in consumer:
                topic = message.topic
                data = message.value
                
                if topic in self.topic_handlers:
                    # Process message in thread pool to avoid blocking
                    future = self.executor.submit(self.topic_handlers[topic], data)
                    success = future.result(timeout=30)  # 30 second timeout
                    
                    if success:
                        logger.debug(f"Successfully processed {topic} message")
                    else:
                        logger.warning(f"Failed to process {topic} message")
                else:
                    logger.warning(f"No handler for topic: {topic}")
        
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            consumer.close()
            self.executor.shutdown(wait=True)

if __name__ == "__main__":
    db_config = DatabaseConfig()
    kafka_config = KafkaConfig()
    
    consumer = MobilityDataConsumer(db_config, kafka_config)
    asyncio.run(consumer.start_consuming())