"""
MQTT Subscriber for IoT Sensor Data
Handles real-time sensor data from vehicles and infrastructure
"""

import json
import logging
import asyncio
import paho.mqtt.client as mqtt
from typing import Dict, Any, Callable
from datetime import datetime
import redis
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MQTTToKafkaBridge:
    def __init__(self):
        self.mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        self.mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        # Initialize Kafka producer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
        # Initialize Redis for caching
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Topic mapping: MQTT topic -> Kafka topic
        self.topic_mapping = {
            'vehicle/+/gps': 'gps-data',
            'vehicle/+/sensors': 'sensor-data',
            'ferry/+/ais': 'ais-data',
            'station/+/turnstile': 'turnstile-data',
            'station/+/sensors': 'sensor-data',
            'train/+/sensors': 'sensor-data',
            'weather/+': 'weather-data',
            'events/+': 'special-events'
        }
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to all relevant topics
            for mqtt_topic in self.topic_mapping.keys():
                client.subscribe(mqtt_topic)
                logger.info(f"Subscribed to MQTT topic: {mqtt_topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        logger.warning(f"Disconnected from MQTT broker, return code {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Process incoming MQTT messages"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Add timestamp if not present
            if 'timestamp' not in payload:
                payload['timestamp'] = datetime.utcnow().isoformat()
            
            # Process based on topic type
            processed_data = self._process_mqtt_data(topic, payload)
            
            if processed_data:
                # Forward to appropriate Kafka topic
                kafka_topic = self._get_kafka_topic(topic)
                if kafka_topic:
                    self._send_to_kafka(kafka_topic, processed_data, topic)
                
                # Cache critical data in Redis
                self._cache_data(topic, processed_data)
            
        except Exception as e:
            logger.error(f"Error processing MQTT message from {msg.topic}: {e}")
    
    def _process_mqtt_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate MQTT data based on topic"""
        try:
            if 'vehicle/' in topic and '/gps' in topic:
                return self._process_gps_data(topic, payload)
            elif 'vehicle/' in topic and '/sensors' in topic:
                return self._process_vehicle_sensor_data(topic, payload)
            elif 'ferry/' in topic and '/ais' in topic:
                return self._process_ais_data(topic, payload)
            elif 'station/' in topic and '/turnstile' in topic:
                return self._process_turnstile_data(topic, payload)
            elif '/sensors' in topic:
                return self._process_sensor_data(topic, payload)
            elif 'weather/' in topic:
                return self._process_weather_data(topic, payload)
            elif 'events/' in topic:
                return self._process_event_data(topic, payload)
            else:
                logger.warning(f"Unknown topic pattern: {topic}")
                return None
        except Exception as e:
            logger.error(f"Error processing data for topic {topic}: {e}")
            return None
    
    def _process_gps_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process GPS data from vehicles"""
        vehicle_id = topic.split('/')[1]
        
        return {
            'vehicle_id': vehicle_id,
            'latitude': float(payload['latitude']),
            'longitude': float(payload['longitude']),
            'speed': payload.get('speed', 0.0),
            'heading': payload.get('heading', 0.0),
            'timestamp': payload['timestamp'],
            'route_id': payload.get('route_id'),
            'stop_sequence': payload.get('stop_sequence'),
            'occupancy_level': payload.get('occupancy_level', 0)
        }
    
    def _process_vehicle_sensor_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data from vehicles"""
        vehicle_id = topic.split('/')[1]
        
        return {
            'sensor_id': f"{vehicle_id}_{payload.get('sensor_type', 'unknown')}",
            'sensor_type': payload.get('sensor_type', 'unknown'),
            'location_id': vehicle_id,
            'value': float(payload['value']),
            'unit': payload.get('unit', ''),
            'timestamp': payload['timestamp'],
            'metadata': json.dumps({
                'vehicle_id': vehicle_id,
                'additional_data': payload.get('metadata', {})
            })
        }
    
    def _process_ais_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process AIS data from ferries"""
        vessel_id = topic.split('/')[1]
        
        return {
            'vessel_id': vessel_id,
            'mmsi': payload.get('mmsi'),
            'latitude': float(payload['latitude']),
            'longitude': float(payload['longitude']),
            'speed_over_ground': payload.get('speed_over_ground', 0.0),
            'course_over_ground': payload.get('course_over_ground', 0.0),
            'heading': payload.get('heading', 0.0),
            'timestamp': payload['timestamp'],
            'vessel_type': payload.get('vessel_type', 'ferry'),
            'destination': payload.get('destination'),
            'eta': payload.get('eta')
        }
    
    def _process_turnstile_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process turnstile data from stations"""
        station_id = topic.split('/')[1]
        
        return {
            'turnstile_id': payload.get('turnstile_id', f"{station_id}_turnstile"),
            'stop_id': station_id,
            'direction': payload['direction'],  # 'entry' or 'exit'
            'count': int(payload.get('count', 1)),
            'timestamp': payload['timestamp']
        }
    
    def _process_sensor_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process general sensor data"""
        location_id = topic.split('/')[1]
        
        return {
            'sensor_id': payload.get('sensor_id', f"{location_id}_{payload.get('sensor_type')}"),
            'sensor_type': payload['sensor_type'],
            'location_id': location_id,
            'value': float(payload['value']),
            'unit': payload.get('unit', ''),
            'timestamp': payload['timestamp'],
            'metadata': json.dumps(payload.get('metadata', {}))
        }
    
    def _process_weather_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process weather data"""
        location = topic.split('/')[1]
        
        return {
            'location': location,
            'temperature': payload.get('temperature'),
            'humidity': payload.get('humidity'),
            'precipitation': payload.get('precipitation'),
            'wind_speed': payload.get('wind_speed'),
            'wind_direction': payload.get('wind_direction'),
            'visibility': payload.get('visibility'),
            'weather_condition': payload.get('condition'),
            'timestamp': payload['timestamp']
        }
    
    def _process_event_data(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process special event data"""
        return {
            'event_id': payload['event_id'],
            'name': payload['name'],
            'description': payload.get('description'),
            'event_type': payload.get('event_type'),
            'start_time': payload['start_time'],
            'end_time': payload['end_time'],
            'location': payload.get('location'),
            'expected_attendance': payload.get('expected_attendance'),
            'impact_radius': payload.get('impact_radius'),
            'transport_impact_level': payload.get('transport_impact_level', 3)
        }
    
    def _get_kafka_topic(self, mqtt_topic: str) -> str:
        """Map MQTT topic to Kafka topic"""
        for pattern, kafka_topic in self.topic_mapping.items():
            if self._topic_matches(mqtt_topic, pattern):
                return kafka_topic
        return None
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports + wildcard)"""
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')
        
        if len(topic_parts) != len(pattern_parts):
            return False
        
        for i, (topic_part, pattern_part) in enumerate(zip(topic_parts, pattern_parts)):
            if pattern_part != '+' and pattern_part != topic_part:
                return False
        
        return True
    
    def _send_to_kafka(self, kafka_topic: str, data: Dict[str, Any], mqtt_topic: str):
        """Send processed data to Kafka"""
        try:
            key = data.get('vehicle_id') or data.get('vessel_id') or data.get('sensor_id')
            
            future = self.kafka_producer.send(kafka_topic, value=data, key=key)
            future.add_callback(lambda metadata: logger.debug(f"Sent to Kafka topic {kafka_topic}"))
            future.add_errback(lambda exc: logger.error(f"Failed to send to Kafka: {exc}"))
            
        except Exception as e:
            logger.error(f"Error sending to Kafka topic {kafka_topic}: {e}")
    
    def _cache_data(self, mqtt_topic: str, data: Dict[str, Any]):
        """Cache critical data in Redis for fast access"""
        try:
            if 'vehicle/' in mqtt_topic and '/gps' in mqtt_topic:
                cache_key = f"vehicle:{data['vehicle_id']}:latest_position"
                self.redis_client.hset(cache_key, mapping={
                    'latitude': data['latitude'],
                    'longitude': data['longitude'],
                    'timestamp': data['timestamp']
                })
                self.redis_client.expire(cache_key, 300)  # 5 minutes TTL
            
            elif 'ferry/' in mqtt_topic and '/ais' in mqtt_topic:
                cache_key = f"ferry:{data['vessel_id']}:latest_position"
                self.redis_client.hset(cache_key, mapping={
                    'latitude': data['latitude'],
                    'longitude': data['longitude'],
                    'timestamp': data['timestamp']
                })
                self.redis_client.expire(cache_key, 300)
            
            elif '/sensors' in mqtt_topic:
                cache_key = f"sensor:{data['sensor_id']}:latest_reading"
                self.redis_client.hset(cache_key, mapping={
                    'value': data['value'],
                    'timestamp': data['timestamp'],
                    'sensor_type': data['sensor_type']
                })
                self.redis_client.expire(cache_key, 600)  # 10 minutes TTL
                
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def start(self):
        """Start MQTT subscriber"""
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            logger.info(f"Starting MQTT subscriber on {self.mqtt_broker}:{self.mqtt_port}")
            self.mqtt_client.loop_forever()
        except Exception as e:
            logger.error(f"Error starting MQTT subscriber: {e}")
    
    def stop(self):
        """Stop MQTT subscriber"""
        self.mqtt_client.disconnect()
        self.kafka_producer.close()
        logger.info("MQTT subscriber stopped")

if __name__ == "__main__":
    bridge = MQTTToKafkaBridge()
    
    try:
        bridge.start()
    except KeyboardInterrupt:
        logger.info("Shutting down MQTT subscriber...")
        bridge.stop()