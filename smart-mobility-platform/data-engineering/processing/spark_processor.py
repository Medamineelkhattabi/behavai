"""
Apache Spark Data Processing Pipeline
Handles batch and streaming data transformation, aggregation, and feature engineering
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobilityDataProcessor:
    def __init__(self, app_name: str = "MobilityPlatform"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Database connection properties
        self.db_properties = {
            "user": os.getenv("DB_USER", "mobility_user"),
            "password": os.getenv("DB_PASSWORD", "mobility_pass"),
            "driver": "org.postgresql.Driver"
        }
        self.db_url = os.getenv("DB_URL", "jdbc:postgresql://localhost:5432/mobility_platform")
        
    def read_from_kafka(self, topics: List[str], starting_offsets: str = "latest") -> DataFrame:
        """Read streaming data from Kafka topics"""
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")) \
            .option("subscribe", ",".join(topics)) \
            .option("startingOffsets", starting_offsets) \
            .load()
    
    def read_from_database(self, table_name: str, conditions: Optional[str] = None) -> DataFrame:
        """Read batch data from PostgreSQL database"""
        query = f"(SELECT * FROM {table_name}"
        if conditions:
            query += f" WHERE {conditions}"
        query += ") as subquery"
        
        return self.spark.read \
            .jdbc(url=self.db_url, table=query, properties=self.db_properties)
    
    def write_to_database(self, df: DataFrame, table_name: str, mode: str = "append"):
        """Write DataFrame to PostgreSQL database"""
        df.write \
            .jdbc(url=self.db_url, table=table_name, mode=mode, properties=self.db_properties)
    
    def process_gps_stream(self, kafka_df: DataFrame) -> DataFrame:
        """Process GPS data stream from Kafka"""
        # Parse JSON from Kafka value
        gps_schema = StructType([
            StructField("vehicle_id", StringType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("speed", DoubleType(), True),
            StructField("heading", DoubleType(), True),
            StructField("timestamp", StringType(), True),
            StructField("route_id", StringType(), True),
            StructField("occupancy_level", IntegerType(), True)
        ])
        
        gps_df = kafka_df \
            .filter(col("topic") == "gps-data") \
            .select(from_json(col("value").cast("string"), gps_schema).alias("data")) \
            .select("data.*") \
            .withColumn("timestamp", to_timestamp(col("timestamp"))) \
            .withColumn("processing_time", current_timestamp())
        
        # Add derived features
        gps_df = gps_df \
            .withColumn("hour_of_day", hour(col("timestamp"))) \
            .withColumn("day_of_week", dayofweek(col("timestamp"))) \
            .withColumn("is_weekend", when(dayofweek(col("timestamp")).isin([1, 7]), 1).otherwise(0))
        
        return gps_df
    
    def process_passenger_flows(self, ticketing_df: DataFrame) -> DataFrame:
        """Process and aggregate passenger flow data"""
        # Calculate passenger flows between stops
        flows_df = ticketing_df \
            .filter(col("transaction_type").isin(["tap_in", "tap_out"])) \
            .withColumn("hour_window", date_trunc("hour", col("timestamp"))) \
            .groupBy("stop_id", "route_id", "hour_window", "passenger_category") \
            .agg(
                count(when(col("transaction_type") == "tap_in", 1)).alias("entries"),
                count(when(col("transaction_type") == "tap_out", 1)).alias("exits"),
                countDistinct("card_id").alias("unique_passengers")
            ) \
            .withColumn("net_flow", col("entries") - col("exits"))
        
        return flows_df
    
    def calculate_congestion_metrics(self, gps_df: DataFrame, flows_df: DataFrame) -> DataFrame:
        """Calculate real-time congestion metrics"""
        # Vehicle-based congestion (speed and occupancy)
        vehicle_congestion = gps_df \
            .withColumn("time_window", date_trunc("minute", col("timestamp"))) \
            .groupBy("route_id", "time_window") \
            .agg(
                avg("speed").alias("avg_speed"),
                avg("occupancy_level").alias("avg_occupancy"),
                count("vehicle_id").alias("vehicle_count")
            ) \
            .withColumn("speed_congestion_score", 
                       when(col("avg_speed") < 10, 100)
                       .when(col("avg_speed") < 20, 75)
                       .when(col("avg_speed") < 30, 50)
                       .otherwise(25)) \
            .withColumn("occupancy_congestion_score", col("avg_occupancy")) \
            .withColumn("overall_congestion_score", 
                       (col("speed_congestion_score") + col("occupancy_congestion_score")) / 2)
        
        # Station-based congestion (passenger flows)
        station_congestion = flows_df \
            .withColumn("congestion_level", 
                       when(col("net_flow") > 100, 90)
                       .when(col("net_flow") > 50, 70)
                       .when(col("net_flow") > 20, 50)
                       .otherwise(30))
        
        return vehicle_congestion, station_congestion
    
    def detect_anomalies(self, df: DataFrame, metric_col: str, window_size: int = 24) -> DataFrame:
        """Detect anomalies using statistical methods"""
        # Calculate rolling statistics
        window_spec = Window.partitionBy("location_id").orderBy("timestamp") \
            .rowsBetween(-window_size, -1)
        
        anomaly_df = df \
            .withColumn("rolling_mean", avg(metric_col).over(window_spec)) \
            .withColumn("rolling_std", stddev(metric_col).over(window_spec)) \
            .withColumn("z_score", 
                       (col(metric_col) - col("rolling_mean")) / col("rolling_std")) \
            .withColumn("is_anomaly", 
                       when(abs(col("z_score")) > 3, 1).otherwise(0)) \
            .withColumn("anomaly_severity", 
                       when(abs(col("z_score")) > 4, "high")
                       .when(abs(col("z_score")) > 3, "medium")
                       .otherwise("low"))
        
        return anomaly_df
    
    def create_mobility_graph_features(self, gps_df: DataFrame, stops_df: DataFrame) -> DataFrame:
        """Create graph-based features for mobility network"""
        # Calculate travel times between consecutive stops
        window_spec = Window.partitionBy("vehicle_id", "route_id") \
            .orderBy("timestamp")
        
        travel_times = gps_df \
            .withColumn("prev_timestamp", lag("timestamp").over(window_spec)) \
            .withColumn("prev_stop_sequence", lag("stop_sequence").over(window_spec)) \
            .filter(col("prev_timestamp").isNotNull()) \
            .withColumn("travel_time_seconds", 
                       unix_timestamp("timestamp") - unix_timestamp("prev_timestamp")) \
            .filter(col("travel_time_seconds") > 0) \
            .groupBy("route_id", "prev_stop_sequence", "stop_sequence") \
            .agg(
                avg("travel_time_seconds").alias("avg_travel_time"),
                stddev("travel_time_seconds").alias("std_travel_time"),
                count("*").alias("trip_count")
            )
        
        # Calculate stop connectivity and centrality measures
        stop_connections = travel_times \
            .groupBy("prev_stop_sequence") \
            .agg(
                count("stop_sequence").alias("out_degree"),
                avg("avg_travel_time").alias("avg_outbound_time")
            ) \
            .union(
                travel_times \
                .groupBy("stop_sequence") \
                .agg(
                    count("prev_stop_sequence").alias("in_degree"),
                    avg("avg_travel_time").alias("avg_inbound_time")
                )
            )
        
        return travel_times, stop_connections
    
    def aggregate_hourly_metrics(self, df: DataFrame, group_cols: List[str], 
                                metric_cols: List[str]) -> DataFrame:
        """Aggregate metrics by hour for time series analysis"""
        hourly_agg = df \
            .withColumn("hour_window", date_trunc("hour", col("timestamp"))) \
            .groupBy(group_cols + ["hour_window"]) \
            .agg(*[avg(col).alias(f"avg_{col}") for col in metric_cols] +
                 *[max(col).alias(f"max_{col}") for col in metric_cols] +
                 *[min(col).alias(f"min_{col}") for col in metric_cols] +
                 *[stddev(col).alias(f"std_{col}") for col in metric_cols])
        
        return hourly_agg
    
    def create_time_series_features(self, df: DataFrame, timestamp_col: str = "timestamp") -> DataFrame:
        """Create time-based features for ML models"""
        features_df = df \
            .withColumn("year", year(col(timestamp_col))) \
            .withColumn("month", month(col(timestamp_col))) \
            .withColumn("day", dayofmonth(col(timestamp_col))) \
            .withColumn("hour", hour(col(timestamp_col))) \
            .withColumn("minute", minute(col(timestamp_col))) \
            .withColumn("day_of_week", dayofweek(col(timestamp_col))) \
            .withColumn("day_of_year", dayofyear(col(timestamp_col))) \
            .withColumn("week_of_year", weekofyear(col(timestamp_col))) \
            .withColumn("is_weekend", 
                       when(dayofweek(col(timestamp_col)).isin([1, 7]), 1).otherwise(0)) \
            .withColumn("is_rush_hour", 
                       when((hour(col(timestamp_col)).between(7, 9)) | 
                            (hour(col(timestamp_col)).between(17, 19)), 1).otherwise(0)) \
            .withColumn("season", 
                       when(month(col(timestamp_col)).isin([12, 1, 2]), "winter")
                       .when(month(col(timestamp_col)).isin([3, 4, 5]), "spring")
                       .when(month(col(timestamp_col)).isin([6, 7, 8]), "summer")
                       .otherwise("autumn"))
        
        return features_df
    
    def process_weather_impact(self, weather_df: DataFrame, mobility_df: DataFrame) -> DataFrame:
        """Analyze weather impact on mobility patterns"""
        # Join weather data with mobility data
        weather_mobility = mobility_df \
            .join(weather_df, 
                  (date_trunc("hour", mobility_df.timestamp) == 
                   date_trunc("hour", weather_df.timestamp)), "left") \
            .withColumn("weather_impact_score",
                       when(col("precipitation") > 10, 0.8)
                       .when(col("precipitation") > 5, 0.6)
                       .when((col("temperature") < 0) | (col("temperature") > 35), 0.7)
                       .when(col("wind_speed") > 20, 0.5)
                       .otherwise(1.0))
        
        return weather_mobility
    
    def run_batch_processing(self):
        """Run batch processing pipeline"""
        logger.info("Starting batch processing pipeline...")
        
        # Read historical data
        gps_data = self.read_from_database("gps_data", 
            f"timestamp >= '{(datetime.now() - timedelta(days=7)).isoformat()}'")
        ticketing_data = self.read_from_database("ticketing_data",
            f"timestamp >= '{(datetime.now() - timedelta(days=7)).isoformat()}'")
        weather_data = self.read_from_database("weather_data",
            f"timestamp >= '{(datetime.now() - timedelta(days=7)).isoformat()}'")
        
        # Process passenger flows
        flows = self.process_passenger_flows(ticketing_data)
        
        # Calculate congestion metrics
        vehicle_congestion, station_congestion = self.calculate_congestion_metrics(gps_data, flows)
        
        # Create time series features
        gps_features = self.create_time_series_features(gps_data)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(gps_features, "occupancy_level")
        
        # Create mobility graph features
        stops_data = self.read_from_database("stops")
        travel_times, stop_connections = self.create_mobility_graph_features(gps_data, stops_data)
        
        # Aggregate hourly metrics
        hourly_metrics = self.aggregate_hourly_metrics(
            gps_features, 
            ["route_id", "vehicle_id"],
            ["speed", "occupancy_level"]
        )
        
        # Write processed data back to database
        self.write_to_database(flows, "passenger_flows")
        self.write_to_database(vehicle_congestion, "congestion_metrics")
        self.write_to_database(anomalies.filter(col("is_anomaly") == 1), "anomaly_detections")
        
        logger.info("Batch processing pipeline completed")
    
    def run_streaming_processing(self):
        """Run streaming processing pipeline"""
        logger.info("Starting streaming processing pipeline...")
        
        # Read from Kafka
        kafka_stream = self.read_from_kafka(["gps-data", "sensor-data", "ticketing-data"])
        
        # Process GPS stream
        gps_stream = self.process_gps_stream(kafka_stream)
        
        # Write streaming results to database
        query = gps_stream.writeStream \
            .format("jdbc") \
            .option("url", self.db_url) \
            .option("dbtable", "gps_data_processed") \
            .option("user", self.db_properties["user"]) \
            .option("password", self.db_properties["password"]) \
            .option("driver", self.db_properties["driver"]) \
            .option("checkpointLocation", "/tmp/streaming-checkpoint") \
            .outputMode("append") \
            .start()
        
        query.awaitTermination()
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()

if __name__ == "__main__":
    processor = MobilityDataProcessor()
    
    try:
        # Run batch processing
        processor.run_batch_processing()
        
        # Run streaming processing (this will run indefinitely)
        # processor.run_streaming_processing()
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
    finally:
        processor.stop()