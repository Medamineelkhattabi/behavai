FROM apache/spark:3.5.0-python3

# Switch to root for installations
USER root

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional data processing dependencies
RUN pip install --no-cache-dir \
    kafka-python==2.0.2 \
    paho-mqtt==1.6.1 \
    pyspark==3.5.0 \
    pandas==2.1.4 \
    numpy==1.24.3

# Copy application code
COPY data-engineering/ /app/data-engineering/
COPY config/ /app/config/

# Set working directory
WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user for Spark
RUN useradd -m -u 1001 sparkuser && chown -R sparkuser:sparkuser /app
USER sparkuser

# Expose Spark ports
EXPOSE 4040 7077 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:4040 || exit 1

# Default command
CMD ["python", "data-engineering/processing/spark_processor.py"]