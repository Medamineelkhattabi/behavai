# Smart Mobility Platform Setup Script for Windows
# Run this script in PowerShell as Administrator

Write-Host "Setting up Smart Mobility Platform..." -ForegroundColor Green

# Create requirements.txt
@"
# Core Data Engineering
kafka-python==2.0.2
paho-mqtt==1.6.1
pyspark==3.5.0
pandas==2.1.4
numpy==1.24.3
sqlalchemy==2.0.23
psycopg2-binary==2.9.9

# Machine Learning & AI
torch==2.1.2
torch-geometric==2.4.0
transformers==4.36.2
scikit-learn==1.3.2
tensorflow==2.15.0
stable-baselines3==2.2.1
gymnasium==0.29.1

# API & Web Framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2
streamlit==1.29.0
plotly==5.17.0
dash==2.16.1

# AI Assistant & NLP
langchain==0.0.352
openai==1.6.1
chromadb==0.4.18
sentence-transformers==2.2.2

# Data Visualization
matplotlib==3.8.2
seaborn==0.13.0
networkx==3.2.1
folium==0.15.1
streamlit-folium==0.15.0

# Monitoring & Deployment
prometheus-client==0.19.0
docker==6.1.3

# Testing & Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
requests==2.31.0

# Utilities
python-dotenv==1.0.0
redis==5.0.1
schedule==1.2.1
asyncpg==0.29.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8

# Create docker-compose.yml for Windows
@"
version: '3.8'

services:
  # Database Services
  postgres:
    image: postgres:15
    container_name: mobility-postgres
    environment:
      POSTGRES_DB: mobility_platform
      POSTGRES_USER: mobility_user
      POSTGRES_PASSWORD: mobility_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - mobility-network

  redis:
    image: redis:7-alpine
    container_name: mobility-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - mobility-network

  # Message Queue
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    container_name: mobility-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - mobility-network

  kafka:
    image: confluentinc/cp-kafka:latest
    container_name: mobility-kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
    networks:
      - mobility-network

  # MQTT Broker
  mqtt:
    image: eclipse-mosquitto:2
    container_name: mobility-mqtt
    ports:
      - "1883:1883"
      - "9001:9001"
    networks:
      - mobility-network

volumes:
  postgres_data:
  redis_data:

networks:
  mobility-network:
    driver: bridge
"@ | Out-File -FilePath "docker-compose.yml" -Encoding UTF8

# Create .env file
@"
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mobility_platform
DB_USER=mobility_user
DB_PASSWORD=mobility_pass

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=mobility-platform-consumer

# API Configuration
API_BASE_URL=http://localhost:8000
JWT_SECRET=your-super-secret-jwt-key-change-in-production

# AI Configuration (Optional - add your OpenAI key)
OPENAI_API_KEY=your-openai-api-key-here

# Paths
MODEL_PATH=./models
LOG_LEVEL=INFO
"@ | Out-File -FilePath ".env" -Encoding UTF8

Write-Host "Project files created successfully!" -ForegroundColor Green
Write-Host "Next: Run 'pip install -r requirements.txt'" -ForegroundColor Yellow