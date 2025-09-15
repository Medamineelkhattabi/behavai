# Smart Mobility Platform

An AI-powered behavior-aware smart mobility platform that integrates heterogeneous data sources into a unified analytics engine for real-time monitoring, predictive modeling, and optimization of passenger and vehicle flows.

## 🚀 Features

### Core Capabilities
- **Real-time Data Integration**: GPS from buses, AIS from ferries, train/metro sensors, ticketing and turnstile logs
- **AI-Powered Analytics**: LSTM/Transformer forecasting, Graph Neural Networks, Reinforcement Learning optimization
- **Anomaly Detection**: Multi-method anomaly detection with real-time alerting
- **Conversational AI**: RAG-powered chatbot for operational queries and insights
- **Interactive Dashboard**: Real-time visualization with heatmaps, flow forecasts, and recommendations

### Technical Architecture
- **Data Engineering**: Kafka/MQTT ingestion, Apache Spark processing, PostgreSQL/BigQuery storage
- **AI/ML Models**: Time-series forecasting, graph-based flow prediction, adaptive scheduling
- **API Backend**: FastAPI with ML model endpoints and real-time services
- **Frontend**: Streamlit dashboard with interactive visualizations
- **Deployment**: Docker containers with Kubernetes orchestration
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 📋 Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (for production deployment)
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Apache Kafka
- MQTT Broker

## 🛠️ Quick Start

### Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd smart-mobility-platform
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start infrastructure services**
```bash
docker-compose up -d postgres redis kafka zookeeper mqtt
```

4. **Initialize the database**
```bash
psql -h localhost -U mobility_user -d mobility_platform -f data-engineering/storage/init.sql
```

5. **Start the API backend**
```bash
cd api-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. **Start the dashboard**
```bash
cd dashboard
streamlit run app.py --server.port 8501
```

7. **Access the applications**
- Dashboard: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build all services**
```bash
docker-compose build
```

2. **Start the complete platform**
```bash
docker-compose up -d
```

3. **Verify deployment**
```bash
docker-compose ps
curl http://localhost:8000/health
```

### Kubernetes Deployment

1. **Create namespace**
```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

2. **Deploy configuration and secrets**
```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
```

3. **Deploy infrastructure**
```bash
kubectl apply -f deployment/kubernetes/database.yaml
kubectl apply -f deployment/kubernetes/redis.yaml
kubectl apply -f deployment/kubernetes/kafka.yaml
```

4. **Deploy applications**
```bash
kubectl apply -f deployment/kubernetes/api-backend.yaml
kubectl apply -f deployment/kubernetes/dashboard.yaml
```

5. **Set up monitoring**
```bash
kubectl apply -f deployment/kubernetes/monitoring.yaml
```

6. **Configure ingress**
```bash
kubectl apply -f deployment/kubernetes/ingress.yaml
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Data Ingestion  │    │   Processing    │
│                 │    │                  │    │                 │
│ • GPS Data      │───▶│ • Kafka/MQTT     │───▶│ • Apache Spark  │
│ • AIS Data      │    │ • Real-time      │    │ • Feature Eng.  │
│ • Sensor Data   │    │ • Batch ETL      │    │ • Data Quality  │
│ • Ticketing     │    │                  │    │                 │
│ • Weather       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Backend    │    │   AI/ML Models  │
│                 │    │                  │    │                 │
│ • Dashboard     │◀───│ • FastAPI        │◀───│ • LSTM/GNN      │
│ • Visualizations│    │ • Authentication │    │ • Anomaly Det.  │
│ • AI Assistant  │    │ • WebSocket      │    │ • RL Optimizer  │
│                 │    │                  │    │ • RAG System    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitoring    │    │     Storage      │    │   Deployment    │
│                 │    │                  │    │                 │
│ • Prometheus    │    │ • PostgreSQL     │    │ • Docker        │
│ • Grafana       │    │ • Redis Cache    │    │ • Kubernetes    │
│ • Alerting      │    │ • Vector DB      │    │ • Auto-scaling  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🤖 AI/ML Components

### 1. Time-Series Forecasting
- **LSTM Models**: Passenger demand prediction with 24-hour horizon
- **Transformer Models**: Advanced sequence modeling for complex patterns
- **Features**: Temporal encoding, weather integration, event impact modeling

### 2. Graph Neural Networks
- **Network Representation**: Stops as nodes, routes as edges with dynamic weights
- **Flow Prediction**: GNN-based passenger flow forecasting between locations
- **Centrality Analysis**: Identification of critical network nodes

### 3. Reinforcement Learning
- **Adaptive Scheduling**: PPO/DQN agents for vehicle dispatch optimization
- **Dynamic Pricing**: RL-based demand distribution strategies
- **Route Optimization**: Real-time route adjustment based on conditions

### 4. Anomaly Detection
- **Multi-Method Approach**: Isolation Forest, Autoencoders, LSTM-AE
- **Real-time Detection**: Streaming anomaly identification
- **Classification**: Automatic categorization of anomaly types

### 5. Conversational AI
- **RAG System**: Retrieval-Augmented Generation for operational queries
- **Knowledge Base**: Dynamic integration of system documentation and real-time data
- **Multi-Modal**: Support for text, voice, and visual interactions

## 📊 Data Pipeline

### Data Sources
- **Vehicle GPS**: Real-time location, speed, occupancy
- **AIS Data**: Ferry positions, destinations, ETAs
- **Sensor Data**: Station occupancy, environmental conditions
- **Ticketing Systems**: Passenger flows, payment data
- **External APIs**: Weather, traffic, special events

### Processing Pipeline
1. **Ingestion**: Kafka/MQTT for real-time streaming
2. **Validation**: Data quality checks and cleansing
3. **Transformation**: Feature engineering and normalization
4. **Storage**: Time-series data in PostgreSQL, caching in Redis
5. **Analysis**: Batch and streaming analytics with Spark

### Data Models
- **Operational**: Real-time system state and performance
- **Analytical**: Historical patterns and trend analysis
- **Predictive**: Future state forecasting and scenario modeling

## 🔧 Configuration

### Environment Variables
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mobility_platform
DB_USER=mobility_user
DB_PASSWORD=mobility_pass

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=mobility-platform-consumer

# API
API_BASE_URL=http://localhost:8000
JWT_SECRET=your-jwt-secret

# AI/ML
OPENAI_API_KEY=your-openai-key
MODEL_PATH=/app/models
```

### Scaling Configuration
- **API Backend**: Horizontal Pod Autoscaler (2-10 replicas)
- **Database**: Read replicas for analytics workloads
- **Cache**: Redis cluster for high availability
- **Processing**: Spark cluster auto-scaling

## 🔍 Monitoring & Observability

### Metrics
- **System Health**: Uptime, response times, error rates
- **ML Performance**: Model accuracy, prediction latency
- **Business KPIs**: Passenger satisfaction, operational efficiency
- **Infrastructure**: Resource utilization, scaling events

### Dashboards
- **Operational**: Real-time system status and alerts
- **Analytical**: Performance trends and insights
- **Technical**: Infrastructure metrics and health

### Alerting
- **Critical**: System outages, data pipeline failures
- **Warning**: Performance degradation, capacity issues
- **Info**: Deployment events, scaling activities

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
locust -f tests/load/locustfile.py
```

### Model Validation
```bash
python -m ai-models.validation.model_validator
```

## 🚀 Deployment Strategies

### Development
- Docker Compose for local development
- Hot reloading for rapid iteration
- Mock data generators for testing

### Staging
- Kubernetes deployment with reduced resources
- Production-like data pipeline
- Automated testing and validation

### Production
- Multi-zone Kubernetes deployment
- Auto-scaling and load balancing
- Comprehensive monitoring and alerting
- Blue-green deployments

## 📈 Performance Optimization

### API Optimization
- Response caching with Redis
- Database connection pooling
- Async processing for ML inference
- Request rate limiting

### ML Model Optimization
- Model quantization and pruning
- Batch inference for efficiency
- GPU acceleration where applicable
- Model versioning and A/B testing

### Data Pipeline Optimization
- Stream processing for real-time data
- Partitioning strategies for large datasets
- Compression and serialization optimization
- Incremental processing patterns

## 🔐 Security

### Authentication & Authorization
- JWT-based API authentication
- Role-based access control (RBAC)
- API key management for external integrations
- Session management and timeout policies

### Data Security
- Encryption at rest and in transit
- PII data anonymization
- Audit logging for sensitive operations
- Secure credential management with Kubernetes secrets

### Network Security
- TLS/SSL for all communications
- Network policies in Kubernetes
- API rate limiting and DDoS protection
- Regular security scanning and updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and embeddings
- Apache Foundation for Kafka and Spark
- PostgreSQL Global Development Group
- Redis Labs
- Kubernetes community
- Streamlit team
- FastAPI developers

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation wiki
- Join our community discussions

---

**Smart Mobility Platform** - Transforming urban transportation through AI and data science.
