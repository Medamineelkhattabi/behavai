"""
RAG Knowledge Base for Smart Mobility Platform
Manages document ingestion, vector storage, and retrieval for conversational AI
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PDFLoader, CSVLoader
from langchain.schema import Document

# Database connection
import asyncpg
import redis

logger = logging.getLogger(__name__)

class MobilityKnowledgeBase:
    """Knowledge base for mobility domain information"""
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 db_connection_string: str = None,
                 redis_url: str = "redis://localhost:6379"):
        
        self.chroma_db_path = chroma_db_path
        self.embedding_model_name = embedding_model
        self.db_connection_string = db_connection_string
        self.redis_url = redis_url
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.redis_client = None
        self.db_pool = None
        
        # Text splitter for documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Document types and loaders
        self.document_loaders = {
            '.txt': TextLoader,
            '.pdf': PDFLoader,
            '.csv': CSVLoader
        }
        
        # Knowledge categories
        self.knowledge_categories = {
            'operational_procedures': 'Standard operating procedures and guidelines',
            'system_documentation': 'Technical system documentation',
            'performance_metrics': 'KPIs and performance measurement guidelines',
            'troubleshooting': 'Problem resolution and troubleshooting guides',
            'regulatory_compliance': 'Compliance and regulatory requirements',
            'best_practices': 'Industry best practices and recommendations',
            'historical_insights': 'Historical data analysis and insights',
            'real_time_data': 'Current system status and real-time information'
        }
    
    async def initialize(self):
        """Initialize knowledge base components"""
        try:
            logger.info("Initializing knowledge base...")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="mobility_knowledge",
                metadata={"description": "Smart mobility platform knowledge base"}
            )
            logger.info("ChromaDB collection initialized")
            
            # Initialize Redis for caching
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Redis client initialized")
            
            # Initialize database connection pool
            if self.db_connection_string:
                self.db_pool = await asyncpg.create_pool(self.db_connection_string)
                logger.info("Database connection pool initialized")
            
            # Load initial knowledge base
            await self._load_initial_knowledge()
            
            logger.info("Knowledge base initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    async def _load_initial_knowledge(self):
        """Load initial knowledge base content"""
        try:
            # Load static documents
            await self._load_static_documents()
            
            # Load system documentation
            await self._load_system_documentation()
            
            # Load operational procedures
            await self._load_operational_procedures()
            
            # Update with recent data
            await self._update_real_time_knowledge()
            
        except Exception as e:
            logger.error(f"Failed to load initial knowledge: {e}")
    
    async def _load_static_documents(self):
        """Load static documents from files"""
        try:
            docs_path = Path("./knowledge-base/documents")
            if not docs_path.exists():
                logger.warning("Documents directory not found, creating sample documents")
                await self._create_sample_documents()
                return
            
            for file_path in docs_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in self.document_loaders:
                    await self._process_document(file_path)
            
            logger.info("Static documents loaded")
            
        except Exception as e:
            logger.error(f"Failed to load static documents: {e}")
    
    async def _create_sample_documents(self):
        """Create sample knowledge base documents"""
        try:
            docs_path = Path("./knowledge-base/documents")
            docs_path.mkdir(parents=True, exist_ok=True)
            
            # Sample operational procedures
            operational_doc = """
# Smart Mobility Platform Operational Procedures

## Daily Operations Checklist
1. Check system health status
2. Review overnight alerts and anomalies
3. Verify data pipeline integrity
4. Monitor vehicle fleet status
5. Assess passenger demand patterns

## Incident Response Procedures
### High Congestion Alert
- Immediately assess affected routes
- Deploy additional vehicles if available
- Notify passengers of delays
- Implement alternative routing

### Vehicle Breakdown Protocol
- Dispatch replacement vehicle
- Update passenger information systems
- Log incident for maintenance review
- Adjust schedule for affected route

## Performance Monitoring
- Monitor KPIs hourly during peak times
- Generate daily performance reports
- Weekly trend analysis
- Monthly optimization reviews

## Emergency Procedures
### System Outage
1. Activate backup systems
2. Switch to manual operations
3. Notify all stakeholders
4. Begin system recovery procedures

### Weather-Related Disruptions
- Monitor weather conditions continuously
- Adjust schedules based on conditions
- Implement safety protocols
- Communicate with passengers
"""
            
            with open(docs_path / "operational_procedures.txt", "w") as f:
                f.write(operational_doc)
            
            # Sample system documentation
            system_doc = """
# Smart Mobility Platform System Documentation

## Architecture Overview
The Smart Mobility Platform consists of:
- Data ingestion layer (Kafka/MQTT)
- Processing layer (Apache Spark)
- AI/ML models (LSTM, GNN, RL)
- API layer (FastAPI)
- Frontend dashboard (Streamlit)

## Data Flow
1. Real-time data collection from vehicles and infrastructure
2. Stream processing and feature engineering
3. ML model inference for predictions
4. Results storage and API serving
5. Dashboard visualization and alerts

## Key Components

### Prediction Models
- LSTM for demand forecasting
- GNN for passenger flow prediction
- RL for adaptive scheduling
- Anomaly detection for system monitoring

### Data Sources
- GPS data from vehicles
- AIS data from ferries
- Sensor data from stations
- Ticketing system data
- Weather feeds
- Special events data

### Performance Metrics
- Prediction accuracy (MAE, RMSE, MAPE)
- System latency
- Throughput
- Availability
- User satisfaction scores

## Troubleshooting Guide

### Common Issues
1. Model prediction failures
   - Check data quality
   - Verify model health
   - Review feature engineering

2. High system latency
   - Check database performance
   - Monitor API response times
   - Review caching effectiveness

3. Data pipeline failures
   - Verify data source connectivity
   - Check Kafka/MQTT brokers
   - Review Spark job status
"""
            
            with open(docs_path / "system_documentation.txt", "w") as f:
                f.write(system_doc)
            
            # Sample best practices
            best_practices_doc = """
# Smart Mobility Best Practices

## Data Quality Management
- Implement data validation at ingestion
- Monitor data freshness and completeness
- Establish data quality metrics
- Regular data profiling and cleansing

## Model Performance Optimization
- Regular model retraining schedules
- A/B testing for model improvements
- Feature importance analysis
- Performance monitoring and alerting

## Operational Excellence
- Automated deployment pipelines
- Comprehensive monitoring and logging
- Disaster recovery procedures
- Security best practices

## Passenger Experience
- Real-time information accuracy
- Personalized recommendations
- Accessibility considerations
- Multi-channel communication

## Sustainability
- Energy-efficient routing
- Occupancy optimization
- Environmental impact monitoring
- Carbon footprint reduction
"""
            
            with open(docs_path / "best_practices.txt", "w") as f:
                f.write(best_practices_doc)
            
            logger.info("Sample documents created")
            
        except Exception as e:
            logger.error(f"Failed to create sample documents: {e}")
    
    async def _process_document(self, file_path: Path):
        """Process and index a document"""
        try:
            # Load document
            loader_class = self.document_loaders[file_path.suffix]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Split documents into chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(doc_chunks):
                    chunks.append({
                        'content': chunk,
                        'metadata': {
                            'source': str(file_path),
                            'chunk_id': i,
                            'category': self._categorize_document(file_path.name),
                            'last_updated': datetime.utcnow().isoformat()
                        }
                    })
            
            # Generate embeddings and store
            await self._store_chunks(chunks)
            
            logger.info(f"Processed document: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
    
    def _categorize_document(self, filename: str) -> str:
        """Categorize document based on filename"""
        filename_lower = filename.lower()
        
        if 'procedure' in filename_lower or 'operation' in filename_lower:
            return 'operational_procedures'
        elif 'system' in filename_lower or 'technical' in filename_lower:
            return 'system_documentation'
        elif 'troubleshoot' in filename_lower or 'problem' in filename_lower:
            return 'troubleshooting'
        elif 'best_practice' in filename_lower or 'guideline' in filename_lower:
            return 'best_practices'
        elif 'compliance' in filename_lower or 'regulation' in filename_lower:
            return 'regulatory_compliance'
        else:
            return 'system_documentation'
    
    async def _store_chunks(self, chunks: List[Dict[str, Any]]):
        """Store document chunks with embeddings"""
        try:
            if not chunks:
                return
            
            # Generate embeddings
            contents = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # Prepare data for ChromaDB
            ids = [f"doc_{i}_{hash(chunk['content'])}" for i, chunk in enumerate(chunks)]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(chunks)} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
    
    async def _load_system_documentation(self):
        """Load system documentation from database"""
        try:
            if not self.db_pool:
                return
            
            # Get system metrics and documentation
            async with self.db_pool.acquire() as conn:
                # Load route information
                routes = await conn.fetch("""
                    SELECT route_id, name, description, transport_mode_id
                    FROM routes
                """)
                
                # Load stop information
                stops = await conn.fetch("""
                    SELECT stop_id, name, latitude, longitude, zone
                    FROM stops
                """)
                
                # Create documentation from system data
                system_info = {
                    'routes': [dict(route) for route in routes],
                    'stops': [dict(stop) for stop in stops]
                }
                
                await self._store_system_info(system_info)
            
        except Exception as e:
            logger.error(f"Failed to load system documentation: {e}")
    
    async def _store_system_info(self, system_info: Dict[str, Any]):
        """Store system information as knowledge"""
        try:
            # Create documentation from system data
            route_docs = []
            for route in system_info['routes']:
                doc_content = f"""
Route: {route['name']} (ID: {route['route_id']})
Transport Mode: {route.get('transport_mode_id', 'Unknown')}
Description: {route.get('description', 'No description available')}
"""
                route_docs.append({
                    'content': doc_content,
                    'metadata': {
                        'category': 'system_documentation',
                        'type': 'route_info',
                        'route_id': route['route_id'],
                        'last_updated': datetime.utcnow().isoformat()
                    }
                })
            
            stop_docs = []
            for stop in system_info['stops']:
                doc_content = f"""
Stop: {stop['name']} (ID: {stop['stop_id']})
Location: {stop['latitude']}, {stop['longitude']}
Zone: {stop.get('zone', 'Unknown')}
"""
                stop_docs.append({
                    'content': doc_content,
                    'metadata': {
                        'category': 'system_documentation',
                        'type': 'stop_info',
                        'stop_id': stop['stop_id'],
                        'last_updated': datetime.utcnow().isoformat()
                    }
                })
            
            # Store all documentation
            all_docs = route_docs + stop_docs
            if all_docs:
                await self._store_chunks(all_docs)
            
        except Exception as e:
            logger.error(f"Failed to store system info: {e}")
    
    async def _load_operational_procedures(self):
        """Load operational procedures and guidelines"""
        try:
            # This would load from operational databases or files
            # For now, we'll use the sample documents created earlier
            pass
            
        except Exception as e:
            logger.error(f"Failed to load operational procedures: {e}")
    
    async def _update_real_time_knowledge(self):
        """Update knowledge base with real-time information"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                # Get recent anomalies
                anomalies = await conn.fetch("""
                    SELECT location_id, anomaly_type, severity_score, description, detected_at
                    FROM anomaly_detections
                    WHERE detected_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY detected_at DESC
                    LIMIT 50
                """)
                
                # Get current system status
                system_status = await conn.fetch("""
                    SELECT location_id, congestion_level, avg_wait_time, timestamp
                    FROM congestion_metrics
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                    ORDER BY timestamp DESC
                """)
                
                # Create real-time knowledge documents
                await self._store_real_time_info(anomalies, system_status)
            
        except Exception as e:
            logger.error(f"Failed to update real-time knowledge: {e}")
    
    async def _store_real_time_info(self, anomalies, system_status):
        """Store real-time information as knowledge"""
        try:
            docs = []
            
            # Anomaly information
            for anomaly in anomalies:
                doc_content = f"""
Recent Anomaly Alert:
Location: {anomaly['location_id']}
Type: {anomaly['anomaly_type']}
Severity: {anomaly['severity_score']}
Description: {anomaly['description']}
Detected: {anomaly['detected_at']}
"""
                docs.append({
                    'content': doc_content,
                    'metadata': {
                        'category': 'real_time_data',
                        'type': 'anomaly_alert',
                        'location_id': anomaly['location_id'],
                        'timestamp': anomaly['detected_at'].isoformat(),
                        'last_updated': datetime.utcnow().isoformat()
                    }
                })
            
            # System status
            for status in system_status:
                doc_content = f"""
Current System Status:
Location: {status['location_id']}
Congestion Level: {status['congestion_level']}%
Average Wait Time: {status['avg_wait_time']} minutes
As of: {status['timestamp']}
"""
                docs.append({
                    'content': doc_content,
                    'metadata': {
                        'category': 'real_time_data',
                        'type': 'system_status',
                        'location_id': status['location_id'],
                        'timestamp': status['timestamp'].isoformat(),
                        'last_updated': datetime.utcnow().isoformat()
                    }
                })
            
            if docs:
                await self._store_chunks(docs)
                logger.info(f"Stored {len(docs)} real-time knowledge chunks")
            
        except Exception as e:
            logger.error(f"Failed to store real-time info: {e}")
    
    async def search_knowledge(self, query: str, 
                             categories: Optional[List[str]] = None,
                             limit: int = 10,
                             similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information"""
        try:
            # Check cache first
            cache_key = f"knowledge_search:{hash(query)}:{categories}:{limit}"
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Build where clause for categories
            where_clause = None
            if categories:
                where_clause = {"category": {"$in": categories}}
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause
            )
            
            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity = 1.0 - distance
                    
                    if similarity >= similarity_threshold:
                        search_results.append({
                            'content': doc,
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': similarity,
                            'id': results['ids'][0][i]
                        })
            
            # Cache results
            self.redis_client.setex(
                cache_key, 
                300,  # 5 minutes
                json.dumps(search_results, default=str)
            )
            
            logger.info(f"Found {len(search_results)} relevant documents for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def get_contextual_information(self, query: str, 
                                       context_type: str = "general") -> Dict[str, Any]:
        """Get contextual information based on query type"""
        try:
            context_info = {}
            
            # Determine relevant categories based on context type
            if context_type == "operational":
                categories = ['operational_procedures', 'troubleshooting', 'real_time_data']
            elif context_type == "technical":
                categories = ['system_documentation', 'troubleshooting']
            elif context_type == "performance":
                categories = ['performance_metrics', 'historical_insights', 'real_time_data']
            else:
                categories = None
            
            # Search knowledge base
            search_results = await self.search_knowledge(
                query=query,
                categories=categories,
                limit=5
            )
            
            context_info['relevant_documents'] = search_results
            
            # Get real-time system status if relevant
            if any(word in query.lower() for word in ['current', 'now', 'status', 'real-time']):
                context_info['real_time_status'] = await self._get_current_system_status()
            
            # Get historical context if relevant
            if any(word in query.lower() for word in ['trend', 'history', 'past', 'previous']):
                context_info['historical_context'] = await self._get_historical_context(query)
            
            return context_info
            
        except Exception as e:
            logger.error(f"Failed to get contextual information: {e}")
            return {}
    
    async def _get_current_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            if not self.db_pool:
                return {}
            
            async with self.db_pool.acquire() as conn:
                # Get current congestion levels
                congestion = await conn.fetch("""
                    SELECT location_id, congestion_level, timestamp
                    FROM congestion_metrics
                    WHERE timestamp >= NOW() - INTERVAL '15 minutes'
                    ORDER BY timestamp DESC
                    LIMIT 20
                """)
                
                # Get recent anomalies
                anomalies = await conn.fetch("""
                    SELECT location_id, anomaly_type, severity_score
                    FROM anomaly_detections
                    WHERE detected_at >= NOW() - INTERVAL '1 hour'
                    AND resolved_at IS NULL
                    ORDER BY severity_score DESC
                    LIMIT 10
                """)
                
                return {
                    'congestion_levels': [dict(row) for row in congestion],
                    'active_anomalies': [dict(row) for row in anomalies],
                    'last_updated': datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Failed to get current system status: {e}")
            return {}
    
    async def _get_historical_context(self, query: str) -> Dict[str, Any]:
        """Get historical context relevant to query"""
        try:
            # This would analyze the query and fetch relevant historical data
            # For now, return placeholder
            return {
                'message': 'Historical context analysis not yet implemented',
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical context: {e}")
            return {}
    
    async def add_document(self, content: str, 
                          metadata: Dict[str, Any]) -> bool:
        """Add a new document to the knowledge base"""
        try:
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Prepare chunks with metadata
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'last_updated': datetime.utcnow().isoformat()
                })
                
                chunk_docs.append({
                    'content': chunk,
                    'metadata': chunk_metadata
                })
            
            # Store chunks
            await self._store_chunks(chunk_docs)
            
            logger.info(f"Added document with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def update_knowledge_base(self):
        """Update knowledge base with latest information"""
        try:
            logger.info("Updating knowledge base...")
            
            # Update real-time information
            await self._update_real_time_knowledge()
            
            # Clear old cached searches
            cache_keys = self.redis_client.keys("knowledge_search:*")
            if cache_keys:
                self.redis_client.delete(*cache_keys)
            
            logger.info("Knowledge base updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            # Get collection stats
            collection_count = self.collection.count()
            
            # Get category distribution
            all_docs = self.collection.get()
            category_counts = {}
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    category = metadata.get('category', 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'total_documents': collection_count,
                'category_distribution': category_counts,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                self.redis_client.close()
            
            if self.db_pool:
                await self.db_pool.close()
            
            logger.info("Knowledge base cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize knowledge base
        kb = MobilityKnowledgeBase(
            db_connection_string="postgresql://mobility_user:mobility_pass@localhost:5432/mobility_platform"
        )
        
        await kb.initialize()
        
        # Test search
        results = await kb.search_knowledge("How to handle high congestion?")
        print(f"Found {len(results)} relevant documents")
        
        for result in results:
            print(f"Score: {result['similarity_score']:.3f}")
            print(f"Content: {result['content'][:200]}...")
            print("---")
        
        # Get contextual information
        context = await kb.get_contextual_information(
            "What is the current system status?",
            context_type="operational"
        )
        
        print(f"Context: {context}")
        
        # Cleanup
        await kb.cleanup()
    
    # Run example
    asyncio.run(main())