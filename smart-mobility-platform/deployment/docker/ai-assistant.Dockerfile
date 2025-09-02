FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional AI dependencies
RUN pip install --no-cache-dir \
    langchain==0.0.352 \
    openai==1.6.1 \
    chromadb==0.4.18 \
    sentence-transformers==2.2.2

# Copy application code
COPY ai-assistant/ ./ai-assistant/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app

# Create directories for data
RUN mkdir -p /app/data/chroma_db /app/data/knowledge-base

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["python", "-m", "ai-assistant.app.main"]