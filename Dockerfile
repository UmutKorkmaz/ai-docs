FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY analytics_dashboard.py .
COPY assessment/ ./assessment/
COPY integration_setup.py .

# Create data directories
RUN mkdir -p data logs static

# Set environment variables
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:///data/analytics.db
ENV REDIS_URL=redis://redis:6379
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/dashboard || exit 1

# Run the application
CMD ["uvicorn", "analytics_dashboard:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]