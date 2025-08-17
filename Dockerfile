FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY ui ./ui
COPY data ./data
COPY README.md .

# Set Python path
ENV PYTHONPATH=/app

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
PORT=${PORT:-8080}\n\
echo "Starting Streamlit on port $PORT"\n\
exec streamlit run ui/app_streamlit.py \\\n\
    --server.port=$PORT \\\n\
    --server.address=0.0.0.0 \\\n\
    --server.headless=true \\\n\
    --browser.gatherUsageStats=false \\\n\
    --server.enableCORS=false \\\n\
    --server.enableXsrfProtection=false \\\n\
    --logger.level=info' > /app/start.sh

# Make script executable
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8080

# Start the application
CMD ["/app/start.sh"]

