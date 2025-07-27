FROM --platform=linux/amd64 python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model download script
COPY download_models.py .

# Download all required models and data during build
RUN python download_models.py

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Default command - will be overridden by docker run
CMD ["python", "main.py", "Default Persona", "Default Job"]