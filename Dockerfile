# # Use official Python slim image
# FROM python:3.10-slim

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PORT=8080

# # Set working directory
# WORKDIR /app

# # Copy requirements (from backend/)
# COPY backend/requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# # Copy backend code only
# COPY backend ./backend

# # Expose Cloud Run port
# EXPOSE 8080

# # Run FastAPI app with environment variable substitution
# CMD uvicorn backend.app:app --host 0.0.0.0 --port $PORT













# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend ./backend

# Download Haar cascade during build
RUN wget -P /app/backend/ https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Expose port
EXPOSE 8080

# Start the application
CMD uvicorn backend.app:app --host 0.0.0.0 --port $PORT