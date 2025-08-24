# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Copy requirements (from backend/)
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend code only
COPY backend ./backend

# Expose Cloud Run port
EXPOSE 8080

# Run FastAPI app with environment variable substitution
CMD uvicorn backend.app:app --host 0.0.0.0 --port $PORT