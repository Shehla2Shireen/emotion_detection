# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install uvicorn gunicorn

# Copy backend code only
COPY backend ./backend
COPY backend/artifacts ./backend/artifacts

# Expose Cloud Run port
EXPOSE 8080

# Start FastAPI app on $PORT (required by Cloud Run)
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
