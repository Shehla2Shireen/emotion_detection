# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements (from backend/)
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install uvicorn gunicorn

# Copy backend code only
COPY backend ./backend

# Expose Cloud Run port (default 8080, but Cloud Run sets $PORT dynamically)
EXPOSE 8080

# Run FastAPI app (must listen on $PORT for Cloud Run)
# CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
# Run FastAPI app (must listen on $PORT for Cloud Run)
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]


