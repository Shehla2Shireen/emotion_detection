# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose ports
EXPOSE 8000 8501

# Run backend and frontend together
CMD uvicorn backend.app:app --host 0.0.0.0 --port 8000 & \
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
