# Use official Python slim image
FROM python:3.10-slim

# Set environment variables to prevent Python buffering and set Streamlit config
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn uvicorn

# Copy project files
COPY . .

# Expose ports for backend and frontend
EXPOSE 8000 8501

# Install tini to manage multiple processes
RUN apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*

# Start both backend and frontend
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD sh -c "uvicorn backend.app:app --host 0.0.0.0 --port $PORT & streamlit run frontend/app.py"
