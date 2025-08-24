# Use official Python slim image
FROM python:3.10-slim

# Set environment variables for Streamlit
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set working directory
WORKDIR /app

# Copy requirements (from frontend/)
COPY frontend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install streamlit requests pillow

# Copy frontend code only
COPY frontend ./frontend

# Expose Cloud Run port (Cloud Run sets $PORT automatically, defaults to 8080)
EXPOSE 8080

# Start Streamlit app (must listen on $PORT for Cloud Run)
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "${PORT}", "--server.address", "0.0.0.0"]
