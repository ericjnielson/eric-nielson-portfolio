FROM python:3.11-slim AS base

# Set Python to run in unbuffered mode and disable writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# System deps needed for building scientific stack and OpenCV runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Cloud Run listens on $PORT (default 8080)
EXPOSE 8080

CMD ["python", "app.py"]

