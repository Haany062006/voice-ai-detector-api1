FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run requires port 8080
EXPOSE 8080

# START FASTAPI SERVER (THIS IS THE KEY)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
