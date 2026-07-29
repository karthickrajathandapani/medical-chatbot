# ==========================================================================
# Dockerfile
# PURPOSE: package the chatbot so it can run on Hugging Face Spaces (or any
# other Docker-friendly free host) with zero manual server setup.
# ==========================================================================

FROM python:3.11-slim

# System deps some ML wheels need to build/run smoothly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (separate layer = faster rebuilds when you
# only change app code, not requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
