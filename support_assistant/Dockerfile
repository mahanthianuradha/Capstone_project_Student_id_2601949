# ============================================================
# Zepto Support Assistant Dockerfile
# ============================================================

# Use a small official Python image.
FROM python:3.11-slim

# Prevent Python from creating .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Make Python output appear immediately in Docker logs.
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# Copy the dependency file first.

# Docker can cache this layer, making later builds faster.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY main.py .

# Copy the complete policy corpus.
COPY docs ./docs

# Create the local ChromaDB directory.
RUN mkdir -p /app/chroma_db

# Document the port used by Uvicorn.
EXPOSE 7860

# Start the FastAPI application.

# 0.0.0.0 makes the application accessible outside
# the Docker container.
CMD [
    "uvicorn",
    "main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "7860"
]