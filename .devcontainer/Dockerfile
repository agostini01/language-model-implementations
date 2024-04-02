FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio torchtyping