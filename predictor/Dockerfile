FROM python:3.12-slim

# Install system dependencies, including OpenGL and GLib libraries
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1 

# Set the working directory inside the container
WORKDIR /app

# Copy the application code into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
