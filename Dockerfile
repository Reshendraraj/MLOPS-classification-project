FROM python:slim

# Environment variables (no spaces around '='!)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Optional: run training pipeline during image build (not always recommended)
RUN python pipeline/training_pipeline.py

# Expose the port
EXPOSE 5000

# Run application (space after CMD, fix syntax)
CMD ["python", "application.py"]
