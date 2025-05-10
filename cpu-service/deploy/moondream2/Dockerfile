FROM python:3.9-slim

# Add libvips installation
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY common /app/common
RUN pip install --no-cache-dir -r common/requirements.txt

COPY cpu-service/deploy/moondream2/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY cpu-service/ /app/

CMD ["python", "cpu_processor.py"]
