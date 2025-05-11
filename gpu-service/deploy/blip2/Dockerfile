FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY common /app/common
RUN pip install --no-cache-dir -r common/requirements.txt
COPY gpu-service/deploy/blip2/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
# ➜ copia lista de processors
COPY gpu-service/deploy/blip2/processors.json /app/
ENV PROCESSORS_FILE=/app/processors.json

COPY gpu-service/ /app/
CMD ["python", "gpu_processor.py"]
