FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY common /app/common
RUN pip install --no-cache-dir -r common/requirements.txt
COPY gpu-service/deploy/__default__/requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt
COPY gpu-service/ /app/
CMD ["python3", "gpu_processor.py"]
