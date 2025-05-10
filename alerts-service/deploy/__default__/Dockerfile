FROM python:3.9-slim

# libs de vídeo e GUI (Qt backend do OpenCV)
RUN apt-get update \
 && apt-get install -y \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libxrender1 \
      libxext6 \
      libsm6 \
      libxcb1 \
      libx11-xcb1 \
      libfontconfig1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY common /app/common
RUN pip install --no-cache-dir -r common/requirements.txt

COPY alerts-service/deploy/__default__/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY alerts-service/alerts_router.py /app/

ENV VISUALIZE=true
# expomos o DISPLAY do host
CMD ["python", "alerts_router.py"]
