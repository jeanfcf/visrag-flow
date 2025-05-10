FROM python:3.9-slim

RUN apt-get update \
 && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY common /app/common
RUN pip install --no-cache-dir -r common/requirements.txt

COPY alerts-service/deploy/default/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY alerts-service/alerts_router.py /app/
CMD ["python", "alerts_router.py"]
