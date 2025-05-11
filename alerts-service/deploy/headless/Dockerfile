FROM python:3.9-slim

RUN apt-get update \
 && apt-get install -y \
      libgl1-mesa-glx \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# common (logger, monitoring)
COPY common /app/common
RUN pip install --no-cache-dir -r common/requirements.txt

# service requirements (headless OpenCV!)
COPY alerts-service/deploy/headless/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ➜ copia lista de rules
COPY alerts-service/deploy/headless/rules.json /app/
ENV RULES_FILE=/app/rules.json

COPY alerts-service/ /app/

ENV VISUALIZE=false
CMD ["python", "alerts_router.py"]
