FROM python:3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY common /app/common

COPY ingestion-service/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion-service/ /app/

CMD ["python", "ingestion.py"]
