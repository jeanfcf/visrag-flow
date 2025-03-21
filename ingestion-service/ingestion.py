import os
import time
import cv2
import base64
import json
import pika
import signal
import sys
import time

from common.logger_config import setup_logger
from common.monitoring import ResourceMonitor
import psutil

CAMERA_URL = os.getenv("CAMERA_URL", "rtsp://meu_endereco_de_camera")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
FRAMES_QUEUE = os.getenv("FRAMES_QUEUE", "frames_queue")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

logger = setup_logger("IngestionService")
stop_requested = False

def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    credentials = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            return pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=credentials))
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Falha de conexão com RabbitMQ ({i+1}/{retries}). Retentando em {delay}s...")
            time.sleep(delay)
    raise Exception("Não foi possível conectar ao RabbitMQ após várias tentativas")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("[INGESTION] Recebido sinal de encerramento...")
    stop_requested = True

def main():
    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = connection.channel()
    channel.queue_declare(queue=FRAMES_QUEUE)

    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        logger.error(f"[INGESTION] Falha ao abrir stream: {CAMERA_URL}")
        return

    logger.info("[INGESTION] Capturando frames...")

    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    while not stop_requested:
        ret, frame = cap.read()
        if not ret:
            logger.warning("[INGESTION] Falha ao ler frame. Retrying...")
            time.sleep(1)
            continue

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("[INGESTION] Erro na compressão JPEG.")
            continue

        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        message = {
            "camera_id": "cam_001",
            "timestamp": time.time(),
            "frame_b64": frame_b64
        }

        try:
            channel.basic_publish(
                exchange="",
                routing_key=FRAMES_QUEUE,
                body=json.dumps(message).encode("utf-8")
            )
            logger.debug("[INGESTION] Frame enviado para frames_queue.")
        except Exception as e:
            logger.exception("[INGESTION] Erro ao publicar mensagem no RabbitMQ.")

    cap.release()
    monitor.stop()
    monitor.join()
    connection.close()
    logger.info("[INGESTION] Encerrado com sucesso.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
