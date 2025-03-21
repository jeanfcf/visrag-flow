import os
import time
import cv2
import base64
import json
import pika
import signal
from common.logger_config import setup_logger
from common.monitoring import ResourceMonitor

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
            conn = pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=credentials))
            logger.info("Conexão com RabbitMQ estabelecida.")
            return conn
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"Falha de conexão com RabbitMQ ({i+1}/{retries}). Retentando em {delay}s...")
            time.sleep(delay)
    raise Exception("Não foi possível conectar ao RabbitMQ após várias tentativas")

def connect_rtsp_stream(url, retries=10, delay=5):
    for i in range(retries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            logger.info(f"Conexão RTSP estabelecida após {i+1} tentativa(s).")
            return cap
        else:
            logger.warning(f"Falha ao abrir stream RTSP (tentativa {i+1}/{retries}). Retentando em {delay}s...")
            time.sleep(delay)
    logger.error(f"Não foi possível conectar à stream RTSP após {retries} tentativas.")
    return None


def signal_handler(sig, frame):
    global stop_requested
    logger.info("Recebido sinal de encerramento.")
    stop_requested = True

def main():
    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = connection.channel()
    channel.queue_declare(queue=FRAMES_QUEUE)

    cap = connect_rtsp_stream(CAMERA_URL)
    if cap is None:
        logger.error("Encerrando devido à falha na conexão RTSP.")
        return

    logger.info("Iniciando captura de frames do RTSP.")
    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    while not stop_requested:
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Falha ao ler frame; tentando reconectar o RTSP...")
                cap.release()
                time.sleep(2)
                cap = connect_rtsp_stream(CAMERA_URL)
                if cap is None:
                    logger.error("Não foi possível reconectar ao RTSP. Tentando novamente em 5 segundos.")
                    time.sleep(5)
                continue

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.error("Erro ao comprimir frame.")
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
                logger.debug("Frame enviado para fila.")
            except pika.exceptions.AMQPConnectionError:
                logger.error("Conexão com RabbitMQ perdida durante publicação. Tentando reconectar...")
                connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
                channel = connection.channel()
                channel.queue_declare(queue=FRAMES_QUEUE)
                channel.basic_publish(
                    exchange="",
                    routing_key=FRAMES_QUEUE,
                    body=json.dumps(message).encode("utf-8")
                )
                logger.info("Frame reenviado após reconexão.")

        except Exception as e:
            logger.exception("Erro no loop de ingestão:")
            time.sleep(2)

    cap.release()
    monitor.stop()
    monitor.join()
    connection.close()
    logger.info("Encerrado com sucesso.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
