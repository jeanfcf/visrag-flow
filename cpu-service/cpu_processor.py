import os
import json
import base64
import cv2
import numpy as np
import pika
import time
import signal

from motion_detection import is_motion_detected
from common.logger_config import setup_logger
from common.monitoring import ResourceMonitor
import psutil

logger = setup_logger("CPUService")
stop_requested = False

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
FRAMES_QUEUE = os.getenv("FRAMES_QUEUE", "frames_queue")
GPU_TASKS_QUEUE = os.getenv("GPU_TASKS_QUEUE", "gpu_tasks_queue")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    credentials = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            return pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=credentials))
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"Falha de conexão com RabbitMQ ({i+1}/{retries}). Retentando em {delay}s...")
            time.sleep(delay)
    raise Exception("Não foi possível conectar ao RabbitMQ após várias tentativas")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("[CPU] Recebido sinal de encerramento...")
    stop_requested = True

def main():
    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = connection.channel()

    channel.queue_declare(queue=FRAMES_QUEUE)
    channel.queue_declare(queue=GPU_TASKS_QUEUE)

    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    def callback(ch, method, properties, body):
        try:
            if stop_requested:
                ch.stop_consuming()
                return

            message_str = body.decode("utf-8")
            message = json.loads(message_str)

            frame_b64 = message.get("frame_b64")
            if not frame_b64:
                logger.warning("[CPU] Mensagem sem frame_b64. Ignorando.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            frame_bin = base64.b64decode(frame_b64)
            frame_array = np.frombuffer(frame_bin, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning("[CPU] Frame corrompido ou inválido.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            if is_motion_detected(frame):
                logger.info("[CPU] Movimento detectado -> tentando enviar para GPU queue.")
                try:
                    channel.basic_publish(
                        exchange="",
                        routing_key=GPU_TASKS_QUEUE,
                        body=json.dumps(message).encode("utf-8")
                    )
                    logger.info("[CPU] Mensagem enviada com sucesso para GPU queue.")
                except Exception as pub_err:
                    logger.error(f"[CPU] Erro ao enviar mensagem para GPU queue: {pub_err}. Mensagem descartada.")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.exception("[CPU] Erro durante processamento do frame:")
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(
        queue=FRAMES_QUEUE,
        on_message_callback=callback,
        auto_ack=False
    )

    logger.info("[CPU] Aguardando frames na fila...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("[CPU] Interrompido via teclado.")

    monitor.stop()
    monitor.join()
    connection.close()
    logger.info("[CPU] Encerrado com sucesso.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
