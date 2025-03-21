import os
import json
import base64
import cv2
import numpy as np
import pika
import time
import signal
import hashlib

from motion_detection import is_motion_detected
from common.logger_config import setup_logger
from common.monitoring import ResourceMonitor

logger = setup_logger("CPUService")
stop_requested = False
last_sent_hash = None

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
FRAMES_QUEUE = os.getenv("FRAMES_QUEUE", "frames_queue")
GPU_TASKS_QUEUE = os.getenv("GPU_TASKS_QUEUE", "gpu_tasks_queue")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

# Declaramos "channel" como variável global (será definida em main)
channel = None

def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    credentials = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=credentials))
            logger.info("Conexão com RabbitMQ estabelecida (CPUService).")
            return conn
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"Falha de conexão com RabbitMQ ({i+1}/5) no CPUService. Retentando em {delay}s...")
            time.sleep(delay)
    raise Exception("CPUService: Não foi possível conectar ao RabbitMQ após várias tentativas")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("CPUService: Recebido sinal de encerramento.")
    stop_requested = True

def main():
    global channel
    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = connection.channel()
    channel.queue_declare(queue=FRAMES_QUEUE)
    channel.queue_declare(queue=GPU_TASKS_QUEUE)

    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    def callback(ch, method, properties, body):
        global channel  # Usamos a variável global channel para atualizar, se necessário
        global last_sent_hash
        try:
            if stop_requested:
                ch.stop_consuming()
                return

            logger.debug("CPUService: Processando frame recebido.")
            message_str = body.decode("utf-8")
            message = json.loads(message_str)
            frame_b64 = message.get("frame_b64")
            if not frame_b64:
                logger.warning("CPUService: Mensagem sem frame_b64. Ignorando.")
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            # Calcula hash para detectar duplicatas
            current_hash = hashlib.md5(frame_b64.encode('utf-8')).hexdigest()
            if last_sent_hash is not None and current_hash == last_sent_hash:
                logger.info("CPUService: Frame duplicado detectado. Ignorando envio para GPU.")
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return
            last_sent_hash = current_hash

            frame_bin = base64.b64decode(frame_b64)
            frame_array = np.frombuffer(frame_bin, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("CPUService: Frame corrompido ou inválido.")
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            if is_motion_detected(frame):
                logger.info("CPUService: Movimento detectado -> tentando enviar para GPU queue.")
                try:
                    channel.basic_publish(
                        exchange="",
                        routing_key=GPU_TASKS_QUEUE,
                        body=json.dumps(message).encode("utf-8")
                    )
                    logger.info("CPUService: Mensagem enviada para GPU queue com sucesso.")
                except pika.exceptions.AMQPConnectionError as e:
                    logger.error("CPUService: Conexão perdida ao enviar para GPU. Tentando reconectar.")
                    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
                    channel = connection.channel()
                    channel.queue_declare(queue=GPU_TASKS_QUEUE)
                    channel.basic_publish(
                        exchange="",
                        routing_key=GPU_TASKS_QUEUE,
                        body=json.dumps(message).encode("utf-8")
                    )
                    logger.info("CPUService: Mensagem reenviada após reconexão.")
            else:
                logger.debug("CPUService: Nenhum movimento relevante detectado.")

            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.exception("CPUService: Erro no processamento do frame:")
            channel.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(
        queue=FRAMES_QUEUE,
        on_message_callback=callback,
        auto_ack=False
    )
    logger.info("CPUService: Aguardando frames na fila...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("CPUService: Interrompido via teclado.")

    monitor.stop()
    monitor.join()
    connection.close()
    logger.info("CPUService: Encerrado com sucesso.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
