import warnings
warnings.simplefilter("ignore", FutureWarning)

import os
import json
import base64
import cv2
import numpy as np
import pika
import signal
import time

from object_detection import load_model, run_inference
from common.logger_config import setup_logger
from common.monitoring import ResourceMonitor

logger = setup_logger("GPUService")
stop_requested = False

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
GPU_TASKS_QUEUE = os.getenv("GPU_TASKS_QUEUE", "gpu_tasks_queue")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

model = load_model()

def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    credentials = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=credentials))
            logger.info("GPUService: Conexão com RabbitMQ estabelecida.")
            return conn
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"GPUService: Falha de conexão com RabbitMQ ({i+1}/{retries}). Retentando em {delay}s...")
            time.sleep(delay)
    raise Exception("GPUService: Não foi possível conectar ao RabbitMQ após várias tentativas")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("GPUService: Recebido sinal de encerramento.")
    stop_requested = True

def main():
    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = connection.channel()
    channel.queue_declare(queue=GPU_TASKS_QUEUE)

    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    def callback(ch, method, properties, body):
        try:
            if stop_requested:
                ch.stop_consuming()
                return

            logger.debug("GPUService: Iniciando processamento de frame.")
            message_str = body.decode("utf-8")
            message = json.loads(message_str)
            frame_b64 = message.get("frame_b64", "")
            frame_bin = base64.b64decode(frame_b64)
            frame_array = np.frombuffer(frame_bin, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("GPUService: Frame inválido ou corrompido.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            detections = run_inference(model, frame)
            cars = [det for det in detections if det['name'].lower() == 'car']
            if len(cars) > 0:
                logger.info(f"GPUService: Carros detectados: {len(cars)}. Detalhes: {cars}")
            else:
                logger.info("GPUService: Nenhum carro detectado neste frame.")

            logger.debug("GPUService: Finalizando processamento de frame.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.exception("GPUService: Erro no processamento:")
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=GPU_TASKS_QUEUE, on_message_callback=callback, auto_ack=False)
    logger.info("GPUService: Aguardando tarefas de GPU...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("GPUService: Interrompido via teclado.")

    monitor.stop()
    monitor.join()
    connection.close()
    logger.info("GPUService: Encerrado com sucesso.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
