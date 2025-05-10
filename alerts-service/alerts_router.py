import os
import json
import signal
import time
import pika
import requests
from common.logger_config import setup_logger

logger = setup_logger("AlertsService")
stop_requested = False

# Filas
ROUTER_QUEUE             = os.getenv("ROUTER_QUEUE", "router_queue")
CPU_TASKS_QUEUE          = os.getenv("CPU_TASKS_QUEUE", "frames_queue")
CPU_RESPONSE_QUEUE       = os.getenv("CPU_RESPONSE_QUEUE", "cpu_response_queue")
GPU_TASKS_QUEUE          = os.getenv("GPU_TASKS_QUEUE", "gpu_tasks_queue")
GPU_RESPONSE_QUEUE       = os.getenv("GPU_RESPONSE_QUEUE", "gpu_response_queue")
RETRIEVE_QUEUE           = os.getenv("RETRIEVE_QUEUE", "retrieve_queue")
EVIDENCE_RESPONSE_QUEUE  = os.getenv("EVIDENCE_RESPONSE_QUEUE", "evidence_response_queue")
ALERTS_QUEUE             = os.getenv("ALERTS_QUEUE", "alerts_queue")  # se precisar log interno
WEBHOOK_URL              = os.getenv("WEBHOOK_URL")

RABBITMQ_HOST  = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER  = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS  = os.getenv("RABBITMQ_PASS", "pass")

def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    creds = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            conn = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, credentials=creds)
            )
            logger.info(f"AlertsService: Connected to RabbitMQ ({i+1}/{retries}).")
            return conn
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"AlertsService: Connection failed ({i+1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("AlertsService: Could not connect to RabbitMQ.")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("AlertsService: Termination signal received.")
    stop_requested = True

def frame_callback(ch, method, props, body):
    msg = json.loads(body)
    pipeline = msg.get("pipeline", [])
    logger.debug(f"AlertsService: New frame for routing: {msg['camera_id']}")

    # sempre publica copy em CPU ou GPU conforme pipeline[0]
    if pipeline:
        step = pipeline[0]
        name = step["name"].lower()
        if "cpu" in name:
            ch.basic_publish(exchange="", routing_key=CPU_TASKS_QUEUE, body=body)
            logger.debug("AlertsService: Routed to CPU.")
        elif "gpu" in name:
            ch.basic_publish(exchange="", routing_key=GPU_TASKS_QUEUE, body=body)
            logger.debug("AlertsService: Routed to GPU.")
        else:
            logger.warning("AlertsService: Pipeline step desconhecido.")
    else:
        logger.warning("AlertsService: Pipeline vazio—descartando frame.")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def cpu_response_callback(ch, method, props, body):
    msg = json.loads(body)
    # remove o primeiro passo CPU
    pipeline = msg.get("pipeline", [])
    if pipeline and "cpu" in pipeline[0]["name"].lower():
        pipeline.pop(0)
    msg["pipeline"] = pipeline

    # repassa ao próximo passo
    if pipeline and "gpu" in pipeline[0]["name"].lower():
        ch.basic_publish(exchange="", routing_key=GPU_TASKS_QUEUE, body=json.dumps(msg).encode())
        logger.debug("AlertsService: CPU result → routed to GPU.")
    else:
        logger.debug("AlertsService: CPU result → sem próximo passo.")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def gpu_response_callback(ch, method, props, body):
    msg = json.loads(body)
    # remove o passo GPU
    pipeline = msg.get("pipeline", [])
    if pipeline and "gpu" in pipeline[0]["name"].lower():
        criteria = pipeline[0].get("criteria", {})
        pipeline.pop(0)
    msg["pipeline"] = pipeline

    # decide alerta final
    gpu_res = msg.get("gpu_results", {})
    count = gpu_res.get("yolov5", [])
    threshold = criteria.get("object_count", 0)
    if len(count) >= threshold:
        # pede evidência ao ingestion
        req = {
            "camera_id": msg["camera_id"],
            "timestamp": msg["timestamp"],
            "event_type": msg.get("event_type")
        }
        ch.basic_publish(exchange="", routing_key=RETRIEVE_QUEUE, body=json.dumps(req).encode())
        logger.info("AlertsService: Valid alert—evidence requested.")
    else:
        logger.info("AlertsService: GPU result abaixo do threshold—sem alerta.")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def evidence_response_callback(ch, method, props, body):
    ev = json.loads(body)
    alert = {
        "camera_id": ev["camera_id"],
        "event_type": ev["event_type"],
        "timestamp": ev["timestamp"],
        "evidence": ev["evidence"]
    }
    # dispara webhook
    try:
        r = requests.post(WEBHOOK_URL, json=alert, timeout=5)
        r.raise_for_status()
        logger.info("AlertsService: Webhook enviado com sucesso.")
    except Exception as e:
        logger.error(f"AlertsService: Falha ao enviar webhook: {e}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = connection.channel()

    # declara filas
    channel.queue_declare(queue=ROUTER_QUEUE, durable=True)
    channel.queue_declare(queue=CPU_TASKS_QUEUE)
    channel.queue_declare(queue=CPU_RESPONSE_QUEUE)
    channel.queue_declare(queue=GPU_TASKS_QUEUE)
    channel.queue_declare(queue=GPU_RESPONSE_QUEUE)
    channel.queue_declare(queue=RETRIEVE_QUEUE)
    channel.queue_declare(queue=EVIDENCE_RESPONSE_QUEUE)

    # associa callbacks
    channel.basic_consume(queue=ROUTER_QUEUE, on_message_callback=frame_callback)
    channel.basic_consume(queue=CPU_RESPONSE_QUEUE, on_message_callback=cpu_response_callback)
    channel.basic_consume(queue=GPU_RESPONSE_QUEUE, on_message_callback=gpu_response_callback)
    channel.basic_consume(queue=EVIDENCE_RESPONSE_QUEUE, on_message_callback=evidence_response_callback)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("AlertsService: waiting for messages…")
    channel.start_consuming()

if __name__ == "__main__":
    main()
