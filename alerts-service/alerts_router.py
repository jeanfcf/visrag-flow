import os
import json
import signal
import time
import pika
import requests
from common.logger_config import setup_logger

logger = setup_logger("AlertsService")
stop_requested = False

# Estado para rastrear duração de movimento por câmera
motion_start_times = {}

# Filas
ROUTER_QUEUE             = os.getenv("ROUTER_QUEUE", "router_queue")
CPU_TASKS_QUEUE          = os.getenv("CPU_TASKS_QUEUE", "frames_queue")
CPU_RESPONSE_QUEUE       = os.getenv("CPU_RESPONSE_QUEUE", "cpu_response_queue")
GPU_TASKS_QUEUE          = os.getenv("GPU_TASKS_QUEUE", "gpu_tasks_queue")
GPU_RESPONSE_QUEUE       = os.getenv("GPU_RESPONSE_QUEUE", "gpu_response_queue")
RETRIEVE_QUEUE           = os.getenv("RETRIEVE_QUEUE", "retrieve_queue")
EVIDENCE_RESPONSE_QUEUE  = os.getenv("EVIDENCE_RESPONSE_QUEUE", "evidence_response_queue")
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

def _route_next(ch, msg):
    req_id = msg.get("id")
    pipeline = msg.get("pipeline", [])
    cam_id   = msg["camera_id"]
    ts       = msg["timestamp"]

    if not pipeline:
        # Fim: todos critérios satisfeitos
        req = {
            "id": req_id,
            "camera_id": cam_id,
            "timestamp": ts,
            "event_type": msg.get("event_type")
        }
        ch.basic_publish(exchange="", routing_key=RETRIEVE_QUEUE, body=json.dumps(req).encode())
        logger.info(f"AlertsService: [id={req_id}] Todos critérios atendidos — evidence requested.")
    else:
        step = pipeline[0]
        name = step["name"].lower()
        body = json.dumps(msg).encode()
        if "cpu" in name:
            ch.basic_publish(exchange="", routing_key=CPU_TASKS_QUEUE, body=body)
            logger.debug(f"AlertsService: [id={req_id}] Routed to CPU → step '{step['name']}'.")
        elif "gpu" in name:
            ch.basic_publish(exchange="", routing_key=GPU_TASKS_QUEUE, body=body)
            logger.debug(f"AlertsService: [id={req_id}] Routed to GPU → step '{step['name']}'.")
        else:
            logger.warning(f"AlertsService: [id={req_id}] Pipeline step desconhecido '{step['name']}'.")

def frame_callback(ch, method, props, body):
    msg = json.loads(body)
    req_id = msg.get("id")
    cam_id = msg["camera_id"]
    ts     = msg["timestamp"]
    logger.debug(f"AlertsService: [id={req_id}] Novo frame recebido [camera={cam_id} ts={ts}].")
    _route_next(ch, msg)
    ch.basic_ack(delivery_tag=method.delivery_tag)

def cpu_response_callback(ch, method, props, body):
    msg = json.loads(body)
    req_id   = msg.get("id")
    cam_id   = msg.get("camera_id")
    ts       = msg.get("timestamp", time.time())
    pipeline = msg.get("pipeline", [])
    step     = pipeline.pop(0) if pipeline and "cpu" in pipeline[0]["name"].lower() else {}
    msg["pipeline"] = pipeline

    criteria = step.get("criteria", {})
    # motion_detected
    if "motion_detected" in criteria:
        detected = msg.get("cpu_results", {}).get("motion_detection", False)
        logger.debug(f"AlertsService: [id={req_id}] CPU result: motion_detected={detected}.")
        if not detected:
            motion_start_times.pop(cam_id, None)
            logger.info(f"AlertsService: [id={req_id}] motion_detected=False → descartando.")
            return ch.basic_ack(delivery_tag=method.delivery_tag)
        if cam_id not in motion_start_times:
            motion_start_times[cam_id] = ts
        elapsed = ts - motion_start_times[cam_id]
    else:
        elapsed = None

    # motion_duration
    if "motion_duration" in criteria:
        req_dur = criteria["motion_duration"]
        logger.debug(f"AlertsService: [id={req_id}] elapsed={elapsed:.1f}s, required={req_dur}s.")
        if elapsed is None or elapsed < req_dur:
            logger.info(f"AlertsService: [id={req_id}] motion_duration={elapsed:.1f}s < {req_dur}s → descartando.")
            return ch.basic_ack(delivery_tag=method.delivery_tag)
        # ─── movimento validado, enviou evidence ───
        # reset para não disparar de novo até este movimento parar
        motion_start_times.pop(cam_id, None)
        # e, opcionalmente, marque "alert_sent" pra este cam_id
    _route_next(ch, msg)
    ch.basic_ack(delivery_tag=method.delivery_tag)

def gpu_response_callback(ch, method, props, body):
    msg = json.loads(body)
    req_id   = msg.get("id")
    pipeline = msg.get("pipeline", [])
    step     = pipeline.pop(0) if pipeline and "gpu" in pipeline[0]["name"].lower() else {}
    msg["pipeline"] = pipeline

    criteria = step.get("criteria", {})
    if "object_count" in criteria:
        results = msg.get("gpu_results", {}).get("yolov5", [])
        count   = len(results)
        req_count = criteria["object_count"]
        logger.debug(f"AlertsService: [id={req_id}] object_count={count}, required={req_count}.")
        if count < req_count:
            logger.info(f"AlertsService: [id={req_id}] object_count={count} < {req_count} → descartando.")
            return ch.basic_ack(delivery_tag=method.delivery_tag)

    _route_next(ch, msg)
    ch.basic_ack(delivery_tag=method.delivery_tag)

def evidence_response_callback(ch, method, props, body):
    ev = json.loads(body)
    req_id = ev.get("id")
    alert = {
        "id": req_id,
        "camera_id": ev["camera_id"],
        "event_type": ev["event_type"],
        "timestamp": ev["timestamp"],
        "evidence": ev.get("evidence")
    }
    try:
        r = requests.post(WEBHOOK_URL, json=alert, timeout=5)
        r.raise_for_status()
        logger.info(f"AlertsService: [id={req_id}] Webhook enviado com sucesso.")
    except Exception as e:
        logger.error(f"AlertsService: [id={req_id}] Falha ao enviar webhook: {e}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    connection = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = connection.channel()

    channel.queue_declare(queue=ROUTER_QUEUE, durable=True)
    channel.queue_declare(queue=CPU_TASKS_QUEUE)
    channel.queue_declare(queue=CPU_RESPONSE_QUEUE)
    channel.queue_declare(queue=GPU_TASKS_QUEUE)
    channel.queue_declare(queue=GPU_RESPONSE_QUEUE)
    channel.queue_declare(queue=RETRIEVE_QUEUE)
    channel.queue_declare(queue=EVIDENCE_RESPONSE_QUEUE)

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
