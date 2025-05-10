import os
import time
import cv2
import base64
import json
import pika
import signal
import datetime
from collections import deque

from common.logger_config import setup_logger
from common.monitoring import ResourceMonitor

logger = setup_logger("IngestionService")
stop_requested = False
frame_buffer = None

# Filas
CAMERA_CONFIG_QUEUE      = os.getenv("CAMERA_CONFIG_QUEUE", "camera_config_queue")
ROUTER_QUEUE             = os.getenv("ROUTER_QUEUE", "router_queue")
RETRIEVE_QUEUE           = os.getenv("RETRIEVE_QUEUE", "retrieve_queue")
EVIDENCE_RESPONSE_QUEUE  = os.getenv("EVIDENCE_RESPONSE_QUEUE", "evidence_response_queue")

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
            logger.info("IngestionService: Connected to RabbitMQ.")
            return conn
        except pika.exceptions.AMQPConnectionError:
            logger.warning(f"IngestionService: Connection failed ({i+1}/{retries}). Retrying...")
            time.sleep(delay)
    raise Exception("IngestionService: Could not connect to RabbitMQ.")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("IngestionService: Termination signal received.")
    stop_requested = True

def get_initial_camera_config(channel):
    logger.info("IngestionService: Waiting for initial camera config...")
    while not stop_requested:
        method, header, body = channel.basic_get(queue=CAMERA_CONFIG_QUEUE, auto_ack=True)
        if body:
            cfg = json.loads(body.decode())
            logger.info("IngestionService: Initial camera config received.")
            return cfg
        time.sleep(1)
    return None

def check_for_config_update(channel):
    method, header, body = channel.basic_get(queue=CAMERA_CONFIG_QUEUE, auto_ack=True)
    if body:
        try:
            cfg = json.loads(body.decode())
            logger.info("IngestionService: New camera config received.")
            return cfg
        except Exception as e:
            logger.error(f"IngestionService: Error parsing config: {e}")
    return None

def check_retrieve_request(channel):
    method, header, body = channel.basic_get(queue=RETRIEVE_QUEUE, auto_ack=True)
    if body:
        try:
            req = json.loads(body.decode())
            logger.info("IngestionService: Evidence retrieve request received.")
            return req
        except Exception as e:
            logger.error(f"IngestionService: Error parsing retrieve request: {e}")
    return None

def connect_rtsp_stream(url, retries=10, delay=5):
    for i in range(retries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            logger.info(f"IngestionService: Connected to RTSP on attempt {i+1}.")
            return cap
        logger.warning(f"IngestionService: RTSP connect failed (attempt {i+1}).")
        time.sleep(delay)
    return None

def generate_evidence_video(buffer, output_path):
    if not buffer:
        logger.warning("IngestionService: Empty buffer—no evidence.")
        return None
    _, sample = buffer[0]
    h, w, _ = sample.shape
    fps = len(buffer) / 15.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for ts, frame in buffer:
        vw.write(frame)
    vw.release()
    logger.info(f"IngestionService: Evidence video saved to {output_path}")
    return output_path

def main():
    global stop_requested, frame_buffer

    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = conn.channel()

    # Declarações de fila (ajuste: router_queue como durable)
    channel.queue_declare(queue=CAMERA_CONFIG_QUEUE)
    channel.queue_declare(queue=ROUTER_QUEUE, durable=True)
    channel.queue_declare(queue=RETRIEVE_QUEUE)
    channel.queue_declare(queue=EVIDENCE_RESPONSE_QUEUE)

    cfg = get_initial_camera_config(channel)
    if not cfg:
        logger.error("IngestionService: No config—exiting.")
        return

    camera_cfg = cfg["camera_config"]
    CAMERA_URL    = camera_cfg["rtsp_url"]
    CAMERA_ID     = camera_cfg["camera_id"]
    frame_buffer  = deque(maxlen=1)  # será redefinido após ler FPS

    cap = connect_rtsp_stream(CAMERA_URL)
    if not cap:
        logger.error("IngestionService: Cannot open RTSP—exiting.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 5
    frame_buffer = deque(maxlen=int(fps * 15))
    logger.info(f"IngestionService: FPS={fps:.1f}, buffer_size={frame_buffer.maxlen}")

    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while not stop_requested:
        new_cfg = check_for_config_update(channel)
        if new_cfg and new_cfg["camera_config"]["camera_id"] == CAMERA_ID:
            logger.info("IngestionService: Restarting stream with new config.")
            cap.release()
            CAMERA_URL = new_cfg["camera_config"]["rtsp_url"]
            cap = connect_rtsp_stream(CAMERA_URL)
            frame_buffer.clear()

        req = check_retrieve_request(channel)
        if req:
            filename = f"evidence_{CAMERA_ID}_{int(time.time())}.mp4"
            os.makedirs("/app/evidences", exist_ok=True)
            path = os.path.join("/app/evidences", filename)
            vid = generate_evidence_video(frame_buffer, path)
            if vid:
                resp = {
                    "camera_id": CAMERA_ID,
                    "event_type": "evidence_response",
                    "timestamp": time.time(),
                    "evidence": {"video": vid}
                }
                channel.basic_publish(
                    exchange="",
                    routing_key=EVIDENCE_RESPONSE_QUEUE,
                    body=json.dumps(resp).encode()
                )

        ret, frame = cap.read()
        if not ret:
            logger.warning("IngestionService: Frame read error—reconnecting.")
            cap.release()
            time.sleep(2)
            cap = connect_rtsp_stream(CAMERA_URL)
            continue

        ts = time.time()
        frame_buffer.append((ts, frame))
        _, buf = cv2.imencode(".jpg", frame)
        msg = {
            "camera_id": CAMERA_ID,
            "timestamp": ts,
            "frame_b64": base64.b64encode(buf).decode(),
            "pipeline": cfg["analytical_config"]["pipeline"]
        }
        channel.basic_publish(
            exchange="",
            routing_key=ROUTER_QUEUE,
            body=json.dumps(msg).encode()
        )

    cap.release()
    monitor.stop()
    monitor.join()
    conn.close()
    logger.info("IngestionService: Shutdown complete.")

if __name__ == "__main__":
    main()
