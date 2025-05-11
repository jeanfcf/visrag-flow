#!/usr/bin/env python3
import os
import sys
import time
import json
import uuid
import signal
import threading
import base64
import inspect
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import pika

from common.logger_config import setup_logger
from common.monitoring    import ResourceMonitor

logger = setup_logger("IngestionService")
stop_requested = False

# ---------------------------- globals ----------------------------
frame_buffer = None
buffer_lock  = threading.Lock()
STREAM_FPS   = 15               # fallback se câmera não reportar FPS
# -----------------------------------------------------------------

# Filas e credenciais RabbitMQ
CAMERA_CONFIG_QUEUE     = os.getenv("CAMERA_CONFIG_QUEUE", "camera_config_queue")
ROUTER_QUEUE            = os.getenv("ROUTER_QUEUE", "router_queue")
RETRIEVE_QUEUE          = os.getenv("RETRIEVE_QUEUE", "retrieve_queue")
EVIDENCE_RESPONSE_QUEUE = os.getenv("EVIDENCE_RESPONSE_QUEUE", "evidence_response_queue")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

# Inicializados após leitura da 1ª configuração
CAMERA_ID  = None
CAMERA_URL = None

# -------------------------- parâmetros ---------------------------
EVIDENCE_DURATION = 15          # segundos totais (antes+depois)
PRE_EVENT_RATIO   = 0.5         # 50 % antes | 50 % depois
# ----------------------------------------------------------------


# =================================================================
# Utilidades
# =================================================================
def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    creds = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            conn = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, credentials=creds)
            )
            logger.info(f"IngestionService: Connected to RabbitMQ ({i+1}/{retries}).")
            return conn
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"IngestionService: Connection failed ({i+1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("IngestionService: Could not connect to RabbitMQ.")


def signal_handler(sig, frame):
    global stop_requested
    logger.info("IngestionService: Termination requested.")
    stop_requested = True


def get_initial_camera_config():
    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    ch   = conn.channel()
    ch.queue_declare(queue=CAMERA_CONFIG_QUEUE)
    logger.info("IngestionService: Waiting for initial camera config…")
    while not stop_requested:
        _, _, body = ch.basic_get(queue=CAMERA_CONFIG_QUEUE, auto_ack=True)
        if body:
            cfg = json.loads(body.decode())
            conn.close()
            logger.info("IngestionService: Initial camera config received.")
            return cfg
        time.sleep(1)
    conn.close()
    return None


def connect_rtsp_stream(url, retries=10, delay=5):
    """
    Conecta ao stream RTSP. Atualiza STREAM_FPS global se FPS válido (>1).
    """
    global STREAM_FPS
    for i in range(retries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                STREAM_FPS = fps
            logger.info(f"IngestionService: RTSP connected (attempt {i+1}) | FPS={STREAM_FPS:.2f}")
            return cap
        logger.warning(f"IngestionService: RTSP connect failed (attempt {i+1}).")
        time.sleep(delay)
    return None


def generate_evidence_video(frames, output_path):
    """
    Recebe lista [(ts, frame), …] em ordem cronológica e gera .mp4.
    """
    if not frames:
        logger.warning("IngestionService: Empty buffer — no evidence.")
        return None
    h, w, _ = frames[0][1].shape
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    vw      = cv2.VideoWriter(output_path, fourcc, STREAM_FPS, (w, h))
    for _, frame in frames:
        vw.write(frame)
    vw.release()
    logger.info(f"IngestionService: Evidence video saved to {output_path}")
    return output_path


def publish_evidence_response(req_id, video_path):
    try:
        conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
        ch   = conn.channel()
        ch.queue_declare(queue=EVIDENCE_RESPONSE_QUEUE)
        resp = {
            "id":          req_id,
            "camera_id":   CAMERA_ID,
            "event_type":  "evidence_response",
            "timestamp":   time.time(),
            "evidence":    {"video": video_path}
        }
        ch.basic_publish(exchange="",
                         routing_key=EVIDENCE_RESPONSE_QUEUE,
                         body=json.dumps(resp).encode())
        logger.info(f"IngestionService: Evidence response published [id={req_id}].")
    except Exception as e:
        logger.error(f"IngestionService: Failed to publish evidence [id={req_id}]: {e}")
    finally:
        try: conn.close()
        except: pass


# =================================================================
# Lógica de evidência (antes + depois)
# =================================================================
def handle_evidence_request(req_id, event_ts):
    """
    Gera vídeo de EVIDENCE_DURATION segundos: metade antes, metade depois.
    Usa STREAM_FPS para contagem exata de quadros.
    """
    pre_frames  = int(EVIDENCE_DURATION * PRE_EVENT_RATIO   * STREAM_FPS)
    post_frames = int(EVIDENCE_DURATION * (1 - PRE_EVENT_RATIO) * STREAM_FPS)

    # -------------- quadros antes do evento -----------------------
    with buffer_lock:
        pre_clip = list(frame_buffer)[-pre_frames:]   # últimos N quadros

    # -------------- aguarda quadros depois ------------------------
    while not stop_requested:
        with buffer_lock:
            post_clip_all = [f for f in frame_buffer if f[0] > event_ts]
        if len(post_clip_all) >= post_frames:
            post_clip = post_clip_all[:post_frames]
            break
        time.sleep(0.05)  # aguarda mais quadros chegarem

    full_clip = pre_clip + post_clip

    # -------------- grava vídeo & publica resposta ---------------
    os.makedirs("/app/evidences", exist_ok=True)
    filename = f"evidence_{CAMERA_ID}_{int(event_ts)}.mp4"
    path     = os.path.join("/app/evidences", filename)
    if generate_evidence_video(full_clip, path):
        publish_evidence_response(req_id, path)


# =================================================================
# Threads
# =================================================================
def capture_thread_fn(pipeline, executor):
    """
    Captura frames, mantém buffer circular e publica no ROUTER_QUEUE.
    """
    global frame_buffer, STREAM_FPS

    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    ch   = conn.channel()
    ch.queue_declare(queue=ROUTER_QUEUE)
    ch.queue_declare(queue=CAMERA_CONFIG_QUEUE)

    cap = connect_rtsp_stream(CAMERA_URL)
    if not cap:
        return

    # garante tamanho do buffer de 15 s
    frame_buffer = deque(maxlen=int(STREAM_FPS * EVIDENCE_DURATION))
    logger.info(f"IngestionService: FPS={STREAM_FPS:.2f}, buffer_size={frame_buffer.maxlen}")

    while not stop_requested:
        # Atualização de configuração (ex.: nova URL da câmera)
        try:
            _, _, body = ch.basic_get(queue=CAMERA_CONFIG_QUEUE, auto_ack=True)
            if body:
                new_cfg = json.loads(body.decode())
                if new_cfg["camera_config"]["camera_id"] == CAMERA_ID:
                    cap.release()
                    logger.info("IngestionService: Camera URL updated, reconnecting…")
                    cap = connect_rtsp_stream(new_cfg["camera_config"]["rtsp_url"])
                    with buffer_lock:
                        frame_buffer.clear()
                        frame_buffer = deque(maxlen=int(STREAM_FPS * EVIDENCE_DURATION))
        except Exception:
            pass

        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1)
            cap = connect_rtsp_stream(CAMERA_URL)
            continue

        ts = time.time()
        with buffer_lock:
            frame_buffer.append((ts, frame))

        _, buf = cv2.imencode(".jpg", frame)
        request_id = str(uuid.uuid4())
        msg = {
            "id":         request_id,
            "camera_id":  CAMERA_ID,
            "timestamp":  ts,
            "frame_b64":  base64.b64encode(buf).decode(),
            "pipeline":   pipeline
        }
        ch.basic_publish(exchange="",
                         routing_key=ROUTER_QUEUE,
                         body=json.dumps(msg).encode())
        logger.debug(f"IngestionService: Published frame [id={request_id}]")

    cap.release()
    conn.close()
    logger.info("IngestionService: Capture thread exiting.")


def retrieve_thread_fn(executor):
    """
    Consome retrieve_queue e delega geração de evidência ao executor.
    """
    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    ch   = conn.channel()
    ch.queue_declare(queue=RETRIEVE_QUEUE)
    ch.basic_qos(prefetch_count=1)

    def on_retrieve(ch, method, props, body):
        if stop_requested:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            return

        req      = json.loads(body.decode())
        req_id   = req.get("id")
        event_ts = time.time()  # marca momento do disparo
        logger.info(f"IngestionService: retrieve request [id={req_id}] received.")
        executor.submit(handle_evidence_request, req_id, event_ts)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_consume(queue=RETRIEVE_QUEUE, on_message_callback=on_retrieve)
    logger.info(f"IngestionService: Started listening on '{RETRIEVE_QUEUE}'.")
    try:
        ch.start_consuming()
    except Exception as e:
        logger.error(f"IngestionService: retrieve thread error: {e}")
    finally:
        conn.close()
        logger.info("IngestionService: Retrieve thread exiting.")


# =================================================================
# Main
# =================================================================
def main():
    global CAMERA_ID, CAMERA_URL

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor = ResourceMonitor(interval=int(os.getenv("MONITOR_INTERVAL", "5")),
                              logger=logger)
    monitor.start()

    cfg = get_initial_camera_config()
    if not cfg:
        return

    CAMERA_ID  = cfg["camera_config"]["camera_id"]
    CAMERA_URL = cfg["camera_config"]["rtsp_url"]
    pipeline   = cfg["analytical_config"]["pipeline"]

    executor = ThreadPoolExecutor(max_workers=2)

    t_cap     = threading.Thread(target=capture_thread_fn, args=(pipeline, executor), daemon=True)
    t_retriev = threading.Thread(target=retrieve_thread_fn,  args=(executor,),        daemon=True)
    t_cap.start()
    t_retriev.start()

    while not stop_requested:
        time.sleep(0.5)

    logger.info("IngestionService: Shutdown initiated.")
    t_cap.join()
    t_retriev.join()
    executor.shutdown(wait=True)
    monitor.stop()
    monitor.join()
    logger.info("IngestionService: Shutdown complete.")


if __name__ == "__main__":
    main()
