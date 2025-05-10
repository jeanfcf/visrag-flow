import os
import json
import signal
import time
import base64
import threading

import cv2
import numpy as np
import pika
import requests
from common.logger_config import setup_logger

logger = setup_logger("AlertsService")
stop_requested = False

# Estado para rastrear duração de movimento por câmera
motion_start_times = {}

# feature flag de visualização
VISUALIZE = os.getenv("VISUALIZE", "false").lower() in ("1", "true", "yes")

# Quanto tempo (em segundos) mantemos cada bounding box na tela
BOX_TTL_SEC = float(os.getenv("BOX_TTL_SEC", "2.0"))

# Estado compartilhado para visualização
frame_lock = threading.Lock()
current_frame = None
status_text = []
detection_boxes = []  # cada item: {"box": {...}, "ts": timestamp}

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
    if VISUALIZE:
        cv2.destroyAllWindows()


def _route_next(ch, msg):
    req_id = msg.get("id")
    pipeline = msg.get("pipeline", [])
    cam_id   = msg["camera_id"]
    ts       = msg["timestamp"]

    if not pipeline:
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
    global current_frame, status_text, detection_boxes
    msg = json.loads(body)
    req_id = msg.get("id")
    cam_id = msg["camera_id"]
    ts     = msg["timestamp"]
    logger.debug(f"AlertsService: [id={req_id}] Novo frame recebido [camera={cam_id} ts={ts}].")

    if VISUALIZE and "frame_b64" in msg:
        try:
            img_bytes = base64.b64decode(msg["frame_b64"])
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            with frame_lock:
                current_frame = frame
                status_text.clear()
                # remove boxes expiradas
                now = time.time()
                detection_boxes[:] = [
                    db for db in detection_boxes
                    if now - db["ts"] < BOX_TTL_SEC
                ]
        except Exception as e:
            logger.error(f"AlertsService: erro ao decodificar frame: {e}")

    _route_next(ch, msg)
    ch.basic_ack(delivery_tag=method.delivery_tag)


def cpu_response_callback(ch, method, props, body):
    global status_text
    msg = json.loads(body)
    req_id   = msg.get("id")
    cam_id   = msg.get("camera_id")
    ts       = msg.get("timestamp", time.time())
    results  = msg.get("cpu_results", {})
    logger.debug(f"AlertsService: CPU result for [id={req_id}]: {results}.")

    # atualiza motion_start_times
    if results.get("motion_detection", False):
        if cam_id not in motion_start_times:
            motion_start_times[cam_id] = ts
    else:
        motion_start_times.pop(cam_id, None)

    # calcula elapsed mesmo antes do threshold
    elapsed = None
    if cam_id in motion_start_times:
        elapsed = ts - motion_start_times[cam_id]

    if VISUALIZE:
        with frame_lock:
            status_text.append("CPU: " + ", ".join(f"{k}={v}" for k, v in results.items()))
            if elapsed is not None:
                status_text.append(f"elapsed={elapsed:.1f}s")

    pipeline = msg.get("pipeline", []).copy()
    step = pipeline.pop(0) if pipeline and "cpu" in pipeline[0]["name"].lower() else {}
    msg["pipeline"] = pipeline

    criteria = step.get("criteria", {})
    # motion_detected
    if "motion_detected" in criteria:
        if not results.get("motion_detection", False):
            logger.info(f"AlertsService: [id={req_id}] motion_detected=False → descartando.")
            return ch.basic_ack(delivery_tag=method.delivery_tag)

    # motion_duration
    if "motion_duration" in criteria:
        req_dur = criteria["motion_duration"]
        if elapsed is None or elapsed < req_dur:
            logger.info(f"AlertsService: [id={req_id}] motion_duration={elapsed:.1f}s < {req_dur}s → descartando.")
            return ch.basic_ack(delivery_tag=method.delivery_tag)
        motion_start_times.pop(cam_id, None)

    _route_next(ch, msg)
    ch.basic_ack(delivery_tag=method.delivery_tag)


def gpu_response_callback(ch, method, props, body):
    global status_text, detection_boxes
    msg = json.loads(body)
    req_id = msg.get("id")
    detections = msg.get("gpu_results", {}).get("yolov5", [])
    logger.debug(f"AlertsService: GPU result for [id={req_id}]: {len(detections)} objects.")

    if VISUALIZE:
        now = time.time()
        with frame_lock:
            status_text.append(f"GPU: {len(detections)} detectados")
            # adiciona boxes com timestamp atual
            for det in detections:
                detection_boxes.append({"box": det, "ts": now})

    pipeline = msg.get("pipeline", []).copy()
    step = pipeline.pop(0) if pipeline and "gpu" in pipeline[0]["name"].lower() else {}
    msg["pipeline"] = pipeline

    if "object_count" in step.get("criteria", {}):
        count   = len(detections)
        req_count = step["criteria"]["object_count"]
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


def display_loop():
    """Thread que exibe a janela com frames, bounding boxes e status."""
    cv2.namedWindow("Alerts", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Alerts", 640, 640)
    while not stop_requested:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
            texts = list(status_text)
            boxes = [db["box"] for db in detection_boxes]
        if frame is not None:
            for det in boxes:
                x1, y1 = int(det["xmin"]), int(det["ymin"])
                x2, y2 = int(det["xmax"]), int(det["ymax"])
                label = f'{det.get("name","")}: {det.get("confidence",0):.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            y = 20
            for t in texts:
                cv2.putText(frame, t, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                y += 20

            cv2.imshow("Alerts", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                signal_handler(None, None)
                break
        else:
            time.sleep(0.05)
    cv2.destroyAllWindows()


def main():
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    channel = conn.channel()

    channel.queue_declare(queue=ROUTER_QUEUE)
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

    if VISUALIZE:
        threading.Thread(target=display_loop, daemon=True).start()
        logger.info("AlertsService: Visualização ativada (janela OpenCV).")

    logger.info("AlertsService: aguardando mensagens…")
    channel.start_consuming()


if __name__ == "__main__":
    main()
