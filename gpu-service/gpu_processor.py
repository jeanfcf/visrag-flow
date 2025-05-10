import os
import json
import base64
import cv2
import numpy as np
import pika
import signal
import time

from common.logger_config import setup_logger
from common.monitoring    import ResourceMonitor
from processes.yolov5     import process_frame as yolov5_process_frame

logger = setup_logger("GPUService")
stop_requested = False

# Filas
GPU_TASKS_QUEUE      = os.getenv("GPU_TASKS_QUEUE", "gpu_tasks_queue")
GPU_RESPONSE_QUEUE   = os.getenv("GPU_RESPONSE_QUEUE", "gpu_response_queue")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    creds = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=creds))
            logger.info("GPUService: RabbitMQ connection established.")
            return conn
        except pika.exceptions.AMQPConnectionError:
            logger.warning(f"GPUService: Connection failed ({i+1}/{retries}). Retrying…")
            time.sleep(delay)
    raise Exception("GPUService: Unable to connect to RabbitMQ.")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("GPUService: Termination signal received.")
    stop_requested = True

def main():
    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    ch = conn.channel()
    ch.queue_declare(queue=GPU_TASKS_QUEUE)
    ch.queue_declare(queue=GPU_RESPONSE_QUEUE)

    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    def callback(ch, method, props, body):
        if stop_requested:
            ch.stop_consuming()
            return

        msg = json.loads(body.decode())
        req_id = msg.get("id")
        cam_id = msg.get("camera_id")
        ts     = msg.get("timestamp")
        logger.debug(f"GPUService: [id={req_id}] Received GPU task for camera={cam_id} ts={ts}")

        frame_b64 = msg.get("frame_b64", "")
        b = base64.b64decode(frame_b64)
        arr = np.frombuffer(b, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning(f"GPUService: [id={req_id}] invalid frame—skipping.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        results = {}
        pipeline = msg.get("pipeline", []).copy()
        step = pipeline.pop(0)
        
        if step["name"].lower().startswith("gpu"):
            proc   = step["parameters"].get("processor")
            params = step["parameters"]
            logger.debug(f"GPUService: [id={req_id}] Executing '{proc}' with params {params}")
            if proc == "yolov5":
                detections = yolov5_process_frame(frame)
                target = params.get("target", "").lower()
                thresh = params.get("threshold", 0.5)
                filtered = [
                    d for d in detections
                    if d.get("name","").lower() == target and d.get("confidence",0) >= thresh
                ]
                results["yolov5"] = filtered

        msg["gpu_results"] = results

        ch.basic_publish(
            exchange="",
            routing_key=GPU_RESPONSE_QUEUE,
            body=json.dumps(msg).encode()
        )
        logger.info(f"GPUService: [id={req_id}] Published GPU response: {results}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_consume(queue=GPU_TASKS_QUEUE, on_message_callback=callback, auto_ack=False)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        pass

    monitor.stop()
    monitor.join()
    conn.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
