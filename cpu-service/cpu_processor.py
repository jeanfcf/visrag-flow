import os
import json
import base64
import cv2
import pika
import signal
import time
import numpy as np

from processes.motion_detection import process_frame as motion_process_frame
from processes.moondream2     import process_frame as moondream2_process_frame
from common.logger_config     import setup_logger
from common.monitoring        import ResourceMonitor

logger = setup_logger("CPUService")
stop_requested = False

# Filas
FRAMES_QUEUE       = os.getenv("CPU_TASKS_QUEUE", "frames_queue")
CPU_RESPONSE_QUEUE = os.getenv("CPU_RESPONSE_QUEUE", "cpu_response_queue")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    creds = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=creds))
            logger.info("CPUService: RabbitMQ connection established.")
            return conn
        except pika.exceptions.AMQPConnectionError:
            logger.warning(f"CPUService: RabbitMQ connection failed ({i+1}/{retries}).")
            time.sleep(delay)
    raise Exception("CPUService: Unable to connect to RabbitMQ.")

def signal_handler(sig, frame):
    global stop_requested
    logger.info("CPUService: Termination signal received.")
    stop_requested = True

def main():
    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    ch = conn.channel()
    ch.queue_declare(queue=FRAMES_QUEUE)
    ch.queue_declare(queue=CPU_RESPONSE_QUEUE)

    monitor = ResourceMonitor(interval=5, logger=logger)
    monitor.start()

    def callback(ch, method, props, body):
        if stop_requested:
            ch.stop_consuming()
            return

        msg = json.loads(body.decode())
        frame_b64 = msg.get("frame_b64")
        frame = None
        if frame_b64:
            binf = base64.b64decode(frame_b64)
            arr = np.frombuffer(binf, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("CPUService: invalid frame—skipping.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        results = {}
        # executa só os processadores CPU configurados
        for step in msg.get("pipeline", []):
            if step.get("name","").lower().startswith("cpu"):
                proc = step["parameters"].get("processor")
                if proc == "motion_detection":
                    thresh = step["parameters"].get("threshold", 30)
                    results["motion_detection"] = motion_process_frame(frame, threshold=thresh)
                if proc == "moondream2":
                    results["moondream2"] = moondream2_process_frame(frame)

        msg["cpu_results"] = results

        ch.basic_publish(
            exchange="",
            routing_key=CPU_RESPONSE_QUEUE,
            body=json.dumps(msg).encode()
        )
        logger.debug(f"CPUService: result sent to AlertsService: {results}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_consume(queue=FRAMES_QUEUE, on_message_callback=callback, auto_ack=False)

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
