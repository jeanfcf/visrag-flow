#!/usr/bin/env python3
import os, json, base64, time, signal
import cv2, numpy as np, pika
from common.logger_config  import setup_logger
from common.monitoring     import ResourceMonitor
from processes             import PROCESSORS    # ← registry dinâmico

logger = setup_logger("CPUService")
stop_requested = False

# ─── filas & broker ─────────────────────────────────────────────────────────
FRAMES_QUEUE       = os.getenv("CPU_TASKS_QUEUE", "frames_queue")
CPU_RESPONSE_QUEUE = os.getenv("CPU_RESPONSE_QUEUE", "cpu_response_queue")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

# ─── conexão resiliente ─────────────────────────────────────────────────────
def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    creds = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            return pika.BlockingConnection(
                pika.ConnectionParameters(host=host, credentials=creds)
            )
        except pika.exceptions.AMQPConnectionError:
            logger.warning(f"CPUService: connection failed ({i+1}/{retries}).")
            time.sleep(delay)
    raise RuntimeError("CPUService: unable to connect to RabbitMQ.")

# ─── sinalização ────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True
    logger.info("CPUService: termination requested.")

# ─── consumer callback ──────────────────────────────────────────────────────
def make_callback():
    def callback(ch, method, props, body):
        if stop_requested:
            ch.basic_nack(method.delivery_tag, requeue=True)
            return

        msg = json.loads(body)
        req_id = msg.get("id")
        step   = msg["pipeline"].pop(0)          # remove passo corrente
        proc_id = step["parameters"].get("processor")
        logger.debug(f"[id={req_id}] CPU '{proc_id}' → {step.get('name')}")
        fn = PROCESSORS.get(proc_id)
        if fn is None:
            logger.error(f"[id={req_id}] processor '{proc_id}' não encontrado.")
            ch.basic_ack(method.delivery_tag); return

        # decodifica frame JPEG → BGR
        arr   = np.frombuffer(base64.b64decode(msg["frame_b64"]), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        params = {k: v for k, v in step["parameters"].items() if k != "processor"}
        result = fn(frame, **params)  # ← execução atômica
        if isinstance(result, (np.bool_, np.generic)):
                    result = result.item()
        msg["cpu_results"] = {proc_id: result}
        msg["_last_step"]  = step

        ch.basic_publish(exchange="", routing_key=CPU_RESPONSE_QUEUE,
                         body=json.dumps(msg).encode())
        logger.debug(f"[id={req_id}] CPU '{proc_id}' → {result}")
        ch.basic_ack(method.delivery_tag)
    return callback

# ─── main ───────────────────────────────────────────────────────────────────
def main():
    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    ch   = conn.channel()
    ch.queue_declare(queue=FRAMES_QUEUE)
    ch.queue_declare(queue=CPU_RESPONSE_QUEUE)

    monitor = ResourceMonitor(interval=int(os.getenv("MONITOR_INTERVAL", "5")),
                              logger=logger)
    monitor.start()

    ch.basic_consume(queue=FRAMES_QUEUE,
                     on_message_callback=make_callback(),
                     auto_ack=False)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info(f"CPUService: processors disponíveis → {list(PROCESSORS)}")
    logger.info("CPUService: waiting for tasks…")
    ch.start_consuming()

if __name__ == "__main__":
    main()
