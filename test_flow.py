#!/usr/bin/env python3
import pika, json, base64, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# === Configurações RabbitMQ ===
RABBIT_HOST = 'localhost'
RABBIT_USER = 'user'
RABBIT_PASS = 'pass'

CAMERA_CONFIG_QUEUE = 'camera_config_queue'
ROUTER_QUEUE        = 'router_queue'

# === Exemplo de config.json ===
CONFIG = {
    "camera_config": {
        "camera_id": "cam_001",
        "rtsp_url": "rtsp://192.168.15.12:8561/stream0",
        "active_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "active_hours": {"start": "08:00", "end": "18:00"}
    },
    "analytical_config": {
        "event_type": "people_counting",
        "pipeline": [
            {
                "name": "cpu-step",
                "version": "1.0",
                "parameters": {
                    "processor": "motion_detection",
                    "threshold": 10
                },
                "criteria": {
                    "motion_detected": True,
                    "motion_duration": 5
                }
            },
            {
                "name": "gpu-step",
                "version": "1.0",
                "parameters": {
                    "processor": "yolov5",
                    "target": "person",
                    "threshold": 0.5
                },
                "criteria": {
                    "object_detected": True,
                    "object_count": 1
                }
            }
        ]
    }
}

# === Webhook HTTP server ===
class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode()
        print("\n*** Webhook received payload: ***")
        print(json.dumps(json.loads(body), indent=2))
        self.send_response(200)
        self.end_headers()

def start_webhook_server():
    server = HTTPServer(('0.0.0.0', 8000), WebhookHandler)
    print("▶️ Webhook HTTP server running on port 8000")
    server.serve_forever()

# === Publicadores RabbitMQ ===
def get_connection():
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    return pika.BlockingConnection(pika.ConnectionParameters(host=RABBIT_HOST, credentials=creds))

def publish_config():
    conn = get_connection()
    ch = conn.channel()
    ch.queue_declare(queue=CAMERA_CONFIG_QUEUE)
    ch.basic_publish(
        exchange='',
        routing_key=CAMERA_CONFIG_QUEUE,
        body=json.dumps(CONFIG).encode()
    )
    print("✅ Published camera config")
    conn.close()

def publish_test_frame():
    # leia um JPEG de teste na mesma pasta: test.jpg
    with open("test.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    msg = {
        "camera_id": CONFIG["camera_config"]["camera_id"],
        "timestamp": time.time(),
        "frame_b64": img_b64,
        "pipeline": CONFIG["analytical_config"]["pipeline"]
    }

    conn = get_connection()
    ch = conn.channel()
    ch.queue_declare(queue=ROUTER_QUEUE, durable=True)
    ch.basic_publish(
        exchange='',
        routing_key=ROUTER_QUEUE,
        body=json.dumps(msg).encode()
    )
    print("✅ Published test frame to router_queue")
    conn.close()

if __name__ == "__main__":
    # 1) sobe servidor de webhook em background
    threading.Thread(target=start_webhook_server, daemon=True).start()
    time.sleep(1)

    # 2) publica config.json
    publish_config()

    # 3) espera ingestion pegar config
    time.sleep(2)

    # 4) publica frame de teste
    # publish_test_frame()

    print("⏳ Waiting for webhook callback... (CTRL+C to exit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("✋ Exiting.")
