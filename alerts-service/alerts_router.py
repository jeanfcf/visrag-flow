#!/usr/bin/env python3
import os, json, signal, time, base64, threading
from collections import defaultdict
from common.monitoring     import ResourceMonitor

import cv2, numpy as np, pika, requests

from common.logger_config import setup_logger
from rules import REGISTRY as RULES_REG          # ⬅ registry de critérios

logger = setup_logger("AlertsService")
logger.info(f"Rules registry carregado → {list(RULES_REG)}")

# ---------------------------------------------------------------------------#
# Mapeia qual critério pertence a qual processor.                            #
#  ‑ chaves = id do processor (igual ao usado em step.parameters.processor)  #
#  ‑ valores = conjunto de critérios aceitos                                 #
# Se um processor não aparecer aqui, TODOS os critérios são aceitos          #
# (back‑compat).                                                             #
# ---------------------------------------------------------------------------#
RULES_CRIT = {
    "motion": {"detected", "duration"},
    "object_detection": {"count"},
    # acrescente outros processors aqui ↓
    # "clip"   : {...},
}

stop_requested = False
state: dict = defaultdict(dict)                    # ⬅ estado compartilhado

# ░░ VISUALIZAÇÃO ░░─────────────────────────────────────────────────────────
VISUALIZE    = os.getenv("VISUALIZE", "false").lower() in ("1","true","yes")
BOX_TTL_SEC  = float(os.getenv("BOX_TTL_SEC", "2.0"))

frame_lock      = threading.Lock()
current_frame   = None         # último frame decodificado
status_text     = {}           # linhas de texto a exibir
detection_boxes = []           # ⇢ [{"box":det, "ts":timestamp}, …]

# ░░ RABBITMQ / FILAS ░░─────────────────────────────────────────────────────
ROUTER_QUEUE        = os.getenv("ROUTER_QUEUE",        "router_queue")
CPU_TASKS_QUEUE     = os.getenv("CPU_TASKS_QUEUE",     "frames_queue")
CPU_RESPONSE_QUEUE  = os.getenv("CPU_RESPONSE_QUEUE",  "cpu_response_queue")
GPU_TASKS_QUEUE     = os.getenv("GPU_TASKS_QUEUE",     "gpu_tasks_queue")
GPU_RESPONSE_QUEUE  = os.getenv("GPU_RESPONSE_QUEUE",  "gpu_response_queue")
RETRIEVE_QUEUE      = os.getenv("RETRIEVE_QUEUE",      "retrieve_queue")
EVIDENCE_QUEUE      = os.getenv("EVIDENCE_RESPONSE_QUEUE", "evidence_response_queue")
WEBHOOK_URL         = os.getenv("WEBHOOK_URL")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "pass")

# ░░ HELPERS ░░──────────────────────────────────────────────────────────────
def create_connection_with_retry(host, user, pwd, retries=5, delay=2):
    creds = pika.PlainCredentials(user, pwd)
    for i in range(retries):
        try:
            return pika.BlockingConnection(pika.ConnectionParameters(host, credentials=creds))
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(f"RabbitMQ connection failed ({i+1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("AlertsService: cannot connect to RabbitMQ")

def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True
    logger.info("AlertsService: termination requested.")
    if VISUALIZE:
        cv2.destroyAllWindows()

def _route_next(ch, msg):
    """Envia msg para a próxima etapa da pipeline ou para retrieve_queue."""
    pipeline = msg.get("pipeline", [])
    logger.debug(f"[id={msg.get('id')}] Próximo passo da pipeline: {pipeline[0].get('name','name-not-found') if pipeline else 'END'}")
    if not pipeline:
        ch.basic_publish(exchange="", routing_key=RETRIEVE_QUEUE,
                         body=json.dumps({
                             "id":          msg["id"],
                             "camera_id":   msg["camera_id"],
                             "timestamp":   msg["timestamp"],
                             "event_type":  msg.get("event_type")
                         }).encode())
        logger.info(f"[id={msg['id']}] critérios satisfeitos – evidência requisitada")
        return
    nxt = pipeline[0]
    rk  = CPU_TASKS_QUEUE if "cpu" in nxt["name"].lower() else GPU_TASKS_QUEUE
    ch.basic_publish(exchange="", routing_key=rk, body=json.dumps(msg).encode())

def _evaluate(step: dict, msg: dict) -> bool:
    global detection_boxes, status_text

    """
    Executa **uma única vez** o avaliador correspondente a step["rule"]["name"].
    O avaliador próprio (em rules/*.py) olha todos os critérios necessários.
    """
    rule = step.get("rule") or {}
    rule_id   = rule.get("name")
    criteria  = rule.get("criteria", {})          # pode estar vazio

    # step sem rule => nada a validar
    if not rule_id:
        return True

    # 1) sanity‑check: critérios permitidos para esta rule
    allowed = RULES_CRIT.get(rule_id)             # None ⇒ qualquer crit ok
    if allowed is not None:
        bad = [k for k in criteria if k not in allowed]
        if bad:
            logger.warning(
                f"[id={msg.get('id')}] Rule '{rule_id}': critérios não reconhecidos → {bad}"
            )

    # 2) localiza avaliador
    fn = RULES_REG.get(rule_id)
    if fn is None:
        logger.warning(
            f"[id={msg.get('id')}] Rule '{rule_id}' sem avaliador registrado; ignorando."
        )
        return False                               # não reprova o fluxo

    # 3) executa avaliador (ele decide aprovado/reprovado)
    try:
        ok, info = fn(msg, step, state)
        if VISUALIZE and info:
            with frame_lock:
                status_text.update(info)
        if not ok:
            logger.debug(f"[id={msg.get('id')}] Rule '{rule_id}' reprovou.")
        return ok
    except Exception as e:
        logger.error(
            f"[id={msg.get('id')}] Avaliador da rule '{rule_id}' disparou exceção: {e}"
        )
        return False


# ░░ CALLBACKs ░░────────────────────────────────────────────────────────────
def frame_callback(ch, mth, _, body):
    global current_frame, status_text, detection_boxes
    msg = json.loads(body)

    # ─── visualização ───
    if VISUALIZE and "frame_b64" in msg:
        try:
            img = cv2.imdecode(
                np.frombuffer(base64.b64decode(msg["frame_b64"]), np.uint8),
                cv2.IMREAD_COLOR)
            with frame_lock:
                current_frame = img
                detection_boxes[:] = [db for db in detection_boxes
                                      if time.time()-db["ts"] < BOX_TTL_SEC]
        except Exception as e:
            logger.error(f"[id={msg['id']}]decoding frame: {e}")

    _route_next(ch, msg)
    ch.basic_ack(mth.delivery_tag)

def cpu_response_callback(ch, mth, _, body):
    global detection_boxes, status_text
    msg  = json.loads(body)
    step = msg.pop("_last_step", None) or (msg["pipeline"][0] if msg["pipeline"] else {})
    logger.debug(f"[id={msg['id']}] cpu response: {msg.get('cpu_results', {})} ")

    # ─── visualização ───
    if VISUALIZE:
        with frame_lock:
            status_text["CPU"] = f"CPU: {msg.get('cpu_results')}"

    # ─── aplica critérios ───
    if not _evaluate(step, msg):
        return ch.basic_ack(mth.delivery_tag)
    

    _route_next(ch, msg)
    ch.basic_ack(mth.delivery_tag)


def gpu_response_callback(ch, mth, _, body):
    """
    Processa respostas do passo GPU.
    Se qualquer erro ocorrer, registra‑o e devolve a mensagem para a fila.
    """
    try:
        global detection_boxes, status_text

        # ───── decodifica mensagem ──────────────────────────────────────────
        msg  = json.loads(body)
        step = msg.pop("_last_step", None) or (
               msg["pipeline"][0] if msg["pipeline"] else {})
        detections = msg.get("gpu_results", {}).get("yolov5", [])
        logger.debug(f"[id={msg['id']}] gpu response: {msg.get('gpu_results', {})}")

        # ───── VISUALIZAÇÃO ────────────────────────────────────────────────
        if VISUALIZE:
            now = time.time()
            with frame_lock:
                status_text["GPU"] = f"GPU: {len(detections)} detections"
                for det in detections:
                    detection_boxes.append({"box": det, "ts": now})

        # ───── REGRAS / CRITÉRIOS ──────────────────────────────────────────
        if not _evaluate(step, msg):                 # reprova → só ACK
            ch.basic_ack(mth.delivery_tag)
            return

        _route_next(ch, msg)

        # ───── tudo OK ─────────────────────────────────────────────────────
        ch.basic_ack(mth.delivery_tag)

    except Exception as exc:
        # log completo com stack‑trace
        logger.exception(f"gpu_response_callback falhou: {exc}")

        # devolve mensagem à fila para tentar novamente mais tarde
        try:
            ch.basic_nack(mth.delivery_tag, requeue=True)
        except Exception:                 # se já foi ack/nack ou outro erro
            pass                          # evita exceção dentro do except


def evidence_response_callback(ch, mth, _, body):
    ev  = json.loads(body)
    try:
        requests.post(WEBHOOK_URL, json=ev, timeout=5).raise_for_status()
        logger.info(f"[id={ev['id']}] webhook enviado.")
    except Exception as e:
        logger.error(f"[id={ev['id']}] webhook falhou: {e}")
    ch.basic_ack(mth.delivery_tag)

# ░░ VISUALIZAÇÃO THREAD ░░──────────────────────────────────────────────────
def display_loop():
    cv2.namedWindow("Alerts", cv2.WINDOW_NORMAL); cv2.resizeWindow("Alerts", 640, 640)
    while not stop_requested:
        with frame_lock:
            frm  = None if current_frame is None else current_frame.copy()
            texts= list(status_text.values())
            boxes= [db["box"] for db in detection_boxes]
        if frm is not None:
            for det in boxes:
                x1,y1,x2,y2 = map(int, (det["xmin"],det["ymin"],det["xmax"],det["ymax"]))
                cv2.rectangle(frm,(x1,y1),(x2,y2),(0,255,0),2)
                lbl=f'{det.get("name","")}:{det.get("confidence",0):.2f}'
                cv2.putText(frm,lbl,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            y=18
            for t in texts:
                cv2.putText(frm,t,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1); y+=18
            cv2.imshow("Alerts",frm)
            if cv2.waitKey(1)&0xFF==ord('q'):
                signal_handler(None,None); break
        else:
            time.sleep(0.05)
    cv2.destroyAllWindows()

# ░░ MAIN ░░─────────────────────────────────────────────────────────────────
def main():
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    conn = create_connection_with_retry(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS)
    ch   = conn.channel()
    for q in (ROUTER_QUEUE,CPU_TASKS_QUEUE,CPU_RESPONSE_QUEUE,
              GPU_TASKS_QUEUE,GPU_RESPONSE_QUEUE,RETRIEVE_QUEUE,EVIDENCE_QUEUE):
        ch.queue_declare(queue=q)

    monitor = ResourceMonitor(interval=int(os.getenv("MONITOR_INTERVAL", "5")),
                              logger=logger)
    monitor.start()

    ch.basic_consume(queue=ROUTER_QUEUE,        on_message_callback=frame_callback)
    ch.basic_consume(queue=CPU_RESPONSE_QUEUE,  on_message_callback=cpu_response_callback)
    ch.basic_consume(queue=GPU_RESPONSE_QUEUE,  on_message_callback=gpu_response_callback)
    ch.basic_consume(queue=EVIDENCE_QUEUE,      on_message_callback=evidence_response_callback)

    if VISUALIZE:
        threading.Thread(target=display_loop, daemon=True).start()
        logger.info("Visualização ativada")

    logger.info("AlertsService pronto; aguardando mensagens…")
    ch.start_consuming()

if __name__ == "__main__":
    main()
