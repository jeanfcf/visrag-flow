"""
Regra “object_detection” – avaliadora de contagem de objetos YOLOv5
-------------------------------------------------------------------
criteria aceitos dentro de step["rule"]["criteria"]:
    • count : int  → número mínimo de detecções do alvo
"""

from common.logger_config import setup_logger
import time

logger = setup_logger(__name__)

# ────────────────────────────────────────────────────────────────────────────
def evaluate(msg: dict, step: dict, state: dict) -> tuple[bool, dict]:
    """
    Retorna (aprovado, overlay_info)

    overlay_info é um dicionário pequeno; se vazio nada será mostrado.
    """
    try:
        req_id  = msg.get("id", "‑")
        rule    = step.get("rule") or {}
        crit    = rule.get("criteria") or {}
        target  = step.get("parameters").get("target")     # opcional p/ overlay
        now_ts     = msg.get("timestamp", time.time())
        now_ts_local = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))
        # lista de detecções vindas do YOLO
        detections = msg.get("gpu_results", {}).get("yolov5") or []
        n_detect   = len(detections)

        # ------------------ avaliação ---------------------------------------
        min_count  = crit.get("count", 1)
        aprovado   = n_detect >= min_count

        # ------------------ overlay -----------------------------------------
        overlay = {
            "object": f"{target}: {n_detect}/{min_count} - now={now_ts_local}",
        }

        logger.debug(
            f"[id={req_id}] object_detection → {n_detect=} {min_count=} "
            f"{'OK' if aprovado else 'FAIL'} - {now_ts_local}"
        )
        return aprovado, overlay

    except Exception as e:
        logger.error(f"[id={msg.get('id','‑')}] Erro na regra object_detection: {e}")
        return False, {}
