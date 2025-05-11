"""
Regra “motion” – avaliadora de critérios para o processor cpu.motion
--------------------------------------------------------------------
criteria aceitos dentro de step["rule"]["criteria"]:
    ─ detected : bool   → True  ⇒ precisa haver movimento
                           False ⇒ precisa NÃO haver movimento
    ─ duration : float  → tempo (s) que o estado acima deve permanecer
"""

import time
from common.logger_config import setup_logger

logger = setup_logger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# … imports e logger …

def evaluate(msg: dict, step: dict, state: dict) -> tuple[bool, dict]:
    """
    Retorna (bool_aprovado, info_overlay).
    info_overlay é um dicionário pequeno só com o que vai ser mostrado.
    """
    cam_id = msg["camera_id"]
    req_id = msg["id"]
    rule   = step.get("rule") or {}

    crit   = rule.get("criteria", {})
    if not crit:
        return False, {}                     # nenhuma regra → reprova p/ segurança

    # ---------- estado detectado ----------
    res_motion = msg.get("cpu_results", {}).get("motion", False)
    detected   = (
        res_motion.get("motion_detected", False)
        if isinstance(res_motion, dict) else bool(res_motion)
    )
    expected   = crit.get("detected", detected)
    now_ts     = msg.get("timestamp", time.time())
    now_ts_local = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_ts))

    key = ("motion_cond_start", expected)
    cond_start = state[key].get(cam_id)

    # ---------- lógica  ----------
    if detected != expected:
        state[key].pop(cam_id, None)
        return False, {"motion": f"motion: (detectado={detected}) - {now_ts_local}"}

    dur_required = crit.get("duration")
    if dur_required is None:
        return True, {"motion": f"motion: ok - {now_ts_local}"}         # sem duração → aprovado
    
    if cond_start is None:
        state[key][cam_id] = now_ts
        return False, {"motion": f"elapsed {expected} motion: 0.0 s - {now_ts_local}"}  # recém‑iniciado

    elapsed = now_ts - cond_start
    if elapsed < dur_required:
        return False, {"motion": f"elapsed {expected} motion: {elapsed:.1f}/{dur_required}s - {now_ts_local}"}

    state[key].pop(cam_id, None)
    return True, {"motion": f"elapsed {expected} motion: {elapsed:.1f}s - {now_ts_local}"}

