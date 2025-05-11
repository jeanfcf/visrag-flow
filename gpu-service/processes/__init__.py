"""
Registry dinâmico de processors (CPU ou GPU).

• Carrega somente os módulos listados em:
      1) env PROCESSORS_FILE  -> caminho p/ arquivo JSON
      2) env PROCESSORS_LIST  -> nomes separados por vírgula
   Se nenhum for especificado, carrega **todos**.

• Cada módulo precisa expor `load()` ou `_load()` que devolve
  a callable `process_frame(frame, **kwargs)`.
"""
import os, json
from importlib import import_module
from pathlib   import Path

# ── lê whitelists ───────────────────────────────────────────────────────────
_allowed = None

file_path = os.getenv("PROCESSORS_FILE")
if file_path and os.path.exists(file_path):
    with open(file_path, encoding="utf-8") as fp:
        _allowed = json.load(fp).get("processors", [])

if _allowed is None:
    env_lst = os.getenv("PROCESSORS_LIST")
    if env_lst:
        _allowed = [x.strip() for x in env_lst.split(",") if x.strip()]

# None → carrega tudo, lista → filtra
def _is_allowed(name: str) -> bool:
    return _allowed is None or name in _allowed

# ── registra ────────────────────────────────────────────────────────────────
PROCESSORS = {}

for p in Path(__file__).parent.glob("*.py"):
    if p.stem.startswith("__") or not _is_allowed(p.stem):
        continue
    mod = import_module(f"{__name__}.{p.stem}")

    loader = getattr(mod, "load", None) or getattr(mod, "_load", None)
    if loader is None:
        continue                                                 # sem loader
    PROCESSORS[p.stem] = loader()                                # instancia
