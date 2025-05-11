"""
Registry de avaliadores de critérios.

Idêntico ao dos processors, mas usa env/arquivo *CRITERIA_FILE*.
"""
import os, json
from importlib import import_module
from pathlib   import Path

_allowed = None
file_path = os.getenv("RULES_FILE")
if file_path and os.path.exists(file_path):
    with open(file_path, encoding="utf-8") as fp:
        _allowed = json.load(fp).get("rules", [])

if _allowed is None:
    env_lst = os.getenv("RULES_LIST")
    if env_lst:
        _allowed = [x.strip() for x in env_lst.split(",") if x.strip()]

def _is_allowed(name: str) -> bool:
    return _allowed is None or name in _allowed

REGISTRY = {}
for p in Path(__file__).parent.glob("*.py"):
    if p.stem.startswith("__") or not _is_allowed(p.stem):
        continue
    mod = import_module(f"{__name__}.{p.stem}")
    if hasattr(mod, "evaluate"):
        REGISTRY[p.stem] = mod.evaluate
