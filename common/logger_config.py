# common/logger_config.py

import logging
import sys
import os

def setup_logger(name):
    # Nível e formatos vindos de env
    level     = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt       = os.getenv("LOG_FMT","%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d ▶ %(message)s")
    datef     = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
    file_path = os.getenv("LOG_FILE_PATH", "/app/logs/service.log")

    # Cria pasta do arquivo, se necessário
    log_dir = os.path.dirname(file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass

    # Formatter para todos os handlers
    formatter = logging.Formatter(fmt=fmt, datefmt=datef)

    # Console handler
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setFormatter(formatter)

    # File handler (append)
    file_h = logging.FileHandler(file_path, mode='a')
    file_h.setFormatter(formatter)

    # Logger principal
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Evita múltiplas adições
    if not logger.handlers:
        logger.addHandler(console_h)
        logger.addHandler(file_h)
    return logger
