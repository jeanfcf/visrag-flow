import logging
import sys
import os

def setup_logger(name):
    # pega nível de log de LOG_LEVEL (DEBUG/INFO/WARNING/ERROR/CRITICAL), default INFO
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt   = os.getenv("LOG_FMT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    datef = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")

    formatter = logging.Formatter(fmt=fmt, datefmt=datef)
    handler   = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
