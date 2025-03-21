import logging
import sys

def setup_logger(name):
    """
    Cria e configura um logger básico que escreve no stdout,
    com formatação consistente para datas e níveis de log.
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Evita duplicar handlers se chamar setup_logger múltiplas vezes
    logger.propagate = False

    return logger
