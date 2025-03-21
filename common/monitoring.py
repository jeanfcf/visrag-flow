import psutil
import time
import threading
import logging

class ResourceMonitor(threading.Thread):
    """
    Thread que periodicamente registra o uso de CPU e memória.
    Pode ser estendida para enviar métricas ao Prometheus, CloudWatch, etc.
    """
    def __init__(self, interval=5, logger=None):
        super().__init__()
        self.interval = interval
        self.stop_flag = False
        self.logger = logger or logging.getLogger("ResourceMonitor")

    def run(self):
        while not self.stop_flag:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_info = psutil.virtual_memory()
            self.logger.info(f"[MONITOR] CPU: {cpu_percent}%, MEM: {mem_info.percent}%")
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag = True
