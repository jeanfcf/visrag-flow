# common/monitoring.py

import os
import time
import threading
import logging
import psutil
import importlib.util

# tenta importar GPUtil para métricas de GPU
spec = importlib.util.find_spec("GPUtil")
if spec:
    import GPUtil
else:
    GPUtil = None

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=None, logger=None):
        super().__init__(daemon=True)
        # pega intervalo de MONITOR_INTERVAL (s) da env, default 5
        self.interval = interval or int(os.getenv("MONITOR_INTERVAL", 5))
        self.logger = logger or logging.getLogger("ResourceMonitor")
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            # CPU e memória
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent

            msg = f"[MONITOR] CPU: {cpu:.1f}%, MEM: {mem:.1f}%"

            # adiciona métricas de GPU se GPUtil estiver instalado
            if GPUtil:
                gpus = GPUtil.getGPUs()
                gpu_msgs = []
                for gpu in gpus:
                    load = gpu.load * 100  # de 0.0–1.0 para %
                    used = gpu.memoryUsed
                    total = gpu.memoryTotal
                    gpu_msgs.append(f"GPU{{id={gpu.id}}} load: {load:.1f}%, VRAM: {used}/{total}MB")
                if gpu_msgs:
                    msg += ", " + ", ".join(gpu_msgs)

            self.logger.info(msg)
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag = True
