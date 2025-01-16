import ray
from ray import serve
import time
from typing import Dict
import psutil
import GPUtil

class ServiceMonitor:
    def __init__(self, interval: int = 10):
        self.interval = interval
        self.metrics = {}

    def update_metrics(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            self.metrics[f"gpu_{gpu.id}_util"] = gpu.load * 100
            self.metrics[f"gpu_{gpu.id}_memory"] = gpu.memoryUsed

        controller = serve.get_deployment("WhisperService").get_handle()
        queue_info = ray.get(controller.get_queue_info.remote())
        self.metrics["queue_length"] = queue_info["queued_queries"]

        self.metrics["cpu_util"] = psutil.cpu_percent()
        self.metrics["memory_util"] = psutil.virtual_memory().percent

        return self.metrics

    def start_monitoring(self):
        while True:
            self.update_metrics()
            time.sleep(self.interval)

