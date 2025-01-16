#!/bin/bash

python -c "
from src.monitoring import ServiceMonitor

monitor = ServiceMonitor()
monitor.start_monitoring()
"
