#!/bin/bash

ray start --head

python -m ray.dashboard.dashboard --host 0.0.0.0 --port 8265 &

python -c "
import ray
from ray import serve
from src.service import WhisperService

serve.start(http_options={'host': '0.0.0.0', 'port': 8000})
WhisperService.deploy('config/service_config.yaml')

while True:
  ray.get([])
"
