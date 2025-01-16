import ray
from ray import serve
import yaml
import torch
from typing import Dict, Any
from .model import WhisperWrapper
from .preprocessing import AudioPreprocessor
from .monitoring import ServiceMonitor

@serve.deployment(
    num_replicas=8,
    ray_actor_options={"num_gpus": 1},
)

class WhisperService:
    def __init__(self, config_path: str):
        # load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(f"cuda:{serve.get_current_replica_context().replica_id}")
        self.config['model']['device'] = str(self.device)

        self.model = WhisperWrapper(self.config['model'])
        self.preprocessor = AudioPreprocessor(self.config['model'])

        print(f"Initialized replica on {self.device}")

    async def __call__(self, request) -> Dict[str, Any]:
        try:
            audio_bytes = await request.body()

            audio_tensor = self.preprocessor.process(audio_bytes)

            result = self.model.transcribe(audio_tensor)

            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}
