import pytest
import ray
from ray import serve
import requests
import numpy as np

@pytest.fixture
def service_url():
    return "http://localhost:8000/WhisperService"

def test_service_health(service_url):
    sample_rate = 16000
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2*np.pi*440*t)

    audio_bytes = audio.tobytes()

    response = requests.post(serivce_url, data=audio_bytes)
    assert response.status_code == 200

    result = response.json()
    assert "status" in result
    assert result["status"] in ["success", "error"]

def test_queue_metrics():
    controller = serve.get_deployment("WhisperService").get_handle()
    queue_info = ray.get(controller.get_queue_info.remote())
    assert "queued_queries" in queue_info
