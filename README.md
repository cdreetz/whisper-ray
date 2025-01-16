# Distributed Whisper Transcription Service

Deploy OpenAI's Whisper model across 8x V100 GPUs using Ray Serve for efficient, scalable audio transcription.

## Project Structure
```
whisper-service/
─ README.md
─ requirements.txt
─ src/
  ─ model.py            # Whisper model wrapper
  ─ preprocessing.py    # Audio preprocessing utilities
  ─ monitoring.py       # Custom monitoring metrics
─ scripts/
  ─ install.sh          # Setup script
  ─ start_service.sh    # Service startup
  ─ monitor.sh          # Monitoring startup
─ config/
  ─ service_config.yaml # Configuration
─ tests/
  ─ test_service.py     # Basic tests

```

## Quick Start

1. Clone and setup:
```bash
git clone <your-repo>
cd whisper-service
bash scripts/install.sh
```

2. Start the service:
```bash
bash scripts/start_service.sh
```

3. View dashboards:
- Ray dashboard: http://localhost:8265
- Serve metrics: http://localhost:8000/metrics

## Monitoring & Dashboards

### Ray Dashboard (Port 8265)
The Ray dashboard provides real-time monitoring of:
- GPU utilization
- Memory usage
- Queue status
- Active tasks
- Error logs

To access from remote:
```bash
# SSH tunnel
ssh -L 8265:localhost:8265 your-instance-ip

# Then open in browser
http://localhost:8265
```

### Serve Metrics (Port 8000)
Provides service-specific metrics:
- Request latencies
- Queue lengths
- Success/error rates
- Replica health

## Configuration

Edit `config/service_config.yaml`:
```yaml
serve:
  max_concurrent_queries: 2
  max_queued_requests: 100
  num_replicas: 8  # One per GPU
  graceful_shutdown_wait_loop_s: 2

model:
  name: "openai/whisper-large-v3"
  device: "cuda"
  batch_size: 1
  max_audio_length: 3600  # 1 hour
```

## API Usage

Send transcription requests:
```python
import requests

def transcribe(audio_path):
    with open(audio_path, "rb") as f:
        files = {"audio": f}
        response = requests.post(
            "http://localhost:8000/transcribe",
            files=files
        )
    return response.json()
```

## Monitoring Examples

1. GPU Utilization:
```bash
# View GPU stats
watch -n 1 nvidia-smi

# Or use Ray dashboard
http://localhost:8265/#/gpu
```

2. Queue Status:
```python
import ray
from ray import serve

# Get queue metrics
controller = serve.get_deployment("WhisperService").get_handle()
queue_info = ray.get(controller.get_queue_info.remote())
print(f"Queued requests: {queue_info['queued_queries']}")
```

## Common Issues

1. GPU Memory Errors:
- Check `nvidia-smi` for memory leaks
- Adjust `max_concurrent_queries` in config
- Consider reducing batch size

2. Queue Buildup:
- Monitor queue length in Ray dashboard
- Adjust `max_queued_requests`
- Consider adding more replicas if possible

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request with tests

## License

MIT License - See LICENSE file for details