import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from typing import Dict, Any, Optional, List
from torch.nn.attention import SDPBackend, sdpa_kernel

class WhisperWrapper:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device(config['device'])
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=config.get('attn_implementation', 'sdpa'),
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

        self.model.generation_config.cache_implementation = "static"
        self.model.generation_config.max_new_tokens = config.get('max_new_tokens', 256)

        if config.get('use_compile', False):
            torch.set_float32_matmul_precision("high")
            self.model.forward = torch.compile(
                self.model.forward,
                mode="reduce-overhead",
                fullgraph=True
            )

        self.model.eval()

        self.chunk_length_s = config.get('chunk_length_s', 30)
        self.batch_size = config.get('batch_size', 16)

    def transcribe(
        self,
        audio_input: torch.Tensor,
        language: Optional[str] = None,
        task: Optional[str] = "transcribe",
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        try:
            generate_kwargs = {
                "task": task,
                "language": language,
                "max_new_tokens": self.model.generation_config.max_new_tokens,
                "num_beams": 1,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "return_timestamps": return_timestamps
            }

            if audio_input.shape[-1] > self.chunk_length_s * 16000:
                return self._process_long_audio(audio_input, generate_kwargs)

            with torch.inference_mode():
                with sdpa_kernel(SDPBackend.MATH):
                    input_features = self.processor(
                        audio_input,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features.to(self.device)

                    predicted_ids = self.model.generate(
                        input_features,
                        **generate_kwargs
                    )

                    transcription = self.processor.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )

                    return {
                        "status": "success",
                        "text": transcription[0],
                        "language": language
                    }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _process_long_audio(
        self,
        audio_input: torch.Tensor,
        generate_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process long audio files using chunked inference"""
        chunk_length = int(self.chunk_length_s * 16000)
        audio_chunks = []

        for i in range(0, audio_input.shape[-1], chunk_length):
            chunk = audio_input[i:i + chunk_length]
            audio_chunks.append(chunk)

        all_transcriptions = []
        for i in range(0, len(audio_chunks), self.batch_size):
            batch = audio_chunks[i:i + self.batch_size]
            batch_features = self.processor(
                batch,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).input_features.to(self.device)

            with torch.inference_mode():
                with sdpa_kernel(SDPBackend.MATH):
                    predicted_ids = self.model.generate(
                        batch_features,
                        **generate_kwargs
                    )
                    transcriptions = self.processor.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )
                    all_transcriptions.extend(transcriptions)

        full_text = " ".join(all_transcriptions)

        return {
            "status": "success",
            "text": full_text,
            "language": generate_kwargs.get("language"),
            "chunks": len(audio_chunks)
        }

