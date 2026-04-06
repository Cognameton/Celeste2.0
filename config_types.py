# /home/head-node/ai-lab/celeste/config_types.py
from typing import Any

from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    # Backend & model
    backend: str                    # "llama_cpp" | "llama_server" | "transformers"
    model_path: str
    dtype: str = "float16"          # (used by transformers)
    device: str = "cuda"            # (used by transformers)
    max_new_tokens: int = 512

    # llama.cpp settings
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    split_mode: int = 1
    main_gpu: int = 0
    tensor_split: str | None = None
    n_batch: int = 512
    n_ubatch: int = 512
    flash_attn: bool = False
    offload_kqv: bool = True
    llama_verbose: bool = False
    llama_server_executable: str | None = None

    # Memory / persistence
    embedding_model: str = "intfloat/e5-small-v2"   # local path or name
    use_chroma: bool = True
    persist_dir: str = "./persist"
    data_dir: str = "./data"
    file_rag_enabled: bool = True
    file_rag_dirs: list[str] = Field(default_factory=list)
    file_rag_top_k: int = 4
    file_rag_use_embeddings: bool = False
    file_rag_embedding_device: str = "auto"
    file_rag_share_embedder: bool = False

    # Behavior
    system_preamble: str = "You are an offline agent."
    top_k: int = 6

    # Text to speech
    tts_enabled: bool = False
    tts_backend: str = "pyttsx3"
    tts_voice: str | None = None
    tts_rate: int | None = None
    tts_volume: float | None = None
    tts_piper_model: str | None = None
    tts_piper_config: str | None = None
    tts_piper_executable: str = "piper"
    tts_piper_speaker: str | None = None
    tts_output_dir: str | None = None
    tts_presets: dict[str, str] = Field(default_factory=dict)
    tts_default_preset: str | None = None

    # Optional nested config blocks from config.yaml
    reflection: dict[str, Any] = Field(default_factory=dict)
    behavior_flags: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
