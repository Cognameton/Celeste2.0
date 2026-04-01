import os
from typing import Any, Callable

from agent import Agent
from app_config import load_config, save_config
from config_types import AgentConfig
from model_runner import discover_local_models, normalize_path


class CelesteService:
    def __init__(self, config_path: str):
        self.config_path = os.path.abspath(config_path)
        self.cfg: AgentConfig = load_config(self.config_path)
        self.agent: Agent | None = None

    def start(self, status_cb: Callable[[str], None] | None = None) -> AgentConfig:
        self.reload(status_cb=status_cb)
        return self.cfg

    def reload(
        self,
        overrides: dict[str, Any] | None = None,
        *,
        persist: bool = False,
        status_cb: Callable[[str], None] | None = None,
    ) -> AgentConfig:
        next_cfg = load_config(self.config_path)
        if status_cb:
            status_cb("Loading Celeste configuration...")
        if overrides:
            updated = next_cfg.model_dump()
            updated.update(overrides)
            next_cfg = AgentConfig(**updated)
            next_cfg.model_path = normalize_path(next_cfg.model_path, base_dir=os.path.dirname(self.config_path))
        if persist:
            save_config(self.config_path, next_cfg)
        self.shutdown()
        self.cfg = next_cfg
        self.agent = Agent(self.cfg, status_cb=status_cb)
        return self.cfg

    def chat(self, message: str) -> tuple[str, str | None, str | None]:
        if self.agent is None:
            self.start()
        assert self.agent is not None
        return self.agent.respond(message)

    def speak(self, text: str) -> None:
        if self.agent is None:
            self.start()
        assert self.agent is not None
        if getattr(self.agent.tts, "enabled", False):
            self.agent.tts.speak(text)

    def shutdown(self) -> None:
        if self.agent is not None:
            self.agent.close()
            self.agent = None

    def available_models(self) -> list[str]:
        models = discover_local_models(limit=64)
        if self.cfg.model_path not in models:
            models.insert(0, self.cfg.model_path)
        return models

    def rag_directories(self) -> list[str]:
        return list(getattr(self.cfg, "file_rag_dirs", []) or [])

    def add_rag_directory(self, directory: str) -> tuple[AgentConfig, dict[str, Any]]:
        if self.agent is None:
            self.start()
        normalized = normalize_path(directory, base_dir=os.path.dirname(self.config_path))
        if not os.path.isdir(normalized):
            raise ValueError(f"Directory does not exist: {normalized}")
        directories = self.rag_directories()
        if normalized not in directories:
            directories.append(normalized)
        self.cfg.file_rag_dirs = directories
        assert self.agent is not None
        stats = self.agent.set_file_rag_dirs(directories)
        save_config(self.config_path, self.cfg)
        return self.cfg, stats

    def remove_rag_directory(self, directory: str) -> tuple[AgentConfig, dict[str, Any]]:
        if self.agent is None:
            self.start()
        normalized = normalize_path(directory, base_dir=os.path.dirname(self.config_path))
        directories = [path for path in self.rag_directories() if path != normalized]
        self.cfg.file_rag_dirs = directories
        assert self.agent is not None
        stats = self.agent.set_file_rag_dirs(directories)
        save_config(self.config_path, self.cfg)
        return self.cfg, stats

    def reindex_rag(self) -> tuple[AgentConfig, dict[str, Any]]:
        if self.agent is None:
            self.start()
        assert self.agent is not None
        stats = self.agent.reindex_file_rag()
        save_config(self.config_path, self.cfg)
        return self.cfg, stats

    def build_deep_rag_index(
        self,
        progress_cb: Callable[[str], None] | None = None,
    ) -> tuple[AgentConfig, dict[str, Any]]:
        if self.agent is None:
            self.start()
        assert self.agent is not None
        stats = self.agent.file_rag.build_deep_index(progress_cb=progress_cb)
        save_config(self.config_path, self.cfg)
        return self.cfg, stats
