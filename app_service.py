import os
from typing import Any, Callable

from agent import Agent
from app_config import load_config, save_config
from config_types import AgentConfig
from model_runner import discover_local_models, normalize_path
from graph_facts import record_deep_index_graph_facts


class CelesteService:
    def __init__(self, config_path: str):
        self.config_path = os.path.abspath(config_path)
        self.cfg: AgentConfig = load_config(self.config_path)
        self.agent: Agent | None = None
        self._reflection_flag_cb: Any | None = None

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
        if self._reflection_flag_cb is not None:
            self.agent.reflection_flag_cb = self._reflection_flag_cb
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

    def set_reflection_flag_cb(self, cb: Any) -> None:
        self._reflection_flag_cb = cb
        if self.agent is not None:
            self.agent.reflection_flag_cb = cb

    def get_rulebook(self) -> list[dict[str, Any]]:
        if self.agent is None:
            self.start()
        assert self.agent is not None
        return self.agent.playbook.get_rules()

    def update_rulebook_rule(self, rule_id: int, text: str) -> bool:
        if self.agent is None:
            return False
        return self.agent.playbook.update_by_id(rule_id, text)

    def delete_rulebook_rule(self, rule_id: int) -> bool:
        if self.agent is None:
            return False
        return self.agent.playbook.delete_by_id(rule_id)

    def available_models(self) -> list[str]:
        models = discover_local_models(limit=64)
        if self.cfg.model_path not in models:
            models.insert(0, self.cfg.model_path)
        return models

    def rag_directories(self) -> list[str]:
        raw = list(getattr(self.cfg, "file_rag_dirs", []) or [])
        base = os.path.dirname(self.config_path)
        return [normalize_path(p, base_dir=base) for p in raw]

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
        runner = self.agent.llm

        def _pre_semantic() -> None:
            offloaded = runner.offload_from_gpu()
            if offloaded and progress_cb:
                progress_cb("LLM temporarily moved to CPU — GPU free for semantic encoding.", 64)

        def _post_semantic() -> None:
            if getattr(runner, "_gpu_offloaded", False):
                if progress_cb:
                    progress_cb("Semantic encoding complete — reloading LLM onto GPU...", 96)
            runner.restore_to_gpu()

        stats = self.agent.file_rag.build_deep_index(
            progress_cb=progress_cb,
            pre_semantic_cb=_pre_semantic,
            post_semantic_cb=_post_semantic,
        )
        record_deep_index_graph_facts(self.agent.mem, stats)
        save_config(self.config_path, self.cfg)
        return self.cfg, stats

    def purge_engram_memory(self, seconds: int | None = None) -> tuple[AgentConfig, dict[str, Any]]:
        if self.agent is None:
            self.start()
        assert self.agent is not None
        stats = self.agent.purge_engram_memory(seconds=seconds)
        return self.cfg, stats

    def set_engram_auto_prune(self, enabled: bool) -> tuple[AgentConfig, dict[str, Any]]:
        if self.agent is None:
            self.start()
        assert self.agent is not None
        stats = self.agent.set_engram_auto_prune(enabled)
        save_config(self.config_path, self.cfg)
        return self.cfg, stats
