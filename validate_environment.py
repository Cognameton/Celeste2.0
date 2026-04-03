#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Iterable

from app_config import load_config
from config_types import AgentConfig
from model_runner import resolve_llama_server_executable, resolve_model_path


@dataclass
class ValidationCheck:
    level: str
    subject: str
    message: str


def _is_path_like(value: str) -> bool:
    candidate = (value or "").strip()
    return (
        candidate.startswith("/")
        or candidate.startswith("~")
        or candidate.startswith(".")
        or (len(candidate) >= 2 and candidate[1] == ":")
        or os.path.exists(os.path.expandvars(os.path.expanduser(candidate)))
    )


def _check_writable_dir(path: str, subject: str) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    normalized = os.path.abspath(os.path.expandvars(os.path.expanduser(path or "")))
    if not normalized:
        checks.append(ValidationCheck("ERROR", subject, "Path is empty."))
        return checks
    try:
        os.makedirs(normalized, exist_ok=True)
    except OSError as exc:
        checks.append(ValidationCheck("ERROR", subject, f"Cannot create directory: {normalized} ({exc})"))
        return checks
    if os.path.isdir(normalized) and os.access(normalized, os.W_OK):
        checks.append(ValidationCheck("OK", subject, f"Writable directory: {normalized}"))
    else:
        checks.append(ValidationCheck("ERROR", subject, f"Directory is not writable: {normalized}"))
    return checks


def validate_config(cfg: AgentConfig, *, config_path: str | None = None) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    try:
        model_path = resolve_model_path(cfg.model_path, base_dir=config_dir)
        checks.append(ValidationCheck("OK", "model_path", f"Model found: {model_path}"))
    except Exception as exc:
        checks.append(ValidationCheck("ERROR", "model_path", f"Model not found: {cfg.model_path} ({exc})"))

    backend = (cfg.backend or "").lower().strip()
    if backend == "llama_server":
        server_bin = resolve_llama_server_executable(cfg)
        if server_bin:
            checks.append(ValidationCheck("OK", "llama_server_executable", f"llama-server found: {server_bin}"))
        else:
            checks.append(
                ValidationCheck(
                    "ERROR",
                    "llama_server_executable",
                    "llama-server binary not found in config path, repo vendor build, or PATH.",
                )
            )
    elif backend == "llama_cpp":
        checks.append(ValidationCheck("OK", "backend", "Configured for llama_cpp Python backend."))
    elif backend == "transformers":
        checks.append(ValidationCheck("OK", "backend", "Configured for transformers backend."))
    else:
        checks.append(ValidationCheck("ERROR", "backend", f"Unknown backend: {cfg.backend}"))

    embedding_model = (cfg.embedding_model or "").strip()
    if _is_path_like(embedding_model):
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(embedding_model)))
        if os.path.exists(expanded):
            checks.append(ValidationCheck("OK", "embedding_model", f"Embedding model path exists: {expanded}"))
        else:
            checks.append(
                ValidationCheck("ERROR", "embedding_model", f"Embedding model path not found: {expanded}")
            )
    elif embedding_model:
        checks.append(
            ValidationCheck(
                "WARN",
                "embedding_model",
                f"Embedding model '{embedding_model}' is not a local path; first load may require internet access.",
            )
        )
    else:
        checks.append(ValidationCheck("ERROR", "embedding_model", "Embedding model is not configured."))

    checks.extend(_check_writable_dir(cfg.data_dir, "data_dir"))
    checks.extend(_check_writable_dir(cfg.persist_dir, "persist_dir"))

    rag_dirs = list(cfg.file_rag_dirs or [])
    if rag_dirs:
        for idx, directory in enumerate(rag_dirs, start=1):
            expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(directory or "")))
            if os.path.isdir(expanded):
                checks.append(ValidationCheck("OK", f"file_rag_dirs[{idx}]", f"Document directory found: {expanded}"))
            else:
                checks.append(
                    ValidationCheck(
                        "WARN",
                        f"file_rag_dirs[{idx}]",
                        f"Document directory not found yet: {expanded}",
                    )
                )
    else:
        checks.append(
            ValidationCheck(
                "WARN",
                "file_rag_dirs",
                "No document directories configured. Chat works, but file RAG will have no library to index.",
            )
        )

    if cfg.tts_enabled and (cfg.tts_backend or "").lower().strip() == "piper":
        piper_exe = (cfg.tts_piper_executable or "").strip()
        resolved = shutil.which(piper_exe) if piper_exe else None
        if not resolved and piper_exe:
            expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(piper_exe)))
            resolved = expanded if os.path.isfile(expanded) else None
        if resolved:
            checks.append(ValidationCheck("OK", "tts_piper_executable", f"Piper executable found: {resolved}"))
        else:
            checks.append(
                ValidationCheck(
                    "ERROR",
                    "tts_piper_executable",
                    f"Piper executable not found: {piper_exe or '(empty)'}",
                )
            )

        if cfg.tts_piper_model and os.path.isfile(os.path.expanduser(os.path.expandvars(cfg.tts_piper_model))):
            checks.append(ValidationCheck("OK", "tts_piper_model", "Piper voice model found."))
        else:
            checks.append(ValidationCheck("ERROR", "tts_piper_model", "Piper voice model path is missing/invalid."))

        if cfg.tts_piper_config and os.path.isfile(os.path.expanduser(os.path.expandvars(cfg.tts_piper_config))):
            checks.append(ValidationCheck("OK", "tts_piper_config", "Piper voice config found."))
        elif cfg.tts_piper_config:
            checks.append(ValidationCheck("ERROR", "tts_piper_config", "Piper voice config path is invalid."))

    return checks


def validate_config_file(config_path: str) -> list[ValidationCheck]:
    cfg = load_config(config_path)
    return validate_config(cfg, config_path=config_path)


def format_checks(checks: Iterable[ValidationCheck]) -> str:
    return "\n".join(f"[{check.level}] {check.subject}: {check.message}" for check in checks)


def has_errors(checks: Iterable[ValidationCheck]) -> bool:
    return any(check.level == "ERROR" for check in checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Celeste config.yaml before launch.")
    parser.add_argument("--config", default="config.yaml", help="Path to the Celeste config file.")
    args = parser.parse_args()

    checks = validate_config_file(args.config)
    print(format_checks(checks))
    return 1 if has_errors(checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
