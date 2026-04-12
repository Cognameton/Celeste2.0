from __future__ import annotations

import os
import platform
import socket
from typing import Any


def record_runtime_graph_facts(mem: Any, cfg: Any, *, source_ref: str = "agent-startup") -> None:
    graph = getattr(mem, "graph", None)
    if graph is None or not getattr(graph, "enabled", False):
        return

    machine_name = socket.gethostname() or "local-machine"
    platform_name = platform.system() or "Unknown"
    graph.connect(
        "machine",
        machine_name,
        "runs_on",
        "platform",
        platform_name,
        src_name=machine_name,
        dst_name=platform_name,
        evidence=f"The current machine runtime platform is {platform_name}.",
        source_ref=source_ref,
    )
    graph.connect(
        "machine",
        machine_name,
        "uses_backend",
        "backend",
        str(getattr(cfg, "backend", "unknown")),
        src_name=machine_name,
        dst_name=str(getattr(cfg, "backend", "unknown")),
        evidence=f"Celeste is configured to use the {getattr(cfg, 'backend', 'unknown')} backend.",
        source_ref=source_ref,
    )
    graph.connect(
        "machine",
        machine_name,
        "stores_data_in",
        "path",
        str(getattr(cfg, "data_dir", "")),
        src_name=machine_name,
        dst_name=str(getattr(cfg, "data_dir", "")),
        evidence=f"Celeste stores its runtime data in {getattr(cfg, 'data_dir', '')}.",
        source_ref=source_ref,
    )

    try:
        import torch

        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        graph.observe_node(
            "runtime",
            "torch",
            name="PyTorch runtime",
            metadata={
                "torch_version": str(torch.__version__),
                "cuda_compiled": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": device_count,
            },
            text=(
                f"PyTorch runtime detected: version={torch.__version__}, "
                f"cuda_available={bool(torch.cuda.is_available())}, devices={device_count}."
            ),
            source_ref=source_ref,
        )
        if torch.cuda.is_available():
            for idx in range(device_count):
                try:
                    gpu_name = torch.cuda.get_device_name(idx)
                except Exception:
                    gpu_name = f"CUDA GPU {idx}"
                graph.connect(
                    "machine",
                    machine_name,
                    "has_gpu",
                    "gpu",
                    f"cuda:{idx}",
                    src_name=machine_name,
                    dst_name=gpu_name,
                    dst_metadata={"device_index": idx, "gpu_name": gpu_name},
                    evidence=f"The runtime detected GPU {idx}: {gpu_name}.",
                    source_ref=source_ref,
                )
    except Exception:
        pass


def record_file_rag_graph_facts(mem: Any, cfg: Any, file_rag: Any, *, source_ref: str = "file-rag") -> None:
    graph = getattr(mem, "graph", None)
    if graph is None or not getattr(graph, "enabled", False):
        return
    rag_key = "file-rag"
    graph.observe_node(
        "component",
        rag_key,
        name="File RAG",
        metadata={
            "enabled": bool(getattr(cfg, "file_rag_enabled", False)),
            "embedding_device": str(getattr(file_rag, "device", "unknown")),
            "share_embedder": bool(getattr(cfg, "file_rag_share_embedder", False)),
            "multi_gpu": bool(getattr(cfg, "file_rag_multi_gpu", False)),
        },
        text=(
            "File RAG runtime configured with "
            f"device={getattr(file_rag, 'device', 'unknown')}, "
            f"share_embedder={bool(getattr(cfg, 'file_rag_share_embedder', False))}, "
            f"multi_gpu={bool(getattr(cfg, 'file_rag_multi_gpu', False))}."
        ),
        source_ref=source_ref,
    )
    for directory in list(getattr(cfg, "file_rag_dirs", []) or []):
        norm = os.path.abspath(os.path.expanduser(os.path.expandvars(str(directory))))
        graph.connect(
            "component",
            rag_key,
            "indexes_directory",
            "path",
            norm,
            src_name="File RAG",
            dst_name=norm,
            evidence=f"File RAG is configured to index {norm}.",
            source_ref=source_ref,
        )


def record_deep_index_graph_facts(mem: Any, stats: dict[str, Any], *, source_ref: str = "deep-index") -> None:
    graph = getattr(mem, "graph", None)
    if graph is None or not getattr(graph, "enabled", False):
        return
    files_indexed = int(stats.get("files_indexed", 0) or 0)
    chunks_indexed = int(stats.get("chunks_indexed", 0) or 0)
    semantic_ready = bool(stats.get("semantic_index_ready", False))
    graph.observe_node(
        "event",
        "deep-index-last-build",
        name="Deep index last build",
        metadata={
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "semantic_index_ready": semantic_ready,
            "deep_index_ready": bool(stats.get("deep_index_ready", False)),
        },
        text=(
            f"Deep index completed with {files_indexed} files and {chunks_indexed} chunks. "
            f"Semantic index ready={semantic_ready}."
        ),
        source_ref=source_ref,
    )
