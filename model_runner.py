# model_runner.py — llama.cpp runtime with adaptive kwargs filtering
from typing import Optional, List, Any, Dict, Iterator, Callable
from app_paths import runtime_root
from config_types import AgentConfig
import atexit
import inspect
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

PROJECT_ROOT = runtime_root()
HOME_DIR = os.path.expanduser("~")
_LLAMA_CPP_LIB_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "vendor", "llama.cpp", "build", "bin", "libllama.so"),
    os.path.join(PROJECT_ROOT, "vendor", "llama.cpp", "build", "src", "libllama.so"),
    "/home/head-node/Dev/ai-lab/llama.cpp/build/bin/libllama.so",
    "/home/head-node/ai-lab/llama.cpp/build/bin/libllama.so",
]
_LLAMA_CPP_BIN_DIR_CANDIDATES = [
    os.path.dirname(path) for path in _LLAMA_CPP_LIB_CANDIDATES
]
_LLAMA_SERVER_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "vendor", "llama.cpp", "build", "bin", "llama-server"),
    os.path.join(PROJECT_ROOT, "vendor", "llama.cpp", "build", "bin", "Release", "llama-server.exe"),
    os.path.join(PROJECT_ROOT, "vendor", "llama.cpp", "build", "bin", "llama-server.exe"),
    "/home/head-node/Dev/ai-lab/llama.cpp/build/bin/llama-server",
    "/home/head-node/ai-lab/llama.cpp/build/bin/llama-server",
]
_MODEL_SEARCH_ROOTS = [
    "/media/head-node/ollama-models/models",
    os.path.join(HOME_DIR, "Celeste", "models"),
    os.path.join(HOME_DIR, "Dev"),
    os.path.join(HOME_DIR, "Downloads"),
    os.path.join(HOME_DIR, "models"),
    PROJECT_ROOT,
]
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}


def normalize_path(path: str, base_dir: str | None = None) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path.strip()))
    if not os.path.isabs(expanded):
        expanded = os.path.join(base_dir or PROJECT_ROOT, expanded)
    return os.path.abspath(expanded)


def _prepare_llama_cpp_env(env: Dict[str, str] | None = None) -> Dict[str, str]:
    target = dict(os.environ if env is None else env)
    for lib_path in _LLAMA_CPP_LIB_CANDIDATES:
        if os.path.isfile(lib_path):
            target.setdefault("LLAMA_CPP_LIB_PATH", os.path.dirname(lib_path))
            target.setdefault("LLAMA_CPP_LIB", lib_path)
            break
    target.setdefault("LLAMA_CUBLAS", "1")
    for lib_dir in _LLAMA_CPP_BIN_DIR_CANDIDATES:
        if os.path.isdir(lib_dir):
            current = target.get("LD_LIBRARY_PATH", "")
            if lib_dir not in (current.split(":") if current else []):
                target["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
            break
    return target


def _walk_model_files(root: str):
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for filename in filenames:
            if not filename.endswith(".gguf"):
                continue
            if filename.startswith("ggml-vocab"):
                continue
            yield os.path.join(dirpath, filename)


def discover_local_models(limit: int = 8) -> List[str]:
    found: List[str] = []
    seen = set()
    for root in _MODEL_SEARCH_ROOTS:
        for path in _walk_model_files(root):
            if path in seen:
                continue
            seen.add(path)
            found.append(path)
            if len(found) >= limit:
                return found
    return found


def resolve_model_path(model_path: str, base_dir: str | None = None) -> str:
    normalized = normalize_path(model_path, base_dir=base_dir)
    if os.path.isfile(normalized):
        return normalized

    requested_name = os.path.basename(normalized)
    matches: List[str] = []
    for root in _MODEL_SEARCH_ROOTS:
        for candidate in _walk_model_files(root):
            if os.path.basename(candidate) == requested_name:
                matches.append(candidate)
                if len(matches) > 1:
                    break
        if len(matches) > 1:
            break

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(normalized)


def discover_llama_server_candidates(cfg: AgentConfig | None = None) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()

    if cfg is not None:
        configured = (getattr(cfg, "llama_server_executable", None) or "").strip()
        if configured:
            candidates.append(normalize_path(configured, base_dir=PROJECT_ROOT))

    for path in _LLAMA_SERVER_CANDIDATES:
        candidates.append(path)

    which_path = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if which_path:
        candidates.append(which_path)

    out: List[str] = []
    for path in candidates:
        normalized = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def resolve_llama_server_executable(cfg: AgentConfig | None = None) -> str | None:
    return next((path for path in discover_llama_server_candidates(cfg) if os.path.isfile(path)), None)

class LLMRunner:
    def __init__(self, cfg: AgentConfig, status_cb: Callable[[str], None] | None = None):
        self.cfg = cfg
        self._status_cb = status_cb
        self._last_status: str | None = None
        self.backend: str = ""
        self.server_proc: subprocess.Popen[str] | None = None
        self.server_url: str | None = None
        self.server_log_path: str | None = None
        self.server_log_handle = None
        backend = (cfg.backend or "").lower()
        if backend == "llama_cpp":
            try:
                self._init_llama_cpp()
            except (AttributeError, ImportError, OSError, RuntimeError) as exc:
                if not self._has_llama_server():
                    raise
                self._init_llama_server(exc)
            except Exception:
                raise
        elif backend == "llama_server":
            self._init_llama_server()
        elif backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _emit_status(self, message: str) -> None:
        if self._status_cb and message != self._last_status:
            self._last_status = message
            self._status_cb(message)

    # ---------------- llama.cpp backend ----------------
    def _init_llama_cpp(self):
        os.environ.update(_prepare_llama_cpp_env())
        from llama_cpp import Llama
        model_path = resolve_model_path(self.cfg.model_path, base_dir=PROJECT_ROOT)
        self.cfg.model_path = model_path
        self._emit_status(f"Loading model with llama.cpp: {os.path.basename(model_path)}")

        kwargs: Dict[str, Any] = dict(
            model_path=model_path,
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            n_gpu_layers=self.cfg.n_gpu_layers,
            split_mode=self.cfg.split_mode,
            n_batch=self.cfg.n_batch,
            n_ubatch=self.cfg.n_ubatch,
            flash_attn=self.cfg.flash_attn,
            offload_kqv=self.cfg.offload_kqv,
            logits_all=False,
            verbose=self.cfg.llama_verbose,
        )

        # Optional extras if present in config.yaml
        if isinstance(self.cfg.main_gpu, int):
            kwargs["main_gpu"] = self.cfg.main_gpu

        tensor_split = self.cfg.tensor_split
        if isinstance(tensor_split, str) and tensor_split.strip():
            try:
                kwargs["tensor_split"] = [float(x) for x in tensor_split.split(",")]
            except Exception:
                pass

        rope_scaling_type = getattr(self.cfg, "rope_scaling_type", None)
        if isinstance(rope_scaling_type, str) and rope_scaling_type:
            kwargs["rope_scaling_type"] = rope_scaling_type

        rope_freq_base = getattr(self.cfg, "rope_freq_base", None)
        if isinstance(rope_freq_base, (int, float)):
            kwargs["rope_freq_base"] = float(rope_freq_base)

        rope_freq_scale = getattr(self.cfg, "rope_freq_scale", None)
        if isinstance(rope_freq_scale, (int, float)):
            kwargs["rope_freq_scale"] = float(rope_freq_scale)

        seed = getattr(self.cfg, "seed", None)
        if isinstance(seed, int):
            kwargs["seed"] = seed

        self.llm = Llama(**kwargs)
        self.backend = "llama_cpp"

    def _has_llama_server(self) -> bool:
        return resolve_llama_server_executable(self.cfg) is not None

    def _split_mode_name(self) -> str:
        split_mode_map = {0: "none", 1: "layer", 2: "row"}
        return split_mode_map.get(self.cfg.split_mode, "layer")

    def _server_env(self) -> Dict[str, str]:
        return _prepare_llama_cpp_env(os.environ.copy())

    def _pick_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _wait_for_server(self, timeout_s: float = 300.0) -> None:
        if not self.server_url:
            raise RuntimeError("llama-server URL not initialized")
        deadline = time.time() + timeout_s
        last_error = ""
        next_hint_at = 0.0
        self._emit_status("Waiting for llama-server to finish loading the model...")
        while time.time() < deadline:
            if self.server_proc and self.server_proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{self.server_url}/health", timeout=2) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("status") == "ok":
                    self._emit_status("Language model ready.")
                    return
            except Exception as exc:
                last_error = str(exc)
            now = time.time()
            if now >= next_hint_at:
                hint = self._startup_status_hint()
                if hint:
                    self._emit_status(hint)
                next_hint_at = now + 5.0
            time.sleep(0.5)
        if self.server_log_path and os.path.isfile(self.server_log_path):
            with open(self.server_log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:]
        else:
            tail = last_error
        raise RuntimeError(f"llama-server failed to become healthy: {tail}")

    def _startup_status_hint(self) -> str | None:
        if not self.server_log_path or not os.path.isfile(self.server_log_path):
            return None
        try:
            with open(self.server_log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:]
        except OSError:
            return None

        model_name = os.path.basename(self.cfg.model_path)
        if "load_tensors: loading model tensors" in tail:
            return "Loading model tensors into memory/GPU. Large models can take several minutes."
        if "llama_model_loader: loaded meta data" in tail:
            return f"Model metadata loaded for {model_name}. Finishing tensor load..."
        if "main: loading model" in tail:
            return f"Loading model {model_name}..."
        if "HTTP server is listening" in tail:
            return "llama-server is up. Waiting for the model to become ready..."
        return None

    def _cleanup_server(self) -> None:
        if self.server_proc and self.server_proc.poll() is None:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
                self.server_proc.wait(timeout=3)
        self.server_proc = None
        self.server_url = None
        if self.server_log_handle is not None:
            try:
                self.server_log_handle.close()
            except Exception:
                pass
            self.server_log_handle = None

    def _server_request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.server_url:
            raise RuntimeError("llama-server not initialized")
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server request failed ({exc.code}): {body}") from exc

    def _server_stream_request(self, path: str, payload: Dict[str, Any]) -> Iterator[str]:
        if not self.server_url:
            raise RuntimeError("llama-server not initialized")
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    content = chunk.get("content", "")
                    if content:
                        yield content
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server stream failed ({exc.code}): {body}") from exc

    def _server_chat_request(self, messages: List[Dict[str, str]], payload: Dict[str, Any]) -> str:
        chat_payload = dict(payload)
        chat_payload["messages"] = messages
        res = self._server_request("/chat/completions", chat_payload)
        choices = res.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""

    def _server_chat_stream_request(self, messages: List[Dict[str, str]], payload: Dict[str, Any]) -> Iterator[str]:
        chat_payload = dict(payload)
        chat_payload["messages"] = messages
        if not self.server_url:
            raise RuntimeError("llama-server not initialized")
        request = urllib.request.Request(
            f"{self.server_url}/chat/completions",
            data=json.dumps(chat_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    parsed = json.loads(data)
                    choices = parsed.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server chat stream failed ({exc.code}): {body}") from exc

    def _init_llama_server(self, prior_error: Exception | None = None):
        model_path = resolve_model_path(self.cfg.model_path, base_dir=PROJECT_ROOT)
        self.cfg.model_path = model_path
        server_bin = resolve_llama_server_executable(self.cfg)
        if not server_bin:
            if prior_error:
                raise prior_error
            raise RuntimeError("llama-server binary not found")

        self._emit_status(f"Launching llama-server for {os.path.basename(model_path)}...")
        port = self._pick_port()
        self.server_url = f"http://127.0.0.1:{port}"
        log_dir = tempfile.gettempdir()
        os.makedirs(log_dir, exist_ok=True)
        self.server_log_path = os.path.join(log_dir, f"celeste-llama-server-{port}.log")
        cmd = [
            server_bin,
            "-m",
            model_path,
            "-c",
            str(self.cfg.n_ctx),
            "-b",
            str(self.cfg.n_batch),
            "-ub",
            str(self.cfg.n_ubatch),
            "-t",
            str(self.cfg.n_threads),
            "-tb",
            str(self.cfg.n_threads),
            "-ngl",
            "999" if self.cfg.n_gpu_layers == -1 else str(self.cfg.n_gpu_layers),
            "-sm",
            self._split_mode_name(),
            "-mg",
            str(self.cfg.main_gpu),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-webui",
            "--no-warmup",
            "--jinja",
            "--reasoning-format",
            "none",
        ]
        model_name = os.path.basename(model_path).lower()
        if "mistral-nemo" in model_name:
            cmd.extend(["--chat-template", "mistral-v7-tekken"])
        if isinstance(self.cfg.tensor_split, str) and self.cfg.tensor_split.strip():
            cmd.extend(["-ts", self.cfg.tensor_split])
        if self.cfg.flash_attn:
            cmd.append("-fa")

        log_handle = open(self.server_log_path, "w", encoding="utf-8")
        self.server_log_handle = log_handle
        self.server_proc = subprocess.Popen(
            cmd,
            env=self._server_env(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        atexit.register(self._cleanup_server)
        self._wait_for_server()
        self.backend = "llama_server"

    # ---------------- transformers backend ----------------
    def _init_transformers(self):
        # Not used for your GGUF model, kept for portability.
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.cfg.dtype, torch.float16)
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.backend = "transformers"

    def shutdown(self) -> None:
        if self.backend == "llama_server":
            self._cleanup_server()

    # ---------------- utilities ----------------
    def count_tokens(self, text: str) -> int:
        try:
            if self.backend == "llama_cpp":
                return len(self.llm.tokenize(text.encode("utf-8"), add_bos=True))
            elif self.backend == "llama_server":
                payload = self._server_request("/tokenize", {"content": text})
                return len(payload.get("tokens", []))
            else:
                return len(self.tok(text, return_tensors="pt").input_ids[0])
        except Exception:
            return max(1, len(text) // 4)

    # Helper: call llama with only the kwargs this build supports
    def _llama_call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Inspect accepted parameters of __call__ for this installed version
        try:
            params = set(inspect.signature(self.llm.__call__).parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in params}
            return self.llm(prompt, **filtered)
        except TypeError:
            # Fallback: keep only the most common/stable args
            keep = ("max_tokens", "temperature", "top_p", "stop", "repeat_penalty")
            filtered = {k: v for k, v in kwargs.items() if k in keep}
            return self.llm(prompt, **filtered)

    # ---------------- generation ----------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 256,  # may be ignored if unsupported
    ) -> str:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature = 0.7 if temperature is None else temperature
        top_p = 0.9 if top_p is None else top_p
        stop = stop or ["</s>", "<|endoftext|>", "\nUser:", "\nSystem:"]

        if self.backend == "llama_cpp":
            res = self._llama_call(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n,  # harmless if filtered out
                # You can also try mirostat if supported:
                # mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1,
            )
            return res["choices"][0]["text"]
        if self.backend == "llama_server":
            payload: Dict[str, Any] = {
                "prompt": prompt,
                "n_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "repeat_penalty": repeat_penalty,
                "stream": False,
                "cache_prompt": True,
            }
            if repeat_last_n:
                payload["repeat_last_n"] = repeat_last_n
            res = self._server_request("/completion", payload)
            return res.get("content", "")

        # transformers path
        import torch
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tok.eos_token_id,
            repetition_penalty=repeat_penalty,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):]

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 256,
    ) -> Iterator[str]:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature = 0.7 if temperature is None else temperature
        top_p = 0.9 if top_p is None else top_p
        stop = stop or ["</s>", "<|endoftext|>", "\nUser:", "\nSystem:"]

        if self.backend == "llama_server":
            payload: Dict[str, Any] = {
                "prompt": prompt,
                "n_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "repeat_penalty": repeat_penalty,
                "stream": True,
                "cache_prompt": True,
            }
            if repeat_last_n:
                payload["repeat_last_n"] = repeat_last_n
            yield from self._server_stream_request("/completion", payload)
            return

        yield self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 256,
    ) -> str:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature = 0.7 if temperature is None else temperature
        top_p = 0.9 if top_p is None else top_p
        stop = stop or ["</s>", "<|endoftext|>", "\nUser:", "\nSystem:"]

        if self.backend == "llama_server":
            payload: Dict[str, Any] = {
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "repeat_penalty": repeat_penalty,
                "stream": False,
            }
            if repeat_last_n:
                payload["repeat_last_n"] = repeat_last_n
            return self._server_chat_request(messages, payload)

        prompt = "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages) + "\nAssistant:"
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 256,
    ) -> Iterator[str]:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature = 0.7 if temperature is None else temperature
        top_p = 0.9 if top_p is None else top_p
        stop = stop or ["</s>", "<|endoftext|>", "\nUser:", "\nSystem:"]

        if self.backend == "llama_server":
            payload: Dict[str, Any] = {
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "stream": True,
            }
            if repeat_last_n:
                payload["repeat_last_n"] = repeat_last_n
            yield from self._server_chat_stream_request(messages, payload)
            return

        prompt = "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages) + "\nAssistant:"
        yield from self.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
        )
