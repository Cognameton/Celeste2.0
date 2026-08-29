"""Executor — sandboxed tool registry for Celeste's ReAct loop.

Tools are registered with a risk level:
  safe    — auto-execute (read-only, bounded output)
  confirm — routed through confirm_cb before executing
  blocked — never runs

The confirm_cb defaults to auto-approve. Wire a GUI callback to surface
confirmation prompts to the user.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


MAX_READ_BYTES   = 32_000
MAX_OUTPUT_BYTES =  8_000
TIMEOUT_S        = 30


@dataclass
class ToolResult:
    tool: str
    output: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, str]        # name → description
    risk: str                   # "safe" | "confirm" | "blocked"
    handler: Callable[..., str]


class Executor:
    def __init__(
        self,
        *,
        confirm_cb: Callable[[str, dict[str, Any]], bool] | None = None,
        working_dir: Path | str | None = None,
    ):
        # Default: auto-approve all confirmable tools.
        # Pass a real callback to surface confirmation to the GUI.
        self.confirm_cb = confirm_cb or (lambda tool, args: True)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    # ---- registry ----

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return [t.name for t in self._tools.values() if t.risk != "blocked"]

    def for_prompt(self) -> str:
        lines = [
            "Call a tool by including this exact line in your response (no other text on that line):",
            "",
            '  TOOL_CALL: {"tool": "tool_name", "args": {"arg": "value"}}',
            "",
            "Available tools:",
        ]
        for t in self._tools.values():
            if t.risk == "blocked":
                continue
            args_str = ", ".join(f"{k}: {v}" for k, v in t.args.items())
            note = " [requires confirmation]" if t.risk == "confirm" else ""
            lines.append(f"- {t.name}({args_str}): {t.description}{note}")
        lines += [
            "",
            "Use tools only when you genuinely need information you don't already have.",
            "After receiving a [Tool: name] result, continue reasoning or provide your answer.",
        ]
        return "\n".join(lines)

    # ---- execution ----

    def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(tool_name, "", f"Unknown tool: {tool_name!r}")
        if tool.risk == "blocked":
            return ToolResult(tool_name, "", f"Tool {tool_name!r} is blocked")
        if tool.risk == "confirm" and not self.confirm_cb(tool_name, args):
            return ToolResult(tool_name, "", f"Tool {tool_name!r} was not confirmed")
        try:
            output = tool.handler(**{k: str(v) for k, v in args.items()})
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + f"\n[truncated at {MAX_OUTPUT_BYTES} bytes]"
            return ToolResult(tool_name, output)
        except Exception as exc:
            return ToolResult(tool_name, "", f"Tool error: {exc}")

    # ---- default tools ----

    def _register_defaults(self) -> None:
        self.register(Tool(
            "read_file",
            "Read a file's text contents",
            {"path": "absolute or relative file path"},
            "safe",
            self._read_file,
        ))
        self.register(Tool(
            "list_dir",
            "List the contents of a directory",
            {"path": "directory path"},
            "safe",
            self._list_dir,
        ))
        self.register(Tool(
            "grep",
            "Search for a pattern in files",
            {"pattern": "regex pattern", "path": "file or directory", "recursive": "true or false (default false)"},
            "safe",
            self._grep,
        ))
        self.register(Tool(
            "run_python",
            "Execute a Python snippet in a subprocess and return stdout/stderr",
            {"code": "Python code to execute"},
            "confirm",
            self._run_python,
        ))
        self.register(Tool(
            "web_fetch",
            "Fetch the text content of a URL (HTML tags stripped)",
            {"url": "https:// URL to fetch"},
            "safe",
            self._web_fetch,
        ))

    def _read_file(self, path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.working_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Not found: {path}")
        size = p.stat().st_size
        with p.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_READ_BYTES)
        if size > MAX_READ_BYTES:
            content += f"\n[file truncated — {size - MAX_READ_BYTES} bytes not shown]"
        return content

    def _list_dir(self, path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.working_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Not found: {path}")
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"[dir]  {e.name}/")
            else:
                lines.append(f"[file] {e.name}  ({e.stat().st_size:,} bytes)")
        return "\n".join(lines) if lines else "(empty)"

    def _grep(self, pattern: str, path: str, recursive: str = "false") -> str:
        p = Path(path)
        if not p.is_absolute():
            p = self.working_dir / p
        flags = ["-rn"] if recursive.lower() == "true" else ["-n"]
        result = subprocess.run(
            ["grep"] + flags + ["-E", "--include=*.py", "--include=*.md",
                                "--include=*.txt", "--include=*.yaml",
                                pattern, str(p)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        out = result.stdout
        if not out:
            return "(no matches)"
        lines = out.splitlines()
        if len(lines) > 80:
            return "\n".join(lines[:80]) + f"\n[{len(lines) - 80} more lines not shown]"
        return out

    def _run_python(self, code: str) -> str:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=TIMEOUT_S,
            cwd=str(self.working_dir),
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts) if parts else "(no output)"

    def _web_fetch(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http:// or https://")
        req = urllib.request.Request(url, headers={"User-Agent": "Synthia/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                raw = resp.read(MAX_READ_BYTES).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.reason}")
        # Strip HTML tags and collapse whitespace
        text = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
