"""Persistent self-state for Synthia.

The self-state lives on the filesystem as a directory of markdown files.
Each file represents a different facet of the agent:

    IDENTITY.md   anchor — name, lineage, operator, purpose. Operator-only.
    SOUL.md       persona, voice, values. Slowly editable.
    AGENTS.md     operating procedure. Reflection and heartbeat write here.
    USER.md       what is known about the operator. Reflection writes here.
    TOOLS.md      tool conventions.

The directory is its own git repository so every change to self-state is
auditable and revertible. On first run the directory is bootstrapped from
``self_template/`` and an initial "genesis" commit is made.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app_paths import default_self_path, resource_path


SELF_FILES = ("IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md", "TOOLS.md")
IMMUTABLE_BY_AGENT = frozenset({"IDENTITY.md"})
BOOTSTRAP_FILE = "BOOTSTRAP.md"

_GIT_AUTHOR = ("user.name=Synthia", "user.email=synthia@local")


@dataclass
class SelfState:
    root: Path

    @classmethod
    def initialize(cls, root: Path | str | None = None,
                   template_dir: Path | str | None = None) -> "SelfState":
        root_path = Path(root) if root else Path(default_self_path())
        template_path = Path(template_dir) if template_dir else Path(resource_path("self_template"))

        if not root_path.exists():
            if not template_path.exists():
                raise FileNotFoundError(
                    f"Self template not found at {template_path}. "
                    "Cannot bootstrap self-state without a template."
                )
            shutil.copytree(template_path, root_path)
            _git_init(root_path)

        return cls(root=root_path)

    # ---- read ----

    def read(self, name: str) -> str:
        path = self.root / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @property
    def name(self) -> str:
        """Parse the agent's name from IDENTITY.md ('I am X.') with fallback."""
        for line in self.read("IDENTITY.md").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("i am "):
                candidate = stripped[5:].rstrip(" .!?\n\t\"'")
                if candidate:
                    return candidate
        return "Synthia"

    def all_for_prompt(self, files: Iterable[str] = SELF_FILES) -> str:
        sections = []
        for name in files:
            text = self.read(name).strip()
            if text:
                sections.append(f"=== {name} ===\n{text}")
        return "\n\n".join(sections)

    # ---- write ----

    def write(self, name: str, content: str, *, message: str, by_agent: bool = True) -> None:
        if by_agent and name in IMMUTABLE_BY_AGENT:
            raise PermissionError(
                f"{name} is immutable to the agent. Operator must edit directly."
            )
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        _git_commit(self.root, [name], message)

    def append_section(self, name: str, heading: str, body: str, *, message: str) -> None:
        existing = self.read(name)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        section = f"\n\n### {heading}\n_{stamp}_\n\n{body.rstrip()}\n"
        self.write(name, existing + section, message=message)

    def write_skill(self, slug: str, content: str, *, message: str) -> None:
        """Create or update a skill at self/skills/<slug>/SKILL.md."""
        skill_dir = self.root / "skills" / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        rel = f"skills/{slug}/SKILL.md"
        _git_commit(self.root, [rel], message)

    # ---- bootstrap ----

    def has_bootstrap(self) -> bool:
        return (self.root / BOOTSTRAP_FILE).exists()

    def consume_bootstrap(self) -> str:
        path = self.root / BOOTSTRAP_FILE
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
        path.unlink()
        _git_commit(self.root, [BOOTSTRAP_FILE], "First awakening complete; bootstrap consumed.")
        return text

    # ---- audit ----

    def history(self, limit: int = 20) -> list[tuple[str, str, str]]:
        result = subprocess.run(
            ["git", "log", f"-n{limit}", "--pretty=format:%h\t%cI\t%s"],
            cwd=self.root, capture_output=True, text=True, check=False,
        )
        rows = []
        for line in result.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                rows.append((parts[0], parts[1], parts[2]))
        return rows


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    _commit(root, "Genesis: self-state initialized from template.")


def _git_commit(root: Path, files: list[str], message: str) -> None:
    subprocess.run(["git", "add", "--"] + files, cwd=root, check=True)
    _commit(root, message, allow_empty=False)


def _commit(root: Path, message: str, allow_empty: bool = False) -> None:
    cmd = ["git"]
    for kv in _GIT_AUTHOR:
        cmd += ["-c", kv]
    cmd += ["commit", "-q", "-m", message]
    if allow_empty:
        cmd.append("--allow-empty")
    subprocess.run(cmd, cwd=root, check=False)
