"""ProjectStore — persistent project tracker for Celeste 2.0.

Projects live in self/projects/<slug>.md as markdown files with YAML frontmatter.
Active projects are injected into the system prompt so Celeste maintains
awareness of ongoing work as a research partner.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Project:
    slug: str
    name: str
    status: str      # "active" | "paused" | "done"
    goal: str
    description: str
    created_at: str
    updated_at: str
    body: str        # full raw file content


def build_project_content(
    name: str,
    status: str,
    goal: str,
    description: str,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> str:
    now = _now()
    front = (
        f"---\n"
        f"name: {name}\n"
        f"status: {status}\n"
        f"goal: {goal}\n"
        f"created_at: {created_at or now}\n"
        f"updated_at: {updated_at or now}\n"
        f"---\n\n"
    )
    return front + description.strip() + "\n\n## Notes\n"


def append_note_to_content(content: str, note: str) -> str:
    stamp = _now()
    content = re.sub(
        r"^(updated_at:\s*).*$",
        f"updated_at: {stamp}",
        content,
        flags=re.MULTILINE,
    )
    note_block = f"\n### {stamp}\n{note.strip()}\n"
    if "## Notes" in content:
        return content.rstrip() + note_block
    return content.rstrip() + "\n\n## Notes\n" + note_block


def set_status_in_content(content: str, status: str) -> str:
    stamp = _now()
    content = re.sub(r"^(status:\s*).*$", f"status: {status}", content, flags=re.MULTILINE)
    content = re.sub(r"^(updated_at:\s*).*$", f"updated_at: {stamp}", content, flags=re.MULTILINE)
    return content


class ProjectStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- read ----

    def load(self) -> list[Project]:
        projects = []
        for path in sorted(self.root.glob("*.md")):
            project = self._parse_project(path)
            if project is not None:
                projects.append(project)
        return projects

    def active(self) -> list[Project]:
        return [p for p in self.load() if p.status == "active"]

    def get(self, slug: str) -> Project | None:
        path = self.root / f"{slug}.md"
        return self._parse_project(path) if path.exists() else None

    def exists(self, slug: str) -> bool:
        return (self.root / f"{slug}.md").exists()

    def for_prompt(self) -> str:
        active = self.active()
        if not active:
            return ""
        lines = []
        for p in active:
            lines.append(f"• {p.slug} — {p.name}")
            if p.goal:
                lines.append(f"  Goal: {p.goal}")
        return "Active projects:\n" + "\n".join(lines)

    # ---- parse ----

    def _parse_project(self, path: Path) -> Project | None:
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        slug = path.stem
        front, body = _split_frontmatter(text)
        meta = _parse_yaml(front)
        return Project(
            slug=slug,
            name=str(meta.get("name", slug)).strip(),
            status=str(meta.get("status", "active")).strip(),
            goal=str(meta.get("goal", "")).strip(),
            description=body.strip(),
            created_at=str(meta.get("created_at", "")).strip(),
            updated_at=str(meta.get("updated_at", "")).strip(),
            body=text,
        )

    # ---- write ----

    def create(
        self,
        name: str,
        goal: str,
        description: str,
        status: str = "active",
    ) -> Project:
        slug = self._name_to_slug(name)
        base, i = slug, 2
        while self.exists(slug):
            slug = f"{base}-{i}"
            i += 1
        path = self.root / f"{slug}.md"
        path.write_text(build_project_content(name, status, goal, description), encoding="utf-8")
        _git_add_commit(self.root.parent, f"projects/{slug}.md", f"Add project: {name}")
        return self._parse_project(path)  # type: ignore[return-value]

    def add_note(self, slug: str, note: str) -> bool:
        path = self.root / f"{slug}.md"
        if not path.exists():
            return False
        content = append_note_to_content(path.read_text(encoding="utf-8"), note)
        path.write_text(content, encoding="utf-8")
        _git_add_commit(self.root.parent, f"projects/{slug}.md", f"Project {slug}: note added")
        return True

    def set_status(self, slug: str, status: str) -> bool:
        path = self.root / f"{slug}.md"
        if not path.exists():
            return False
        content = set_status_in_content(path.read_text(encoding="utf-8"), status)
        path.write_text(content, encoding="utf-8")
        _git_add_commit(self.root.parent, f"projects/{slug}.md", f"Project {slug}: status → {status}")
        return True

    def delete(self, slug: str) -> bool:
        path = self.root / f"{slug}.md"
        if not path.exists():
            return False
        _git_rm_commit(self.root.parent, f"projects/{slug}.md", f"Delete project: {slug}")
        return True

    @staticmethod
    def _name_to_slug(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return slug or "project"

    @staticmethod
    def valid_slug(slug: str) -> bool:
        return bool(re.match(r"^[a-z0-9][a-z0-9-]*$", slug))


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    end = text.find("---", 3)
    if end == -1:
        return "", text
    front = text[3:end].strip()
    body = text[end + 3:].strip()
    return front, body


def _git_add_commit(git_root: Path, rel_path: str, message: str) -> None:
    subprocess.run(["git", "add", "--", rel_path], cwd=git_root, check=False)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=git_root, check=False)


def _git_rm_commit(git_root: Path, rel_path: str, message: str) -> None:
    subprocess.run(["git", "rm", "-q", "--force", "--", rel_path], cwd=git_root, check=False)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=git_root, check=False)


def _parse_yaml(front: str) -> dict[str, Any]:
    if not front:
        return {}
    if _YAML_AVAILABLE:
        try:
            result = _yaml.safe_load(front)
            return result if isinstance(result, dict) else {}
        except Exception:
            pass
    meta: dict[str, Any] = {}
    for line in front.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta
