"""SkillsStore — reads and manages Celeste's skill library.

Skills live in self/skills/<slug>/SKILL.md. Each file has YAML frontmatter
with name, description, when_to_use, and status fields, followed by optional
body content.

Skills are part of self-state — tracked in the self/ git repo. The heartbeat
can propose new skill stubs (status: draft); the operator activates them by
changing status to active.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_VALID_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass
class Skill:
    slug: str
    name: str
    description: str
    when_to_use: str = ""
    status: str = "active"
    body: str = ""


class SkillsStore:
    """Read-only view of the skills directory. Write operations go through SelfState."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def load(self) -> list[Skill]:
        if not self.root.exists():
            return []
        skills: list[Skill] = []
        for skill_dir in sorted(self.root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            skill = _parse_skill(skill_dir.name, skill_file.read_text(encoding="utf-8"))
            if skill is not None:
                skills.append(skill)
        return skills

    def for_prompt(self, include_drafts: bool = False) -> str:
        skills = [
            s for s in self.load()
            if s.status == "active" or (include_drafts and s.status == "draft")
        ]
        if not skills:
            return ""
        lines = []
        for s in skills:
            line = f"- **{s.name}**: {s.description}"
            if s.when_to_use:
                line += f" — when: {s.when_to_use}"
            lines.append(line)
        return "\n".join(lines)

    def exists(self, slug: str) -> bool:
        return (self.root / slug / "SKILL.md").exists()

    @staticmethod
    def valid_slug(slug: str) -> bool:
        return bool(_VALID_SLUG_RE.match(slug)) and ".." not in slug


def _parse_skill(slug: str, text: str) -> Skill | None:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    try:
        fm = _parse_flat_yaml(m.group(1))
    except Exception:
        return None
    name = str(fm.get("name", slug)).strip()
    if not name:
        return None
    body = text[m.end():].strip()
    return Skill(
        slug=slug,
        name=name,
        description=str(fm.get("description", "")).strip(),
        when_to_use=str(fm.get("when_to_use", "")).strip(),
        status=str(fm.get("status", "active")).strip().lower(),
        body=body,
    )


def _parse_flat_yaml(text: str) -> dict[str, str]:
    """Parse flat key: value YAML (no nesting, no lists)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        result[key.strip()] = val.strip().strip("\"'")
    return result


def build_skill_content(
    name: str,
    description: str,
    when_to_use: str,
    *,
    status: str = "draft",
    note: str = "",
) -> str:
    """Build the content of a new SKILL.md file."""
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        f"when_to_use: {when_to_use}",
        f"status: {status}",
        "---",
        "",
        f"## {name}",
        "",
        description,
    ]
    if note:
        lines += ["", f"_{note}_"]
    return "\n".join(lines) + "\n"
