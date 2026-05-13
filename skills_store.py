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
import subprocess
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
    confidence: float = 1.0     # 0.0–1.0; starts at 1.0 (benefit of the doubt)
    sample_count: int = 0       # number of outcome samples recorded
    body: str = ""


class SkillsStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    # ---- read ----

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

    def get(self, slug: str) -> Skill | None:
        path = self.root / slug / "SKILL.md"
        if not path.exists():
            return None
        return _parse_skill(slug, path.read_text(encoding="utf-8"))

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

    # ---- write ----

    def create(self, slug: str, content: str, *, message: str = "") -> None:
        skill_dir = self.root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        _git_add_commit(
            self.root.parent,
            f"skills/{slug}/SKILL.md",
            message or f"Add skill: {slug}",
        )

    def promote(self, slug: str) -> bool:
        """Promote a draft skill to active. Returns True if status changed."""
        path = self.root / slug / "SKILL.md"
        if not path.exists():
            return False
        original = path.read_text(encoding="utf-8")
        updated = re.sub(
            r"^(status:\s*)draft\s*$", r"\g<1>active", original, flags=re.MULTILINE
        )
        if updated == original:
            return False
        path.write_text(updated, encoding="utf-8")
        _git_add_commit(self.root.parent, f"skills/{slug}/SKILL.md", f"Promote skill to active: {slug}")
        return True

    def deprecate(self, slug: str) -> bool:
        """Mark a skill as deprecated."""
        path = self.root / slug / "SKILL.md"
        if not path.exists():
            return False
        original = path.read_text(encoding="utf-8")
        updated = re.sub(
            r"^(status:\s*)\S+\s*$", r"\g<1>deprecated", original, flags=re.MULTILINE
        )
        if updated == original:
            return False
        path.write_text(updated, encoding="utf-8")
        _git_add_commit(self.root.parent, f"skills/{slug}/SKILL.md", f"Deprecate skill: {slug}")
        return True

    def update_confidence(self, slug: str, outcome_value: float) -> None:
        """Record one outcome sample and rewrite confidence/sample_count in frontmatter."""
        path = self.root / slug / "SKILL.md"
        if not path.exists():
            return
        skill = self.get(slug)
        if skill is None or skill.status not in ("active", "draft"):
            return
        # Windowed average — window capped at 20 so recent turns count more
        window = min(skill.sample_count, 19)
        new_conf = round((skill.confidence * window + outcome_value) / (window + 1), 3)
        new_count = skill.sample_count + 1
        content = path.read_text(encoding="utf-8")
        content = _set_frontmatter_field(content, "confidence", str(new_conf))
        content = _set_frontmatter_field(content, "sample_count", str(new_count))
        # Auto-flag underperforming skills for review
        if new_count >= 10 and new_conf < 0.3:
            content = _set_frontmatter_field(content, "needs_review", "true")
        path.write_text(content, encoding="utf-8")
        _git_add_commit(
            self.root.parent,
            f"skills/{slug}/SKILL.md",
            f"Skill confidence update: {slug} → {new_conf:.2f} (n={new_count})",
        )


def _git_add_commit(git_root: Path, rel_path: str, message: str) -> None:
    subprocess.run(["git", "add", "--", rel_path], cwd=git_root, check=False)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=git_root, check=False)


def _set_frontmatter_field(content: str, key: str, value: str) -> str:
    """Set a key in YAML frontmatter, adding it if absent."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return content
    fm = m.group(1)
    body_after = content[m.end():]
    pattern = re.compile(r"^(" + re.escape(key) + r":\s*).*$", re.MULTILINE)
    new_fm, n = pattern.subn(f"{key}: {value}", fm)
    if n == 0:
        new_fm = fm.rstrip() + f"\n{key}: {value}"
    return f"---\n{new_fm}\n---\n{body_after}"


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
    try:
        confidence = float(fm.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    try:
        sample_count = int(fm.get("sample_count", 0))
    except (TypeError, ValueError):
        sample_count = 0
    return Skill(
        slug=slug,
        name=name,
        description=str(fm.get("description", "")).strip(),
        when_to_use=str(fm.get("when_to_use", "")).strip(),
        status=str(fm.get("status", "active")).strip().lower(),
        confidence=confidence,
        sample_count=sample_count,
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
