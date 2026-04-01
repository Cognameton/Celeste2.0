import os

class Playbook:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("# Agent Playbook (Self-Improvement Notes)\n\n")

    def read(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def update(self, improvements_markdown: str):
        if not improvements_markdown or not improvements_markdown.strip():
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n## Update\n")
            f.write(improvements_markdown.strip() + "\n")
