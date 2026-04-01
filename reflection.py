import json
from typing import Dict, Tuple, Optional
from model_runner import LLMRunner, AgentConfig

REFLECT_PROMPT = """[Instruction]
You are to introspect on the prior exchange. Produce three sections as JSON with keys:
- "critique": a brief, pointed critique of the assistant's answer (max 120 words).
- "improvements": a list of 3-6 short bullet rules that would improve future answers in similar tasks.
- "self_state": a short JSON object updating identity/focus/skills if needed.

[Conversation]
[User]: {user}
[Assistant]: {answer}

[Output JSON]
"""

class Reflector:
    def __init__(self, llm: LLMRunner, cfg: AgentConfig):
        self.llm = llm
        self.cfg = cfg

    def reflect(self, user: str, answer: str, prior_state: Dict) -> Tuple[Optional[str], Optional[str], Dict]:
        prompt = REFLECT_PROMPT.format(user=user, answer=answer)
        out = self.llm.generate(prompt, max_new_tokens=256).strip()
        try:
            j = json.loads(out)
            critique = j.get("critique", "").strip()
            improvements = j.get("improvements", [])
            if isinstance(improvements, list):
                improvements = "\n".join(f"- {r}" for r in improvements)
            elif not isinstance(improvements, str):
                improvements = ""
            new_state = prior_state.copy()
            if "self_state" in j and isinstance(j["self_state"], dict):
                new_state.update(j["self_state"])
            return critique, improvements, new_state
        except Exception:
            return None, None, prior_state
