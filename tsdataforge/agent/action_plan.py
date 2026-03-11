from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionPlanItem:
    action_id: str
    title: str
    kind: str
    status: str
    rationale: str
    trigger: str = "always"
    command_hint: str | None = None
    target: str | None = None
    related_artifacts: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        target = f" (`{self.target}`)" if self.target else ""
        hint = f" Hint: `{self.command_hint}`." if self.command_hint else ""
        return f"- **{self.status}** `{self.action_id}` {self.title}{target}. {self.rationale}{hint}"


__all__ = ["ActionPlanItem"]
