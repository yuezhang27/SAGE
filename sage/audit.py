"""Lightweight audit logger for SAGE pipeline observability.

Prints structured, concise log lines so the multi-agent workflow is
auditable: who is running, what context they received, and key
paper-relevant metrics at each step.
"""

from __future__ import annotations

import os
from typing import Any

_INDENT = "  "
_TAG = "[AUDIT]"


def _truncate(text: str, max_len: int = 120) -> str:
    text = str(text).replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def log_agent_start(agent_name: str, step: int, context_keys: list[str]) -> None:
    """Called at the beginning of each agent's run()."""
    print(f"\n{_TAG} [{step}/10] >>> {agent_name} starting")
    print(f"{_TAG} {_INDENT}Context sources (ACL): {context_keys}")


def log_agent_end(agent_name: str, step: int, output_summary: str) -> None:
    """Called at the end of each agent's run()."""
    print(f"{_TAG} [{step}/10] <<< {agent_name} done | {output_summary}")


def log_detail(label: str, value: Any) -> None:
    """Print a single audit detail line."""
    print(f"{_TAG} {_INDENT}{label}: {_truncate(value)}")


def log_section(title: str) -> None:
    """Print a section header within an agent's audit block."""
    print(f"{_TAG} {_INDENT}-- {title} --")


# ---- Agent-specific helpers ------------------------------------------------

_AGENT_STEP: dict[str, int] = {
    "path_generation": 1,
    "ontologist": 2,
    "scientist": 3,
    "hypothesis_expansion": 4,
    "novelty_debate": 5,
    "explainability": 6,
    "dataset_discovery": 7,
    "coding": 8,
    "results_interpreter": 9,
    "summary": 10,
}


def step_for(agent_name: str) -> int:
    return _AGENT_STEP.get(agent_name, 0)
