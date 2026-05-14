from typing import Any


class MemoryStore:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._acl: dict[str, list[str]] = {
            "path_generation": ["__user_query__"],
            "ontologist": ["path_generation"],
            "scientist": ["ontologist"],
            "hypothesis_expansion": ["scientist", "ontologist"],
            "novelty_debate": ["hypothesis_expansion"],
            "explainability": ["hypothesis_expansion"],
            "dataset_discovery": ["hypothesis_expansion"],
            "coding": ["hypothesis_expansion", "dataset_discovery"],
            "results_interpreter": ["hypothesis_expansion", "coding", "dataset_discovery"],
            "summary": ["results_interpreter", "hypothesis_expansion", "novelty_debate"],
        }

    def write(self, agent_name: str, output: Any) -> None:
        self._store[agent_name] = output

    def read(self, agent_name: str) -> dict[str, Any]:
        allowed = self._acl.get(agent_name, [])
        return {src: self._store[src] for src in allowed if src in self._store}

    def set_user_query(self, query: str) -> None:
        self._store["__user_query__"] = query
