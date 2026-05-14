from abc import ABC, abstractmethod
from typing import Any

from sage.memory_store import MemoryStore


class BaseAgent(ABC):
    name: str

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    @abstractmethod
    def run(self) -> dict[str, Any]:
        ...

    def read_context(self) -> dict[str, Any]:
        return self.memory.read(self.name)

    def write_output(self, output: dict[str, Any]) -> None:
        self.memory.write(self.name, output)
