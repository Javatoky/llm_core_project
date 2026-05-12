"""Agent 数据结构"""

from dataclasses import dataclass


@dataclass
class AgentResult:
    answer: str
    called_tools: list[str]
