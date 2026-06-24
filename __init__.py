"""TaskAgent-Flow: a lightweight micro-framework for multi-step LLM task
automation using a local Ollama model, with a built-in self-healing
fallback loop for failed steps.
"""

from agent_flow.engine import TaskEngine, TaskExecutionError
from agent_flow.llm_client import OllamaClient
from agent_flow.models import StepDefinition, StepResult, TaskDefinition

__version__ = "0.1.0"

__all__ = [
    "TaskEngine",
    "TaskExecutionError",
    "OllamaClient",
    "StepDefinition",
    "StepResult",
    "TaskDefinition",
]
