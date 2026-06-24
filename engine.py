import yaml
import json
import os
from typing import List
from agent_flow.models import TaskDefinition, StepResult, StepDefinition
from agent_flow.llm_client import OllamaClient
from agent_flow.validators import validate_output

DEFAULT_MAX_RETRIES = 3


class TaskExecutionError(Exception):
    pass


class TaskEngine:
    def __init__(self, task_path: str, model: str = None, host: str = None, max_retries: int = DEFAULT_MAX_RETRIES,
                 verbose: bool = True):
        self.task_path = task_path
        self.max_retries = max_retries
        self.verbose = verbose
        self.task = self._load_task()

        chosen_model = model or self.task.model
        self.client = OllamaClient(host=host, model=chosen_model)
        self.context_history = ""

    def _load_task(self) -> TaskDefinition:
        if not os.path.exists(self.task_path):
            raise FileNotFoundError(f"Task definition file not found: {self.task_path}")

        with open(self.task_path, 'r', encoding='utf-8') as f:
            if self.task_path.endswith('.json'):
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        return TaskDefinition(**data)

    def run(self) -> List[StepResult]:
        results = []

        for step in self.task.steps:
            attempts = 0
            success = False
            step_output = ""
            failure_reason = ""

            while attempts < self.max_retries and not success:
                attempts += 1

                prompt = f"You are an expert AI agent executing a multi-step task workflow.\n"
                if self.context_history:
                    prompt += f"Here is the context and results from your previous steps:\n{self.context_history}\n"

                prompt += f"Your current instruction (Step {step.id}): {step.action}\n"
                prompt += f"Required explicit output format: {step.expected_output}\n"

                if failure_reason:
                    prompt += f"\n⚠️ ATTENTION: Your previous attempt failed validation because: {failure_reason}. Please rewrite your answer to completely fix this issue."

                try:
                    step_output = self.client.generate(prompt)
                    is_valid, failure_reason = validate_output(step_output)
                    if is_valid:
                        success = True
                except Exception as e:
                    failure_reason = f"Network or execution error: {str(e)}"

            if not success:
                raise TaskExecutionError(
                    f"Step {step.id} ('{step.action}') failed validation after {self.max_retries} attempts. Reason: {failure_reason}")

            self.context_history += f"\n[Step {step.id} Result]:\n{step_output}\n"
            results.append(StepResult(step_id=step.id, attempts=attempts, output=step_output))

        return results