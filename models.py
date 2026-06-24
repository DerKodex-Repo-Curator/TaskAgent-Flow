from pydantic import BaseModel
from typing import List, Optional

class StepDefinition(BaseModel):
    id: int
    action: str
    expected_output: str

class TaskDefinition(BaseModel):
    task_name: str
    model: Optional[str] = None
    steps: List[StepDefinition]

class StepResult(BaseModel):
    step_id: int
    attempts: int
    output: str