from pydantic import BaseModel
from typing import List, Optional

class PlanStep(BaseModel):
    step_number: int
    action: str          # e.g., "search", "summarize", "write"
    description: str     # detailed description
    tool: Optional[str] = None  # optional tool name

class Plan(BaseModel):
    steps: List[PlanStep]