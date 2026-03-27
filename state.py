from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """State of the agent throughout the workflow."""
    messages: List[Dict[str, str]]          # conversation history
    user_input: str                         # original user request
    plan: List[str]                         # list of steps to execute (string steps)
    execution_results: List[Dict[str, Any]] # results from each tool call
    draft: str                              # current version of the output
    reflection: str                         # critique of the draft
    iteration: int                          # current refinement iteration
    max_refinements: int                    # stop after this many loops