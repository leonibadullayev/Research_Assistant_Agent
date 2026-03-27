from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import planner, executor, composer, reflector, refiner, should_continue


def build_graph(max_refinements: int = 2):
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("composer", composer)
    workflow.add_node("reflector", reflector)
    workflow.add_node("refiner", refiner)

    # Set entry point
    workflow.set_entry_point("planner")

    # Define edges
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "composer")
    workflow.add_edge("composer", "reflector")
    workflow.add_edge("reflector", "refiner")
    workflow.add_conditional_edges(
        "refiner",
        should_continue,
        {"reflector": "reflector", "end": END},
    )

    return workflow.compile()