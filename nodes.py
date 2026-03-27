from state import AgentState
from utils import get_llm, get_llm_for_planning
from tools import search_arxiv


def planner(state: AgentState) -> AgentState:
    """Generate a plan based on user input."""
    llm = get_llm_for_planning()
    prompt = f"""You are a research planning assistant.
Given the user's request, produce a numbered list of steps to follow.
Each step should be a concrete action like:
1. Search for papers on topic X
2. Summarize key findings
3. Identify open questions
4. Write literature review sections
Only output the list, no extra text.

User request: {state['user_input']}"""

    response = llm.invoke(prompt)
    # Parse numbered list
    plan = []
    for line in response.content.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            plan.append(line)
    state["plan"] = plan
    return state


def executor(state: AgentState) -> AgentState:
    """Execute the plan steps, using tools where needed."""
    results = []
    for step in state["plan"]:
        if "search" in step.lower():
            # Extract query from step text
            query = step.split("search")[-1].strip().strip(":")
            if not query or len(query) < 3:
                query = state["user_input"]
            papers = search_arxiv(query)
            results.append({"step": step, "type": "search", "data": papers})
        elif "pdf" in step.lower() or "fetch" in step.lower():
            results.append({"step": step, "type": "pdf", "data": "PDF fetching not implemented in this demo"})
        elif "zotero" in step.lower():
            results.append({"step": step, "type": "zotero", "data": "Zotero addition not implemented in this demo"})
        else:
            results.append({"step": step, "type": "pending", "data": None})
    state["execution_results"] = results
    return state


def composer(state: AgentState) -> AgentState:
    """Compose an initial draft from execution results."""
    context = ""
    for res in state["execution_results"]:
        if res["type"] == "search" and isinstance(res["data"], list):
            context += f"Papers found for step '{res['step']}':\n"
            for paper in res["data"][:3]:
                context += f"- {paper['title']} ({', '.join(paper['authors'])})\n"
                context += f"  Abstract: {paper['abstract'][:300]}...\n\n"

    llm = get_llm()
    prompt = f"""You are a research assistant. Based on the following search results,
write a concise literature review section addressing the user's request.
Use proper academic style.

User request: {state['user_input']}

Search results:
{context}"""

    draft = llm.invoke(prompt)
    state["draft"] = draft.content
    return state


def reflector(state: AgentState) -> AgentState:
    """Critique the current draft."""
    llm = get_llm()
    prompt = f"""You are a critical reviewer. Evaluate the following draft and provide constructive feedback.
Focus on:
- Missing key papers or themes
- Lack of critical analysis
- Structure and flow
- Areas needing more detail
Keep the feedback concise and actionable.

Draft:
{state['draft']}"""

    critique = llm.invoke(prompt)
    state["reflection"] = critique.content
    return state


def refiner(state: AgentState) -> AgentState:
    """Improve the draft based on the reflection."""
    llm = get_llm()
    prompt = f"""You are a researcher. Revise the following draft to address the reviewer's feedback.
Improve clarity, add missing elements, and enhance the academic tone.

Original draft:
{state['draft']}

Reviewer feedback:
{state['reflection']}

Improved draft:"""

    improved = llm.invoke(prompt)
    state["draft"] = improved.content
    state["iteration"] += 1
    return state


def should_continue(state: AgentState) -> str:
    """Decide whether to refine again or finish."""
    if state["iteration"] < state["max_refinements"]:
        return "reflector"
    return "end"