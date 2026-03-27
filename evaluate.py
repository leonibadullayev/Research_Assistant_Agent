import time
import json
from dotenv import load_dotenv
from graph import build_graph
from state import AgentState
from utils import get_llm

load_dotenv()

TEST_QUERIES = [
    "Draft a literature review on contrastive learning in computer vision.",
    "Summarize recent advances in LLM agents for scientific research.",
    "What are the main challenges in applying transformers to time series forecasting? Provide a structured overview.",
]


def evaluate_query(query, max_refinements=2):
    initial_state: AgentState = {
        "messages": [],
        "user_input": query,
        "plan": [],
        "execution_results": [],
        "draft": "",
        "reflection": "",
        "iteration": 0,
        "max_refinements": max_refinements,
    }
    app = build_graph(max_refinements=max_refinements)
    start = time.time()
    final_state = app.invoke(initial_state)
    elapsed = time.time() - start

    # Quality rating using another LLM call
    llm = get_llm()
    rating_prompt = f"""Rate the following literature review draft on a scale of 1-5 based on:
    - Relevance to query
    - Depth of analysis
    - Use of recent papers
    - Structure and clarity

    Query: {query}

    Draft:
    {final_state['draft']}

    Provide only the number (1-5)."""
    rating = llm.invoke(rating_prompt).content.strip()
    try:
        rating = int(rating)
    except ValueError:
        rating = None

    return {
        "query": query,
        "elapsed": elapsed,
        "iterations": final_state["iteration"],
        "rating": rating,
        "draft_length": len(final_state["draft"]),
    }


def main():
    results = []
    for q in TEST_QUERIES:
        print(f"Evaluating: {q}")
        res = evaluate_query(q)
        results.append(res)
        print(f"  Time: {res['elapsed']:.2f}s, Iterations: {res['iterations']}, Rating: {res['rating']}")
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to eval_results.json")


if __name__ == "__main__":
    main()