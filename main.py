import os
import sys
from dotenv import load_dotenv
from graph import build_graph
from state import AgentState

load_dotenv()

def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set in environment or .env file.")
        print("Get a free key at https://console.groq.com")
        sys.exit(1)

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("Enter your research request: ").strip()
        if not user_input:
            user_input = "Draft a literature review on transformer-based time series forecasting. Include recent papers from arXiv."

    initial_state: AgentState = {
        "messages": [],
        "user_input": user_input,
        "plan": [],
        "execution_results": [],
        "draft": "",
        "reflection": "",
        "iteration": 0,
        "max_refinements": 2
    }

    app = build_graph(max_refinements=2)
    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL DRAFT:")
    print("="*50)
    print(final_state["draft"])
    print("\n" + "="*50)
    if final_state["reflection"]:
        print("Final Reflection (last critique):")
        print(final_state["reflection"])

if __name__ == "__main__":
    main()