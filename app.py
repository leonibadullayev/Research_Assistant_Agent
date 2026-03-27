import streamlit as st
import os
from dotenv import load_dotenv
from graph import build_graph
from state import AgentState

load_dotenv()

st.set_page_config(page_title="Research Assistant Agent", layout="wide")
st.title("📚 Research Assistant Agent")
st.markdown("An AI agent that plans, retrieves papers, composes drafts, and refines based on critique.")

with st.sidebar:
    st.header("Settings")
    max_refinements = st.slider("Number of refinement loops", 1, 5, 2)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This agent uses:")
    st.markdown("- **Planning**: Step-by-step plan decomposition")
    st.markdown("- **Retrieval**: arXiv search (dynamic)")
    st.markdown("- **Composition**: Draft generation")
    st.markdown("- **Reflexion**: Critique and refinement")
    st.markdown("---")
    st.markdown("Built with LangGraph, Groq (Llama 3.3 70B), and Streamlit")

user_input = st.text_area(
    "Enter your research request:",
    height=150,
    placeholder="E.g., Draft a literature review on transformer-based time series forecasting. Include recent papers from arXiv.",
)

if st.button("Run Agent", type="primary"):
    if not user_input:
        st.warning("Please enter a research request.")
        st.stop()
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Get a free key at https://console.groq.com and add it to your .env file.")
        st.stop()

    initial_state: AgentState = {
        "messages": [],
        "user_input": user_input,
        "plan": [],
        "execution_results": [],
        "draft": "",
        "reflection": "",
        "iteration": 0,
        "max_refinements": max_refinements,
    }

    app = build_graph(max_refinements=max_refinements)

    with st.spinner("Agent is working... (may retry on rate limits)"):
        try:
            final_state = app.invoke(initial_state)
        except RuntimeError as e:
            if "exhausted" in str(e).lower():
                st.error(
                    "⚠️ All Gemini API quotas are exhausted. "
                    "Please wait a few minutes or upgrade your Google AI plan, then try again."
                )
                st.stop()
            raise
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.error(
                    "⚠️ Gemini API rate limit hit. "
                    "The agent retried automatically but quota is still exhausted. "
                    "Please wait a few minutes and try again."
                )
                st.stop()
            raise

    st.subheader("Final Draft")
    st.markdown(final_state["draft"])
    with st.expander("See reflection (last critique)"):
        st.markdown(final_state["reflection"])
    with st.expander("Execution details"):
        st.json(final_state["execution_results"])
        st.write("Plan:", final_state["plan"])