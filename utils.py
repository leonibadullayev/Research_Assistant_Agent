import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Simple in-memory cache for API results
cache = {}

# Groq client (OpenAI-compatible)
_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Default model — fast and capable on Groq's free tier
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Retry settings
MAX_RETRIES = 4
BASE_DELAY = 5  # seconds


class LLMResponse:
    """Simple wrapper so response.content works like LangChain."""
    def __init__(self, text: str):
        self.content = text

    def __str__(self):
        return self.content


def _call_with_retry(model: str, prompt: str, temperature: float) -> str:
    """Call Groq API with retry on rate-limit errors."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "503" in err_str:
                delay = BASE_DELAY * (2 ** attempt)
                print(
                    f"  Rate limited (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Waiting {delay:.0f}s..."
                )
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"All retries exhausted. Last error: {last_error}")


class GroqLLM:
    """A wrapper for Groq's OpenAI-compatible API with retry."""

    def __init__(self, model=DEFAULT_MODEL, temperature=0.2):
        self.model_name = model
        self.temperature = temperature

    def invoke(self, prompt: str) -> LLMResponse:
        if not isinstance(prompt, str):
            prompt = str(prompt)
        text = _call_with_retry(self.model_name, prompt, self.temperature)
        return LLMResponse(text)


def get_llm(temperature=0.2, model=DEFAULT_MODEL):
    """Return a Groq LLM instance."""
    return GroqLLM(model=model, temperature=temperature)


def get_llm_for_planning():
    """Slightly higher temperature for planning creativity."""
    return get_llm(temperature=0.4)


def clear_cache():
    cache.clear()